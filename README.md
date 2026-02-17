# pyrator: System Design and Architecture
**Version:** 1.0

## 1. Introduction & Vision

**pyrator** analyzes inter-annotator agreement (IAA/IRA) when labels live in **ontologies** (trees/DAGs). It supports:

- **Classical, chance-corrected reliability** (κ family, ICC, Krippendorff’s α), including **semantic** extensions that account for **distance in the ontology** (partial credit).
- **Unsupervised triage** via **disagreement feature vectors** and scalable clustering (no $n^2$ matrices).
- **Bayesian pooling** (hierarchical models) to infer **item difficulty**, **annotator reliability/bias**, and **consensus** labels with **uncertainty**.
- Scalability from small guideline sets (e.g., MCG-like) to large ontologies (ICD/UMLS) via **strategy switching** (exact discrete, pruned discrete, embedding).

**Non-expert intuition:** instead of treating all disagreements as equally bad, pyrator uses the *structure* of the ontology to measure “how close” annotators are, and uses generative models to decide whether disagreement is because the **item is hard** or the **annotator tends to be off**.

---

## 2. Core Abstractions & Data Models

### 2.1 Ontology

**Role:** fast structural queries + semantic distances.

**Graph:** directed acyclic graph (DAG) or tree. Cache:
- $depth(node) ∈ ℕ$ (root depth = 0)
- $parents(node)$ (1 in trees, ≥ 1 in DAGs)
- $ancestors(node)$ (including node)
- $root\_path(node)$ (for LCA depth computations)
- $IC(node)= -\log \hat p(node)$ (see IC definition below)
- **Path (tree) distance**

  $d_{\text{path}}(a,b) = \mathrm{depth}(a) + \mathrm{depth}(b) - 2\,\mathrm{depth}(\mathrm{LCA}(a,b))$.

 For DAGs, use **undirected** shortest path on the transitive reduction (or a chosen spanning tree); document the policy.

“Undirected shortest path on the transitive reduction” is fast but can produce unintuitive “cross-branch” detours. Offer an alternative:

path_policy="spanning_tree": distance = depth(a)+depth(b) − 2*depth(best_LCA) (uses DAG LCA policy), i.e., the tree path induced by the chosen LCA. It’s semantically tighter and cheap if you have ancestor bitsets.


**Distances/Similarities** (pickable by `metric`):

- Distances (`get_distance`): `"path"`, `"lca"`, `"lin"`
- Similarities (`get_similarity`): `"resnik"`, `"lca"`, `"lin"`


- **LCA-depth similarity** (convertible to distance)
  $$
  s_{\text{lca}}(a,b) = \frac{2\,\mathrm{depth}(\mathrm{LCA}(a,b))}{\mathrm{depth}(a)+\mathrm{depth}(b)},\quad d_{\text{lca}}=1-s_{\text{lca}}.
  $$
  If `depth(a)+depth(b)=0` (both root), set `s_lca=1` ⇒ `d_lca=0`.


- **Resnik similarity**
  $$
  s_{\text{resnik}}(a,b)=IC(\mathrm{LCA}(a,b)).
  $$

- **Lin similarity** (distance via $d_{\text{lin}}=1-s_{\text{lin}}$)
  $$
  s_{\text{lin}}(a,b)=\frac{2\,IC(\mathrm{LCA}(a,b))}{IC(a)+IC(b)}.
  $$
$s_{lin} = 2*IC(LCA)/(max(IC(a)+IC(b), ε))$ with $ε=1e^{-6}$.

Expose **only** `resnik_norm` (similarity in [0,1]) and `resnik_norm_dist = 1 - IC(LCA)/IC_max` (record corpus + base). Hide raw Resnik from public API or mark as “unsafe for α”.

- **Information Content (IC)**

Let $L\subseteq V$ be the set of **base nodes where raw counts live** (e.g., leaves or atomic concepts). For DAGs, map each corpus occurrence to exactly one $v\in L$.

Let $\mathrm{desc}(c)$ be the set containing $c$ and all of its descendants (transitive closure; deduplicate via unique base nodes). Define
$$
\mathrm{count}_{\text{desc}}(c)=\sum_{v\in \mathrm{desc}(c)\cap L}\mathrm{count}_{\text{corpus}}(v),\qquad
\hat p(c)=\frac{\mathrm{count}_{\text{desc}}(c)+\alpha}{\sum_{v\in L}\mathrm{count}_{\text{corpus}}(v)+\alpha|V|},\qquad
IC(c)=-\log \hat p(c).
$$

**Notes:** (i) Persist corpus choice and $\alpha$ with version metadata; (ii) For DAGs, compute $\mathrm{desc}(c)$ on the transitive reduction and aggregate over **unique** nodes to avoid double-counting. (iii) $IC$ is corpus-dependent; when comparing projects or time windows, pin the reference corpus (or re-estimate and version $IC$).

**DAG LCA tie-breaks:** if multiple LCAs exist, choose the one with **max $IC$** (or, optionally, **max depth**). Keep this policy explicit and stable across runs.

- `lca_policy ∈ {"max_ic","max_depth"}` (default `"max_ic"`).
- `path_policy ∈ {"undirected_tr","spanning_tree"}` (default `"undirected_tr"`).

Record both in run metadata.

Use natural log for `IC`. Counts aggregated over unique base nodes on the **transitive reduction** to avoid double-counting. Persist `{α, corpus_id, log_base}`.


`Ontology.metrics()` → returns a table of {key, type∈{distance, similarity}, definition, version}. Persist chosen keys in run metadata.



**API sketch**
~~~
class Ontology:
    def __init__(self, structure_data, corpus_counts=None, is_dag=False, lca_policy="max_ic"):
        ...

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> 'Ontology': ...
    @classmethod
    def from_json(cls, path: str, **kwargs) -> 'Ontology': ...

    def get_lca(self, a: str, b: str) -> str | None: ...
    def get_ancestors(self, node: str) -> set[str]: ...
    def get_depth(self, node: str) -> int: ...
    def get_information_content(self, node: str) -> float: ...
    def get_distance(self, a: str, b: str, metric: str = 'path') -> float: ...
    def get_similarity(self, a: str, b: str, metric: str = 'lin') -> float: ...

~~~

### 2.2 AnnotationData

DataFrame wrapper with required columns:
`item_id, annotator_id, label_id`; optional: `duration_secs, flagged, comment, timestamp`. Validates labels ∈ ontology.

Allow `label_id` to be a **set**. Default set-distance:
`d_set(A,B) = (1/|A||B|) Σ_{a∈A} Σ_{b∈B} d(a,b)`.
Alternative: Tree-Wasserstein on ancestor-smeared masses (`smear=True`).

### 2.3 AnnotatorModel (user façade)

~~~
class AnnotatorModel:
    def __init__(self, ontology: Ontology, strategy: str = "auto",
                 metric: str = "path", random_state: int | None = 0, **kwargs):
        ...
    def fit(self, data: AnnotationData) -> 'ModelResults':
        ...
~~~

### 2.4 ModelResults

~~~
class ModelResults:
    def get_hard_items(self, top_n=10) -> pd.DataFrame: ...
    def get_annotator_profiles(self) -> pd.DataFrame: ...
    def get_consensus_labels(self) -> pd.Series: ...
    def plots(self) -> 'ResultsPlots': ...
~~~

**LCA (Longest Common Ancestor):** the root-to-LCA path; used for divergence-level features.

---

## 3. Agreement & Reliability Module (Foundations → Semantic)
Introduce `ABSTAIN`.
- α: exclude from coincidence counts (missingness OK).
- κ: drop items with any `ABSTAIN` unless `config.penalize_abstain=True`.

### 3.1 Why “chance-corrected” agreement?

Raw percent agreement inflates reliability when class imbalance is high. Chance-corrected measures discount **expected-by-chance** agreement.

### 3.2 Classical frequentist metrics

- **Cohen’s κ (two raters, nominal).**
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e},
  $$
  where $p_o$ is observed agreement and $p_e$ the chance agreement from marginals. Issues: prevalence/bias paradoxes, two-rater only.

- **Weighted κ (two raters, ordinal).**
  $$
  \kappa_w = 1 - \frac{\sum_{c,k} w_{ck} O_{ck}}{\sum_{c,k} w_{ck} E_{ck}},
  $$
  with weights $w_{ck}\in[0,1]$ (e.g., squared distance along the scale).

- **Fleiss’ κ (many raters, nominal).**
  Generalizes to multiple raters (fixed # per item); still nominal.

- **ICC (continuous).**
  Reliability as variance component ratio; choose form (one-/two-way, random/fixed) carefully.

### 3.3 Krippendorff’s α (flexible; missing data OK)

Works for any #raters, missingness, and **any difference function** $\delta^2(c,k)$.

Let $o_{ck}$ be the **coincidence matrix** (counts of pairable ratings across items), $n_c=\sum_k o_{ck}$, $N=\sum_c n_c$.

- **Observed disagreement**
  $$
  D_o = \frac{1}{N}\sum_{c,k} o_{ck}\,\delta^2(c,k).
  $$
- **Expected disagreement**
  $$
  D_e = \frac{1}{N(N-1)} \sum_{c,k} n_c n_k\,\delta^2(c,k).
  $$
- **Alpha**
  $$
  \alpha = 1 - \frac{D_o}{D_e}.
  $$

For nominal data: $\delta^2(c,k)=\mathbb{1}[c\neq k]$. For ordinal/interval: quadratic distance, etc.

### 3.4 Semantic α for ontologies (key for pyrator)
- **Resnik similarity**
  $$
  s_{\text{resnik}}(a,b)=IC(\mathrm{LCA}(a,b)).
  $$
- **Lin similarity** (distance via $d_{\text{lin}}=1-s_{\text{lin}}$)
  $$
  s_{\text{lin}}(a,b)=\frac{2\,IC(\mathrm{LCA}(a,b))}{IC(a)+IC(b)}.
  $$

Replace $\delta^2$ with an ontology-aware difference:

- **Path-normalized:**
  $$
  \delta^2_{\text{path}}(c,k)=\Big(\frac{d_{\text{path}}(c,k)}{d_{\max}}\Big)^2.
  $$
where $d_{\max}=\max_{a,b\in V} d_{\text{path}}(a,b)$ (or the **observed** max over labeled nodes for normalization; record the choice in run metadata).

- `d_max`: use **observed** max among labels present in data (default), or `"global"` with `K`-wide scan; record choice.
- `δ²_ic = 1 - IC(LCA)/max(max(IC(c),IC(k)), ε)`, then clamp to `[0,1]`.
- **Lin-based:**
  $$
  \delta^2_{\text{lin}}(c,k)=\big(1 - s_{\text{lin}}(c,k)\big)^2.
  $$

- **IC-weighted LCA:**
  $$
  \delta^2_{\text{ic}}(c,k)=1-\frac{IC(\mathrm{LCA}(c,k))}{\max(IC(c),IC(k))}.
  $$
  (square if desired)
Clip the denominator with $\max\!\big(\max(IC(c),IC(k)),\ \varepsilon\big)$ (default $\varepsilon=10^{-6}$).


**Why:** close concepts (parent/child, siblings) incur small penalty; cross-branch concepts incur large penalty. This aligns “agreement” with **semantic consistency** rather than exact code identity.

**Non-expert note:** two coders picking “Acute HF” vs “Heart Failure” are close in the hierarchy—semantic α counts them *mostly agreeing*, unlike nominal α which calls it a full mismatch.

### 3.5 “Semantic κ”

Define a hierarchy-aware weight matrix $w_{ck} = \delta^2_{\text{semantic}}(c,k)$ and compute weighted κ. Use with care (two raters).

Weights are **disagreement penalties**: set $w_{cc}=0$ on the diagonal and $w_{ck}\in(0,1]$ off-diagonal (e.g., $w=\delta^2_{\text{semantic}}$).


### 3.6 When to use what (rules of thumb)

- Quick snapshot, many raters, missing data → **α**.
- Two raters, ordinal scale → **weighted κ**.
- Continuous ratings (scores) → **ICC**.
- Ontology labels → **semantic α** (preferred), report nominal α alongside for transparency.

---

## 4. Modeling Strategies

### Strategy 1 — Vector Featurization & Clustering (Default/Fast)

Item $i$ with annotator labels $\{\ell_{i1},\dots,\ell_{iA}\}$ → feature vector $x_i$.

**Dispersion**
$$
\overline{d}_i = \frac{2}{A(A-1)}\sum_{1\le j<k\le A} d(\ell_{ij},\ell_{ik}),\quad
\max d_i,\ \mathrm{sd}(d_i).
$$

**Consensus-centric**
$$
z_i^{\text{LCA}}=\mathrm{LCA}(\ell_{i1},\dots,\ell_{iA}),\quad
\overline{d}_{i\to\text{LCA}}=\tfrac{1}{A}\sum_j d(\ell_{ij}, z_i^{\text{LCA}}),\quad
\mathrm{depth}(z_i^{\text{LCA}}).
$$

**Disagreement geometry**
Ancestor-pair fraction, sibling-pair fraction, divergence-level entropy $H_i$ from LCA levels.

**Stability**
Jackknife stability (leave-one-annotator-out swing), agreement-vs-depth AUC.

**Depth/branching**
Mean/SD of depths; #unique top-1/2 branches.

**IC variants**
Use $d_{\text{lin}}$ or IC-weighted versions of the above.

**Clustering**
- HDBSCAN (finds dense “hardness pockets”, can label noise).
- MiniBatchKMeans for very large $n$.
- Hardness score: composite $$h_i = w_1\overline d_i + w_2 H_i + w_3(1-\mathrm{AUC}_i)+\cdots$$

*Scales linearly*: featurization $O(n\cdot A^2 \cdot \bar L)$, clustering on $n\times d$.

---

### Strategy 2 — Hierarchical Bayesian Models (Pooling)

Two complementary formulations.

#### 4.1 Distance-based (energy) generative model
Before inference, rescale distances so median pairwise `d` over observed labels = 1. Then use `τ_j, δ_i ~ HalfNormal(1)` as weakly-informative defaults.


**Latents:** $z_i\in\{1..K\}$, $\delta_i>0$ (difficulty), $\tau_j>0$ (reliability), $\beta_j\in\mathbb{R}$ (specificity bias).

**IC-based specificity bias:**
$$
\Pr(\ell_{ij}\mid z_i,\tau_j,\beta_j,\delta_i)\ \propto\
\exp\!\left\{-\frac{\tau_j}{\delta_i}\,d(\ell_{ij},z_i)\ +\ \beta_j\big(IC(z_i)-IC(\ell_{ij})\big)\right\}.
$$
Interpretation: $\beta_j>0$ ⇒ tendency to be **less specific** (favoring lower $IC$); $\beta_j<0$ ⇒ tendency to be **more specific** (favoring higher $IC$).

**Hybrid (recommended for robustness):**
$$
S(\ell,z;\lambda)=\lambda\,\big(\mathrm{depth}(z)-\mathrm{depth}(\ell)\big)\ +\ (1-\lambda)\,\big(IC(z)-IC(\ell)\big),
$$
$$
\Pr(\ell_{ij}\mid z_i,\tau_j,\beta_j,\delta_i)\ \propto\
\exp\!\left\{-\frac{\tau_j}{\delta_i}\,d(\ell_{ij},z_i)\ +\ \beta_j\,S(\ell_{ij},z_i;\lambda)\right\},
$$
with $\lambda\sim \mathrm{Beta}(a,b)$ (e.g., $a=b=2$) to **learn** the mix of structural depth vs corpus-informed specificity.

**Identifiability cautions:** place weakly informative priors $\beta_j\sim \mathcal N(0,\sigma_\beta^2)$ with $\sigma_\beta$ small-to-moderate, and consider centering $IC$ **within depth** (subtract the mean $IC$ at each depth) to reduce collinearity with $d(\cdot,\cdot)$ and $\delta_i$.

- High $\tau_j$: annotator sticks close to $z_i$.
- High $\delta_i$: item dilutes everyone’s precision (hard).
- $\beta_j>0$: **under-specific** tendency (favor lower $IC$ than $IC(z_i)$); $\beta_j<0$: **over-specific** tendency (favor higher $IC$).



**Priors**
$$
z_i\sim\mathrm{Categorical}(\pi),\ \ \pi\ \text{uniform or prevalence-based};
\quad \delta_i\sim\mathrm{HalfNormal}(\sigma_\delta);\quad
\tau_j\sim\mathrm{HalfNormal}(\sigma_\tau);\quad
\beta_j\sim\mathcal{N}(0,\sigma_\beta^2).
$$
Define $\widetilde{IC}(c)=IC(c)-\mathbb{E}[IC\mid \mathrm{depth}(c)]$ and use $\widetilde{IC}$ in $S(\cdot)$ to reduce collinearity with depth.

**Inference backends**
- Exact discrete (PyMC/NUTS) when \(K\lesssim 2000\). Precompute \(K\times K\) \(D\).
- Pruned discrete for large trees (ICD): per-item candidate set
$$
C_i=\Big(\{\ell_{ij}\}_{j}\Big)
\cup \mathrm{Parents}\!\big(\{\ell_{ij}\}_{j}\big)
\cup \mathrm{Siblings}\!\big(\{\ell_{ij}\}_{j}\big)
\cup \mathrm{NN}_k\!\big(\{\ell_{ij}\}_{j};\,d\big),
$$
with total size $\approx 10\text{–}50$ (tune $k$).


- Path-factorized: sample branch-by-branch.
- VI (ADVI/SVI) for speed; confirm on subsample with NUTS.

`C_i` must include: all annotated labels for item `i`, their parents/siblings, current MAP of `z_i` (if available), and `k` nearest by `d`. Log `|C_i|` per item.


#### 4.2 Dawid–Skene-style (confusion-matrix) model (alternative)

For annotator $j$, confusion $\Theta_j$ (rows = true $z$, columns = emitted label $\ell$):

$$
\Pr(\ell_{ij}=\ell\mid z_i=z,\ \Theta_j)=(\Theta_j)_{z,\ell}.
$$
- Dirichlet priors on rows of $\Theta_j$.
- Hierarchical shrinkage: hyperpriors share strength across annotators.
- **Hierarchy-aware:** bias the Dirichlet base to favor **nearby** $\ell$ via a distance-decay prior (see equation below), or add a diagonal boost to encourage self-consistency.


**Hierarchy-aware Dirichlet prior (concrete).** For annotator $j$ and true label $z$, place a Dirichlet prior over emitted labels $\ell$ with concentration
$$
\alpha_{z\ell}\;=\;\alpha_0\,e^{-\gamma\,d(z,\ell)}\;+\;\kappa\,\mathbb{1}[\ell=z],
$$
and draw
$$
\Theta_{j,z,\cdot}\sim\mathrm{Dirichlet}\!\big(\{\alpha_{z\ell}\}_{\ell\in V}\big).
$$

**Hyperpriors & choices.** Use $\gamma\sim\mathrm{HalfNormal}(\sigma_\gamma)$, $\kappa\sim\mathrm{HalfNormal}(\sigma_\kappa)$, and choose $d\in\{d_{\text{path}},\; d_{\text{lin}}\}$ with your stated DAG LCA policy. For large $K$, prune each row to a candidate set $C_z=\{z\}\cup\text{parents/children/siblings of }z\cup\text{top-IC neighbors}$ and renormalize the Dirichlet row.

**Notes.** (i) Avoid double-counting hierarchy effects if your likelihood also penalizes far emissions; (ii) consider a weak diagonal floor (small $\kappa$) when data are sparse; (iii) log and version $d(\cdot,\cdot)$ and the pruning rule for reproducibility.

**Inference:** EM (classic DS) or full Bayesian (NUTS/VI).
**Pros:** explicit per-class confusion; **Cons:** large $K$ ⇒ big parameter surface (prune/factorize).

**Outputs (both models):**
$E[\delta_i]$ (hardness), $E[\tau_j], E[\beta_j]$ (profiles), $p(z_i=k)$ (consensus) + entropy, posterior predictive checks.

---

## 5. Structured Clustering Extras

### 5.1 Item-to-item distance on trees (Tree-Wasserstein)
Use **Wasserstein-1 (a.k.a. Earth Mover’s Distance)** — exposed as function `w1` in the API.
**Aliases:** expose `"emd"` as an alias of `"w1"` in the public API to match common terminology.


Represent item \(i\) as a distribution over nodes (annotator votes; optionally smear mass along ancestor path). Precompute edge flows $F_i(e)$ = mass in subtree under edge $e$.

$$
W_1(i,j) = \sum_{e} w_e\,|F_i(e)-F_j(e)|
$$
with edge weights $w_e\in\{1,\text{IC-weighted}\}$. Build a sparse k-NN on $W_1$ (no dense $n^2$) and run HDBSCAN/Leiden.

### 5.2 Tree edit distance (TED) for structured annotations

Use RTED (robust TED) for predictable runtime; cluster with HAC or spectral on a k-NN graph (avoid dense $n^2$).

### 5.3 Graph kernels & community detection (graphs)

- Weisfeiler–Lehman (WL) kernel for subtree patterns → spectral clustering.
- Annotator graph: nodes=annotators, edge weight = similarity on co-labeled items (agreement, $\kappa$, $\alpha$, or semantic versions). Run Louvain/Leiden to find “schools of thought”.


---

## 6. Plots & Diagnostics

**Agreement dashboards**
- Nominal vs Semantic α side-by-side (per slice), with CIs (bootstrap).
- κ / κ_w distributions; ICC with CI when continuous.
- Agreement-vs-depth curves with AUC.

**Clustering**
- UMAP/**t-SNE** of item disagreement vectors, colored by hardness/cluster.
- Cluster profiles: radar/bars of mean features per cluster.
- Divergence level histograms per cluster.
- Top-1/2 branch spread per cluster.

**Bayesian**
- Forest/caterpillar plots for $E[\tau_j], E[\beta_j]$ with 95% intervals.
- Ranked $E[\delta_i]$ with HPD bands (hardness).
- Posterior entropy of $p(z_i)$ (uncertainty).
- Traces/R-hat/ESS (convergence).
- Posterior predictive overlays: pairwise distance histograms, ancestor/sibling fractions, divergence-level distributions.

**Annotator communities**
- Annotator graph with Louvain communities; heatmap of inter-community (dis)agreement (semantic).

---

## 7. Complexity & Scaling

- Vector strategy: $O(n\cdot A^2\cdot \bar L)$ featurization; clustering linear in $n$.
- Exact Bayes: viable for $K\lesssim 2{,}000$.
- Pruned Bayes: per-item $K'_i\ll K$; near-linear in annotations.
- Embedding Bayes: cost independent of $K$ post-embedding; depends on dim $d$.
- DAGs: LCA tie-breaks; ancestor mass normalization when smearing.

---

## 8. Drift & Association Monitoring

> **Purpose**: Detect, quantify, and explain distribution and behavior shifts that threaten reliability of rater decisions, guideline adherence, and model/LLM outputs. This module complements inter-rater agreement (IRA) by watching the **data**, **raters**, and **models** over time.

---

### What we mean by "drift"

- **Covariate drift**: the **inputs** change (case-mix, demographics, encounter attributes, text characteristics, embeddings).
- **Prior drift**: the **marginal label distribution** changes (e.g., prevalence of guideline categories).
- **Conditional drift**: the **relationship** between inputs and labels changes (e.g., same patients yield different decisions).
- **Behavioral drift**: raters or LLMs change behavior (prompting, tool use, rationale length, calibration) even if inputs look similar.

We monitor these in rolling or fixed **windows** (e.g., 2025-Q2 as baseline vs 2025-10 as current), optionally stratified by high-stakes axes (service line, setting, region, payer, facility type, age bands, SDOH proxies). Monitor the **label marginal** `p(label)` across windows using **KL** (base 2) and **Total Variation** ($0.5 * \sum_i |p_i −q_i|$). Labels are fixed to the baseline union; probabilities are $\varepsilon$-smoothed.


---

### Architecture overview

- **Inputs**: a tidy table with `{window_id, timestamp, doc_id, rater_id?, model_id?, label?, prediction?, features..., embeddings?}`.
- **Baseline**: fixed historical window(s) with pinned bin edges and reference summaries.
- **Monitors**: declarative specs (YAML) that compute metric(s) per window (and stratum), evaluate thresholds, and emit explanations.
- **Artifacts**: Parquet result tables + JSON sidecar metadata (bin edges, χ² tables, residuals) for reproducibility.
- **Views**: time-series dashboards, PSI contribution waterfalls, contingency maps, embedding drift projections.

---

### Metrics
**Log base:** PSI uses natural log ($\ln$); KL/JSD use $\log_2$.


### Population Stability Index (PSI)

**What**: Measures change between two **binned** distributions.

**Formula** (for bin $i$ with baseline share $pᵢ$ and current share $qᵢ$):

$$PSI_i = (p_i - q_i) * ln(p_i / q_i)$$
$$PSI = Σ_i PSI_i$$


- Add a small $\varepsilon$ to avoid division by zero; clip probabilities to $[\varepsilon, 1-\varepsilon]$.

- **Binning**:
  - Numeric: quantiles (default deciles), Freedman–Diaconis, or clinically meaningful cutpoints.
  - Categorical: native categories; combine rare levels into `Other` to stabilize.
- **Interpretation (rule-of-thumb; tune per domain)**:
  - $< 0.10$ none / negligible
  - $0.10 - 0.25$ moderate (watch)
  - $> 0.25$ major shift (investigate)
- **Where to apply**:
  - Case-mix: age, LOS, setting (IP/OP), DRG/ICD clusters, payer, region, facility size.
  - Text: note length, section counts, clinical entity density, embedding cluster IDs.
  - Outputs: guideline recommendation categories, rater decisions, model probability bins, LLM rationale length bins.
- **Explainability**: report **per-bin contributions** $PSI_i$ sorted descending; surface example records near shifting cutpoints.
**PSI bin policy:** Bins are **pinned to the baseline** and **reused** for all future windows. Empty bins are retained (not dropped) and $\varepsilon$-smoothed to preserve comparability. Bin edges are versioned under `artifacts/psi_bins/{col}.json`.

### Cramér’s V (association strength)


**What**: Effect size for **categorical × categorical** relationships, derived from $χ²$.

$$
V=\sqrt{\frac{\chi^2}{n\,\min(r-1,\ c-1)}}
$$

- Use bias-corrected V (Bergsma) for small/imbalanced tables.
- **Use cases**:
  - Feature ↔ label (e.g., setting ↔ guideline class) drift across windows.
  - Rater/Model ↔ gold agreement strength changes over time.
  - Prompt template ↔ error type coupling in LLM operations.
- **Reporting**: show **ΔV** vs baseline with bootstrap CIs; highlight table cells with largest standardized residual change.
We report **bias-corrected Cramér’s V (V\*)** per Bergsma (2013) with **permutation confidence intervals** (e.g., 1000 shuffles). Alerts trigger when the **ΔV\*** CI excludes 0 and $|Δ|$ exceeds thresholds.

### Jensen–Shannon Divergence (JSD)

**What**: Symmetric, bounded divergence for probability vectors (e.g., softmax outputs, token categories).

$$JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) \ \text{where} \ M = 0.5*(P+Q)$$


- Numerically stable with $\varepsilon$-smoothing; return square-rooted JSD when interpretability as a distance helps.
> **JSD note:** With $\log_2$ base, $JSD \in [0,1]$. When a metric (distance) is preferred, report $\sqrt{JSD}$ (also in $[0,1]$).


### Total Variation Distance (tVD) & Hellinger

- **tVD**: $0.5 * \sum_i |p_i - q_i|$ (simple, interpretable upper bound on probability change).
- **Hellinger**: $(1/\sqrt{2}) * \sqrt{( \sum_i (\sqrt{p_i} - \sqrt{q_i})^2 )}$ (bounded, useful for discrete distributions).

### Wasserstein-1 (Earth Mover’s Distance)

- For **ordered** variables (age, risk scores, calibrated probabilities). Captures location + shape shifts; insensitive to binning.

### Maximum Mean Discrepancy (MMD)

- Kernel two-sample test for **embeddings**. Use RBF kernel with median heuristic; report statistic and permutation p-value.
- Practical: subsample uniformly or via coresets to control $O(n^2)$ cost; store seeds for reproducibility.

  **MMD scale guard:** Default cap `n_points_per_window = 2000` (uniform or stratified sampling). Kernel = RBF with median heuristic on standardized embeddings. Sampling is seeded for determinism.

### Mutual Information (MI) & Theil’s U

- Track MI(feature, label) per window; a drop in MI can indicate loss of predictive coupling.
- **Theil’s U (uncertainty coefficient)** adds directionality (how much knowledge of X reduces uncertainty in Y).

### Change-point detection (online)

- **CUSUM** / **Page–Hinkley** for univariate monitors (PSI, V, JSD, calibration error) to detect regime shifts quickly.
- **BOCPD** (Bayesian online change-point detection) optional for smoother alerting in noisy series.

**Change-point defaults:**
- **CUSUM** on z-scored series with $k = 0.5σ, \ h = 5σ$.
- **Page–Hinkley** with $λ = 0.1σ$, $α = 5σ$.
All parameters (including $σ$ estimates and window sizes) are logged to `artifacts/run_meta.json`.
### Calibration Drift (ECE / Brier)

**What:** Probabilities can remain accurate yet become **miscalibrated**.

- **ECE:** Partition predicted probabilities into $M$ equal-frequency bins; $ECE = \sum_m (n_m/N) * |acc_m − conf_m|$. Default $M=15$.
- **Brier:** Mean squared error of probabilistic predictions.

**Reporting:** Track $ΔECE$ and $ΔBrier$ vs baseline with bootstrap CIs.

**Defaults:** warn when $ΔECE ≥ 0.02$, crit when $ΔECE ≥ 0.05$.


### Rater Behavioral Drift

Per-rater monitors across windows:
- **Propensity drift:** PSI/JSD on the rater’s label distribution.
- **Specificity bias ($\beta$):** change in mean label **$IC$** (and/or depth); alert when $|\Delta IC|$ or $|\Delta \text{depth}|$ exceeds thresholds.

- **Latency drift:** PSI/JSD on `duration_secs` bins.
Trigger adjudication when behavioral drift co-occurs with outcome drift.




---

### Windows, baselines, and stratification

- **Baselining**: 4–8 weeks of stable production-like data; **pin bin edges** here for PSI so future comparisons are apples-to-apples.
- **Windowing**: fixed (Q1 vs Q2) and rolling (e.g., last 30 days). Compute vs baseline **and** vs previous window to catch creep.
- **Strata**: compute per subgroup (service line, condition cluster, payer, age band, region). Compare subgroup vs overall to detect disparate drift.

---

### Thresholds & alerting (defaults; tune per program)

| Metric               | Warn  | Critical | Notes                                                                                   |
|----------------------|:-----:|:--------:|-----------------------------------------------------------------------------------------|
| PSI                  | 0.10  | 0.25     | Evaluate per column; require persistence over $\ge 2$ windows to reduce flaps          |
| $\Delta$ Cramér’s $V$| 0.05  | 0.10     | Use CI; alert when CI excludes 0 and $\lvert \Delta \rvert$ exceeds thresholds         |
| JSD (softmax)        | 0.03  | 0.07     | With $\varepsilon=10^{-6}$ smoothing                                                    |
| Wasserstein-1 (prob) | 0.02  | 0.05     | On calibrated probabilities in $[0,1]$                                                  |
| MMD p-value          | <0.10 | <0.01    | With 1000 permutations                                                                   |
| tVD                  | 0.05  | 0.10     | Simple headline shift check                                                              |

---
- **Multiple testing**: control FDR (Benjamini–Hochberg) across many monitors; alert on FDR-significant metrics.
- **Stability**: require $k$ consecutive breaches or use change-point logic before paging humans.
 **Also** consider q-values from [Storey q-value](http://genomics.princeton.edu/storeylab/papers/Storey_FDR_2011.pdf)


### LLM Operational Drift

Treat prompts/tools as features; monitor:
- **Prompt template share** (PSI),
- **Tool call failure rate** (change-point),
- **Latency** (CUSUM on p50/p95),
- **Rationale token counts** (PSI/JSD),
- **Softmax/logit distribution** (JSD).
On crit, rollback to a **frozen baseline prompt** and compare A/B.


---

## Explainability playbook

- **PSI**: list top contributing bins with $(pᵢ, qᵢ, PSIᵢ)$, show representative documents and cutpoint proximity.
- **Cramér’s V**: show contingency tables for baseline/current with standardized residuals; annotate cells with largest swings.
- **Embeddings**: project with UMAP/TSNE for baseline/current; overlay cluster centroids and MMD statistics.
- **LLM ops**: track prompt template, tool call counts, rationale token counts; compute $PSI$ and run change-point on these operational traces.

---

## Implementation details

- **Smoothing**: $\varepsilon=1e-6$ for probabilities; clip to $[\varepsilon, 1-\varepsilon]$.
- **Binning**: store bin edges under `artifacts/psi_bins/{col}.json` and **reuse** across windows.
- **CIs**: bootstrap windows ($≥1000$ resamples) for $PSI, ΔV, JSD$; permutation for V/MMD.
- **Performance**: vectorize with NumPy/Polars; for embeddings, sample deterministically; shard by window/stratum.
- **Fairness**: compute all metrics **by subgroup**; report between-group deltas to catch disparate impact drift early.
- **Storage**: write result tables to Parquet with schema:
    ```python
    import pyarrow as pa

    # Define the schema for the result tables.
    # This schema ensures that the data is stored in a consistent and efficient format
    # when written to Parquet files.

    results_schema = pa.schema([
        # The unique identifier for the monitor.
        pa.field('monitor_id', pa.string(), nullable=False),

        # The unique identifier for the time window of the analysis.
        pa.field('window_id', pa.string(), nullable=False),

        # A map to store strata information (e.g., country, device type).
        # Allows for flexible key-value pair filtering.
        pa.field('stratum', pa.map_(pa.string(), pa.string())),

        # The name of the metric being measured.
        pa.field('metric', pa.string()),

        # The calculated value of the metric.
        pa.field('value', pa.float64()),

        # The difference between the metric's value and its baseline.
        pa.field('delta_from_baseline', pa.float64()),

        # The lower bound of the confidence interval.
        pa.field('ci_low', pa.float64()),

        # The upper bound of the confidence interval.
        pa.field('ci_high', pa.float64()),

        # The p-value from statistical testing.
        pa.field('p_value', pa.float64()),

        # The alert level for the threshold (e.g., 'none', 'warn', 'crit').
        pa.field('threshold_level', pa.string()),

        # A unique hash identifying the run (code + config + data).
        pa.field('run_id', pa.string()),

        # The timestamp when the record was created, with microsecond precision.
        pa.field('created_at', pa.timestamp('us'))
    ])

    # You can print the schema to verify it.
    print(results_schema)
    ```
---

### YAML monitors
```yaml
monitors:
  # In monitor configs, specify absolute vs delta thresholds explicitly.
  - alert:
    semantics: abs        # or: delta
    warn: 0.10
    crit: 0.25
```

Examples:
```yaml
- name: case_mix_age_psi
  metric: psi
  target: age
  bins: { type: quantile, n_bins: 10 }
  stratify: [setting, payer]
  alert: { semantics: abs, warn: 0.10, crit: 0.25 }

- name: assoc_setting_guideline
  metric: cramer_v
  x: setting
  y: guideline_class
  compare: baseline_vs_latest
  alert: { semantics: delta, warn: 0.05, crit: 0.10 }

- name: softmax_jsd
  metric: jsd
  dist_cols: [p0, p1, p2]
  alert: { semantics: abs, warn: 0.03, crit: 0.07 }
```
### Defaults

- $\varepsilon$-smoothing: `1e-6`
- JSD base: `log₂` (report `√JSD` when distance needed)
- PSI bins: deciles (numeric) / native categories (categorical), **pinned to baseline**
- MMD: RBF, median heuristic, `n_perm=1000`, `n_points_per_window=2000`
- Calibration: ECE with `M=15` equal-frequency bins
- FDR: BH at `q=0.10` per monitor family, per run
- Change-point: CUSUM (`k=0.5σ`, `h=5σ`), Page–Hinkley (`λ=0.1σ`, `α=5σ`)

### CLI
```bash
pyrator drift run --config drift.yml --baseline 2025Q2 --current 2025-10
pyrator drift explain --monitor case_mix_age_psi --window 2025-10
pyrator drift board --serve :8080  # quick local dashboard
```
---
### Medical Guidelines & LLM Contexts
- `Guideline programs`: Watch case-mix (age, setting, DRG clusters), label prevalence, rater/model coupling (ΔV), and calibration drift on predicted approval/denial probabilities.

- `Rater management`: per-rater PSI on decision categories, rationale length drift, bias/prevalence-adjusted κ trends; trigger adjudication when behavioral drift coincides with outcome drift.

- `LLM operations`: treat prompts/templates/tools as operational features; compute PSI/JSD and run change-point to catch silent prompt regressions.

### Trading
- Use online change-point (CUSUM/Page–Hinkley) on volatility, spread, regime embeddings.

- Maintain regime-specific baselines; throttle position sizing when critical drift persists.

---
### API & CLI

```python
from pyrator.drift import psi, cramer_v, jsd, w1, mmd, monitor

# PSI on age with quantile bins, stratified by setting
psi_tbl = psi(
    df,
    col="age",
    window_col="window_id",
    bins="quantile",
    n_bins=10,
    stratify=["setting"],
    eps=1e-6,
)

# Association drift between setting and guideline class
v_tbl = cramer_v(
    df,
    x="setting",
    y="guideline_class",
    window_col="window_id",
    bias_correct=True,
)

# Softmax drift on model outputs
jsd_tbl = jsd(df, dist_cols=["p0","p1","p2"], groupby="window_id")

# Embedding drift (subsample for scale)
stat, p = mmd(
    embA,
    embB,
    kernel="rbf",
    sigma="median_heuristic",
    n_perm=1000,
    seed=7,
)
```

### Privacy & PHI

- Use **de-identified** inputs; drop free-text fields unless transformed to non-reversible features (lengths, counts, hashed tokens).
- On-prem only; Parquet + metadata encrypted **at rest** (AES-256) and **in transit** (TLS).
- Access is audit-logged; per-run `run_id` and user ID recorded.

**Reproducibility:**
```python
run_id = SHA256((
    pyrator_version          # e.g., '1.0.0'
    + '||' + code_hash       # repo/tree hash
    + '||' + config_hash     # YAML monitors + thresholds
    + '||' + data_window_hash# stable hash of windowed input IDs
    + '||' + bins_hash       # PSI bin edges
    + '||' + ontology_hash   # structure + counts
    + '||' + deps_fingerprint# e.g., numpy/pandas/pymc versions
).encode('utf-8')).hexdigest()
```

- Each component is persisted in `run_meta` for forensic reconstruction.

## Quick checklist (operational)

- [ ] Baseline windows selected; **PSI bin edges pinned** and versioned
- [ ] Monitors defined for **inputs**, **labels**, **predictions**, **ops traces**
- [ ] Fairness strata chosen (age band, region, payer); subgroup deltas enabled
- [ ] Thresholds tuned; **FDR (BH)** configured; change-point params logged
- [ ] **Calibration** monitors (ECE/Brier) enabled for probabilistic outputs
- [ ] Explanations wired: PSI bin waterfall, χ² residual maps, embedding drift plots
- [ ] **PHI policy** enforced; storage encrypted; access logs active
- [ ] **Reproducibility**: run_id + hashes; seeds pinned; sampling caps set
- [ ] Playbooks documented (recalibration, retraining, prompt rollback, guideline review)

---
### API & Module Layout

~~~bash
pyrator/
  ontology/
    graph.py        # build/caches; paths/ancestors/LCA; IC smoothing
    metrics.py      # path, lin/resnik; DAG LCA policy
    embeddings.py   # poincaré/node2vec with IC weights
  data/
    annotations.py  # loader/validator
  ira/
    kappa.py        # Cohen/Weighted/Fleiss
    icc.py
    krippendorff.py # nominal α + generalized α(δ^2)
    semantic_alpha.py  # δ^2 from ontology distances (path/lin/ic)
  features/
    disagreement.py # x_i features
    hardness.py     # composite hardness score
  cluster/
    vector_cluster.py  # HDBSCAN, MiniBatchKMeans
    knn_graph.py       # ANN kNN (faiss/hnswlib)
    tree_wasserstein.py
    rted.py            # robust tree edit distance wrapper
    annotator_graph.py # build graph + Louvain/Leiden
  bayes/
    model_energy.py    # distance-based likelihood (PyMC/NumPyro)
    model_ds.py        # Dawid–Skene (+ hierarchy-aware priors)
    model_embed.py     # embedding-based VI
    priors.py
    diagnostics.py     # PPCs, summaries
  results/
    results.py
    plots.py
  api.py
~~~

---



## 9. Evaluation, Data Sources & Benchmarks

**Public surrogates (since MCG is proprietary):**
- **ClinPGx** (guidelineAnnotations.json + TSVs): treat CPIC/DPWG/FDA as raters, gene–drug as items, recommendation types/strengths as labels.
- **CREST** (recommendation strength): use sentences/sections as items; collect fresh multi-rater annotations (humans/LLMs) to simulate multi-annotator setup and compare to embedded gold.
- **General guidelines (IDSA/AHRQ)**: scrape → define schema → create new annotation corpus.

**Data modeling tips:**
- Decide the **item** (e.g., gene–drug pair), the **rater** (source organization), and the **label** (recommendation type/strength).
- Build an ontology from tags/sections (tree over categories/strengths).
- Ensure reproducible splits & scripts to map raw JSON/HTML to `(item, rater, label)` tables.

---

## 10. Roadmap

**v0.1**
- Ontology core; AnnotationData; semantic α (δ² path/lin/ic).
- Vector features; HDBSCAN & MB-KMeans.
- Plots: UMAP, cluster profiles, divergence hist, agreement-vs-depth AUC.

**v0.2**
- Bayesian Exact Discrete (PyMC/NUTS); posterior summaries; PPCs.
- Annotator profiles $(τ, β)$; hardness $(δ)$.

**v0.3**
- Pruned Discrete + VI (ADVI/SVI).
- Tree-Wasserstein; annotator graph + Louvain.

**v1.0**
- Embedding solver (Poincaré/node2vec+IC) + VI; mapping latent → nearest labels.
- Extended visualization (ontology heatmaps, hierarchical confusion).

**Future**
- OWL import; active learning (“route hardest k to experts”); fairness audits via annotator communities; plug-ins for annotation platforms.

---

## 11. Appendix: Mathematical Details

### 11.1 Krippendorff’s α computation (multi-rater, missingness)

Per item $i$ with $n_i$ ratings, there are $n_i(n_i-1)$ ordered pairs. The coincidence matrix $o_{ck}$ sums over items the counts of ordered category pairs among raters on that item. With totals $n_c=\sum_k o_{ck}$, $N=\sum_c n_c$:

$$
D_o = \frac{1}{N}\sum_{c,k} o_{ck}\,\delta^2(c,k),\quad
D_e = \frac{1}{N(N-1)}\sum_{c,k} n_c n_k\,\delta^2(c,k),\quad
\alpha = 1 - \frac{D_o}{D_e}
$$
$\log Z_{ij}$ is computed over the active label set (full $V$ for small $K$; pruned $C_i$ when using candidate sets).


- Nominal: $\delta^2(c,k)=\mathbb{1}[c\ne k]$.
- Ordinal: $\delta^2(c,k) = (r(c)-r(k))^2$.
- Semantic: see §3.4 for $\delta^2_{\text{path}}, \delta^2_{\text{lin}}, \delta^2_{\text{ic}}$.

**CIs:** bootstrap items; report α mean ± 95% percentile interval.

### 11.2 Weighted κ (two raters)

Observed $O=(o_{ck})$ and expected $E=(e_{ck})$ from marginals. With weights $w_{ck}\in[0,1]$,

$$
\kappa_w = 1 - \frac{\sum_{c,k} w_{ck} O_{ck}}{\sum_{c,k} w_{ck} E_{ck}}.
$$

For semantic κ, set \(w_{ck}=\delta^2_{\text{semantic}}(c,k)\).

### 11.3 Tree-Wasserstein distance (edge-flow form)

Let $F_i(e)$ be mass below edge $e$ for item $i$, weights $w_e\ge 0$:

$$
W_1(i,j) = \sum_{e} w_e\,|F_i(e)-F_j(e)|.
$$

Compute \(F_i\) in one bottom-up pass per item; distances are L1 on flows.

### 11.4 Bayesian model (energy form) — log-likelihood

For annotation $a=(i,j)$, with $z_i$, define the hybrid specificity term
$$
S(\ell,z;\lambda)=\lambda\big(\mathrm{depth}(z)-\mathrm{depth}(\ell)\big)
+(1-\lambda)\big(IC(z)-IC(\ell)\big).
$$
The log-likelihood becomes
$$
\log p(\ell_{ij}\mid z_i,\tau_j,\beta_j,\delta_i,\lambda)
= -\frac{\tau_j}{\delta_i}\,d(\ell_{ij},z_i)
+ \beta_j\,S(\ell_{ij},z_i;\lambda)
- \log Z_{ij},
$$
with $\lambda\sim \mathrm{Beta}(a,b)$ (e.g., $a=b=2$).

`log Z_{ij}` is computed over the **active** label set (full `V` for small `K`; pruned `C_i` when using candidates).


---

## 12. What this adds beyond the original draft

- A full IRA module: κ family, ICC, Krippendorff’s α and its semantic generalization with explicit formulas and computation recipe.
- Clear handling of DAG LCAs, IC smoothing, and semantic difference functions $\delta^2$.
- A parallel Dawid–Skene path (confusion-matrix) alongside the distance-based model, with hierarchy-aware priors.
- Concrete plots/diagnostics to ship (nominal vs semantic α, PPCs, annotator communities).
- Practical benchmarks (ClinPGx, CREST) and how to model them into `(item, rater, label)` tables.
