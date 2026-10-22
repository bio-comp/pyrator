from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd

from pyrator.drift.monitor import (
    CramerVMonitorConfig,
    JsdMonitorConfig,
    MmdMonitorConfig,
    Monitor,
    PsiMonitorConfig,
    WassersteinMonitorConfig,
)


def test_psi_monitor_config_creation():
    config = PsiMonitorConfig(name="test", col="value", n_bins=20)
    assert config.name == "test"
    assert config.col == "value"
    assert config.n_bins == 20
    assert config.metric == "psi"


def test_cramer_v_monitor_config_creation():
    config = CramerVMonitorConfig(name="test", x="col1", y="col2")
    assert config.name == "test"
    assert config.x == "col1"
    assert config.y == "col2"
    assert config.metric == "cramer_v"


def test_jsd_monitor_config_creation():
    config = JsdMonitorConfig(name="test", dist_cols=["a", "b"])
    assert config.name == "test"
    assert config.dist_cols == ["a", "b"]
    assert config.metric == "jsd"


def test_wasserstein_monitor_config_creation():
    config = WassersteinMonitorConfig(name="test", col="value")
    assert config.name == "test"
    assert config.col == "value"
    assert config.metric == "wasserstein"


def test_mmd_monitor_config_creation():
    config = MmdMonitorConfig(name="test", emb_cols=["emb1", "emb2"])
    assert config.name == "test"
    assert config.emb_cols == ["emb1", "emb2"]
    assert config.metric == "mmd"


def test_monitor_dispatch_by_config_type():
    psi_config = PsiMonitorConfig(name="test", col="value")
    monitor = Monitor(psi_config)
    assert isinstance(monitor.config, PsiMonitorConfig)


def test_base_config_fields():
    config = PsiMonitorConfig(
        name="test",
        col="value",
        warn=0.2,
        crit=0.5,
        semantics="delta",
        window_col="window",
    )
    assert config.warn == 0.2
    assert config.crit == 0.5
    assert config.semantics == "delta"
    assert config.window_col == "window"


def test_monitor_execute_all_metrics() -> None:
    # Test data with multiple columns
    df = pd.DataFrame(
        {
            "window_id": ["base", "base", "base", "curr", "curr", "curr"],
            "val": [1.0, 2.0, 1.5, 3.0, 3.5, 4.0],
            "cat1": ["a", "b", "a", "b", "b", "a"],
            "cat2": ["x", "x", "y", "y", "x", "y"],
            "d1": [0.6, 0.7, 0.5, 0.3, 0.4, 0.2],
            "d2": [0.4, 0.3, 0.5, 0.7, 0.6, 0.8],
            "e1": [0.1, 0.2, 0.15, 0.8, 0.9, 0.85],
            "e2": [0.3, 0.4, 0.35, 0.1, 0.05, 0.15],
            "grp": ["g1", "g1", "g2", "g1", "g2", "g2"],
        }
    )

    # 1. PSI Monitor
    m_psi = Monitor(PsiMonitorConfig(name="psi_m", col="val"))
    res_psi = m_psi.execute(df)
    assert isinstance(res_psi, pd.DataFrame)
    assert len(res_psi) == 1
    assert res_psi.iloc[0]["metric"] == "psi"

    # 2. Cramer's V Monitor
    m_cv = Monitor(CramerVMonitorConfig(name="cv_m", x="cat1", y="cat2"))
    res_cv = m_cv.execute(df)
    assert isinstance(res_cv, pd.DataFrame)
    assert len(res_cv) == 1
    assert res_cv.iloc[0]["metric"] == "cramer_v"

    # 3. JSD Monitor
    m_jsd = Monitor(JsdMonitorConfig(name="jsd_m", dist_cols=["d1", "d2"]))
    res_jsd = m_jsd.execute(df)
    assert isinstance(res_jsd, pd.DataFrame)
    assert len(res_jsd) == 1
    assert res_jsd.iloc[0]["metric"] == "jsd"

    # 4. Wasserstein Monitor
    m_w1 = Monitor(WassersteinMonitorConfig(name="w1_m", col="val"))
    res_w1 = m_w1.execute(df)
    assert isinstance(res_w1, pd.DataFrame)
    assert len(res_w1) == 1
    assert res_w1.iloc[0]["metric"] == "wasserstein"

    # 5. MMD Monitor (must return pd.DataFrame, not tuple)
    m_mmd = Monitor(MmdMonitorConfig(name="mmd_m", emb_cols=["e1", "e2"], n_perm=50, seed=42))
    res_mmd = m_mmd.execute(df)
    assert isinstance(res_mmd, pd.DataFrame)
    assert len(res_mmd) == 1
    assert res_mmd.iloc[0]["metric"] == "mmd"

    # 6. Test FrameLike input (Narwhals DataFrame)
    nw_df = nw.from_native(df)
    res_nw = m_psi.execute(nw_df)
    assert isinstance(res_nw, pd.DataFrame)
    assert len(res_nw) == 1
