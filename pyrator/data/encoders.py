# pyrator/data/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrator.data.backends import get_xp, has_polars, to_pandas, to_polars
from pyrator.types import ArrayLike, FrameLike

if TYPE_CHECKING:
    import polars as pl


@dataclass
class Encoded:
    items: ArrayLike
    annotators: ArrayLike
    labels: ArrayLike
    maps: dict[str, dict[Any, int]]
    backend: str  # "numpy" or "cupy"


class CategoryEncoder:
    """
    Vectorized encoder for (item, annotator, label) -> contiguous int32 ids.
    Produces NumPy or CuPy arrays depending on device.
    """

    def __init__(self, *, device: str | None = None):
        self.device: str | None = device
        self._maps: dict[str, dict[Any, int]] | None = None
        self.backend: str = "cupy" if (device == "gpu") else "numpy"

    def fit(
        self, df_like: FrameLike, *, item_col: str, annotator_col: str, label_col: str
    ) -> CategoryEncoder:
        if has_polars():
            try:
                df = to_polars(df_like).select([item_col, annotator_col, label_col])
            except Exception as e:
                # Convert polars exceptions to KeyError for consistency
                raise KeyError(f"Missing columns: {e}")
            maps: dict[str, dict[Any, int]] = {}
            for col in (item_col, annotator_col, label_col):
                uniq = df.select(col).unique().sort(by=col).to_series().to_list()
                maps[col] = {v: i for i, v in enumerate(uniq)}
            self._maps = {
                "item": maps[item_col],
                "annotator": maps[annotator_col],
                "label": maps[label_col],
            }
            return self

        import numpy as np

        df = to_pandas(df_like)[[item_col, annotator_col, label_col]]
        _maps: dict[str, dict[Any, int]] = {}
        for col in (item_col, annotator_col, label_col):
            vals = df[col].to_numpy()
            uniq = np.unique(vals)
            _maps[col] = {v: int(i) for i, v in enumerate(uniq)}
        self._maps = {
            "item": _maps[item_col],
            "annotator": _maps[annotator_col],
            "label": _maps[label_col],
        }
        return self

    def transform(
        self, df_like: FrameLike, *, item_col: str, annotator_col: str, label_col: str
    ) -> Encoded:
        if self._maps is None:
            raise RuntimeError("CategoryEncoder must be fit() before transform().")
        xp = get_xp(self.device)

        if has_polars():
            try:
                df: pl.DataFrame = to_polars(df_like).select([item_col, annotator_col, label_col])
            except Exception as e:
                # Convert polars exceptions to KeyError for consistency
                raise KeyError(f"Missing columns: {e}")

            # Use the fast, vectorized .replace() expression
            items_s = df[item_col].replace(self._maps["item"], default=-1)
            annos_s = df[annotator_col].replace(self._maps["annotator"], default=-1)
            labels_s = df[label_col].replace(self._maps["label"], default=-1)

            return Encoded(
                xp.asarray(items_s.to_numpy(), dtype=xp.int32),
                xp.asarray(annos_s.to_numpy(), dtype=xp.int32),
                xp.asarray(labels_s.to_numpy(), dtype=xp.int32),
                self._maps,
                self.backend,
            )

        import numpy as np

        df = to_pandas(df_like)[[item_col, annotator_col, label_col]]

        def encode(col_name: str, m: dict[Any, int]) -> np.ndarray:
            arr = df[col_name].to_numpy()
            uniq, inv = np.unique(arr, return_inverse=True)
            id_lookup = np.array([m.get(u, -1) for u in uniq], dtype=np.int32)
            return id_lookup[inv]

        items = encode(item_col, self._maps["item"])
        annos = encode(annotator_col, self._maps["annotator"])
        labels = encode(label_col, self._maps["label"])

        return Encoded(
            xp.asarray(items, dtype=xp.int32),
            xp.asarray(annos, dtype=xp.int32),
            xp.asarray(labels, dtype=xp.int32),
            self._maps,
            self.backend,
        )
