# pyrator/data/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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

    def __init__(
        self,
        *,
        device: str | None = None,
        handle_unknown: Literal["default", "ignore", "error"] = "default",
        unknown_value: int = -1,
    ):
        self.device: str | None = device
        self._maps: dict[str, dict[Any, int]] | None = None
        self.backend: str = "cupy" if (device == "gpu") else "numpy"
        if handle_unknown not in {"default", "ignore", "error"}:
            raise ValueError("handle_unknown must be one of: 'default', 'ignore', 'error'.")
        self.handle_unknown: Literal["default", "ignore", "error"] = handle_unknown
        self.unknown_value = int(unknown_value)

    def _unknown_sentinel(self) -> int:
        if self.handle_unknown == "ignore":
            return -1
        return self.unknown_value

    def _assert_no_unknown_values(
        self, values: list[Any], mapping: dict[Any, int], *, column_name: str
    ) -> None:
        unknown = [value for value in values if value not in mapping]
        if unknown:
            raise ValueError(f"Unknown values in column '{column_name}': {unknown}")

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
        unknown_sentinel = self._unknown_sentinel()

        if has_polars():
            try:
                df: pl.DataFrame = to_polars(df_like).select([item_col, annotator_col, label_col])
            except Exception as e:
                # Convert polars exceptions to KeyError for consistency
                raise KeyError(f"Missing columns: {e}")

            if self.handle_unknown == "error":
                self._assert_no_unknown_values(
                    df[item_col].unique().to_list(), self._maps["item"], column_name=item_col
                )
                self._assert_no_unknown_values(
                    df[annotator_col].unique().to_list(),
                    self._maps["annotator"],
                    column_name=annotator_col,
                )
                self._assert_no_unknown_values(
                    df[label_col].unique().to_list(), self._maps["label"], column_name=label_col
                )

            # Use strict replacement to map known ids and apply fallback for unknowns.
            items_s = df[item_col].replace_strict(self._maps["item"], default=unknown_sentinel)
            annos_s = df[annotator_col].replace_strict(
                self._maps["annotator"], default=unknown_sentinel
            )
            labels_s = df[label_col].replace_strict(self._maps["label"], default=unknown_sentinel)

            return Encoded(
                xp.asarray(items_s.to_numpy(), dtype=xp.int32),
                xp.asarray(annos_s.to_numpy(), dtype=xp.int32),
                xp.asarray(labels_s.to_numpy(), dtype=xp.int32),
                self._maps,
                self.backend,
            )

        import numpy as np

        df = to_pandas(df_like)[[item_col, annotator_col, label_col]]

        if self.handle_unknown == "error":
            self._assert_no_unknown_values(
                df[item_col].drop_duplicates().tolist(), self._maps["item"], column_name=item_col
            )
            self._assert_no_unknown_values(
                df[annotator_col].drop_duplicates().tolist(),
                self._maps["annotator"],
                column_name=annotator_col,
            )
            self._assert_no_unknown_values(
                df[label_col].drop_duplicates().tolist(), self._maps["label"], column_name=label_col
            )

        def encode(col_name: str, m: dict[Any, int]) -> np.ndarray:
            arr = df[col_name].to_numpy()
            uniq, inv = np.unique(arr, return_inverse=True)
            id_lookup = np.array([m.get(u, unknown_sentinel) for u in uniq], dtype=np.int32)
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
