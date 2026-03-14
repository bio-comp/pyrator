"""Input validation schemas for the API."""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class AnnotationSchema(pa.DataFrameModel):
    """Canonical schema for inter-rater agreement data."""

    item_id: Series[str] = pa.Field(coerce=True)
    annotator_id: Series[str] = pa.Field(coerce=True)
    label_id: Series[str] = pa.Field(coerce=True)

    class Config:
        strict = False
