from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
import json
import re

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

NA_TOKENS = {"", "na", "n/a", "null", "none", "-", "--"}

EMBEDDED_SCHEMA: Dict[str, Any] = {'columns': {'run_id': {'required': True, 'type': 'string'}, 'scene': {'required': True, 'type': 'string'}, 'batch_size': {'required': True, 'type': 'string'}, 'algorithm': {'required': True, 'type': 'string'}, 'current_time': {'required': True, 'type': 'string'}, 'mean_reward': {'required': True, 'type': 'string'}, 'training_time_s': {'required': True, 'type': 'string'}, 'total_steps': {'required': True, 'type': 'string'}, 'steps_per_second': {'required': True, 'type': 'string'}, 'avg_system_cpu_percent': {'required': True, 'type': 'string'}, 'avg_tracked_cpu_percent': {'required': True, 'type': 'string'}, 'avg_ram_usage': {'required': True, 'type': 'string'}, 'max_ram_usage': {'required': True, 'type': 'string'}, 'num_cores': {'required': True, 'type': 'string'}, 'total_ram': {'required': True, 'type': 'string'}, 'policy_loss': {'required': True, 'type': 'string'}, 'value_loss': {'required': True, 'type': 'string'}, 'learning_rate': {'required': True, 'type': 'string'}}}

def get_embedded_schema() -> Dict[str, Any]:
    return EMBEDDED_SCHEMA

def _read_csv_any(csv_path: PathLike) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, sep=None, engine="python")

def _normalize_na(series: pd.Series, extra_na: Optional[Iterable[str]]) -> pd.Series:
    s = series.astype(str).str.strip()
    lowered = s.str.lower()
    tokens = set(t.lower() for t in (extra_na or [])) | NA_TOKENS
    s = s.mask(lowered.isin(tokens))
    s = s.replace(r"^\s*$", np.nan, regex=True)
    return s

def _coerce_type_valid(series: pd.Series, rule: Dict[str, Any]) -> bool:
    t = (rule.get("type") or "string").lower()
    s = series.copy()

    if t in {"int", "integer"}:
        vals = pd.to_numeric(s, errors="coerce")
        valid = vals.notna() & np.isclose(vals % 1, 0)
        return bool(valid.all())

    if t in {"float", "number", "numeric"}:
        vals = pd.to_numeric(s, errors="coerce")
        return bool(vals.notna().all())

    if t in {"bool", "boolean"}:
        lowered = s.astype(str).str.lower()
        valid = lowered.isin({"true", "false", "yes", "no", "y", "n", "0", "1"})
        return bool(valid.all())

    if t in {"date", "datetime", "date/time"}:
        vals = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        return bool(vals.notna().all())

    if t in {"email"}:
        pat = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        valid = s.astype(str).str.match(pat, na=False)
        return bool(valid.all())

    return True

def _validate_with_schema(df: pd.DataFrame, schema: Dict[str, Any], extra_na: Optional[Iterable[str]]) -> bool:
    schema_cols = list(schema.get("columns", {}).keys())
    for c in schema_cols:
        if c not in df.columns:
            return False

    for col, rule in schema["columns"].items():
        required = bool(rule.get("required", True))
        col_series = df[col] if col in df.columns else pd.Series([], dtype=object)
        s = _normalize_na(col_series, extra_na)

        if required and s.isna().any():
            return False

        non_missing = s.dropna()

        if not _coerce_type_valid(non_missing, rule):
            return False

        if rule.get("regex"):
            pat = re.compile(rule["regex"])
            if not bool(non_missing.astype(str).str.match(pat, na=False).all()):
                return False

        if rule.get("allowed"):
            allowed = set(map(str, rule["allowed"]))
            if not bool(non_missing.astype(str).isin(allowed).all()):
                return False

        if rule.get("min") is not None or rule.get("max") is not None:
            nums = pd.to_numeric(non_missing, errors="coerce")
            if nums.isna().any():
                return False
            if rule.get("min") is not None and not bool(nums.ge(rule["min"]).all()):
                return False
            if rule.get("max") is not None and not bool(nums.le(rule["max"]).all()):
                return False

    return True

def csv_is_complete(
    csv_path: PathLike,
    schema: Optional[Union[Dict[str, Any], PathLike]] = None,
    extra_na: Optional[Iterable[str]] = None,
) -> bool:

    df = _read_csv_any(csv_path)

    if schema is None:
        df_norm = df.copy()
        for c in df_norm.columns:
            df_norm[c] = _normalize_na(df_norm[c], extra_na)
        return not df_norm.isna().values.any()

    if isinstance(schema, (str, Path)):
        with open(schema, "r", encoding="utf-8") as f:
            schema_dict = json.load(f)
    else:
        schema_dict = schema

    return _validate_with_schema(df, schema_dict, extra_na)

def csv_is_complete_embedded(csv_path: PathLike, extra_na: Optional[Iterable[str]] = None) -> bool:
    return csv_is_complete(csv_path, schema=EMBEDDED_SCHEMA, extra_na=extra_na)
