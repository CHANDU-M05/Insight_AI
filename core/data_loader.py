"""
InsightAI Data Loader
CSV/Excel → DataFrame + automatic profiling.
WHY auto-profiling: every analyst does this first manually.
InsightAI does it automatically on upload — zero questions needed.
The analysis starts before the user types anything.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list = []
    min_val: Optional[str] = None
    max_val: Optional[str] = None
    mean_val: Optional[str] = None


class DataProfile(BaseModel):
    filename: str
    row_count: int
    col_count: int
    total_nulls: int
    null_pct: float
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    date_cols: list[str] = []
    columns: list[ColumnProfile] = []
    top_correlations: list[dict] = []
    warnings: list[str] = []


def load_file(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Load CSV or Excel file into DataFrame.
    Returns (DataFrame, error_message).
    """
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes),
                encoding='utf-8',
                na_values=['NA', 'N/A', 'missing', 'null', 'NULL', '-'],
            )
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(
                pd.io.common.BytesIO(file_bytes),
                na_values=['NA', 'N/A', 'missing', 'null', 'NULL', '-'],
            )
        else:
            return None, "Unsupported format. Upload CSV or Excel."

        # Auto-parse date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass

        # Clean column names
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]

        return df, ""

    except Exception as e:
        return None, f"Error loading file: {e}"


def load_sample(sample_name: str) -> tuple[pd.DataFrame, str]:
    """Load a built-in sample dataset."""
    samples_dir = Path(__file__).parent.parent / "samples"
    path = samples_dir / f"{sample_name}.csv"
    if not path.exists():
        return None, f"Sample {sample_name} not found."
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
        return df, ""
    except Exception as e:
        return None, f"Error loading sample: {e}"


def profile_dataframe(df: pd.DataFrame, filename: str = "dataset") -> DataProfile:
    """
    Generate automatic data profile.
    Runs on upload — no question needed.
    """
    row_count, col_count = df.shape
    total_nulls = int(df.isnull().sum().sum())
    null_pct = round(total_nulls / (row_count * col_count) * 100, 2) if row_count > 0 else 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Column profiles
    columns = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        col_profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_count=null_count,
            null_pct=round(null_count / row_count * 100, 1) if row_count > 0 else 0,
            unique_count=int(series.nunique()),
            sample_values=[str(v) for v in series.dropna().head(3).tolist()],
        )
        if pd.api.types.is_numeric_dtype(series):
            col_profile.min_val = str(round(series.min(), 2))
            col_profile.max_val = str(round(series.max(), 2))
            col_profile.mean_val = str(round(series.mean(), 2))
        columns.append(col_profile)

    # Top correlations
    top_correlations = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    pairs.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": round(float(corr_val), 3),
                    })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_correlations = pairs[:5]

    # Warnings
    warnings = []
    if null_pct > 20:
        warnings.append(f"High null rate: {null_pct}% of values are missing.")
    for col in df.columns:
        if df[col].nunique() == row_count and row_count > 10:
            warnings.append(f"Column '{col}' has all unique values — likely an ID column.")
    if row_count < 10:
        warnings.append("Very small dataset — analysis may not be statistically meaningful.")

    return DataProfile(
        filename=filename,
        row_count=row_count,
        col_count=col_count,
        total_nulls=total_nulls,
        null_pct=null_pct,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        date_cols=date_cols,
        columns=columns,
        top_correlations=top_correlations,
        warnings=warnings,
    )
