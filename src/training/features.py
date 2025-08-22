from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
CAT_COLS = ["month", "day"]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

def engineer_label(df: pd.DataFrame, target_col: str = "risk") -> pd.DataFrame:
    # The UCI dataset has 'area' — we derive a binary ignition/risk label.
    # Label = 1 if burned area > 0.2 hectares else 0.
    if "area" in df.columns:
        df[target_col] = (df["area"].astype(float) > 0.2).astype(int)
    elif target_col not in df.columns:
        raise ValueError("No 'area' column to derive label and no 'risk' given.")
    return df
