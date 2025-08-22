import numpy as np
import pandas as pd

def dict_to_df(payload: dict) -> pd.DataFrame:
    # Preserve column order
    order = ["FFMC","DMC","DC","ISI","temp","RH","wind","rain","month","day"]
    row = {k: payload[k] for k in order}
    return pd.DataFrame([row])
