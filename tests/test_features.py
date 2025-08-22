from src.training.features import build_preprocessor
import pandas as pd

def test_preprocessor_shape():
    pre = build_preprocessor()
    df = pd.DataFrame({
        "FFMC":[85.0], "DMC":[26.2], "DC":[94.3], "ISI":[5.1],
        "temp":[18.0], "RH":[45], "wind":[4.3], "rain":[0.0],
        "month":["aug"], "day":["fri"]
    })
    X = pre.fit_transform(df)
    assert X.shape[0] == 1
