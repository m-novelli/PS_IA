import pandas as pd
import pytest
from app import routes

def test_coerce_dtypes_numeric_and_text(feature_schema):
    df = pd.DataFrame({
        (feature_schema["num"][0] if feature_schema["num"] else "idade"): ["10", "20", None],
        (feature_schema["cat"][0] if feature_schema["cat"] else "sexo"): ["M", None, "F"],
        (feature_schema["txt"][0] if feature_schema["txt"] else "cv_pt"): ["ola", None, "mundo"],
    })
    out = routes._coerce_dtypes(df.copy())
    num_col = feature_schema["num"][0] if feature_schema["num"] else "idade"
    assert out[num_col].dtype.kind in ("i", "f")
    cat_col = feature_schema["cat"][0] if feature_schema["cat"] else "sexo"
    assert (
    pd.api.types.is_string_dtype(out[cat_col]) or
    pd.api.types.is_object_dtype(out[cat_col])
)
    txt_col = feature_schema["txt"][0] if feature_schema["txt"] else "cv_pt"
    assert str(out[txt_col].dtype) == "string"
    assert out[txt_col].isna().sum() == 0

def test_prepare_dataframe_missing_and_unknown(feature_schema):
    known_any = (feature_schema["num"] + feature_schema["cat"] + feature_schema["txt"])
    if not known_any:
        pytest.skip("Schema de features vazio no meta.json; pulando teste.")
    known = known_any[0]
    item = routes.PredictItem(meta={"external_id": "x1"},
                              features={known: 1, "feature_desconhecida": 123})
    df, warns = routes._prepare_dataframe([item])
    assert known in df.columns
    flat = {k for w in warns for k in w.keys()}
    assert "unknown" in flat or any("unknown" in w for w in warns)
