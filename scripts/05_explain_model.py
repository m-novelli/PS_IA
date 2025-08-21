# scripts/04_explain_model.py

import shap
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import json

# === Caminhos ===
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "artifacts" / "modelo_triagem_baseline.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"
FEATURE_CFG_PATH = BASE_DIR / "configs" / "triagem_features.json"

# === Carregar modelo e dados ===
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# === Target e features ===
TARGET_COL = "target_triagem"
y = df[TARGET_COL].astype(int)

with open(FEATURE_CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

exclude = set(cfg.get("EXCLUDE", []))
used_features = (
    cfg.get("NUM_CANDIDATES", []) +
    cfg.get("CAT_CANDIDATES", []) +
    cfg.get("TXT_CANDIDATES", [])
)
used_features = [c for c in used_features if c not in exclude]

X = df[used_features].copy()
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].astype(str).fillna("")

# === Aplicar apenas o transformador (pré-processamento do pipeline) ===
preprocessor = model.named_steps["pre"]
X_transformed = preprocessor.transform(X)

# === SHAP ===
explainer = shap.LinearExplainer(model.named_steps["clf"], X_transformed, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_transformed)

# === Visualização ===
print("Gerando summary plot...")
shap.summary_plot(shap_values, X_transformed, show=True)
