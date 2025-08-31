# scripts/train_xgboost_final.py
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
from joblib import dump

# ================================
# Caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET     = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
FEATURE_MAP = BASE_DIR / "data" / "processed" / "feature_map.json"
CONFIG      = BASE_DIR / "configs" / "triagem_features.json"
MODELS_DIR  = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# Carregamento
# ================================
df = pd.read_csv(DATASET)
with open(CONFIG, "r", encoding="utf-8") as f:
    feats = json.load(f)
with open(FEATURE_MAP, "r", encoding="utf-8") as f:
    fmap = json.load(f)

num_cols = feats["NUM_CANDIDATES"]
cat_cols = feats["CAT_CANDIDATES"]
txt_cols = feats["TXT_CANDIDATES"]
target_col = "target_triagem"
group_cols = [c for c in fmap.get("group", []) if c in df.columns]
assert group_cols, f"Nenhuma coluna de grupo encontrada. Esperava uma de: {fmap.get('group', [])}"

# ================================
# Seleção + limpeza de colunas
# ================================
requested = num_cols + cat_cols + txt_cols
present   = [c for c in requested if c in df.columns]
X = df[present].copy()
y = df[target_col].astype(int).values
groups = df[group_cols[0]].values

# drop 100% NaN e constantes (defensivo)
drop_all_nan = X.columns[X.isna().mean() == 1.0]
if len(drop_all_nan):
    print("Removendo colunas 100% NaN:", list(drop_all_nan))
    X.drop(columns=drop_all_nan, inplace=True)

drop_const = X.columns[X.nunique(dropna=True) <= 1]
if len(drop_const):
    print("Removendo colunas constantes:", list(drop_const))
    X.drop(columns=drop_const, inplace=True)

# re-sync listas
num_cols = [c for c in num_cols if c in X.columns]
cat_cols = [c for c in cat_cols if c in X.columns]
txt_cols = [c for c in txt_cols if c in X.columns]

# ================================
# Split com grupos (mesmo que na comparação)
# ================================
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
y_train, y_test = y[tr_idx], y[te_idx]

# ================================
# Texto: concatena múltiplas colunas
# ================================
if len(txt_cols) > 1:
    X_train["__texto"] = X_train[txt_cols].fillna("").agg(" ".join, axis=1)
    X_test["__texto"]  = X_test[txt_cols].fillna("").agg(" ".join, axis=1)
    X_train.drop(columns=txt_cols, inplace=True)
    X_test.drop(columns=txt_cols, inplace=True)
    txt_input = "__texto"
elif len(txt_cols) == 1:
    txt_input = txt_cols[0]
else:
    txt_input = None

# ================================
# Pré-processamento (igual ao evaluate)
# ================================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if num_cols: transformers.append(("num", num_pipe, num_cols))
if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
if txt_input: transformers.append(("txt", TfidfVectorizer(max_features=1000), txt_input))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# ================================
# XGBoost (mesmos hparams usados na comparação)
# ================================
neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
scale_pos_weight = neg / pos
print(f"scale_pos_weight (treino): {scale_pos_weight:.3f}  [neg={neg}, pos={pos}]")

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.10,
    objective="binary:logistic",
    eval_metric=["logloss", "auc", "aucpr"],
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

# ================================
# Treino
# ================================
model.fit(X_train, y_train)

# ================================
# Avaliação (holdout)
# ================================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "pr_auc":  float(average_precision_score(y_test, y_proba)),
    "f1":      float(f1_score(y_test, y_pred)),
}

print("\nRelatório de classificação:\n")
print(classification_report(y_test, y_pred, digits=3))
print("ROC AUC:", round(metrics["roc_auc"], 3))
print("PR  AUC:", round(metrics["pr_auc"], 3))
print("F1     :", round(metrics["f1"], 3))

# ================================
# Salvamento
# ================================
model_path = MODELS_DIR / "xgboost_pipeline.joblib"
meta_path  = MODELS_DIR / "xgboost_meta.json"

dump(model, model_path)

meta = {
    "type": "xgboost",
    "dataset": DATASET.name,
    "group_col": group_cols[0],
    "target": target_col,
    "features_used": {
        "num": num_cols,
        "cat": cat_cols,
        "txt": [txt_input] if txt_input else []
    },
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "class_balance": {"neg": neg, "pos": pos},
    "scale_pos_weight": scale_pos_weight,
    "metrics_holdout": metrics,
    "split": {"test_size": 0.20, "random_state": 42, "type": "GroupShuffleSplit"}
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✅ Modelo salvo em: {model_path}")
print(f"📝 Metadados salvos em: {meta_path}")
