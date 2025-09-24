# scripts/train_baseline.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from joblib import dump

# ================================
# Caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
CONFIG1 = BASE_DIR / "configs" / "triagem_features.json"   # auto_features
CONFIG2 = BASE_DIR / "data" / "processed" / "feature_map.json"  # feature_map atualizado
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# Carregamento
# ================================
df = pd.read_csv(DATASET)

# tenta ler chaves no formato atual; se não, cai para o feature_map genérico
with open(CONFIG1, "r", encoding="utf-8") as f:
    fmap = json.load(f)

if all(k in fmap for k in ["NUM_CANDIDATES", "CAT_CANDIDATES", "TXT_CANDIDATES"]):
    num_cols = fmap["NUM_CANDIDATES"]
    cat_cols = fmap["CAT_CANDIDATES"]
    txt_cols = fmap["TXT_CANDIDATES"]
else:
    # fallback para o formato "text / candidato_info / vaga_info etc."
    if CONFIG2.exists():
        with open(CONFIG2, "r", encoding="utf-8") as f2:
            fmap2 = json.load(f2)
        txt_cols = fmap2.get("text", [])
        all_avoid = set(
            fmap2.get("id", [])
            + fmap2.get("date", [])
            + txt_cols
            + fmap2.get("personal_info", [])
            + fmap2.get("leakage_risk", [])
            + fmap2.get("group", [])
        )
        candidates = [c for c in df.columns if c not in all_avoid]
        num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in candidates if c not in num_cols]
    else:
        raise ValueError("Não encontrei chaves NUM/CAT/TXT nem feature_map alternativo.")

# alvo
target_col = "target_triagem"
if target_col not in df.columns:
    raise KeyError(f"Coluna alvo '{target_col}' não encontrada em {DATASET}")

# ================================
# Seleção de colunas + sanity checks
# ================================
all_used_cols = list(dict.fromkeys(num_cols + cat_cols + txt_cols))
present = [c for c in all_used_cols if c in df.columns]
missing = [c for c in all_used_cols if c not in df.columns]
if missing:
    print(f"⚠️ Colunas ausentes ignoradas: {missing}")

X = df[present].copy()
y = df[target_col].astype(int).values

# remove colunas inválidas (constantes ou 100% nulas)
invalid = X.columns[(X.nunique(dropna=True) <= 1) | (X.isna().mean() == 1.0)]
if len(invalid):
    print("Removendo colunas inválidas:", list(invalid))
    X = X.drop(columns=invalid)
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]
    txt_cols = [c for c in txt_cols if c in X.columns]

# ================================
# Split estratificado por GRUPO (evita vazamento)
# ================================
with open(CONFIG2, "r", encoding="utf-8") as f2:
    fmap2 = json.load(f2)
group_cols = [c for c in fmap2.get("group", []) if c in df.columns]

if not group_cols:
    raise RuntimeError("Nenhuma coluna de grupo encontrada (esperava uma de: {})".format(fmap2.get("group", [])))

groups = df[group_cols[0]].values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
y_train, y_test = y[tr_idx], y[te_idx]

# ================================
# Texto: colagem opcional
# ================================
txt_input = None
if len(txt_cols) > 1:
    X_train["__texto"] = X_train[txt_cols].fillna("").agg(" ".join, axis=1)
    X_test["__texto"]  = X_test[txt_cols].fillna("").agg(" ".join, axis=1)
    X_train.drop(columns=txt_cols, inplace=True)
    X_test.drop(columns=txt_cols,  inplace=True)
    txt_input = "__texto"
elif len(txt_cols) == 1:
    txt_input = txt_cols[0]

# ================================
# Pré-processamento (com imputação)
# ================================
num_cols_in = [c for c in num_cols if c in X_train.columns]
cat_cols_in = [c for c in cat_cols if c in X_train.columns]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler(with_mean=False)),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore")),
])

transformers = []
if num_cols_in: transformers.append(("num", num_pipe, num_cols_in))
if cat_cols_in: transformers.append(("cat", cat_pipe, cat_cols_in))
if txt_input:   transformers.append(("txt", TfidfVectorizer(max_features=15000, min_df=3), txt_input))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
    sparse_threshold=1.0
)

# ================================
# Modelo
# ================================
clf = LogisticRegression(
    max_iter=2000,
    solver="saga",
    n_jobs=-1,
    class_weight="balanced"
)

model = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

# ================================
# Treinamento
# ================================
model.fit(X_train, y_train)

# ================================
# Avaliação
# ================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "pr_auc":  float(average_precision_score(y_test, y_proba)),
    "f1":      float(f1_score(y_test, y_pred)),
    "n_train": int(len(y_train)),
    "n_test":  int(len(y_test))
}

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, digits=3))
print("ROC AUC:", round(metrics["roc_auc"], 3))
print("PR AUC:",  round(metrics["pr_auc"], 3))
print("F1:",      round(metrics["f1"], 3))

# ================================
# Salvamento de artefatos
# ================================
dump(model, MODELS_DIR / "baseline_pipeline.joblib")

meta = {
    "type": "baseline_logreg",
    "features_used": {
        "num": num_cols_in,
        "cat": cat_cols_in,
        "txt": [txt_input] if txt_input else []
    },
    "metrics": metrics,
    "dataset": str(DATASET.name),
    "group_col": group_cols[0],
    "target": target_col
}
with open(MODELS_DIR / "baseline_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n Baseline treinado e salvo em {MODELS_DIR}/baseline_pipeline.joblib")
