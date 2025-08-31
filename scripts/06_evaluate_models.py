from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    f1_score
)

# ================================
# Função auxiliar: Recall@k
# ================================
def recall_at_k(y_true, y_proba, k=0.2):
    """Calcula recall nos top k% exemplos mais prováveis"""
    n = int(len(y_true) * k)
    idx = np.argsort(y_proba)[::-1][:n]
    return (y_true[idx] == 1).sum() / (y_true == 1).sum()

# ================================
# Caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
CONFIG = BASE_DIR / "configs" / "triagem_features.json"
FEATURE_MAP = BASE_DIR / "data" / "processed" / "feature_map.json"

# ================================
# Carregamento
# ================================
df = pd.read_csv(DATASET)
with open(CONFIG, "r", encoding="utf-8") as f:
    features = json.load(f)
with open(FEATURE_MAP, "r", encoding="utf-8") as f:
    fmap = json.load(f)

num_cols = features["NUM_CANDIDATES"]
cat_cols = features["CAT_CANDIDATES"]
txt_cols = features["TXT_CANDIDATES"]
target_col = "target_triagem"

X = df[num_cols + cat_cols + txt_cols].copy()
y = df[target_col].values

groups = df[fmap["group"][0]].values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
y_train, y_test = y[tr_idx], y[te_idx]

# ================================
# Texto: combina colunas
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
# Pré-processamento (igual em todos)
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
# Modelos com mesmos parâmetros usados individualmente
# ================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "LogReg": LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                           class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="binary:logistic",
        eval_metric=["logloss", "auc", "aucpr"],
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1
    )
}

# ================================
# Avaliação
# ================================
results = []

for name, clf in models.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "recall@10%": recall_at_k(y_test, y_proba, 0.1),
        "recall@20%": recall_at_k(y_test, y_proba, 0.2),
        "recall@50%": recall_at_k(y_test, y_proba, 0.5),
    }
    results.append(metrics)

# ================================
# Resultado final
# ================================
df_results = pd.DataFrame(results)
print("\n📊 Comparação de modelos:\n")
print(df_results.round(3).to_string(index=False))
