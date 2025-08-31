# scripts/train_random_forest.py
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score

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

num_cols = features["NUM_CANDIDATES"]
cat_cols = features["CAT_CANDIDATES"]
txt_cols = features["TXT_CANDIDATES"]
target_col = "target_triagem"

# ================================
# Prepara X e y
# ================================
all_used_cols = num_cols + cat_cols + txt_cols
X = df[[c for c in all_used_cols if c in df.columns]].copy()
y = df[target_col]

# remove colunas inválidas
invalid_cols = X.columns[(X.nunique(dropna=True) <= 1) | (X.isna().mean() == 1.0)]
if len(invalid_cols):
    print("Removendo colunas inválidas:", list(invalid_cols))
    X = X.drop(columns=invalid_cols)
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]
    txt_cols = [c for c in txt_cols if c in X.columns]

# ================================
# Split com grupos
# ================================
with open(FEATURE_MAP, "r", encoding="utf-8") as f:
    fmap = json.load(f)

group_cols = [c for c in fmap.get("group", []) if c in df.columns]
if not group_cols:
    raise RuntimeError("Nenhuma coluna de grupo encontrada no dataset.")

groups = df[group_cols[0]].values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

# ================================
# Texto: combina colunas
# ================================
if len(txt_cols) > 1:
    X_train["__texto"] = X_train[txt_cols].fillna("").agg(" ".join, axis=1)
    X_test["__texto"] = X_test[txt_cols].fillna("").agg(" ".join, axis=1)
    X_train = X_train.drop(columns=txt_cols)
    X_test = X_test.drop(columns=txt_cols)
    txt_input = "__texto"
elif len(txt_cols) == 1:
    txt_input = txt_cols[0]
else:
    txt_input = None

# ================================
# Pré-processamento
# ================================
transformers = []
if num_cols: transformers.append(("num", "passthrough", num_cols))
if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
if txt_input: transformers.append(("txt", TfidfVectorizer(max_features=1000), txt_input))

preprocessor = ColumnTransformer(transformers=transformers)

# ================================
# Pipeline com Random Forest
# ================================
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300,     # um pouco mais robusto que 100
        max_depth=20,
        min_samples_leaf=10,    # deixa a árvore crescer (mais expressivo)
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
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

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, digits=3))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Average Precision (PR AUC):", round(average_precision_score(y_test, y_proba), 3))
print("F1 Score:", round(f1_score(y_test, y_pred), 3))
