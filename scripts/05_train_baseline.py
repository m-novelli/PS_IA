from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score

# ================================
# Caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
CONFIG = BASE_DIR / "configs" / "triagem_features.json"

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

# Verifica colunas ausentes
all_used_cols = num_cols + cat_cols + txt_cols
missing = [col for col in all_used_cols if col not in df.columns]
if missing:
    print(f"⚠️ Colunas ausentes na base e ignoradas: {missing}")

print(f"\nBase: {df.shape} | Target: {target_col}")
print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)
print("Texto:", txt_cols)

# ================================
# Prepara X e y
# ================================
X = df[[col for col in all_used_cols if col in df.columns]].copy()
y = df[target_col]

# Remove colunas inválidas (constantes ou 100% nulas)
invalid_cols = X.columns[(X.nunique(dropna=True) <= 1) | (X.isna().mean() == 1.0)]
if len(invalid_cols):
    print("Removendo colunas inválidas:", list(invalid_cols))
    X = X.drop(columns=invalid_cols)

# ================================
# Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

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
transformers = [
    ("num", StandardScaler(), [c for c in num_cols if c in X_train.columns]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_cols if c in X_train.columns]),
]

if txt_input:
    transformers.append(("txt", TfidfVectorizer(max_features=1000), txt_input))

preprocessor = ColumnTransformer(transformers=transformers)

# ================================
# Pipeline
# ================================
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
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
