from pathlib import Path
import json, re, os, time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from joblib import dump

# ===================== Caminhos =====================
BASE_DIR = Path(__file__).resolve().parents[1]
CSV = BASE_DIR / "data" / "interim" / "dataset_triagem.csv"
CFG = BASE_DIR / "configs" / "triagem_features.json"
OUT_DIR = BASE_DIR / "models" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "modelo_triagem_baseline.pkl"
METRICS_PATH = OUT_DIR / "triagem_baseline_metrics.json"

TARGET_COL = "target_triagem"     # ajuste conforme necessário
GROUP_COL  = "codigo_vaga"

# ===================== Parâmetros =====================
FAST = os.getenv("FAST", "0") == "1"
CONCAT_TEXT = os.getenv("CONCAT_TEXT", "0") == "1"
TEXT_MIN_DF = 5 if FAST else 2
TEXT_MAX_FEATURES = 3000 if FAST else 5000
TEXT_NGRAMS = (1, 1) if FAST else (1, 2)
N_SPLITS = 3 if FAST else 5

EXCLUDE_DEFAULT = {
    "contratado","codigo_vaga","codigo_applicant",
    "status_simplificado","status_candidato","situacao_candidato","situacao"
}
LEAKAGE_NAME_PATTERNS = re.compile(
    r"(status|situac|situacao|resultado|contratad|aprovad|reprovad|negad|proposta|documenta)",
    re.IGNORECASE
)

# ===================== Utils =====================
def recall_at_k_per_group(y_true, y_prob, groups, k=3):
    dfm = pd.DataFrame({"y": y_true, "p": y_prob, "g": groups})
    recalls = []
    for g, chunk in dfm.groupby("g"):
        pos = chunk["y"].sum()
        if pos == 0:
            continue
        topk = chunk.sort_values("p", ascending=False).head(min(k, len(chunk)))
        recalls.append(topk["y"].sum() / pos)
    return float(np.mean(recalls)) if recalls else np.nan

def build_preprocessor(num_cols, cat_cols, txt_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    text_transformers = []
    for c in txt_cols:
        text_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", FunctionTransformer(np.ravel, accept_sparse=False)),
            ("tfidf", TfidfVectorizer(
                min_df=TEXT_MIN_DF,
                max_features=TEXT_MAX_FEATURES,
                ngram_range=TEXT_NGRAMS
            )),
        ])
        text_transformers.append((f"text_{c}", text_pipe, [c]))

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            *text_transformers
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

def canon(name: str) -> str:
    if name is None: return ""
    s = str(name).strip().lower()
    s = (s
         .replace("á","a").replace("à","a").replace("â","a").replace("ã","a")
         .replace("é","e").replace("ê","e").replace("í","i")
         .replace("ó","o").replace("ô","o").replace("õ","o")
         .replace("ú","u").replace("ç","c"))
    return re.sub(r"[^a-z0-9]", "", s)

def map_columns_resilient(df_cols, desired_cols):
    canon_to_real = {canon(c): c for c in df_cols}
    matched, missing = [], []
    for want in desired_cols:
        real = canon_to_real.get(canon(want))
        if real in df_cols:
            matched.append(real)
        else:
            missing.append(want)
    return sorted(set(matched)), sorted(set(missing))

# ===================== Main =====================
def main():
    print("CSV:", CSV)
    df = pd.read_csv(CSV)
    if not CFG.exists():
        raise FileNotFoundError(f"Config não encontrada: {CFG}. Rode antes o 03_auto_features.py.")
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    exclude = set(cfg.get("EXCLUDE", [])) | EXCLUDE_DEFAULT
    num_cols, _ = map_columns_resilient(df.columns, cfg.get("NUM_CANDIDATES", []))
    cat_cols, _ = map_columns_resilient(df.columns, cfg.get("CAT_CANDIDATES", []))
    txt_cols, _ = map_columns_resilient(df.columns, cfg.get("TXT_CANDIDATES", []))

    num_cols = [c for c in num_cols if c not in exclude and not LEAKAGE_NAME_PATTERNS.search(c)]
    cat_cols = [c for c in cat_cols if c not in exclude and not LEAKAGE_NAME_PATTERNS.search(c)]
    txt_cols = [c for c in txt_cols if c not in exclude and not LEAKAGE_NAME_PATTERNS.search(c)]

    used_cols = num_cols + cat_cols + txt_cols
    if not used_cols:
        raise RuntimeError("Nenhuma feature utilizável foi encontrada.")

    assert TARGET_COL in df.columns, f"Alvo '{TARGET_COL}' não encontrado no dataset."
    y = df[TARGET_COL].astype(int)
    X = df[used_cols].copy()
    for c in txt_cols:
        if c in X.columns:
            X[c] = X[c].fillna("").astype(str)

    if CONCAT_TEXT and len(txt_cols) > 1:
        df["texto_unificado"] = df[txt_cols].fillna("").agg(" ".join, axis=1)
        X["texto_unificado"] = df["texto_unificado"]
        txt_cols = ["texto_unificado"]

    groups = df[GROUP_COL] if GROUP_COL in df.columns else None

    pre = build_preprocessor(num_cols, cat_cols, txt_cols)
    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    if groups is not None:
        splitter = GroupKFold(n_splits=N_SPLITS)
        splits = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splits = splitter.split(X, y)

    pr_aucs, roc_aucs, f1s, r3s, r5s = [], [], [], [], []
    for i, (tr, va) in enumerate(splits, start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        gva = groups.iloc[va] if groups is not None else pd.Series(np.arange(len(va)))

        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        yhat = (p >= 0.5).astype(int)

        pr_aucs.append(average_precision_score(yva, p))
        roc_aucs.append(roc_auc_score(yva, p))
        f1s.append(f1_score(yva, yhat))
        r3s.append(recall_at_k_per_group(yva.values, p, gva.values, k=3))
        r5s.append(recall_at_k_per_group(yva.values, p, gva.values, k=5))

    metrics = {
        "PR_AUC_mean": float(np.nanmean(pr_aucs)),
        "ROC_AUC_mean": float(np.nanmean(roc_aucs)),
        "F1_mean": float(np.nanmean(f1s)),
        "Recall@3_mean": float(np.nanmean(r3s)),
        "Recall@5_mean": float(np.nanmean(r5s)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "n_features_txt": len(txt_cols),
        "FAST": FAST,
        "CONCAT_TEXT": CONCAT_TEXT
    }
    print("\nMÉTRICAS:", metrics)

    pipe.fit(X, y)
    dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")

if __name__ == "__main__":
    main()
