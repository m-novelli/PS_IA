# scripts/train_xgboost_final.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from joblib import dump

# ===== MLflow =====
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ================================
# Configuração e caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[2]   # raiz do projeto
# Garante que 'app' seja importável (app/transformers.py)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.transformers import TextConcatTransformer  # <--- usa a classe do módulo

DATASET     = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
FEATURE_MAP = BASE_DIR / "data" / "processed" / "feature_map.json"
CONFIG      = BASE_DIR / "configs" / "triagem_features.json"

MODELS_DIR  = BASE_DIR / "models" / "prod"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_METRIC = "pr_auc"  # métrica de referência para desbalanceados
TEXT_OUT = "text_concat"   # deve bater com ColumnTransformer

# ================================
# Pipeline
# ================================
def build_pipeline(num_cols: list[str], cat_cols: list[str], txt_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if txt_cols:
        # ColumnTransformer recebe o nome da coluna criada pelo TextConcatTransformer
        transformers.append(("txt", TfidfVectorizer(max_features=1000), TEXT_OUT))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.10,
        objective="binary:logistic",
        eval_metric=["logloss", "auc", "aucpr"],
        # scale_pos_weight será definido após construir o pipeline
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("txt_concat", TextConcatTransformer(text_cols=txt_cols, out_col=TEXT_OUT, sep=" ", drop_original=False)),
        ("pre", preprocessor),
        ("clf", clf)
    ])


def main():
    # ================================
    # Carregamento
    # ================================
    df = pd.read_csv(DATASET)
    with open(CONFIG, "r", encoding="utf-8") as f:
        feats = json.load(f)
    with open(FEATURE_MAP, "r", encoding="utf-8") as f:
        fmap = json.load(f)

    num_cols = feats.get("NUM_CANDIDATES", [])
    cat_cols = feats.get("CAT_CANDIDATES", [])
    txt_cols = feats.get("TXT_CANDIDATES", [])
    target_col = "target_triagem"

    group_candidates = fmap.get("group", [])
    group_cols = [c for c in group_candidates if c in df.columns]
    assert group_cols, f"Nenhuma coluna de grupo encontrada. Esperava uma de: {group_candidates}"
    group_col = group_cols[0]

    # ================================
    # Seleção + limpeza de colunas
    # ================================
    requested = num_cols + cat_cols + txt_cols
    present   = [c for c in requested if c in df.columns]
    X = df[present].copy()
    y = df[target_col].astype(int).values
    groups = df[group_col].values

    # drop 100% NaN e constantes (robustez)
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
    # Split com grupos
    # ================================
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_train, y_test = y[tr_idx], y[te_idx]
    groups_test = pd.Series(groups[te_idx])

    # ================================
    # Pipeline + ajuste do peso de classe
    # ================================
    neg = int((y_train == 0).sum())
    pos = max(1, int((y_train == 1).sum()))  # proteção
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight (treino): {scale_pos_weight:.3f}  [neg={neg}, pos={pos}]")

    model = build_pipeline(num_cols, cat_cols, txt_cols)
    model.named_steps["clf"].set_params(scale_pos_weight=scale_pos_weight)

    # ================================
    # MLflow (tracking local opcional)
    # ================================
    mlruns_path = "file:" + str((BASE_DIR / "mlruns").resolve())
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("triagem-candidatos")

    run_name = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("business_goal", "triagem_candidatos")
        mlflow.set_tag("primary_metric", PRIMARY_METRIC)
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.10,
            "scale_pos_weight": scale_pos_weight,
            "tfidf_max_features": 1000,
            "test_size": 0.20,
            "random_state": 42,
            "group_col": group_col,
            "text_cols": ",".join(txt_cols),
        })
        mlflow.log_dict({"num": num_cols, "cat": cat_cols, "txt": txt_cols}, "features_used.json")

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

        mlflow.log_metrics(metrics)

        # ================================
        # Salvamento (compatível com a API)
        # ================================
        model_path = MODELS_DIR / "model.joblib"        # <<--- padronizado para a API
        meta_path  = MODELS_DIR / "meta.json"           # <<--- padronizado para a API
        preds_path = MODELS_DIR / "holdout_predictions.csv"
        refscore_path = MODELS_DIR / "reference_scores.csv"

        dump(model, model_path)

        meta = {
            "type": "xgboost",
            "dataset": DATASET.name,
            "group_col": group_col,
            "target": "target_triagem",
            "features_used": {"num": num_cols, "cat": cat_cols, "txt": txt_cols},
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "class_balance": {"neg": neg, "pos": pos},
            "scale_pos_weight": scale_pos_weight,
            "metrics_holdout": metrics,
            "primary_metric": PRIMARY_METRIC,
            "split": {"test_size": 0.20, "random_state": 42, "type": "GroupShuffleSplit"},
            "version": "1.0.0",
            "mlflow": {"experiment": "triagem-candidatos", "run_id": run.info.run_id},
            # opcional: threshold padrão para a API
            "default_threshold": 0.6
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "score": y_proba,
            "group": groups_test.values
        }).to_csv(preds_path, index=False)

        pd.DataFrame({"score": y_proba}).to_csv(refscore_path, index=False)

        print(f"\nModelo salvo em: {model_path}")
        print(f" Metadados salvos em: {meta_path}")
        print(f" Holdout salvo em: {preds_path}")

        # ================================
        # Artefatos extras (matriz de confusão)
        # ================================
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        fig.tight_layout()
        cm_path = MODELS_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)

        # log como artefatos do run
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(meta_path),  artifact_path="artifacts")
        mlflow.log_artifact(str(preds_path), artifact_path="artifacts")
        mlflow.log_artifact(str(refscore_path), artifact_path="artifacts")
        mlflow.log_artifact(str(cm_path),    artifact_path="artifacts")
        mlflow.log_text(classification_report(y_test, y_pred, digits=3),
                        "artifacts/classification_report.txt")

        # ================================
        # Log do modelo no formato MLflow (signature p/ validação)
        # ================================
        X_ex = X_train.head(5).copy()             # exemplo cru (mesma forma da API)
        y_ex = model.predict_proba(X_ex)[:, 1]
        signature = infer_signature(X_ex, y_ex)

        pip_reqs = "requirements.txt" if (BASE_DIR / "requirements.txt").exists() else None

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_ex,
                signature=signature,
                registered_model_name=None,
                pip_requirements=pip_reqs
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_ex,
                signature=signature,
                registered_model_name=None,
                pip_requirements=pip_reqs
            )

        print(f" MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
