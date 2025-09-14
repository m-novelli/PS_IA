# src/ml/train_pipeline.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from joblib import dump

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.ml.pipeline import build_pipeline

# ====== Paths / Config ======
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"
OUT_DIR  = BASE_DIR / "models" / "prod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET    = "target_triagem"
GROUP_COL = "codigo_vaga"

# Mesmo conjunto cru usado no treino anterior (meta.json antigo)
CAT_COLS = [
    "beneficios.valor_compra_1","beneficios.valor_compra_2","beneficios.valor_venda",
    "formacao_e_idiomas.ano_conclusao","formacao_e_idiomas.cursos",
    "formacao_e_idiomas.instituicao_ensino_superior","formacao_e_idiomas.nivel_academico",
    "formacao_e_idiomas.nivel_espanhol","formacao_e_idiomas.nivel_ingles","formacao_e_idiomas.outro_idioma",
    "informacoes_basicas.cliente","informacoes_basicas.empresa_divisao",
    "informacoes_basicas.limite_esperado_para_contratacao","informacoes_basicas.objetivo_vaga",
    "informacoes_basicas.origem_vaga","informacoes_basicas.prazo_contratacao",
    "informacoes_basicas.prioridade_vaga","informacoes_basicas.tipo_contratacao",
    "informacoes_basicas.titulo_vaga","informacoes_basicas.vaga_sap",
    "informacoes_pessoais.download_cv","informacoes_profissionais.area_atuacao",
    "informacoes_profissionais.certificacoes","informacoes_profissionais.conhecimentos_tecnicos",
    "informacoes_profissionais.nivel_profissional","informacoes_profissionais.outras_certificacoes",
    "informacoes_profissionais.remuneracao","informacoes_profissionais.titulo_profissional",
    "infos_basicas.local","modalidade","perfil_vaga.areas_atuacao","perfil_vaga.bairro",
    "perfil_vaga.cidade","perfil_vaga.equipamentos_necessarios","perfil_vaga.estado",
    "perfil_vaga.faixa_etaria","perfil_vaga.habilidades_comportamentais_necessarias",
    "perfil_vaga.horario_trabalho","perfil_vaga.local_trabalho","perfil_vaga.nivel profissional",
    "perfil_vaga.nivel_academico","perfil_vaga.nivel_espanhol","perfil_vaga.nivel_ingles",
    "perfil_vaga.outro_idioma","perfil_vaga.regiao","perfil_vaga.vaga_especifica_para_pcd",
    "perfil_vaga.viagens_requeridas","titulo_vaga"
]
TXT_COLS = [
    "cv_pt",
    "infos_basicas.objetivo_profissional",
    "perfil_vaga.competencia_tecnicas_e_comportamentais",
    "perfil_vaga.demais_observacoes",
    "perfil_vaga.principais_atividades",
]

DEFAULT_THRESHOLD = 0.6
PRIMARY_METRIC = "pr_auc"

def main():
    # ===== Load =====
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in (CAT_COLS + TXT_COLS + [TARGET, GROUP_COL]) if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas ausentes no CSV: {missing[:12]} ...")

    X = df[CAT_COLS + TXT_COLS].copy()
    y = df[TARGET].astype(int)
    groups = df[GROUP_COL].astype(str)

    # ===== Split por grupo =====
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    groups_test = groups.iloc[te]

    # ===== Pipeline + class weights =====
    pipe = build_pipeline(cat_cols=CAT_COLS, txt_cols=TXT_COLS)
    neg = int((y_tr == 0).sum()); pos = max(1, int((y_tr == 1).sum()))
    spw = neg / pos
    pipe.named_steps["clf"].set_params(scale_pos_weight=spw)

    # ===== MLflow setup =====
    mlruns_path = "file:" + str((BASE_DIR / "mlruns").resolve())
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("triagem-candidatos")

    run_name = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Params/Tags
        mlflow.set_tag("business_goal", "triagem_candidatos")
        mlflow.set_tag("primary_metric", PRIMARY_METRIC)
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": spw,
            "tfidf_max_features": 1000,
            "sim_tfidf_max_features": 2000,
            "test_size": 0.20,
            "random_state": 42,
            "group_col": GROUP_COL,
            "dataset": CSV_PATH.name,
            "default_threshold": DEFAULT_THRESHOLD,
        })
        mlflow.log_dict({"cat": CAT_COLS, "txt": TXT_COLS}, "features_used.json")

        # ===== Fit =====
        pipe.fit(X_tr, y_tr)

        # ===== Avaliação =====
        proba = pipe.predict_proba(X_te)[:, 1]
        y_hat = (proba >= DEFAULT_THRESHOLD).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc":  float(average_precision_score(y_te, proba)),
            "f1":      float(f1_score(y_te, y_hat)),
        }
        print(metrics)
        mlflow.log_metrics(metrics)

        # ===== Artefatos locais =====
        model_path = OUT_DIR / "model.joblib"
        meta_path  = OUT_DIR / "meta.json"
        preds_path = OUT_DIR / "holdout_predictions.csv"
        ref_path   = OUT_DIR / "reference_scores.csv"

        dump(pipe, model_path)

        pd.DataFrame({
            "y_true": y_te.values,
            "y_pred": y_hat,
            "score": proba,
            "group": groups_test.values
        }).to_csv(preds_path, index=False)
        pd.DataFrame({"score": proba}).to_csv(ref_path, index=False)

        # Matriz de confusão + relatório
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_te, y_hat)
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        fig.tight_layout()
        cm_path = OUT_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=120); plt.close(fig)

        # Log artefatos no MLflow
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(preds_path), artifact_path="artifacts")
        mlflow.log_artifact(str(ref_path),   artifact_path="artifacts")
        mlflow.log_artifact(str(cm_path),    artifact_path="artifacts")

        # Classification report como texto
        from sklearn.metrics import classification_report
        report_txt = classification_report(y_te, y_hat, digits=3)
        (OUT_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
        mlflow.log_text(report_txt, "artifacts/classification_report.txt")

        # ===== Meta.json compatível com API =====
        meta = {
            "type": "sklearn-pipeline-xgboost",
            "dataset": CSV_PATH.name,
            "group_col": GROUP_COL,
            "target": TARGET,
            "schema_in": {
                "cat": CAT_COLS,
                "txt": TXT_COLS,
                "num_engineered": ["match_nivel_academico","overlap_skills","sim_cv_vaga"],
                "bow": {"tfidf_max_features": 1000}
            },
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "class_balance": {"neg": neg, "pos": pos},
            "scale_pos_weight": spw,
            "metrics_holdout": metrics,
            "primary_metric": PRIMARY_METRIC,
            "split": {"test_size": 0.20, "random_state": 42, "type": "GroupShuffleSplit"},
            "version": "2.0.0",
            "default_threshold": DEFAULT_THRESHOLD,
            "mlflow": {"experiment": "triagem-candidatos", "run_id": run.info.run_id},
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # ===== Log modelo no MLflow (com assinatura) =====
        X_ex = X_tr.head(5).copy()
        y_ex = pipe.predict_proba(X_ex)[:, 1]
        signature = infer_signature(X_ex, y_ex)

        try:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                input_example=X_ex,
                signature=signature,
                registered_model_name=None
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                input_example=X_ex,
                signature=signature,
                registered_model_name=None
            )

        print(f"[OK] modelo salvo em {model_path}")
        print(f"[OK] meta salvo em    {meta_path}")
        print(f"[OK] holdout          {preds_path}")

if __name__ == "__main__":
    main()
