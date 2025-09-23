import pytest
import pandas as pd
from pathlib import Path
import src.ml.train_pipeline as tp

def test_missing_columns(tmp_path, monkeypatch):
    # criar CSV com colunas erradas
    df = pd.DataFrame({"x": [1,2,3], "y": [0,1,0]})
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)

    monkeypatch.setattr(tp, "CSV_PATH", csv)
    with pytest.raises(SystemExit):
        tp.main()

def test_train_pipeline_minimal(monkeypatch, tmp_path):
    # dataset fake com colunas mínimas
    df = pd.DataFrame({
        "codigo_vaga": ["1","2","3","4"],
        "target_triagem": [0,1,0,1],
        "cv_pt": ["texto"]*4,
        "infos_basicas.objetivo_profissional": ["a"]*4,
        "perfil_vaga.competencia_tecnicas_e_comportamentais": ["b"]*4,
        "perfil_vaga.demais_observacoes": ["c"]*4,
        "perfil_vaga.principais_atividades": ["d"]*4,
    })
    # adicionar colunas categóricas obrigatórias com valores dummy
    for col in tp.CAT_COLS:
        df[col] = "x"

    csv = tmp_path / "ok.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(tp, "CSV_PATH", csv)
    monkeypatch.setattr(tp, "OUT_DIR", tmp_path)

    # mock mlflow para não logar de verdade
    class DummyRun:
        info = type("x", (), {"run_id": "123"})
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(tp.mlflow, "start_run", lambda run_name=None: DummyRun())
    monkeypatch.setattr(tp.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tp.mlflow, "log_dict", lambda *a, **k: None)
    monkeypatch.setattr(tp.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(tp.mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(tp.mlflow, "log_text", lambda *a, **k: None)

    # rodar main (deve treinar e salvar arquivos)
    tp.main()
    assert (tmp_path / "model.joblib").exists()