import json
import pandas as pd
from pathlib import Path
import importlib.util

# importa o módulo original
module_path = Path("src/preprocess/01_json_to_df.py")
spec = importlib.util.spec_from_file_location("json_to_df_module", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_final_csv_columns(tmp_path):
    # cria dados fake
    vagas_data = {"vaga1": {"titulo": "Dev", "modalidade": "remoto"}}
    applicants_data = {"c1": {"nome": "Ana", "email": "ana@email.com"}}
    prospects_data = {
        "vaga1": {
            "titulo": "Dev",
            "modalidade": "remoto",
            "prospects": [{"codigo": "c1", "nome": "Ana", "situacao_candidado": "ok"}],
        }
    }

    # salva jsons
    (tmp_path / "vagas.json").write_text(json.dumps(vagas_data), encoding="utf-8")
    (tmp_path / "applicants.json").write_text(json.dumps(applicants_data), encoding="utf-8")
    (tmp_path / "prospects.json").write_text(json.dumps(prospects_data), encoding="utf-8")

    # carrega
    vagas = mod.load_dict_json_flat(tmp_path / "vagas.json", "codigo_vaga")
    applicants = mod.load_dict_json_flat(tmp_path / "applicants.json", "codigo_applicant")
    prospects = mod.load_prospects(tmp_path / "prospects.json")

    df_total = vagas.merge(prospects, on="codigo_vaga", how="left")
    df_total = df_total.merge(applicants, on="codigo_applicant", how="left", suffixes=("", "_applicant"))

    # aplica as mesmas limpezas do script
    if "titulo_vaga" in df_total.columns:
        df_total = df_total.drop(columns=["titulo"], errors="ignore")
        df_total = df_total.rename(columns={"titulo_vaga": "titulo"})
    if "modalidade_x" in df_total.columns and "modalidade_y" in df_total.columns:
        df_total = df_total.drop(columns=["modalidade_y"], errors="ignore")
        df_total = df_total.rename(columns={"modalidade_x": "modalidade"})

    # checa se só tem as colunas corretas
    expected_cols = {
        "codigo_vaga","modalidade","titulo","codigo_applicant",
        "nome_candidato","situacao_candidato","data_candidatura",
        "ultima_atualizacao","comentario","recrutador",
        "nome","email"
    }
    assert set(df_total.columns) == expected_cols