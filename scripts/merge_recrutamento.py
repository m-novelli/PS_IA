# scripts/merge_recrutamento.py

import pandas as pd
import json
from pathlib import Path

# Caminhos dos arquivos
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "processed" / "dataset_final.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Carregando arquivos JSON...")
    jobs = load_json(RAW_DIR / "Jobs.json")
    prospects = load_json(RAW_DIR / "Prospects.json")
    applicants = load_json(RAW_DIR / "Applicants.json")

    print("Flattening: vagas...")
    df_jobs = pd.json_normalize(jobs, sep="_')
    df_jobs.rename(columns={"codigo": "vaga_id"}, inplace=True)

    print("Flattening: candidatos...")
    df_applicants = pd.json_normalize(applicants, sep="_")
    df_applicants.rename(columns={"codigo": "candidato_id"}, inplace=True)

    print("Explodindo prospects (candidatos por vaga)...")
    df_prospects = []
    for vaga_id, prospecoes in prospects.items():
        for prospect in prospecoes:
            row = prospect.copy()
            row["vaga_id"] = vaga_id
            df_prospects.append(row)

    df_prospects = pd.DataFrame(df_prospects)
    df_prospects.rename(columns={"codigo": "candidato_id"}, inplace=True)

    print("Juntando as tabelas...")
    df = df_prospects.merge(df_jobs, on="vaga_id", how="left")
    df = df.merge(df_applicants, on="candidato_id", how="left")

    print("Limpando dados...")
    df["vaga_id"] = df["vaga_id"].astype(str)
    df["candidato_id"] = df["candidato_id"].astype(str)
    df["status"] = df["situacao"].str.lower().str.strip()

    print("💾 Salvando CSV final...")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nBase final salva em: {OUTPUT_PATH}")
    print(f"Total de linhas: {len(df)} — colunas: {len(df.columns)}")


if __name__ == "__main__":
    main()
