from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== Stopwords em português com fallback se não estiverem disponíveis ====
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS_PT = stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")
    STOPWORDS_PT = stopwords.words("portuguese")

# ================================
# Caminhos
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"

# ================================
# Utilitários de texto
# ================================
import re
import unicodedata

def normalize_text(s):
    """Minimiza e remove acentuação e pontuação."""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = re.sub(r"[^a-z\s]", "", s)
    return s

def jaccard_overlap(text1, text2):
    """Calcula sobreposição Jaccard entre dois textos."""
    set1 = set(normalize_text(text1).split())
    set2 = set(normalize_text(text2).split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def cosine_sim(text1, text2, vectorizer):
    """Calcula similaridade cosseno entre dois textos."""
    vecs = vectorizer.transform([normalize_text(text1), normalize_text(text2)])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

# ================================
# Pipeline principal
# ================================
def main():
    assert INPUT.exists(), f"Arquivo não encontrado: {INPUT}"
    df = pd.read_csv(INPUT)
    print(f"Base carregada: {df.shape}")

    # Verificação de colunas obrigatórias
    required_cols = [
        "perfil_vaga.nivel_academico",
        "formacao_e_idiomas.nivel_academico",
        "perfil_vaga.competencia_tecnicas_e_comportamentais",
        "informacoes_profissionais.conhecimentos_tecnicos",
        "cv_pt",
        "perfil_vaga.principais_atividades"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"Colunas ausentes: {missing_cols}"

    # ========== 1. Match direto de nível acadêmico ==========
    df["match_nivel_academico"] = (
        df["perfil_vaga.nivel_academico"].fillna("").str.lower().str.strip() ==
        df["formacao_e_idiomas.nivel_academico"].fillna("").str.lower().str.strip()
    ).astype(int)

    # ========== 2. Overlap de competências exigidas x oferecidas ==========
    df["overlap_skills"] = df.apply(lambda row: jaccard_overlap(
        row.get("perfil_vaga.competencia_tecnicas_e_comportamentais", ""),
        row.get("informacoes_profissionais.conhecimentos_tecnicos", "")
    ), axis=1)

    # ========== 3. Similaridade textual entre CV e atividades da vaga ==========
    textos_comb = (
        df["cv_pt"].fillna("").astype(str) + " " +
        df["perfil_vaga.principais_atividades"].fillna("").astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_PT, max_features=1000)
    vectorizer.fit(textos_comb)

    df["sim_cv_vaga"] = df.apply(lambda row: cosine_sim(
        row.get("cv_pt", ""),
        row.get("perfil_vaga.principais_atividades", ""),
        vectorizer
    ), axis=1)

    # ========== Salvar ==========
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Recarrega coluna de target da base original, caso necessário
    df_target = pd.read_csv(BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv", usecols=["target_triagem"])
    df["target_triagem"] = df_target["target_triagem"]


    df.to_csv(OUTPUT, index=False)
    print(f"Arquivo salvo: {OUTPUT} | Shape final: {df.shape}")

if __name__ == "__main__":
    main()
