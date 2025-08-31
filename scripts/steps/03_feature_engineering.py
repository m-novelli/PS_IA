from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"

# ================================
# Utilitários de texto
# ================================
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

# ================================
# Pipeline principal
# ================================
def main():
    assert INPUT.exists(), f"Arquivo não encontrado: {INPUT}"
    df = pd.read_csv(INPUT)
    print(f"Base carregada: {df.shape}")

    # === Carrega feature_map para preservar grupos/targets no output ===
    fmap_path = BASE_DIR / "data" / "processed" / "feature_map.json"
    fmap = {}
    if fmap_path.exists():
        fmap = json.loads(fmap_path.read_text(encoding="utf-8"))
    group_cols = [c for c in fmap.get("group", ["codigo_vaga", "codigo_applicant"]) if c in df.columns]
    target_cols = [c for c in fmap.get("target", ["target_triagem", "status_simplificado", "target_contratacao"]) if c in df.columns]

    assert len(target_cols) > 0, "Nenhuma coluna de target encontrada no INPUT."
    # escolha um target principal (prioridade para target_triagem)
    target_main = next((c for c in ["target_triagem", "status_simplificado", "target_contratacao"] if c in target_cols), target_cols[0])

    # Verificação de colunas obrigatórias de FE
    required_cols = [
        "perfil_vaga.nivel_academico",
        "formacao_e_idiomas.nivel_academico",
        "perfil_vaga.competencia_tecnicas_e_comportamentais",
        "informacoes_profissionais.conhecimentos_tecnicos",
        "cv_pt",
        "perfil_vaga.principais_atividades"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")

    # ========== 1) Match direto de nível acadêmico ==========
    df["match_nivel_academico"] = (
        df["perfil_vaga.nivel_academico"].fillna("").str.lower().str.strip() ==
        df["formacao_e_idiomas.nivel_academico"].fillna("").str.lower().str.strip()
    ).astype(int)

    # ========== 2) Overlap de competências exigidas x oferecidas ==========
    df["overlap_skills"] = df.apply(lambda row: jaccard_overlap(
        row.get("perfil_vaga.competencia_tecnicas_e_comportamentais", ""),
        row.get("informacoes_profissionais.conhecimentos_tecnicos", "")
    ), axis=1)

    # ========== 3) Similaridade textual CV vs atividades da vaga (batch) ==========
    cv_series   = df["cv_pt"].fillna("").astype(str).map(normalize_text)
    vaga_series = df["perfil_vaga.principais_atividades"].fillna("").astype(str).map(normalize_text)

    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_PT, max_features=1000)
    # Ajusta no corpus conjunto para ter vocabulário de ambos
    vectorizer.fit(pd.concat([cv_series, vaga_series], axis=0))

    X_cv   = vectorizer.transform(cv_series)
    X_vaga = vectorizer.transform(vaga_series)

    # TF-IDF já é L2-normalizado por padrão, mas normalizamos explicitamente
    X_cv_n   = normalize(X_cv, norm="l2", copy=False)
    X_vaga_n = normalize(X_vaga, norm="l2", copy=False)

    # Similaridade coseno linha a linha (sparse)
    df["sim_cv_vaga"] = (X_cv_n.multiply(X_vaga_n)).sum(axis=1).A1

    # ========== Monta o output ==========
    keep_always = list({target_main, *group_cols})
    fe_cols = ["match_nivel_academico", "overlap_skills", "sim_cv_vaga"]

    out_cols = list(df.columns)
    for c in fe_cols:
        if c not in out_cols:
            out_cols.append(c)

    # sanity: grupo/target presentes
    assert any(c in out_cols for c in group_cols), f"Coluna(s) de grupo ausente(s) no output: {group_cols}"
    assert target_main in out_cols, f"Target {target_main} ausente no output."

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT, index=False)
    print(f"Arquivo salvo: {OUTPUT} | Shape final: {df[out_cols].shape}")
    print(f"[OK] Grupo(s) no output: {group_cols} | Target principal: {target_main}")


if __name__ == "__main__":
    main()
