# src/ml/pipeline.py
from __future__ import annotations
from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.ml.transformers import (
    TypeCoercionTransformer,
    MatchNivelAcademicoEqTransformer,
    OverlapSkillsJaccardTransformer,
    SimilarityTFIDFTransformer,
)

class ConcatTextTransformer(BaseEstimator, TransformerMixin):
    """Concatena colunas textuais em uma única string por linha (sem lambda)."""
    def __init__(self, cols: List[str]):
        self.cols = list(cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cols = [c for c in self.cols if c in X.columns]
        if not cols:
            return pd.Series([""] * len(X))
        return X[cols].fillna("").astype(str).agg(" ".join, axis=1)

def build_pipeline(cat_cols: List[str], txt_cols: List[str]) -> Pipeline:
    # 1) coerção de tipos para entrada crua
    coerce = TypeCoercionTransformer(cat_cols=cat_cols, txt_cols=txt_cols)

    # 2) features engenheiradas (iguais ao seu FE)
    num_features = FeatureUnion([
        ("match_nivel_academico", MatchNivelAcademicoEqTransformer()),
        ("overlap_skills",        OverlapSkillsJaccardTransformer()),
        ("sim_cv_vaga",           SimilarityTFIDFTransformer(
            max_features=2000, min_df=3, max_df=0.9, ngram_min=1, ngram_max=1
        )),
    ])

    # 3) categóricas (OHE)
    cats = ColumnTransformer(
        transformers=[("cats", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
    )

    # 4) BOW TF-IDF do texto concatenado (sem lambda)
    bow_pipe = Pipeline([
        ("concat", ConcatTextTransformer(txt_cols)),
        ("tfidf",  TfidfVectorizer(max_features=1000)),
    ])

    # 5) classificador
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric="logloss",
        tree_method="hist", verbosity=0,
    )

    # 6) pipeline final
    return Pipeline(steps=[
        ("coerce", coerce),
        ("features", FeatureUnion([
            ("cats_encoded", cats),
            ("nums_engineered", num_features),
            ("bag_of_words",    bow_pipe),
        ])),
        ("clf", clf),
    ])
