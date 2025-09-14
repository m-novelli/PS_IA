from __future__ import annotations
from typing import Iterable, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class TextConcatTransformer(BaseEstimator, TransformerMixin):
    """
    Concatena colunas textuais em uma única coluna (default: 'text_concat').
    Mantém colunas originais por padrão; pode opcionalmente removê-las.
    """

    def __init__(
        self,
        text_cols: Iterable[str] | None,
        out_col: str = "text_concat",
        sep: str = " ",
        drop_original: bool = False,
    ):
        self.text_cols = list(text_cols) if text_cols else []
        self.out_col = out_col
        self.sep = sep
        self.drop_original = drop_original

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # guarda apenas colunas que existem no fit
        self._effective_cols_: List[str] = [c for c in self.text_cols if c in X.columns]
        # metadado padrão sklearn
        self.feature_names_in_ = getattr(X, "columns", None)
        return self

    def transform(self, X):
        check_is_fitted(self, "_effective_cols_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        cols = [c for c in self._effective_cols_ if c in X.columns]

        if len(cols) == 0:
            # garante a presença da coluna de saída
            X[self.out_col] = ""
        elif len(cols) == 1:
            col = cols[0]
            # fillna antes de astype(str) para evitar "nan"
            X[self.out_col] = X[col].fillna("").astype(str)
        else:
            X[self.out_col] = (
                X[cols]
                .fillna("")           # primeiro
                .astype(str)          # depois
                .agg(self.sep.join, axis=1)
            )

        if self.drop_original and cols:
            X = X.drop(columns=cols, errors="ignore")

        return X

    def get_feature_names_out(self, input_features=None):
        return [self.out_col]
