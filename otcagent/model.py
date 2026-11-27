"""The AI prediction model: an ensemble of three different learners.

1. Gradient boosting (HistGradientBoosting) on all indicators + patterns.
2. Logistic regression (scaled) on the same features - a stable, low-variance view.
3. k-nearest-neighbours on the shape of the last 10 candles - "history matching":
   how did the market move after the most similar past situations?

Their probabilities are averaged. One ensemble is trained per expiry.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import sequence_features


class EnsembleModel:
    def __init__(self, random_state: int = 7):
        self.gb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
            l2_regularization=1.0, early_stopping=False, random_state=random_state)
        self.lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.05, max_iter=500))
        self.knn = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 KNeighborsClassifier(n_neighbors=75, weights="distance"))
        self.columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns = list(X.columns)
        y = y.astype(int).values
        self.gb.fit(X.values, y)
        self.lr.fit(X.values, y)
        self.knn.fit(sequence_features(X).values, y)
        return self

    def predict_proba_parts(self, X: pd.DataFrame) -> dict:
        X = X[self.columns]
        return {
            "boosting": self.gb.predict_proba(X.values)[:, 1],
            "logistic": self.lr.predict_proba(X.values)[:, 1],
            "history_knn": self.knn.predict_proba(sequence_features(X).values)[:, 1],
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        parts = self.predict_proba_parts(X)
        return np.mean(np.vstack(list(parts.values())), axis=0)
