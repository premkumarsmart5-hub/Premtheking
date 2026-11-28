"""The AI prediction model: an ensemble of three different learners.

1. Gradient boosting (histogram trees) on all indicators + patterns.
2. Logistic regression (L2, standardised) on the same features - a stable view.
3. k-nearest-neighbours on the shape of the last 10 candles - "history matching":
   how did the market move after the most similar past situations?

Their probabilities are averaged. One ensemble is trained per expiry.
Pure numpy (no scikit-learn) so it installs on Termux / Android.
"""
import numpy as np
import pandas as pd

from .features import sequence_features


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class _Scaler:
    """Median imputation + standardisation."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        Xi = self._impute(X)
        self.mu = Xi.mean(axis=0)
        self.sd = Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def _impute(self, X):
        X = np.array(X, dtype=float, copy=True)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.med[c]
        return X

    def transform(self, X):
        return (self._impute(X) - self.mu) / self.sd


class LogisticModel:
    def __init__(self, l2=20.0, iters=25):
        self.l2, self.iters = l2, iters

    def fit(self, X, y):
        self.sc = _Scaler().fit(X)
        Z = np.c_[np.ones(len(X)), self.sc.transform(X)]
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], self.l2)
        reg[0] = 0.0  # no penalty on the intercept
        for _ in range(self.iters):  # Newton-Raphson
            p = _sigmoid(Z @ w)
            g = Z.T @ (p - y) + reg * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict_proba(self, X):
        return _sigmoid(np.c_[np.ones(len(X)), self.sc.transform(X)] @ self.w)


class KNNModel:
    def __init__(self, k=75):
        self.k = k

    def fit(self, X, y):
        self.sc = _Scaler().fit(X)
        self.X = self.sc.transform(X)
        self.sq = (self.X ** 2).sum(axis=1)
        self.y = np.asarray(y, dtype=float)
        return self

    def predict_proba(self, X):
        Q = self.sc.transform(X)
        k = min(self.k, len(self.X))
        out = np.empty(len(Q))
        step = max(16, int(4_000_000 / max(1, len(self.X))))  # keep memory ~32 MB per block
        for s in range(0, len(Q), step):
            q = Q[s:s + step]
            d2 = (q ** 2).sum(axis=1)[:, None] + self.sq[None, :] - 2 * q @ self.X.T
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            d = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0)) + 1e-9
            w = 1.0 / d
            out[s:s + step] = (w * self.y[idx]).sum(axis=1) / w.sum(axis=1)
        return out


class BoostingModel:
    """Gradient-boosted depth-limited histogram trees with log-loss."""

    def __init__(self, n_trees=150, lr=0.05, depth=3, bins=32, min_leaf=60, l2=1.0,
                 subsample=0.8, seed=7, colsample=1.0, always=()):
        self.n_trees, self.lr, self.depth, self.bins = n_trees, lr, depth, bins
        self.min_leaf, self.l2, self.subsample, self.seed = min_leaf, l2, subsample, seed
        self.colsample = colsample
        self.always = tuple(always)  # feature indices every tree may use (e.g. pair identity)

    def _bin(self, X):
        Xb = np.zeros(X.shape, dtype=np.int64)  # bin 0 = missing value
        for j, e in enumerate(self.edges):
            col = X[:, j]
            ok = ~np.isnan(col)
            Xb[ok, j] = np.searchsorted(e, col[ok], side="right") + 1
        return Xb

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, F = X.shape
        qs = np.linspace(0, 1, self.bins)[1:-1]
        self.edges = []
        for j in range(F):
            col = X[~np.isnan(X[:, j]), j]
            self.edges.append(np.unique(np.quantile(col, qs)) if len(col) else np.array([]))
        Xb = self._bin(X).astype(np.int32)
        B = self.bins + 1
        rng = np.random.default_rng(self.seed)
        m = y.mean()
        self.base = float(np.log(np.clip(m, 1e-6, 1 - 1e-6) / np.clip(1 - m, 1e-6, 1)))
        raw = np.full(n, self.base)
        self.trees = []
        rows = np.arange(n)
        n_cols = max(1, int(round(F * self.colsample)))
        for _ in range(self.n_trees):
            p = _sigmoid(raw)
            g, h = p - y, p * (1 - p)
            use = np.where(rng.random(n) < self.subsample)[0]
            if n_cols < F:
                cols = np.union1d(rng.choice(F, n_cols, replace=False), np.array(self.always, dtype=int))
            else:
                cols = np.arange(F)
            Fs = len(cols)
            sub = Xb[np.ix_(use, cols)] + (np.arange(Fs, dtype=np.int32) * B)[None, :]
            gu, hu = g[use], h[use]
            gw = np.broadcast_to(gu[:, None], sub.shape).ravel()
            hw = np.broadcast_to(hu[:, None], sub.shape).ravel()
            node_u = np.zeros(len(use), dtype=np.int32)
            node = np.zeros(n, dtype=np.int64)
            splits = []
            for d in range(self.depth):
                nn = 2 ** d
                idx = (sub + (node_u * (Fs * B))[:, None]).ravel()
                size = nn * Fs * B
                G = np.bincount(idx, gw, size).reshape(nn, Fs, B)
                Hh = np.bincount(idx, hw, size).reshape(nn, Fs, B)
                C = np.bincount(idx, minlength=size).reshape(nn, Fs, B)
                GL, HL, CL = G.cumsum(2), Hh.cumsum(2), C.cumsum(2)
                Gt, Ht, Ct = GL[:, :, -1:], HL[:, :, -1:], CL[:, :, -1:]
                GR, HR, CR = Gt - GL, Ht - HL, Ct - CL
                gain = GL ** 2 / (HL + self.l2) + GR ** 2 / (HR + self.l2) - Gt ** 2 / (Ht + self.l2)
                gain[(CL < self.min_leaf) | (CR < self.min_leaf)] = -np.inf
                flat = gain.reshape(nn, -1)
                best = flat.argmax(axis=1)
                bgain = flat[np.arange(nn), best]
                feat, thr = cols[best // B], best % B
                thr = np.where(bgain > 1e-12, thr, B)  # no useful split -> everything left
                splits.append((feat, thr))
                node = node * 2 + (Xb[rows, feat[node]] > thr[node])
                node_u = node[use].astype(np.int32)
            leaves = 2 ** self.depth
            Gs = np.bincount(node_u, gu, leaves)
            Hs = np.bincount(node_u, hu, leaves)
            val = -self.lr * Gs / (Hs + self.l2)
            raw += val[node]
            self.trees.append((splits, val))
        return self

    def predict_proba(self, X):
        Xb = self._bin(np.asarray(X, dtype=float))
        n = len(Xb)
        raw = np.full(n, self.base)
        rows = np.arange(n)
        for splits, val in self.trees:
            node = np.zeros(n, dtype=np.int64)
            for feat, thr in splits:
                node = node * 2 + (Xb[rows, feat[node]] > thr[node])
            raw += val[node]
        return _sigmoid(raw)


class EnsembleModel:
    def __init__(self, random_state: int = 7, fast: bool = False):
        # fast: each tree sees 50% of rows and 30% of features (like LightGBM's bagging /
        # feature_fraction) - much quicker on many features, and less overfitting
        self.gb = (BoostingModel(seed=random_state, subsample=0.5, colsample=0.3) if fast
                   else BoostingModel(seed=random_state))
        self.lr = LogisticModel()
        self.knn = KNNModel()
        self.columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns = list(X.columns)
        self.gb.always = tuple(i for i, c in enumerate(self.columns)
                               if c.startswith("pair_") or c == "streak")
        yv = y.astype(float).values
        self.gb.fit(X.values, yv)
        self.lr.fit(X.values, yv)
        self.knn.fit(sequence_features(X).values, yv)
        return self

    def predict_proba_parts(self, X: pd.DataFrame) -> dict:
        X = X[self.columns]
        return {
            "boosting": self.gb.predict_proba(X.values),
            "logistic": self.lr.predict_proba(X.values),
            "history_knn": self.knn.predict_proba(sequence_features(X).values),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        parts = self.predict_proba_parts(X)
        return np.mean(np.vstack(list(parts.values())), axis=0)
