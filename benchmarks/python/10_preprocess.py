"""
Benchmark 10 — Preprocessing
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer,
)
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, GroupKFold, LeaveOneOut, LeavePOut,
)
from utils import run, create_suite, header, footer

suite = create_suite("preprocess", "scikit-learn")
header("Benchmark 10 — Preprocessing", "scikit-learn")

# ── Data generators ──────────────────────────────────────

rng = np.random.RandomState(42)

X200 = rng.randn(200, 5) * 50
X500 = rng.randn(500, 10) * 50
X1k = rng.randn(1000, 10) * 50
X5k = rng.randn(5000, 10) * 50
Xpos500 = np.abs(rng.randn(500, 10)) * 50 + 1
Xpos1k = np.abs(rng.randn(1000, 10)) * 50 + 1
y200 = rng.randint(0, 3, 200)
y500 = rng.randint(0, 5, 500)
y1k = rng.randint(0, 5, 1000)

# ── StandardScaler ──────────────────────────────────────

run(suite, "StandardScaler fit", "200x5", lambda: StandardScaler().fit(X200))
run(suite, "StandardScaler fit", "500x10", lambda: StandardScaler().fit(X500))
run(suite, "StandardScaler fit", "1Kx10", lambda: StandardScaler().fit(X1k))
ss = StandardScaler().fit(X500)
run(suite, "StandardScaler transform", "500x10", lambda: ss.transform(X500))
run(suite, "StandardScaler transform", "1Kx10", lambda: ss.transform(X1k))
run(suite, "StandardScaler fit+transform", "5Kx10", lambda: StandardScaler().fit_transform(X5k))

# ── MinMaxScaler ────────────────────────────────────────

run(suite, "MinMaxScaler fit", "200x5", lambda: MinMaxScaler().fit(X200))
run(suite, "MinMaxScaler fit", "500x10", lambda: MinMaxScaler().fit(X500))
run(suite, "MinMaxScaler fit", "1Kx10", lambda: MinMaxScaler().fit(X1k))
mms = MinMaxScaler().fit(X500)
run(suite, "MinMaxScaler transform", "500x10", lambda: mms.transform(X500))
run(suite, "MinMaxScaler transform", "1Kx10", lambda: mms.transform(X1k))

# ── RobustScaler ────────────────────────────────────────

run(suite, "RobustScaler fit", "500x10", lambda: RobustScaler().fit(X500))
run(suite, "RobustScaler fit", "1Kx10", lambda: RobustScaler().fit(X1k))
rs = RobustScaler().fit(X500)
run(suite, "RobustScaler transform", "500x10", lambda: rs.transform(X500))

# ── MaxAbsScaler ────────────────────────────────────────

run(suite, "MaxAbsScaler fit", "500x10", lambda: MaxAbsScaler().fit(X500))
run(suite, "MaxAbsScaler fit", "1Kx10", lambda: MaxAbsScaler().fit(X1k))
mas = MaxAbsScaler().fit(X500)
run(suite, "MaxAbsScaler transform", "500x10", lambda: mas.transform(X500))

# ── Normalizer ──────────────────────────────────────────

run(suite, "Normalizer fit+transform", "500x10", lambda: Normalizer().fit_transform(X500))
run(suite, "Normalizer fit+transform", "1Kx10", lambda: Normalizer().fit_transform(X1k))

# ── PowerTransformer ────────────────────────────────────

run(suite, "PowerTransformer fit", "500x10", lambda: PowerTransformer(method="yeo-johnson").fit(Xpos500))
pt = PowerTransformer(method="yeo-johnson").fit(Xpos500)
run(suite, "PowerTransformer transform", "500x10", lambda: pt.transform(Xpos500))

# ── QuantileTransformer ─────────────────────────────────

run(suite, "QuantileTransformer fit", "500x10", lambda: QuantileTransformer().fit(X500))
qt = QuantileTransformer().fit(X500)
run(suite, "QuantileTransformer transform", "500x10", lambda: qt.transform(X500))

# ── LabelEncoder ────────────────────────────────────────

cats = ["cat", "dog", "fish", "bird", "snake"]
str500 = np.array([cats[i % 5] for i in range(500)])
str1k = np.array([cats[i % 5] for i in range(1000)])

run(suite, "LabelEncoder fit", "500 labels", lambda: LabelEncoder().fit(str500))
run(suite, "LabelEncoder fit", "1K labels", lambda: LabelEncoder().fit(str1k))
le = LabelEncoder().fit(str500)
run(suite, "LabelEncoder transform", "500 labels", lambda: le.transform(str500))
run(suite, "LabelEncoder transform", "1K labels", lambda: le.transform(str1k))

# ── OneHotEncoder ───────────────────────────────────────

str500_2d = str500.reshape(-1, 1)
str1k_2d = str1k.reshape(-1, 1)

run(suite, "OneHotEncoder fit", "500 samples", lambda: OneHotEncoder(sparse_output=False).fit(str500_2d))
run(suite, "OneHotEncoder fit", "1K samples", lambda: OneHotEncoder(sparse_output=False).fit(str1k_2d))
ohe = OneHotEncoder(sparse_output=False).fit(str500_2d)
run(suite, "OneHotEncoder transform", "500 samples", lambda: ohe.transform(str500_2d))

# ── OrdinalEncoder ──────────────────────────────────────

run(suite, "OrdinalEncoder fit", "500 samples", lambda: OrdinalEncoder().fit(str500_2d))
oe = OrdinalEncoder().fit(str500_2d)
run(suite, "OrdinalEncoder transform", "500 samples", lambda: oe.transform(str500_2d))

# ── LabelBinarizer ──────────────────────────────────────

run(suite, "LabelBinarizer fit", "500 samples", lambda: LabelBinarizer().fit(str500))
lbin = LabelBinarizer().fit(str500)
run(suite, "LabelBinarizer transform", "500 samples", lambda: lbin.transform(str500))

# ── trainTestSplit ──────────────────────────────────────

run(suite, "trainTestSplit", "200x5", lambda: train_test_split(X200, y200, test_size=0.2))
run(suite, "trainTestSplit", "500x10", lambda: train_test_split(X500, y500, test_size=0.2))
run(suite, "trainTestSplit", "1Kx10", lambda: train_test_split(X1k, y1k, test_size=0.2))

# ── KFold ───────────────────────────────────────────────

def kfold_iter(n_splits, X):
    kf = KFold(n_splits=n_splits)
    for _ in kf.split(X): pass

run(suite, "KFold (k=5)", "500 samples", lambda: kfold_iter(5, X500))
run(suite, "KFold (k=5)", "1K samples", lambda: kfold_iter(5, X1k))
run(suite, "KFold (k=10)", "1K samples", lambda: kfold_iter(10, X1k))

# ── StratifiedKFold ─────────────────────────────────────

def skfold_iter(n_splits, X, y):
    sf = StratifiedKFold(n_splits=n_splits)
    for _ in sf.split(X, y): pass

run(suite, "StratifiedKFold (k=5)", "500 samples", lambda: skfold_iter(5, X500, y500))
run(suite, "StratifiedKFold (k=5)", "1K samples", lambda: skfold_iter(5, X1k, y1k))

# ── LeaveOneOut ─────────────────────────────────────────

Xsmall = rng.randn(50, 3)

def loo_iter():
    loo = LeaveOneOut()
    for _ in loo.split(Xsmall): pass

run(suite, "LeaveOneOut", "50 samples", loo_iter)

footer(suite, "sklearn-preprocess.json")
