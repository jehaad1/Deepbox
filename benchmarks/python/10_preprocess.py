"""
Benchmark 10 — Preprocessing
scikit-learn
"""

import sys, os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer, KBinsDiscretizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer,
    Binarizer, MultiLabelBinarizer, PolynomialFeatures, SplineTransformer,
)
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, LeaveOneOut, RepeatedKFold,
    ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit, train_test_split,
)
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from utils import run, create_suite, header, footer

warnings.filterwarnings("ignore", message="n_quantiles .* is greater than the total number of samples.*")
warnings.filterwarnings("ignore", message="The current default behavior, quantile_method='linear'.*")

suite = create_suite("preprocess", "scikit-learn")
header("Benchmark 10 — Preprocessing", "scikit-learn")

# ── Data generators ──────────────────────────────────────

rng = np.random.RandomState(42)

X200 = rng.randn(200, 5) * 50
X500 = rng.randn(500, 10) * 50
X1k = rng.randn(1000, 10) * 50
X5k = rng.randn(5000, 10) * 50
XFeat1k = rng.randn(1000, 20) * 50
Xpos500 = np.abs(rng.randn(500, 10)) * 50 + 1
Xpos1k = np.abs(rng.randn(1000, 10)) * 50 + 1
XSpline1k = rng.randn(1000, 1) * 50
y200 = rng.randint(0, 3, 200)
y500 = rng.randint(0, 5, 500)
y1k = rng.randint(0, 5, 1000)
yFeat1k = rng.randint(0, 3, 1000)
groups1k = np.array([i // 10 for i in range(1000)])
yReg1k = np.array([((i * 7) % 31) / 3 for i in range(1000)], dtype=float)
Xmissing500 = rng.randn(500, 10) * 10
Xmissing500[(np.add.outer(np.arange(500), np.arange(10)) % 7) == 0] = np.nan
documents1k = [
    f"{['cat', 'dog', 'fox', 'owl', 'yak'][i % 5]} {['jumps', 'runs', 'sleeps', 'eats', 'looks'][(i * 3) % 5]} in the {['garden', 'forest', 'city', 'desert', 'river'][(i * 7) % 5]}"
    for i in range(1000)
]
multiLabelTargets = [
    [f"label_{i % 5}"] + ([f"group_{i % 4}"] if i % 3 == 0 else [])
    for i in range(1000)
]

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

run(suite, "QuantileTransformer fit", "500x10", lambda: QuantileTransformer(n_quantiles=500).fit(X500))
qt = QuantileTransformer(n_quantiles=500).fit(X500)
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

# ── Advanced v1.0.0 Preprocessing ───────────────────────

run(suite, "KBinsDiscretizer fit+transform", "1Kx10", lambda: KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile", quantile_method="linear").fit_transform(X1k))
run(suite, "SplineTransformer fit+transform", "1Kx1", lambda: SplineTransformer(n_knots=5, degree=3).fit_transform(XSpline1k))
run(suite, "KNNImputer fit+transform", "500x10", lambda: KNNImputer(n_neighbors=3).fit_transform(Xmissing500))
run(suite, "MissingIndicator fit+transform", "500x10", lambda: MissingIndicator().fit_transform(Xmissing500))
run(suite, "SelectKBest fit+transform", "1Kx20", lambda: SelectKBest(score_func=f_classif, k=10).fit_transform(XFeat1k, yFeat1k))
run(suite, "mutual_info_classif", "1Kx20", lambda: mutual_info_classif(XFeat1k, yFeat1k))
run(suite, "f_regression", "1Kx20", lambda: f_regression(XFeat1k, yReg1k))
run(suite, "mutual_info_regression", "1Kx20", lambda: mutual_info_regression(XFeat1k, yReg1k))
run(suite, "SimpleImputer fit+transform", "500x10", lambda: SimpleImputer(strategy="mean").fit_transform(Xmissing500))
run(suite, "Binarizer fit+transform", "1Kx10", lambda: Binarizer(threshold=0.0).fit_transform(X1k))
run(suite, "PolynomialFeatures fit+transform", "200x5→deg2", lambda: PolynomialFeatures(degree=2, include_bias=False).fit_transform(X200))
run(suite, "VarianceThreshold fit+transform", "1Kx20", lambda: VarianceThreshold(threshold=0.1).fit_transform(XFeat1k))
run(suite, "MultiLabelBinarizer fit+transform", "1K label-sets", lambda: MultiLabelBinarizer().fit_transform(multiLabelTargets))
run(suite, "CountVectorizer fit+transform", "1K docs", lambda: CountVectorizer().fit_transform(documents1k))
run(suite, "TfidfVectorizer fit+transform", "1K docs", lambda: TfidfVectorizer().fit_transform(documents1k))
run(suite, "HashingVectorizer transform", "1K docs", lambda: HashingVectorizer(n_features=1024, alternate_sign=True).transform(documents1k))

def shuffle_split_iter():
    splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for _ in splitter.split(X1k): pass

def strat_shuffle_split_iter():
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for _ in splitter.split(X1k, y1k): pass

def ts_split_iter():
    splitter = TimeSeriesSplit(n_splits=5)
    for _ in splitter.split(X1k): pass

def group_kfold_iter():
    splitter = GroupKFold(n_splits=5)
    for _ in splitter.split(X1k, y1k, groups1k): pass

def repeated_kfold_iter():
    splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    for _ in splitter.split(X1k): pass

run(suite, "ShuffleSplit (5)", "1K samples", shuffle_split_iter)
run(suite, "StratifiedShuffleSplit (5)", "1K samples", strat_shuffle_split_iter)
run(suite, "TimeSeriesSplit (5)", "1K samples", ts_split_iter)
run(suite, "GroupKFold (5)", "1K samples", group_kfold_iter)
run(suite, "RepeatedKFold (5x3)", "1K samples", repeated_kfold_iter)
run(suite, "GroupShuffleSplit (5)", "1K samples", lambda: list(GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42).split(X1k, y1k, groups1k)), comparable=False, tags=["sklearn-only"])

footer(suite, "sklearn-preprocess.json")
