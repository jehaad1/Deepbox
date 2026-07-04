"""
Benchmark 05 — Machine Learning
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.cluster import Birch, MeanShift, OPTICS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import run, create_suite, header, footer

suite = create_suite("ml", "scikit-learn")
header("Benchmark 05 — Machine Learning", "scikit-learn")

# ── Data generators ──────────────────────────────────────

rng = np.random.RandomState(42)

def make_reg(n, f, seed=42):
    r = np.random.RandomState(seed)
    X = r.rand(n, f) * 10
    w = np.arange(1, f + 1, dtype=float)
    y = X @ w + (r.rand(n) - 0.5) * 2
    return X, y

def make_cls(n, f, seed=42):
    r = np.random.RandomState(seed)
    X = r.rand(n, f) * 10
    y = (X.sum(axis=1) > f * 5).astype(int)
    return X, y

def make_multiclass(n, f, k=3, seed=42):
    r = np.random.RandomState(seed)
    X = np.zeros((n, f), dtype=float)
    y = np.arange(n) % k
    for i in range(n):
        cls = y[i]
        X[i] = cls * 2.5 + r.rand(f) * 1.25
    return X, y

def make_count_cls(n, f, k=3, seed=42):
    r = np.random.RandomState(seed)
    X = np.zeros((n, f), dtype=float)
    y = np.arange(n) % k
    for i in range(n):
        cls = y[i]
        X[i] = cls + r.randint(0, 4, size=f)
    return X, y

def make_binary_cls(n, f, k=3, seed=42):
    r = np.random.RandomState(seed)
    X = np.zeros((n, f), dtype=float)
    y = np.arange(n) % k
    for i in range(n):
        cls = y[i]
        thresholds = np.array([0.25 if (cls + j) % 3 == 0 else 0.55 for j in range(f)])
        X[i] = (r.rand(f) > thresholds).astype(float)
    return X, y

def make_clust(n, f, seed=42):
    r = np.random.RandomState(seed)
    X = np.vstack([r.rand(n // 3, f) * 2 + i * 5 for i in range(3)])
    return X[:n]

Xr200, yr200 = make_reg(200, 5)
Xr500, yr500 = make_reg(500, 10)
Xc200, yc200 = make_cls(200, 5)
Xc500, yc500 = make_cls(500, 10)
Xmulti200, ymulti200 = make_multiclass(200, 5, 3, 42)
Xcl120, _ycl120 = make_multiclass(120, 2, 3, 84)
Xcount200, ycount200 = make_count_cls(200, 8, 3, 17)
Xcount500, ycount500 = make_count_cls(500, 12, 3, 23)
Xbinary200, ybinary200 = make_binary_cls(200, 8, 3, 29)
Xcl200 = make_clust(200, 5)
Xcl500 = make_clust(500, 5)

# ── Linear Regression ───────────────────────────────────

run(suite, "LinearRegression fit", "200x5", lambda: LinearRegression().fit(Xr200, yr200))
run(suite, "LinearRegression fit", "500x10", lambda: LinearRegression().fit(Xr500, yr500))
lr = LinearRegression().fit(Xr200, yr200)
run(suite, "LinearRegression predict", "200x5", lambda: lr.predict(Xr200))

# ── Ridge ───────────────────────────────────────────────

run(suite, "Ridge fit", "200x5", lambda: Ridge(alpha=1.0).fit(Xr200, yr200))
run(suite, "Ridge fit", "500x10", lambda: Ridge(alpha=1.0).fit(Xr500, yr500))
ridge = Ridge(alpha=1.0).fit(Xr200, yr200)
run(suite, "Ridge predict", "200x5", lambda: ridge.predict(Xr200))

# ── Bayesian Ridge ──────────────────────────────────────

run(suite, "BayesianRidge fit", "200x5", lambda: BayesianRidge().fit(Xr200, yr200))
bayes_ridge = BayesianRidge().fit(Xr200, yr200)
run(suite, "BayesianRidge predict", "200x5", lambda: bayes_ridge.predict(Xr200))

# ── Lasso ───────────────────────────────────────────────

run(suite, "Lasso fit", "200x5", lambda: Lasso(alpha=0.1).fit(Xr200, yr200))
run(suite, "Lasso fit", "500x10", lambda: Lasso(alpha=0.1).fit(Xr500, yr500))
lasso = Lasso(alpha=0.1).fit(Xr200, yr200)
run(suite, "Lasso predict", "200x5", lambda: lasso.predict(Xr200))

# ── ElasticNet ──────────────────────────────────────────

run(suite, "ElasticNet fit", "200x5", lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42).fit(Xr200, yr200))
elastic_net = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42).fit(Xr200, yr200)
run(suite, "ElasticNet predict", "200x5", lambda: elastic_net.predict(Xr200))

# ── Logistic Regression ─────────────────────────────────

run(suite, "LogisticRegression fit", "200x5", lambda: LogisticRegression(max_iter=200).fit(Xc200, yc200))
run(suite, "LogisticRegression fit", "500x10", lambda: LogisticRegression(max_iter=200).fit(Xc500, yc500))
logreg = LogisticRegression(max_iter=200).fit(Xc200, yc200)
run(suite, "LogisticRegression predict", "200x5", lambda: logreg.predict(Xc200))

# ── Linear Discriminant Analysis ────────────────────────

run(suite, "LinearDiscriminantAnalysis fit", "200x5", lambda: LinearDiscriminantAnalysis(n_components=2).fit(Xmulti200, ymulti200))
lda = LinearDiscriminantAnalysis(n_components=2).fit(Xmulti200, ymulti200)
run(suite, "LinearDiscriminantAnalysis transform", "200x5→2", lambda: lda.transform(Xmulti200))

# ── GaussianNB ──────────────────────────────────────────

run(suite, "GaussianNB fit", "200x5", lambda: GaussianNB().fit(Xc200, yc200))
run(suite, "GaussianNB fit", "500x10", lambda: GaussianNB().fit(Xc500, yc500))
gnb = GaussianNB().fit(Xc200, yc200)
run(suite, "GaussianNB predict", "200x5", lambda: gnb.predict(Xc200))

# ── BernoulliNB ─────────────────────────────────────────

run(suite, "BernoulliNB fit", "200x8", lambda: BernoulliNB().fit(Xbinary200, ybinary200))
bernoulli = BernoulliNB().fit(Xbinary200, ybinary200)
run(suite, "BernoulliNB predict", "200x8", lambda: bernoulli.predict(Xbinary200))

# ── MultinomialNB ───────────────────────────────────────

run(suite, "MultinomialNB fit", "200x8", lambda: MultinomialNB().fit(Xcount200, ycount200))
run(suite, "MultinomialNB fit", "500x12", lambda: MultinomialNB().fit(Xcount500, ycount500))
multinomial = MultinomialNB().fit(Xcount200, ycount200)
run(suite, "MultinomialNB predict", "200x8", lambda: multinomial.predict(Xcount200))

# ── KNN Classifier ──────────────────────────────────────

run(suite, "KNeighborsClassifier fit", "200x5", lambda: KNeighborsClassifier(n_neighbors=5).fit(Xc200, yc200))
run(suite, "KNeighborsClassifier fit", "500x10", lambda: KNeighborsClassifier(n_neighbors=5).fit(Xc500, yc500))
knnc = KNeighborsClassifier(n_neighbors=5).fit(Xc200, yc200)
run(suite, "KNeighborsClassifier predict", "200x5", lambda: knnc.predict(Xc200))

# ── Radius-based Neighbors ──────────────────────────────

run(suite, "RadiusNeighborsClassifier fit", "200x5", lambda: RadiusNeighborsClassifier(radius=2.5).fit(Xmulti200, ymulti200))
radius_cls = RadiusNeighborsClassifier(radius=2.5).fit(Xmulti200, ymulti200)
run(suite, "RadiusNeighborsClassifier predict", "200x5", lambda: radius_cls.predict(Xmulti200))

# ── Nearest Centroid ────────────────────────────────────

run(suite, "NearestCentroid fit", "200x5", lambda: NearestCentroid().fit(Xmulti200, ymulti200))
nearest_centroid = NearestCentroid().fit(Xmulti200, ymulti200)
run(suite, "NearestCentroid predict", "200x5", lambda: nearest_centroid.predict(Xmulti200))

# ── KNN Regressor ───────────────────────────────────────

run(suite, "KNeighborsRegressor fit", "200x5", lambda: KNeighborsRegressor(n_neighbors=5).fit(Xr200, yr200))
knnr = KNeighborsRegressor(n_neighbors=5).fit(Xr200, yr200)
run(suite, "KNeighborsRegressor predict", "200x5", lambda: knnr.predict(Xr200))

# ── LinearSVC ───────────────────────────────────────────

run(suite, "LinearSVC fit", "200x5", lambda: LinearSVC(max_iter=200, dual=True).fit(Xc200, yc200))
run(suite, "LinearSVC fit", "500x10", lambda: LinearSVC(max_iter=200, dual=True).fit(Xc500, yc500))
svc = LinearSVC(max_iter=200, dual=True).fit(Xc200, yc200)
run(suite, "LinearSVC predict", "200x5", lambda: svc.predict(Xc200))

# ── LinearSVR ───────────────────────────────────────────

run(suite, "LinearSVR fit", "200x5", lambda: LinearSVR(max_iter=200, dual=True).fit(Xr200, yr200))
svr = LinearSVR(max_iter=200, dual=True).fit(Xr200, yr200)
run(suite, "LinearSVR predict", "200x5", lambda: svr.predict(Xr200))

# ── SGD Models ──────────────────────────────────────────

run(suite, "SGDClassifier fit", "200x5", lambda: SGDClassifier(loss="log_loss", max_iter=200, random_state=42).fit(Xc200, yc200))
sgd_classifier = SGDClassifier(loss="log_loss", max_iter=200, random_state=42).fit(Xc200, yc200)
run(suite, "SGDClassifier predict", "200x5", lambda: sgd_classifier.predict(Xc200))

run(suite, "SGDRegressor fit", "200x5", lambda: SGDRegressor(max_iter=200, random_state=42).fit(Xr200, yr200))
sgd_regressor = SGDRegressor(max_iter=200, random_state=42).fit(Xr200, yr200)
run(suite, "SGDRegressor predict", "200x5", lambda: sgd_regressor.predict(Xr200))

# ── Decision Tree ───────────────────────────────────────

run(suite, "DecisionTreeClassifier fit", "200x5", lambda: DecisionTreeClassifier(max_depth=5).fit(Xc200, yc200))
run(suite, "DecisionTreeClassifier fit", "500x10", lambda: DecisionTreeClassifier(max_depth=5).fit(Xc500, yc500))
dtc = DecisionTreeClassifier(max_depth=5).fit(Xc200, yc200)
run(suite, "DecisionTreeClassifier predict", "200x5", lambda: dtc.predict(Xc200))

run(suite, "DecisionTreeRegressor fit", "200x5", lambda: DecisionTreeRegressor(max_depth=5).fit(Xr200, yr200))
dtr = DecisionTreeRegressor(max_depth=5).fit(Xr200, yr200)
run(suite, "DecisionTreeRegressor predict", "200x5", lambda: dtr.predict(Xr200))

# ── Random Forest ───────────────────────────────────────

run(suite, "RandomForestClassifier fit", "200x5", lambda: RandomForestClassifier(n_estimators=10, max_depth=5).fit(Xc200, yc200), iterations=5)
run(suite, "RandomForestClassifier fit", "500x10", lambda: RandomForestClassifier(n_estimators=10, max_depth=5).fit(Xc500, yc500), iterations=5)
rfc = RandomForestClassifier(n_estimators=10, max_depth=5).fit(Xc200, yc200)
run(suite, "RandomForestClassifier predict", "200x5", lambda: rfc.predict(Xc200))

run(suite, "RandomForestRegressor fit", "200x5", lambda: RandomForestRegressor(n_estimators=10, max_depth=5).fit(Xr200, yr200), iterations=5)
rfr = RandomForestRegressor(n_estimators=10, max_depth=5).fit(Xr200, yr200)
run(suite, "RandomForestRegressor predict", "200x5", lambda: rfr.predict(Xr200))

# ── Gradient Boosting ───────────────────────────────────

run(suite, "GradientBoostingClassifier fit", "200x5", lambda: GradientBoostingClassifier(n_estimators=10, max_depth=3).fit(Xc200, yc200), iterations=5)
gbc = GradientBoostingClassifier(n_estimators=10, max_depth=3).fit(Xc200, yc200)
run(suite, "GradientBoostingClassifier predict", "200x5", lambda: gbc.predict(Xc200))

run(suite, "GradientBoostingRegressor fit", "200x5", lambda: GradientBoostingRegressor(n_estimators=10, max_depth=3).fit(Xr200, yr200), iterations=5)
gbr = GradientBoostingRegressor(n_estimators=10, max_depth=3).fit(Xr200, yr200)
run(suite, "GradientBoostingRegressor predict", "200x5", lambda: gbr.predict(Xr200))

# ── KMeans ──────────────────────────────────────────────

run(suite, "KMeans fit", "200x5 k=3", lambda: KMeans(n_clusters=3, max_iter=50, n_init=1).fit(Xcl200))
run(suite, "KMeans fit", "500x5 k=3", lambda: KMeans(n_clusters=3, max_iter=50, n_init=1).fit(Xcl500))
km = KMeans(n_clusters=3, max_iter=50, n_init=1).fit(Xcl200)
run(suite, "KMeans predict", "200x5", lambda: km.predict(Xcl200))

# ── DBSCAN ──────────────────────────────────────────────

run(suite, "DBSCAN fit", "200x5", lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xcl200))
run(suite, "DBSCAN fit", "500x5", lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xcl500))

# ── PCA ─────────────────────────────────────────────────

run(suite, "PCA fit", "200x5 k=2", lambda: PCA(n_components=2).fit(Xcl200))
run(suite, "PCA fit", "500x5 k=3", lambda: PCA(n_components=3).fit(Xcl500))
pca = PCA(n_components=2).fit(Xcl200)
run(suite, "PCA transform", "200x5", lambda: pca.transform(Xcl200))

# ── TSNE ────────────────────────────────────────────────

run(suite, "TSNE fit", "200x5", lambda: TSNE(n_components=2, perplexity=30).fit_transform(Xcl200), iterations=3, warmup=1)

# ── Additional v1.0.0 estimators ─────────────────────────

run(suite, "AdaBoostClassifier fit", "200x5", lambda: AdaBoostClassifier(n_estimators=20, learning_rate=1.0).fit(Xc200, yc200), iterations=5)
run(suite, "BaggingClassifier fit", "200x5", lambda: BaggingClassifier(n_estimators=10, random_state=42).fit(Xc200, yc200), iterations=5)
run(
    suite,
    "VotingClassifier fit",
    "200x5",
    lambda: VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=100)),
            ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        voting="hard",
    ).fit(Xc200, yc200),
    iterations=5,
)
run(
    suite,
    "StackingClassifier fit",
    "200x5",
    lambda: StackingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=5)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        final_estimator=LogisticRegression(max_iter=100),
    ).fit(Xc200, yc200),
    iterations=5,
)
run(suite, "ExtraTreesClassifier fit", "200x5", lambda: ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=42).fit(Xc200, yc200), iterations=5)
run(suite, "SVC fit", "200x5", lambda: SVC(kernel="rbf", C=10).fit(Xc200, yc200), iterations=3, warmup=1)
svc_kernel = SVC(kernel="rbf", C=10).fit(Xc200, yc200)
run(suite, "SVC predict", "200x5", lambda: svc_kernel.predict(Xc200))
run(suite, "NuSVC fit", "200x5", lambda: NuSVC(nu=0.5, kernel="rbf").fit(Xc200, yc200), iterations=3, warmup=1)
run(suite, "IsolationForest fit", "200x5", lambda: IsolationForest(n_estimators=50, random_state=42).fit(Xcl200), iterations=5)
iforest = IsolationForest(n_estimators=50, random_state=42).fit(Xcl200)
run(suite, "IsolationForest predict", "200x5", lambda: iforest.predict(Xcl200))
run(suite, "LocalOutlierFactor fit", "200x5", lambda: LocalOutlierFactor(n_neighbors=10).fit(Xcl200), iterations=5)
# novelty=True exposes predict() so it can be timed as a standalone op,
# mirroring Deepbox's separate fit()/predict() benchmark cases.
lof = LocalOutlierFactor(n_neighbors=10, novelty=True).fit(Xcl200)
run(suite, "LocalOutlierFactor predict", "200x5", lambda: lof.predict(Xcl200))
run(suite, "GaussianMixture fit", "200x5 k=3", lambda: GaussianMixture(n_components=3, random_state=42).fit(Xcl200), iterations=5)
run(suite, "MiniBatchKMeans fit", "200x5 k=3", lambda: MiniBatchKMeans(n_clusters=3, batch_size=32, max_iter=50, n_init=1, random_state=42).fit(Xcl200), iterations=5)
run(suite, "SpectralClustering fit", "200x5 k=3", lambda: SpectralClustering(n_clusters=3, random_state=42, n_init=1).fit(Xcl200), iterations=3, warmup=1)
run(suite, "Birch fit", "120x2 k=3", lambda: Birch(n_clusters=3, threshold=0.8).fit(Xcl120), iterations=5)
birch = Birch(n_clusters=3, threshold=0.8).fit(Xcl120)
run(suite, "Birch predict", "120x2", lambda: birch.predict(Xcl120))
run(suite, "MeanShift fit", "120x2", lambda: MeanShift(bandwidth=2.5, max_iter=60).fit(Xcl120), iterations=3, warmup=1)
mean_shift = MeanShift(bandwidth=2.5, max_iter=60).fit(Xcl120)
run(suite, "MeanShift predict", "120x2", lambda: mean_shift.predict(Xcl120))
run(suite, "OPTICS fit", "120x2", lambda: OPTICS(min_samples=5, max_eps=3.5).fit(Xcl120), iterations=3, warmup=1)
run(suite, "GaussianRandomProjection fit+transform", "500x10→4", lambda: GaussianRandomProjection(n_components=4, random_state=42).fit_transform(Xr500))

footer(suite, "sklearn-ml.json")
