"""
Benchmark 05 — Machine Learning
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
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

def make_clust(n, f, seed=42):
    r = np.random.RandomState(seed)
    X = np.vstack([r.rand(n // 3, f) * 2 + i * 5 for i in range(3)])
    return X[:n]

Xr200, yr200 = make_reg(200, 5)
Xr500, yr500 = make_reg(500, 10)
Xc200, yc200 = make_cls(200, 5)
Xc500, yc500 = make_cls(500, 10)
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

# ── Lasso ───────────────────────────────────────────────

run(suite, "Lasso fit", "200x5", lambda: Lasso(alpha=0.1).fit(Xr200, yr200))
run(suite, "Lasso fit", "500x10", lambda: Lasso(alpha=0.1).fit(Xr500, yr500))
lasso = Lasso(alpha=0.1).fit(Xr200, yr200)
run(suite, "Lasso predict", "200x5", lambda: lasso.predict(Xr200))

# ── Logistic Regression ─────────────────────────────────

run(suite, "LogisticRegression fit", "200x5", lambda: LogisticRegression(max_iter=200).fit(Xc200, yc200))
run(suite, "LogisticRegression fit", "500x10", lambda: LogisticRegression(max_iter=200).fit(Xc500, yc500))
logreg = LogisticRegression(max_iter=200).fit(Xc200, yc200)
run(suite, "LogisticRegression predict", "200x5", lambda: logreg.predict(Xc200))

# ── GaussianNB ──────────────────────────────────────────

run(suite, "GaussianNB fit", "200x5", lambda: GaussianNB().fit(Xc200, yc200))
run(suite, "GaussianNB fit", "500x10", lambda: GaussianNB().fit(Xc500, yc500))
gnb = GaussianNB().fit(Xc200, yc200)
run(suite, "GaussianNB predict", "200x5", lambda: gnb.predict(Xc200))

# ── KNN Classifier ──────────────────────────────────────

run(suite, "KNeighborsClassifier fit", "200x5", lambda: KNeighborsClassifier(n_neighbors=5).fit(Xc200, yc200))
run(suite, "KNeighborsClassifier fit", "500x10", lambda: KNeighborsClassifier(n_neighbors=5).fit(Xc500, yc500))
knnc = KNeighborsClassifier(n_neighbors=5).fit(Xc200, yc200)
run(suite, "KNeighborsClassifier predict", "200x5", lambda: knnc.predict(Xc200))

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

footer(suite, "sklearn-ml.json")
