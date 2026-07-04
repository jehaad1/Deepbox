"""
Benchmark 02 — Dataset Loading & Generation
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import (
    load_iris, load_breast_cancer, load_diabetes, load_digits, load_linnerud,
    make_biclusters, make_blobs, make_checkerboard, make_circles, make_classification,
    make_friedman1, make_friedman2, make_friedman3, make_gaussian_quantiles,
    make_low_rank_matrix, make_moons, make_regression, make_s_curve, make_sparse_uncorrelated,
    make_spd_matrix, make_swiss_roll,
)
from utils import run, create_suite, header, footer

suite = create_suite("datasets", "scikit-learn")
header("Benchmark 02 — Dataset Loading & Generation", "scikit-learn")

# ── Built-in Loaders ────────────────────────────────────

run(suite, "loadIris", "150x4", lambda: load_iris())
run(suite, "loadBreastCancer", "569x30", lambda: load_breast_cancer())
run(suite, "loadDiabetes", "442x10", lambda: load_diabetes())
run(suite, "loadDigits", "1797x64", lambda: load_digits(), iterations=10)
run(suite, "loadLinnerud", "20x3", lambda: load_linnerud())

# ── makeBlobs ───────────────────────────────────────────

run(suite, "makeBlobs", "100x2 k=3", lambda: make_blobs(n_samples=100, n_features=2, centers=3))
run(suite, "makeBlobs", "500x2 k=3", lambda: make_blobs(n_samples=500, n_features=2, centers=3))
run(suite, "makeBlobs", "1Kx5 k=5", lambda: make_blobs(n_samples=1000, n_features=5, centers=5))
run(suite, "makeBlobs", "2Kx10 k=5", lambda: make_blobs(n_samples=2000, n_features=10, centers=5))
run(suite, "makeBlobs", "5Kx2 k=3", lambda: make_blobs(n_samples=5000, n_features=2, centers=3))
run(suite, "makeBlobs", "10Kx5 k=10", lambda: make_blobs(n_samples=10000, n_features=5, centers=10))
run(suite, "makeBlobs", "20Kx20 k=10", lambda: make_blobs(n_samples=20000, n_features=20, centers=10))

# ── makeCircles ─────────────────────────────────────────

run(suite, "makeCircles", "100 samples", lambda: make_circles(n_samples=100))
run(suite, "makeCircles", "500 samples", lambda: make_circles(n_samples=500))
run(suite, "makeCircles", "1K samples", lambda: make_circles(n_samples=1000))
run(suite, "makeCircles", "5K samples", lambda: make_circles(n_samples=5000))
run(suite, "makeCircles", "10K noise=0.1", lambda: make_circles(n_samples=10000, noise=0.1))
run(suite, "makeCircles", "20K noise=0.05", lambda: make_circles(n_samples=20000, noise=0.05))

# ── makeMoons ───────────────────────────────────────────

run(suite, "makeMoons", "100 samples", lambda: make_moons(n_samples=100))
run(suite, "makeMoons", "500 samples", lambda: make_moons(n_samples=500))
run(suite, "makeMoons", "1K samples", lambda: make_moons(n_samples=1000))
run(suite, "makeMoons", "5K samples", lambda: make_moons(n_samples=5000))
run(suite, "makeMoons", "10K noise=0.1", lambda: make_moons(n_samples=10000, noise=0.1))
run(suite, "makeMoons", "20K noise=0.05", lambda: make_moons(n_samples=20000, noise=0.05))

# ── makeClassification ──────────────────────────────────

run(suite, "makeClassification", "100x10", lambda: make_classification(n_samples=100, n_features=10))
run(suite, "makeClassification", "500x10", lambda: make_classification(n_samples=500, n_features=10))
run(suite, "makeClassification", "1Kx20", lambda: make_classification(n_samples=1000, n_features=20))
run(suite, "makeClassification", "5Kx20", lambda: make_classification(n_samples=5000, n_features=20))
run(suite, "makeClassification", "10Kx50", lambda: make_classification(n_samples=10000, n_features=50))
run(suite, "makeClassification", "20Kx100", lambda: make_classification(n_samples=20000, n_features=100), iterations=10)

# ── makeRegression ──────────────────────────────────────

run(suite, "makeRegression", "100x10", lambda: make_regression(n_samples=100, n_features=10))
run(suite, "makeRegression", "500x10", lambda: make_regression(n_samples=500, n_features=10))
run(suite, "makeRegression", "1Kx20", lambda: make_regression(n_samples=1000, n_features=20))
run(suite, "makeRegression", "5Kx20", lambda: make_regression(n_samples=5000, n_features=20))
run(suite, "makeRegression", "10Kx50", lambda: make_regression(n_samples=10000, n_features=50))
run(suite, "makeRegression", "20Kx100", lambda: make_regression(n_samples=20000, n_features=100), iterations=10)

# ── makeGaussianQuantiles ───────────────────────────────

run(suite, "makeGaussianQuantiles", "100x2 k=3", lambda: make_gaussian_quantiles(n_samples=100, n_features=2, n_classes=3))
run(suite, "makeGaussianQuantiles", "500x5 k=3", lambda: make_gaussian_quantiles(n_samples=500, n_features=5, n_classes=3))
run(suite, "makeGaussianQuantiles", "1Kx5 k=5", lambda: make_gaussian_quantiles(n_samples=1000, n_features=5, n_classes=5))
run(suite, "makeGaussianQuantiles", "5Kx10 k=5", lambda: make_gaussian_quantiles(n_samples=5000, n_features=10, n_classes=5))
run(suite, "makeGaussianQuantiles", "10Kx10 k=5", lambda: make_gaussian_quantiles(n_samples=10000, n_features=10, n_classes=5))

# ── Additional v1.0.0 generators ─────────────────────────

run(suite, "makeFriedman1", "1Kx10", lambda: make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42))
run(suite, "makeFriedman2", "1K", lambda: make_friedman2(n_samples=1000, noise=1.0, random_state=42))
run(suite, "makeFriedman3", "1K", lambda: make_friedman3(n_samples=1000, noise=1.0, random_state=42))
run(suite, "makeSwissRoll", "2K", lambda: make_swiss_roll(n_samples=2000, noise=0.2, random_state=42))
run(suite, "makeSCurve", "2K", lambda: make_s_curve(n_samples=2000, noise=0.2, random_state=42))
run(suite, "makeSparseUncorrelated", "2Kx20", lambda: make_sparse_uncorrelated(n_samples=2000, n_features=20, random_state=42))
run(suite, "makeLowRankMatrix", "500x100 rank=10", lambda: make_low_rank_matrix(n_samples=500, n_features=100, effective_rank=10, random_state=42))
run(suite, "makeSPDMatrix", "100x100", lambda: make_spd_matrix(n_dim=100, random_state=42))
run(suite, "makeBiclusters", "200x200 k=4", lambda: make_biclusters(shape=(200, 200), n_clusters=4, noise=0.1, random_state=42))
run(suite, "makeCheckerboard", "200x200 k=4x4", lambda: make_checkerboard(shape=(200, 200), n_clusters=(4, 4), noise=0.1, random_state=42))

footer(suite, "sklearn-datasets.json")
