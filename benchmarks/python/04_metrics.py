"""
Benchmark 04 — Metrics
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    brier_score_loss,
    confusion_matrix, hamming_loss, jaccard_score, cohen_kappa_score,
    coverage_error, matthews_corrcoef, balanced_accuracy_score, log_loss, roc_auc_score,
    average_precision_score,
    mean_pinball_loss, mean_squared_error, ndcg_score, root_mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, max_error, median_absolute_error,
    mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    top_k_accuracy_score, zero_one_loss,
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score,
)
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, pairwise_distances
from utils import run, create_suite, header, footer

suite = create_suite("metrics", "scikit-learn")
header("Benchmark 04 — Metrics", "scikit-learn")

# ── Data generators ──────────────────────────────────────

rng = np.random.RandomState(42)
rng2 = np.random.RandomState(123)

yt1k = rng.randint(0, 2, 1000)
yp1k = rng.randint(0, 2, 1000)
yt10k = rng2.randint(0, 2, 10000)
yp10k = rng2.randint(0, 2, 10000)

rng3 = np.random.RandomState(42)
prob_yt1k = rng3.randint(0, 2, 1000)
prob_yp1k = np.clip(rng3.rand(1000), 0.001, 0.999)
rng4 = np.random.RandomState(123)
prob_yt10k = rng4.randint(0, 2, 10000)
prob_yp10k = np.clip(rng4.rand(10000), 0.001, 0.999)

rng5 = np.random.RandomState(42)
rt1k = rng5.rand(1000) * 10
rp1k = rt1k + (rng5.rand(1000) - 0.5) * 2
rng6 = np.random.RandomState(123)
rt10k = rng6.rand(10000) * 10
rp10k = rt10k + (rng6.rand(10000) - 0.5) * 2

rng7 = np.random.RandomState(42)
cX200 = np.vstack([rng7.randn(67, 5) + i * 5 for i in range(3)])[:200]
cL200 = np.array([i for i in range(3) for _ in range(67)])[:200]
rng8 = np.random.RandomState(123)
cX500 = np.vstack([rng8.randn(125, 5) + i * 5 for i in range(4)])
cL500 = np.array([i for i in range(4) for _ in range(125)])

rng9 = np.random.RandomState(42)
clab_a500 = np.array([i % 4 for i in range(500)])
clab_b500 = rng9.randint(0, 4, 500)
rng10 = np.random.RandomState(123)
clab_a1k = np.array([i % 5 for i in range(1000)])
clab_b1k = rng10.randint(0, 5, 1000)
rank_true = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
])
rank_score = np.array([
    [0.9, 0.2, 0.8, 0.1],
    [0.3, 0.7, 0.2, 0.6],
    [0.8, 0.1, 0.4, 0.7],
])
multiclass_true = np.array([i % 4 for i in range(1000)])
multiclass_score = np.array([
    [0.7 if j == (i % 4) else 0.1 for j in range(4)]
    for i in range(1000)
], dtype=float)
pairwise_x = np.array([
    [((i * 3 + j * 5) % 17) / 17 for j in range(8)]
    for i in range(300)
], dtype=float)

# ── Classification Metrics ──────────────────────────────

run(suite, "accuracy", "1K", lambda: accuracy_score(yt1k, yp1k))
run(suite, "accuracy", "10K", lambda: accuracy_score(yt10k, yp10k))
run(suite, "precision", "1K", lambda: precision_score(yt1k, yp1k, zero_division=0))
run(suite, "precision", "10K", lambda: precision_score(yt10k, yp10k, zero_division=0))
run(suite, "recall", "1K", lambda: recall_score(yt1k, yp1k, zero_division=0))
run(suite, "recall", "10K", lambda: recall_score(yt10k, yp10k, zero_division=0))
run(suite, "f1Score", "1K", lambda: f1_score(yt1k, yp1k, zero_division=0))
run(suite, "f1Score", "10K", lambda: f1_score(yt10k, yp10k, zero_division=0))
run(suite, "fbetaScore (β=0.5)", "1K", lambda: fbeta_score(yt1k, yp1k, beta=0.5, zero_division=0))
run(suite, "fbetaScore (β=0.5)", "10K", lambda: fbeta_score(yt10k, yp10k, beta=0.5, zero_division=0))
run(suite, "confusionMatrix", "1K", lambda: confusion_matrix(yt1k, yp1k))
run(suite, "confusionMatrix", "10K", lambda: confusion_matrix(yt10k, yp10k))
run(suite, "hammingLoss", "1K", lambda: hamming_loss(yt1k, yp1k))
run(suite, "hammingLoss", "10K", lambda: hamming_loss(yt10k, yp10k))
run(suite, "jaccardScore", "1K", lambda: jaccard_score(yt1k, yp1k))
run(suite, "jaccardScore", "10K", lambda: jaccard_score(yt10k, yp10k))
run(suite, "cohenKappaScore", "1K", lambda: cohen_kappa_score(yt1k, yp1k))
run(suite, "cohenKappaScore", "10K", lambda: cohen_kappa_score(yt10k, yp10k))
run(suite, "matthewsCorrcoef", "1K", lambda: matthews_corrcoef(yt1k, yp1k))
run(suite, "matthewsCorrcoef", "10K", lambda: matthews_corrcoef(yt10k, yp10k))
run(suite, "balancedAccuracy", "1K", lambda: balanced_accuracy_score(yt1k, yp1k))
run(suite, "balancedAccuracy", "10K", lambda: balanced_accuracy_score(yt10k, yp10k))
run(suite, "logLoss", "1K", lambda: log_loss(prob_yt1k, prob_yp1k))
run(suite, "logLoss", "10K", lambda: log_loss(prob_yt10k, prob_yp10k))
run(suite, "rocAucScore", "1K", lambda: roc_auc_score(prob_yt1k, prob_yp1k))
run(suite, "rocAucScore", "10K", lambda: roc_auc_score(prob_yt10k, prob_yp10k))
run(suite, "averagePrecision", "1K", lambda: average_precision_score(prob_yt1k, prob_yp1k))
run(suite, "averagePrecision", "10K", lambda: average_precision_score(prob_yt10k, prob_yp10k))
run(suite, "brierScoreLoss", "1K", lambda: brier_score_loss(prob_yt1k, prob_yp1k))
run(suite, "zeroOneLoss", "1K", lambda: zero_one_loss(yt1k, yp1k))
run(suite, "topKAccuracyScore", "1Kx4", lambda: top_k_accuracy_score(multiclass_true, multiclass_score, k=2))
run(suite, "coverageError", "3x4", lambda: coverage_error(rank_true, rank_score))
run(suite, "ndcgScore", "3x4", lambda: ndcg_score(rank_true, rank_score))

# ── Regression Metrics ──────────────────────────────────

run(suite, "mse", "1K", lambda: mean_squared_error(rt1k, rp1k))
run(suite, "mse", "10K", lambda: mean_squared_error(rt10k, rp10k))
run(suite, "rmse", "1K", lambda: root_mean_squared_error(rt1k, rp1k))
run(suite, "rmse", "10K", lambda: root_mean_squared_error(rt10k, rp10k))
run(suite, "mae", "1K", lambda: mean_absolute_error(rt1k, rp1k))
run(suite, "mae", "10K", lambda: mean_absolute_error(rt10k, rp10k))
run(suite, "r2Score", "1K", lambda: r2_score(rt1k, rp1k))
run(suite, "r2Score", "10K", lambda: r2_score(rt10k, rp10k))

def adjusted_r2(yt, yp, p):
    r2 = r2_score(yt, yp)
    n = len(yt)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

run(suite, "adjustedR2Score", "1K", lambda: adjusted_r2(rt1k, rp1k, 5))
run(suite, "adjustedR2Score", "10K", lambda: adjusted_r2(rt10k, rp10k, 10))
run(suite, "explainedVariance", "1K", lambda: explained_variance_score(rt1k, rp1k))
run(suite, "explainedVariance", "10K", lambda: explained_variance_score(rt10k, rp10k))
run(suite, "maxError", "1K", lambda: max_error(rt1k, rp1k))
run(suite, "maxError", "10K", lambda: max_error(rt10k, rp10k))
run(suite, "medianAbsoluteError", "1K", lambda: median_absolute_error(rt1k, rp1k))
run(suite, "medianAbsoluteError", "10K", lambda: median_absolute_error(rt10k, rp10k))
run(suite, "mape", "1K", lambda: mean_absolute_percentage_error(rt1k, rp1k))
run(suite, "mape", "10K", lambda: mean_absolute_percentage_error(rt10k, rp10k))
run(suite, "meanPinballLoss", "1K", lambda: mean_pinball_loss(rt1k, rp1k, alpha=0.9))

# ── Clustering Metrics ──────────────────────────────────

run(suite, "silhouetteScore", "200x5 k=3", lambda: silhouette_score(cX200, cL200))
run(suite, "silhouetteScore", "500x5 k=4", lambda: silhouette_score(cX500, cL500))
run(suite, "calinskiHarabasz", "200x5 k=3", lambda: calinski_harabasz_score(cX200, cL200))
run(suite, "calinskiHarabasz", "500x5 k=4", lambda: calinski_harabasz_score(cX500, cL500))
run(suite, "daviesBouldin", "200x5 k=3", lambda: davies_bouldin_score(cX200, cL200))
run(suite, "daviesBouldin", "500x5 k=4", lambda: davies_bouldin_score(cX500, cL500))
run(suite, "adjustedRandScore", "500", lambda: adjusted_rand_score(clab_a500, clab_b500))
run(suite, "adjustedRandScore", "1K", lambda: adjusted_rand_score(clab_a1k, clab_b1k))
run(suite, "adjustedMutualInfo", "500", lambda: adjusted_mutual_info_score(clab_a500, clab_b500))
run(suite, "adjustedMutualInfo", "1K", lambda: adjusted_mutual_info_score(clab_a1k, clab_b1k))
run(suite, "normalizedMutualInfo", "500", lambda: normalized_mutual_info_score(clab_a500, clab_b500))
run(suite, "normalizedMutualInfo", "1K", lambda: normalized_mutual_info_score(clab_a1k, clab_b1k))
run(suite, "homogeneityScore", "500", lambda: homogeneity_score(clab_a500, clab_b500))
run(suite, "completenessScore", "500", lambda: completeness_score(clab_a500, clab_b500))
run(suite, "vMeasureScore", "500", lambda: v_measure_score(clab_a500, clab_b500))
run(suite, "fowlkesMallows", "500", lambda: fowlkes_mallows_score(clab_a500, clab_b500))
run(suite, "fowlkesMallows", "1K", lambda: fowlkes_mallows_score(clab_a1k, clab_b1k))
run(suite, "pairwiseEuclidean", "300x8", lambda: pairwise_distances(pairwise_x, metric="euclidean"))
run(suite, "pairwiseCosine", "300x8", lambda: cosine_similarity(pairwise_x))
run(suite, "pairwiseManhattan", "300x8", lambda: manhattan_distances(pairwise_x))

# ── Extended coverage (v1.1 benchmark expansion) ────────
_mrng = np.random.RandomState(7)
cls_sizes = {}
prob_sizes = {}
reg_sizes = {}
for _sz, _n in [("100", 100), ("500", 500), ("5K", 5000)]:
    cls_sizes[_sz] = (_mrng.randint(0, 2, _n), _mrng.randint(0, 2, _n))
    prob_sizes[_sz] = (_mrng.randint(0, 2, _n), np.clip(_mrng.rand(_n), 0.001, 0.999))
    _t = _mrng.rand(_n) * 10
    reg_sizes[_sz] = (_t, _t + (_mrng.rand(_n) - 0.5) * 2)

for _sz, (_yt, _yp) in cls_sizes.items():
    run(suite, "accuracy", _sz, lambda yt=_yt, yp=_yp: accuracy_score(yt, yp))
    run(suite, "precision", _sz, lambda yt=_yt, yp=_yp: precision_score(yt, yp, zero_division=0))
    run(suite, "recall", _sz, lambda yt=_yt, yp=_yp: recall_score(yt, yp, zero_division=0))
    run(suite, "f1Score", _sz, lambda yt=_yt, yp=_yp: f1_score(yt, yp, zero_division=0))
    run(suite, "hammingLoss", _sz, lambda yt=_yt, yp=_yp: hamming_loss(yt, yp))
    run(suite, "jaccardScore", _sz, lambda yt=_yt, yp=_yp: jaccard_score(yt, yp))
    run(suite, "cohenKappaScore", _sz, lambda yt=_yt, yp=_yp: cohen_kappa_score(yt, yp))
    run(suite, "matthewsCorrcoef", _sz, lambda yt=_yt, yp=_yp: matthews_corrcoef(yt, yp))
    run(suite, "balancedAccuracy", _sz, lambda yt=_yt, yp=_yp: balanced_accuracy_score(yt, yp))
    run(suite, "zeroOneLoss", _sz, lambda yt=_yt, yp=_yp: zero_one_loss(yt, yp))
for _sz, (_yt, _yp) in prob_sizes.items():
    run(suite, "logLoss", _sz, lambda yt=_yt, yp=_yp: log_loss(yt, yp))
    run(suite, "brierScoreLoss", _sz, lambda yt=_yt, yp=_yp: brier_score_loss(yt, yp))
for _sz, (_yt, _yp) in reg_sizes.items():
    run(suite, "mse", _sz, lambda yt=_yt, yp=_yp: mean_squared_error(yt, yp))
    run(suite, "rmse", _sz, lambda yt=_yt, yp=_yp: root_mean_squared_error(yt, yp))
    run(suite, "mae", _sz, lambda yt=_yt, yp=_yp: mean_absolute_error(yt, yp))
    run(suite, "r2Score", _sz, lambda yt=_yt, yp=_yp: r2_score(yt, yp))
    run(suite, "explainedVariance", _sz, lambda yt=_yt, yp=_yp: explained_variance_score(yt, yp))
    run(suite, "maxError", _sz, lambda yt=_yt, yp=_yp: max_error(yt, yp))
    run(suite, "mape", _sz, lambda yt=_yt, yp=_yp: mean_absolute_percentage_error(yt, yp))

footer(suite, "sklearn-metrics.json")
