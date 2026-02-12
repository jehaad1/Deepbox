"""
Benchmark 04 — Metrics
scikit-learn
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, hamming_loss, jaccard_score, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score, log_loss, roc_auc_score,
    average_precision_score,
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, max_error, median_absolute_error,
    mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score,
)
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

footer(suite, "sklearn-metrics.json")
