"""
Benchmark 12 — Statistics
SciPy / NumPy
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats as sp
from utils import run, create_suite, header, footer

suite = create_suite("stats", "SciPy")
header("Benchmark 12 — Statistics", "SciPy")

rng = np.random.RandomState(42)

a500 = rng.randn(500) * 10
b500 = rng.randn(500) * 10
a1k = rng.randn(1000) * 10
b1k = rng.randn(1000) * 10
a5k = rng.randn(5000) * 10
b5k = rng.randn(5000) * 10
a10k = rng.randn(10000) * 10
pos500 = np.abs(rng.randn(500)) * 10 + 0.1
pos1k = np.abs(rng.randn(1000)) * 10 + 0.1

# ── Descriptive Statistics ──────────────────────────────

run(suite, "mean", "500", lambda: np.mean(a500))
run(suite, "mean", "1K", lambda: np.mean(a1k))
run(suite, "mean", "10K", lambda: np.mean(a10k))
run(suite, "median", "500", lambda: np.median(a500))
run(suite, "median", "1K", lambda: np.median(a1k))
run(suite, "median", "10K", lambda: np.median(a10k))
run(suite, "mode", "500", lambda: sp.mode(a500, keepdims=False))
run(suite, "mode", "1K", lambda: sp.mode(a1k, keepdims=False))
run(suite, "std", "500", lambda: np.std(a500, ddof=1))
run(suite, "std", "1K", lambda: np.std(a1k, ddof=1))
run(suite, "std", "10K", lambda: np.std(a10k, ddof=1))
run(suite, "variance", "500", lambda: np.var(a500, ddof=1))
run(suite, "variance", "1K", lambda: np.var(a1k, ddof=1))
run(suite, "skewness", "500", lambda: sp.skew(a500))
run(suite, "skewness", "1K", lambda: sp.skew(a1k))
run(suite, "kurtosis", "500", lambda: sp.kurtosis(a500))
run(suite, "kurtosis", "1K", lambda: sp.kurtosis(a1k))
run(suite, "quantile (0.25)", "1K", lambda: np.quantile(a1k, 0.25))
run(suite, "quantile (0.75)", "1K", lambda: np.quantile(a1k, 0.75))
run(suite, "percentile (90)", "1K", lambda: np.percentile(a1k, 90))
run(suite, "geometricMean", "500", lambda: sp.gmean(pos500))
run(suite, "geometricMean", "1K", lambda: sp.gmean(pos1k))
run(suite, "harmonicMean", "500", lambda: sp.hmean(pos500))
run(suite, "harmonicMean", "1K", lambda: sp.hmean(pos1k))
run(suite, "trimMean (10%)", "1K", lambda: sp.trim_mean(a1k, 0.1))
run(suite, "moment (3rd)", "1K", lambda: sp.moment(a1k, 3))

# ── Correlation ─────────────────────────────────────────

run(suite, "pearsonr", "500", lambda: sp.pearsonr(a500, b500))
run(suite, "pearsonr", "1K", lambda: sp.pearsonr(a1k, b1k))
run(suite, "pearsonr", "5K", lambda: sp.pearsonr(a5k, b5k))
run(suite, "spearmanr", "500", lambda: sp.spearmanr(a500, b500))
run(suite, "spearmanr", "1K", lambda: sp.spearmanr(a1k, b1k))
run(suite, "kendalltau", "500", lambda: sp.kendalltau(a500, b500))
run(suite, "corrcoef", "500", lambda: np.corrcoef(a500, b500))
run(suite, "corrcoef", "1K", lambda: np.corrcoef(a1k, b1k))
run(suite, "cov", "500", lambda: np.cov(a500, b500))
run(suite, "cov", "1K", lambda: np.cov(a1k, b1k))

# ── Hypothesis Tests ────────────────────────────────────

run(suite, "ttest_1samp", "500", lambda: sp.ttest_1samp(a500, 0))
run(suite, "ttest_1samp", "1K", lambda: sp.ttest_1samp(a1k, 0))
run(suite, "ttest_ind", "500", lambda: sp.ttest_ind(a500, b500))
run(suite, "ttest_ind", "1K", lambda: sp.ttest_ind(a1k, b1k))
run(suite, "ttest_rel", "500", lambda: sp.ttest_rel(a500, b500))
run(suite, "ttest_rel", "1K", lambda: sp.ttest_rel(a1k, b1k))

rng2 = np.random.RandomState(1)
g1 = rng2.randn(200) * 10
rng3 = np.random.RandomState(2)
g2 = rng3.randn(200) * 10
rng4 = np.random.RandomState(3)
g3 = rng4.randn(200) * 10

run(suite, "f_oneway", "3×200", lambda: sp.f_oneway(g1, g2, g3))
run(suite, "chisquare", "10 bins", lambda: sp.chisquare([16, 18, 16, 14, 12, 12, 9, 10, 11, 12]))
run(suite, "shapiro", "500", lambda: sp.shapiro(a500))
run(suite, "mannwhitneyu", "500", lambda: sp.mannwhitneyu(a500, b500))
run(suite, "mannwhitneyu", "1K", lambda: sp.mannwhitneyu(a1k, b1k))
run(suite, "kruskal", "3×200", lambda: sp.kruskal(g1, g2, g3))
run(suite, "friedmanchisquare", "3×200", lambda: sp.friedmanchisquare(g1, g2, g3))
run(suite, "anderson", "500", lambda: sp.anderson(a500))
run(suite, "kstest", "500", lambda: sp.kstest(a500, "norm"))
run(suite, "kstest", "1K", lambda: sp.kstest(a1k, "norm"))
run(suite, "levene", "500", lambda: sp.levene(a500, b500))
run(suite, "bartlett", "500", lambda: sp.bartlett(a500, b500))
run(suite, "normaltest", "500", lambda: sp.normaltest(a500))
run(suite, "normaltest", "1K", lambda: sp.normaltest(a1k))
run(suite, "wilcoxon", "500", lambda: sp.wilcoxon(a500))

footer(suite, "scipy-stats.json")
