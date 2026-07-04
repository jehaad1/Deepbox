"""
Benchmark 12 — Statistics
SciPy / NumPy
"""

import sys, os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats as sp
from utils import run, create_suite, header, footer

warnings.filterwarnings("ignore", message="As of SciPy 1.17, users must choose a p-value calculation method.*")

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

# ── Advanced v1.0.0 Statistics ──────────────────────────

pvalues = np.array([0.01, 0.04, 0.03, 0.005, 0.02, 0.15, 0.2, 0.001])
observed = np.array([[10, 20, 30], [6, 9, 17]])
table2x2 = np.array([[1, 9], [11, 3]])
kde = sp.gaussian_kde(pos500)
grid1k = np.linspace(pos500.min(), pos500.max(), 1000)

def do_bonferroni():
    return np.minimum(1.0, pvalues * len(pvalues))

def do_holm():
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.empty_like(ranked)
    running = 0.0
    for i, p in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[i] = running
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result

def do_sidak():
    return 1 - (1 - pvalues) ** len(pvalues)

def do_bh():
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = min(1.0, running)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result

def do_bootstrap_mean():
    rng_local = np.random.default_rng(42)
    return sp.bootstrap((a500,), np.mean, n_resamples=500, random_state=rng_local)

run(suite, "zscore", "1K", lambda: sp.zscore(a1k))
run(suite, "sem", "1K", lambda: sp.sem(a1k))
run(suite, "bootstrap(mean,500)", "500", do_bootstrap_mean)
run(suite, "gaussian_kde.evaluate", "1K grid", lambda: kde(grid1k))
run(suite, "bonferroni", "8 pvals", do_bonferroni)
run(suite, "holm", "8 pvals", do_holm)
run(suite, "benjaminiHochberg", "8 pvals", do_bh)
run(suite, "sidak", "8 pvals", do_sidak)
run(suite, "chi2_contingency", "2x3", lambda: sp.chi2_contingency(observed))
run(suite, "fisher_exact", "2x2", lambda: sp.fisher_exact(table2x2))
run(suite, "fligner", "3x200", lambda: sp.fligner(g1, g2, g3))
run(suite, "ks_2samp", "1K", lambda: sp.ks_2samp(a1k, b1k))
run(suite, "median_test", "3x200", lambda: sp.median_test(g1, g2, g3))

# ── Extended coverage (v1.1 benchmark expansion) ────────
a100 = rng.randn(100) * 10
pos100 = np.abs(rng.randn(100)) * 10 + 0.1
pos5k = np.abs(rng.randn(5000)) * 10 + 0.1
for _sz, _d in [("100", a100), ("5K", a5k)]:
    run(suite, "mean", _sz, lambda d=_d: np.mean(d))
    run(suite, "median", _sz, lambda d=_d: np.median(d))
    run(suite, "std", _sz, lambda d=_d: np.std(d, ddof=1))
    run(suite, "variance", _sz, lambda d=_d: np.var(d, ddof=1))
    run(suite, "skewness", _sz, lambda d=_d: sp.skew(d))
    run(suite, "kurtosis", _sz, lambda d=_d: sp.kurtosis(d))
    run(suite, "mode", _sz, lambda d=_d: sp.mode(d, keepdims=False))
    run(suite, "zscore", _sz, lambda d=_d: sp.zscore(d))
    run(suite, "sem", _sz, lambda d=_d: sp.sem(d))
for _sz, _d in [("100", pos100), ("5K", pos5k)]:
    run(suite, "geometricMean", _sz, lambda d=_d: sp.gmean(d))
    run(suite, "harmonicMean", _sz, lambda d=_d: sp.hmean(d))
for _sz, _d in [("500", a500), ("5K", a5k)]:
    run(suite, "quantile (0.25)", _sz, lambda d=_d: np.quantile(d, 0.25))
    run(suite, "quantile (0.75)", _sz, lambda d=_d: np.quantile(d, 0.75))
    run(suite, "percentile (90)", _sz, lambda d=_d: np.percentile(d, 90))

footer(suite, "scipy-stats.json")
