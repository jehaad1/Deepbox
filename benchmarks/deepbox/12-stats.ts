/**
 * Benchmark 12 — Statistics
 * Deepbox vs SciPy / NumPy
 */

import { tensor } from "deepbox/ndarray";
import {
  anderson,
  bartlett,
  chisquare,
  corrcoef,
  cov,
  f_oneway,
  friedmanchisquare,
  geometricMean,
  harmonicMean,
  kendalltau,
  kruskal,
  kstest,
  kurtosis,
  levene,
  mannwhitneyu,
  mean,
  median,
  mode,
  moment,
  normaltest,
  pearsonr,
  percentile,
  quantile,
  shapiro,
  skewness,
  spearmanr,
  std,
  trimMean,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  variance,
  wilcoxon,
} from "deepbox/stats";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("stats");
header("Benchmark 12 — Statistics");

// ── Data ────────────────────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function makeArray(n: number, seed: number): number[] {
  const rand = seededRng(seed);
  return Array.from({ length: n }, () => rand() * 20 - 10);
}

function makePositiveArray(n: number, seed: number): number[] {
  const rand = seededRng(seed);
  return Array.from({ length: n }, () => rand() * 10 + 0.1);
}

const a500 = tensor(makeArray(500, 42));
const b500 = tensor(makeArray(500, 99));
const a1k = tensor(makeArray(1000, 42));
const b1k = tensor(makeArray(1000, 99));
const a5k = tensor(makeArray(5000, 42));
const b5k = tensor(makeArray(5000, 99));
const a10k = tensor(makeArray(10000, 42));
const pos500 = tensor(makePositiveArray(500, 42));
const pos1k = tensor(makePositiveArray(1000, 42));

// ── Descriptive Statistics ──────────────────────────────

run(suite, "mean", "500", () => mean(a500));
run(suite, "mean", "1K", () => mean(a1k));
run(suite, "mean", "10K", () => mean(a10k));
run(suite, "median", "500", () => median(a500));
run(suite, "median", "1K", () => median(a1k));
run(suite, "median", "10K", () => median(a10k));
run(suite, "mode", "500", () => mode(a500));
run(suite, "mode", "1K", () => mode(a1k));
run(suite, "std", "500", () => std(a500));
run(suite, "std", "1K", () => std(a1k));
run(suite, "std", "10K", () => std(a10k));
run(suite, "variance", "500", () => variance(a500));
run(suite, "variance", "1K", () => variance(a1k));
run(suite, "skewness", "500", () => skewness(a500));
run(suite, "skewness", "1K", () => skewness(a1k));
run(suite, "kurtosis", "500", () => kurtosis(a500));
run(suite, "kurtosis", "1K", () => kurtosis(a1k));
run(suite, "quantile (0.25)", "1K", () => quantile(a1k, 0.25));
run(suite, "quantile (0.75)", "1K", () => quantile(a1k, 0.75));
run(suite, "percentile (90)", "1K", () => percentile(a1k, 90));
run(suite, "geometricMean", "500", () => geometricMean(pos500));
run(suite, "geometricMean", "1K", () => geometricMean(pos1k));
run(suite, "harmonicMean", "500", () => harmonicMean(pos500));
run(suite, "harmonicMean", "1K", () => harmonicMean(pos1k));
run(suite, "trimMean (10%)", "1K", () => trimMean(a1k, 0.1));
run(suite, "moment (3rd)", "1K", () => moment(a1k, 3));

// ── Correlation ─────────────────────────────────────────

run(suite, "pearsonr", "500", () => pearsonr(a500, b500));
run(suite, "pearsonr", "1K", () => pearsonr(a1k, b1k));
run(suite, "pearsonr", "5K", () => pearsonr(a5k, b5k));
run(suite, "spearmanr", "500", () => spearmanr(a500, b500));
run(suite, "spearmanr", "1K", () => spearmanr(a1k, b1k));
run(suite, "kendalltau", "500", () => kendalltau(a500, b500));
run(suite, "corrcoef", "500", () => corrcoef(a500, b500));
run(suite, "corrcoef", "1K", () => corrcoef(a1k, b1k));
run(suite, "cov", "500", () => cov(a500, b500));
run(suite, "cov", "1K", () => cov(a1k, b1k));

// ── Hypothesis Tests ────────────────────────────────────

run(suite, "ttest_1samp", "500", () => ttest_1samp(a500, 0));
run(suite, "ttest_1samp", "1K", () => ttest_1samp(a1k, 0));
run(suite, "ttest_ind", "500", () => ttest_ind(a500, b500));
run(suite, "ttest_ind", "1K", () => ttest_ind(a1k, b1k));
run(suite, "ttest_rel", "500", () => ttest_rel(a500, b500));
run(suite, "ttest_rel", "1K", () => ttest_rel(a1k, b1k));

const g1 = tensor(makeArray(200, 1));
const g2 = tensor(makeArray(200, 2));
const g3 = tensor(makeArray(200, 3));

run(suite, "f_oneway", "3×200", () => f_oneway(g1, g2, g3));
run(suite, "chisquare", "10 bins", () =>
  chisquare(tensor([16, 18, 16, 14, 12, 12, 9, 10, 11, 12]))
);
run(suite, "shapiro", "500", () => shapiro(a500));
run(suite, "mannwhitneyu", "500", () => mannwhitneyu(a500, b500));
run(suite, "mannwhitneyu", "1K", () => mannwhitneyu(a1k, b1k));
run(suite, "kruskal", "3×200", () => kruskal(g1, g2, g3));
run(suite, "friedmanchisquare", "3×200", () => friedmanchisquare(g1, g2, g3));
run(suite, "anderson", "500", () => anderson(a500));
run(suite, "kstest", "500", () => kstest(a500, "norm"));
run(suite, "kstest", "1K", () => kstest(a1k, "norm"));
run(suite, "levene", "500", () => levene("median", a500, b500));
run(suite, "bartlett", "500", () => bartlett(a500, b500));
run(suite, "normaltest", "500", () => normaltest(a500));
run(suite, "normaltest", "1K", () => normaltest(a1k));
run(suite, "wilcoxon", "500", () => wilcoxon(a500));

footer(suite, "deepbox-stats.json");
