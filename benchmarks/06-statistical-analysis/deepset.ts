/**
 * Benchmark 06: Statistical Analysis — Deepbox vs NumPy/SciPy
 *
 * Compares statistical operations: descriptive stats, correlations,
 * percentiles, and distribution analysis.
 */

import { tensor } from "deepbox/ndarray";
import {
  corrcoef,
  kurtosis,
  mean,
  median,
  pearsonr,
  percentile,
  skewness,
  std,
  variance,
} from "deepbox/stats";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("statistical-analysis");
header("Benchmark 06: Statistical Analysis");

// ── Data Generation ───────────────────────────────────────

function makeData(n: number, seed: number): number[] {
  let state = seed >>> 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state / 2 ** 32) * 2 - 1;
  };
  return Array.from({ length: n }, () => rand() * 100);
}

const data1k = tensor(makeData(1000, 42));
const data10k = tensor(makeData(10000, 123));
const data100k = tensor(makeData(100000, 456));

const dataA = tensor(makeData(1000, 789));
const dataB = tensor(makeData(1000, 101));

const dataA10k = tensor(makeData(10000, 202));
const dataB10k = tensor(makeData(10000, 303));

// ── Descriptive Statistics ────────────────────────────────

run(suite, "mean", "1K", () => {
  mean(data1k);
});
run(suite, "mean", "10K", () => {
  mean(data10k);
});
run(suite, "mean", "100K", () => {
  mean(data100k);
});

run(suite, "median", "1K", () => {
  median(data1k);
});
run(suite, "median", "10K", () => {
  median(data10k);
});
run(suite, "median", "100K", () => {
  median(data100k);
});

run(suite, "std", "1K", () => {
  std(data1k);
});
run(suite, "std", "10K", () => {
  std(data10k);
});
run(suite, "std", "100K", () => {
  std(data100k);
});

run(suite, "variance", "1K", () => {
  variance(data1k);
});
run(suite, "variance", "10K", () => {
  variance(data10k);
});
run(suite, "variance", "100K", () => {
  variance(data100k);
});

run(suite, "skewness", "1K", () => {
  skewness(data1k);
});
run(suite, "skewness", "10K", () => {
  skewness(data10k);
});

run(suite, "kurtosis", "1K", () => {
  kurtosis(data1k);
});
run(suite, "kurtosis", "10K", () => {
  kurtosis(data10k);
});

// ── Percentiles ───────────────────────────────────────────

run(suite, "percentile (50th)", "1K", () => {
  percentile(data1k, 50);
});
run(suite, "percentile (50th)", "10K", () => {
  percentile(data10k, 50);
});
run(suite, "percentile (95th)", "10K", () => {
  percentile(data10k, 95);
});
run(suite, "percentile (50th)", "100K", () => {
  percentile(data100k, 50);
});

// ── Correlations ──────────────────────────────────────────

run(suite, "pearsonr", "1K", () => {
  pearsonr(dataA, dataB);
});
run(suite, "pearsonr", "10K", () => {
  pearsonr(dataA10k, dataB10k);
});

// Correlation matrix on multi-dim data
const multiData1k = tensor(
  Array.from({ length: 5 }, () => makeData(1000, Math.floor(Math.random() * 10000)))
);

run(suite, "corrcoef (5 variables)", "1K", () => {
  corrcoef(multiData1k);
});

footer(suite, "deepbox-stats-ops.json");
