/**
 * Benchmark 11 — Random Number Generation
 * Deepbox vs NumPy
 */

import { arange } from "deepbox/ndarray";
import {
  beta,
  binomial,
  choice,
  exponential,
  gamma,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  randn,
  setSeed,
  shuffle,
  uniform,
} from "deepbox/random";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("random");
header("Benchmark 11 — Random Number Generation");

setSeed(42);

// ── rand ────────────────────────────────────────────────

run(suite, "rand", "1K", () => rand([1000]));
run(suite, "rand", "10K", () => rand([10000]));
run(suite, "rand", "100K", () => rand([100000]));
run(suite, "rand", "1M", () => rand([1000000]));

// ── randn ───────────────────────────────────────────────

run(suite, "randn", "1K", () => randn([1000]));
run(suite, "randn", "10K", () => randn([10000]));
run(suite, "randn", "100K", () => randn([100000]));
run(suite, "randn", "1M", () => randn([1000000]));

// ── randint ─────────────────────────────────────────────

run(suite, "randint", "1K", () => randint(0, 100, [1000]));
run(suite, "randint", "10K", () => randint(0, 100, [10000]));
run(suite, "randint", "100K", () => randint(0, 100, [100000]));
run(suite, "randint", "1M", () => randint(0, 100, [1000000]));

// ── uniform ─────────────────────────────────────────────

run(suite, "uniform", "1K", () => uniform(0, 1, [1000]));
run(suite, "uniform", "10K", () => uniform(0, 1, [10000]));
run(suite, "uniform", "100K", () => uniform(0, 1, [100000]));
run(suite, "uniform", "1M", () => uniform(0, 1, [1000000]));

// ── normal ──────────────────────────────────────────────

run(suite, "normal", "1K", () => normal(0, 1, [1000]));
run(suite, "normal", "10K", () => normal(0, 1, [10000]));
run(suite, "normal", "100K", () => normal(0, 1, [100000]));
run(suite, "normal", "1M", () => normal(0, 1, [1000000]));

// ── binomial ────────────────────────────────────────────

run(suite, "binomial", "1K", () => binomial(10, 0.5, [1000]));
run(suite, "binomial", "10K", () => binomial(10, 0.5, [10000]));
run(suite, "binomial", "100K", () => binomial(10, 0.5, [100000]));

// ── poisson ─────────────────────────────────────────────

run(suite, "poisson", "1K", () => poisson(5, [1000]));
run(suite, "poisson", "10K", () => poisson(5, [10000]));
run(suite, "poisson", "100K", () => poisson(5, [100000]));

// ── exponential ─────────────────────────────────────────

run(suite, "exponential", "1K", () => exponential(1.0, [1000]));
run(suite, "exponential", "10K", () => exponential(1.0, [10000]));
run(suite, "exponential", "100K", () => exponential(1.0, [100000]));

// ── gamma ───────────────────────────────────────────────

run(suite, "gamma", "1K", () => gamma(2.0, 1.0, [1000]));
run(suite, "gamma", "10K", () => gamma(2.0, 1.0, [10000]));
run(suite, "gamma", "100K", () => gamma(2.0, 1.0, [100000]));

// ── beta ────────────────────────────────────────────────

run(suite, "beta", "1K", () => beta(2.0, 5.0, [1000]));
run(suite, "beta", "10K", () => beta(2.0, 5.0, [10000]));
run(suite, "beta", "100K", () => beta(2.0, 5.0, [100000]));

// ── choice ──────────────────────────────────────────────

const pool1k = arange(0, 1000);

run(suite, "choice (replace)", "100 from 1K", () => choice(pool1k, 100, true));
run(suite, "choice (replace)", "500 from 1K", () => choice(pool1k, 500, true));
run(suite, "choice (no replace)", "100 from 1K", () => choice(pool1k, 100, false));
run(suite, "choice (no replace)", "500 from 1K", () => choice(pool1k, 500, false));

// ── shuffle ─────────────────────────────────────────────

run(suite, "shuffle", "1K", () => {
  const t = arange(0, 1000);
  shuffle(t);
});
run(suite, "shuffle", "10K", () => {
  const t = arange(0, 10000);
  shuffle(t);
});

// ── permutation ─────────────────────────────────────────

run(suite, "permutation", "1K", () => permutation(1000));
run(suite, "permutation", "10K", () => permutation(10000));
run(suite, "permutation", "100K", () => permutation(100000));

footer(suite, "deepbox-random.json");
