"""
Benchmark 11 — Random Number Generation
NumPy
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils import run, create_suite, header, footer

suite = create_suite("random", "NumPy")
header("Benchmark 11 — Random Number Generation", "NumPy")

rng = np.random.RandomState(42)

# ── rand ────────────────────────────────────────────────

run(suite, "rand", "1K", lambda: rng.rand(1000))
run(suite, "rand", "10K", lambda: rng.rand(10000))
run(suite, "rand", "100K", lambda: rng.rand(100000))
run(suite, "rand", "1M", lambda: rng.rand(1000000))

# ── randn ───────────────────────────────────────────────

run(suite, "randn", "1K", lambda: rng.randn(1000))
run(suite, "randn", "10K", lambda: rng.randn(10000))
run(suite, "randn", "100K", lambda: rng.randn(100000))
run(suite, "randn", "1M", lambda: rng.randn(1000000))

# ── randint ─────────────────────────────────────────────

run(suite, "randint", "1K", lambda: rng.randint(0, 100, 1000))
run(suite, "randint", "10K", lambda: rng.randint(0, 100, 10000))
run(suite, "randint", "100K", lambda: rng.randint(0, 100, 100000))
run(suite, "randint", "1M", lambda: rng.randint(0, 100, 1000000))

# ── uniform ─────────────────────────────────────────────

run(suite, "uniform", "1K", lambda: rng.uniform(0, 1, 1000))
run(suite, "uniform", "10K", lambda: rng.uniform(0, 1, 10000))
run(suite, "uniform", "100K", lambda: rng.uniform(0, 1, 100000))
run(suite, "uniform", "1M", lambda: rng.uniform(0, 1, 1000000))

# ── normal ──────────────────────────────────────────────

run(suite, "normal", "1K", lambda: rng.normal(0, 1, 1000))
run(suite, "normal", "10K", lambda: rng.normal(0, 1, 10000))
run(suite, "normal", "100K", lambda: rng.normal(0, 1, 100000))
run(suite, "normal", "1M", lambda: rng.normal(0, 1, 1000000))

# ── binomial ────────────────────────────────────────────

run(suite, "binomial", "1K", lambda: rng.binomial(10, 0.5, 1000))
run(suite, "binomial", "10K", lambda: rng.binomial(10, 0.5, 10000))
run(suite, "binomial", "100K", lambda: rng.binomial(10, 0.5, 100000))

# ── poisson ─────────────────────────────────────────────

run(suite, "poisson", "1K", lambda: rng.poisson(5, 1000))
run(suite, "poisson", "10K", lambda: rng.poisson(5, 10000))
run(suite, "poisson", "100K", lambda: rng.poisson(5, 100000))

# ── exponential ─────────────────────────────────────────

run(suite, "exponential", "1K", lambda: rng.exponential(1.0, 1000))
run(suite, "exponential", "10K", lambda: rng.exponential(1.0, 10000))
run(suite, "exponential", "100K", lambda: rng.exponential(1.0, 100000))

# ── gamma ───────────────────────────────────────────────

run(suite, "gamma", "1K", lambda: rng.gamma(2.0, 1.0, 1000))
run(suite, "gamma", "10K", lambda: rng.gamma(2.0, 1.0, 10000))
run(suite, "gamma", "100K", lambda: rng.gamma(2.0, 1.0, 100000))

# ── beta ────────────────────────────────────────────────

run(suite, "beta", "1K", lambda: rng.beta(2.0, 5.0, 1000))
run(suite, "beta", "10K", lambda: rng.beta(2.0, 5.0, 10000))
run(suite, "beta", "100K", lambda: rng.beta(2.0, 5.0, 100000))

# ── choice ──────────────────────────────────────────────

pool1k = np.arange(1000)

run(suite, "choice (replace)", "100 from 1K", lambda: rng.choice(pool1k, 100, replace=True))
run(suite, "choice (replace)", "500 from 1K", lambda: rng.choice(pool1k, 500, replace=True))
run(suite, "choice (no replace)", "100 from 1K", lambda: rng.choice(pool1k, 100, replace=False))
run(suite, "choice (no replace)", "500 from 1K", lambda: rng.choice(pool1k, 500, replace=False))

# ── shuffle ─────────────────────────────────────────────

def do_shuffle(n):
    a = np.arange(n)
    rng.shuffle(a)

run(suite, "shuffle", "1K", lambda: do_shuffle(1000))
run(suite, "shuffle", "10K", lambda: do_shuffle(10000))

# ── permutation ─────────────────────────────────────────

run(suite, "permutation", "1K", lambda: rng.permutation(1000))
run(suite, "permutation", "10K", lambda: rng.permutation(10000))
run(suite, "permutation", "100K", lambda: rng.permutation(100000))

footer(suite, "numpy-random.json")
