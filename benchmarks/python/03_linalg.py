"""
Benchmark 03 — Linear Algebra
NumPy / SciPy
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.linalg as la
from utils import run, create_suite, header, footer

suite = create_suite("linalg", "NumPy/SciPy")
header("Benchmark 03 — Linear Algebra", "NumPy/SciPy")

# ── Matrix generators ───────────────────────────────────

rng = np.random.RandomState(42)

def rand_matrix(n, m=None, seed=42):
    r = np.random.RandomState(seed)
    return r.randn(n, m if m else n)

def sym_pos_def(n, seed=42):
    r = np.random.RandomState(seed)
    A = r.randn(n, n)
    return A @ A.T + n * np.eye(n)

def upper_tri(n, seed=42):
    r = np.random.RandomState(seed)
    U = np.triu(r.randn(n, n))
    np.fill_diagonal(U, np.abs(np.diag(U)) + 1)
    return U

m10 = rand_matrix(10)
m25 = rand_matrix(25)
m50 = rand_matrix(50)
m100 = rand_matrix(100)
m200 = rand_matrix(200)
sym25 = sym_pos_def(25)
sym50 = sym_pos_def(50)
sym100 = sym_pos_def(100)
sym200 = sym_pos_def(200)
tri25 = upper_tri(25)
tri50 = upper_tri(50)
tri100 = upper_tri(100)
b25 = rand_matrix(25, 1, 99)
b50 = rand_matrix(50, 1, 99)
b100 = rand_matrix(100, 1, 99)
b200 = rand_matrix(200, 1, 99)
rect50x30 = rand_matrix(50, 30)
rect100x50 = rand_matrix(100, 50)

# ── Determinant ─────────────────────────────────────────

run(suite, "det", "10x10", lambda: np.linalg.det(m10))
run(suite, "det", "25x25", lambda: np.linalg.det(m25))
run(suite, "det", "50x50", lambda: np.linalg.det(m50))
run(suite, "det", "100x100", lambda: np.linalg.det(m100))

# ── Trace ───────────────────────────────────────────────

run(suite, "trace", "25x25", lambda: np.trace(m25))
run(suite, "trace", "100x100", lambda: np.trace(m100))
run(suite, "trace", "200x200", lambda: np.trace(m200))

# ── Norm ────────────────────────────────────────────────

run(suite, "norm (fro)", "25x25", lambda: np.linalg.norm(m25, 'fro'))
run(suite, "norm (fro)", "50x50", lambda: np.linalg.norm(m50, 'fro'))
run(suite, "norm (fro)", "100x100", lambda: np.linalg.norm(m100, 'fro'))

# ── Condition Number ────────────────────────────────────

run(suite, "cond", "25x25", lambda: np.linalg.cond(m25))
run(suite, "cond", "50x50", lambda: np.linalg.cond(m50))

# ── Matrix Rank ─────────────────────────────────────────

run(suite, "matrixRank", "25x25", lambda: np.linalg.matrix_rank(m25))
run(suite, "matrixRank", "50x50", lambda: np.linalg.matrix_rank(m50))

# ── Slogdet ─────────────────────────────────────────────

run(suite, "slogdet", "25x25", lambda: np.linalg.slogdet(m25))
run(suite, "slogdet", "50x50", lambda: np.linalg.slogdet(m50))
run(suite, "slogdet", "100x100", lambda: np.linalg.slogdet(m100))

# ── Inverse ─────────────────────────────────────────────

run(suite, "inv", "10x10", lambda: np.linalg.inv(m10))
run(suite, "inv", "25x25", lambda: np.linalg.inv(m25))
run(suite, "inv", "50x50", lambda: np.linalg.inv(m50))
run(suite, "inv", "100x100", lambda: np.linalg.inv(m100))

# ── Pseudo-Inverse ──────────────────────────────────────

run(suite, "pinv", "25x25", lambda: np.linalg.pinv(m25))
run(suite, "pinv", "50x50", lambda: np.linalg.pinv(m50))
run(suite, "pinv", "50x30", lambda: np.linalg.pinv(rect50x30))

# ── SVD ─────────────────────────────────────────────────

run(suite, "svd", "10x10", lambda: np.linalg.svd(m10))
run(suite, "svd", "25x25", lambda: np.linalg.svd(m25))
run(suite, "svd", "50x50", lambda: np.linalg.svd(m50))
run(suite, "svd", "100x100", lambda: np.linalg.svd(m100), iterations=10)

# ── QR ──────────────────────────────────────────────────

run(suite, "qr", "10x10", lambda: np.linalg.qr(m10))
run(suite, "qr", "25x25", lambda: np.linalg.qr(m25))
run(suite, "qr", "50x50", lambda: np.linalg.qr(m50))
run(suite, "qr", "100x100", lambda: np.linalg.qr(m100), iterations=10)

# ── LU ──────────────────────────────────────────────────

run(suite, "lu", "10x10", lambda: la.lu(m10))
run(suite, "lu", "25x25", lambda: la.lu(m25))
run(suite, "lu", "50x50", lambda: la.lu(m50))
run(suite, "lu", "100x100", lambda: la.lu(m100))

# ── Cholesky ────────────────────────────────────────────

run(suite, "cholesky", "25x25", lambda: np.linalg.cholesky(sym25))
run(suite, "cholesky", "50x50", lambda: np.linalg.cholesky(sym50))
run(suite, "cholesky", "100x100", lambda: np.linalg.cholesky(sym100))

# ── Eigenvalues (symmetric) ─────────────────────────────

run(suite, "eigvalsh", "25x25", lambda: np.linalg.eigvalsh(sym25))
run(suite, "eigvalsh", "50x50", lambda: np.linalg.eigvalsh(sym50))
run(suite, "eigvalsh", "100x100", lambda: np.linalg.eigvalsh(sym100))

# ── Solve ───────────────────────────────────────────────

run(suite, "solve", "25x25", lambda: np.linalg.solve(sym25, b25))
run(suite, "solve", "50x50", lambda: np.linalg.solve(sym50, b50))
run(suite, "solve", "100x100", lambda: np.linalg.solve(sym100, b100))

# ── Solve Triangular ────────────────────────────────────

run(suite, "solveTriangular", "25x25", lambda: la.solve_triangular(tri25, b25))
run(suite, "solveTriangular", "50x50", lambda: la.solve_triangular(tri50, b50))
run(suite, "solveTriangular", "100x100", lambda: la.solve_triangular(tri100, b100))

# ── Least Squares ───────────────────────────────────────

run(suite, "lstsq", "50x30", lambda: np.linalg.lstsq(rect50x30, b50, rcond=None))
run(suite, "lstsq", "100x50", lambda: np.linalg.lstsq(rect100x50, b100, rcond=None))

# ── Matmul ──────────────────────────────────────────────

run(suite, "matmul", "25x25", lambda: m25 @ m25)
run(suite, "matmul", "50x50", lambda: m50 @ m50)
run(suite, "matmul", "100x100", lambda: m100 @ m100)
run(suite, "matmul", "200x200", lambda: m200 @ m200, iterations=10)

footer(suite, "numpy-linalg.json")
