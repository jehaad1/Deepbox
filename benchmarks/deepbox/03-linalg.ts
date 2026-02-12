/**
 * Benchmark 03 — Linear Algebra
 * Deepbox vs NumPy/SciPy
 */

import {
  cholesky,
  cond,
  det,
  eigvalsh,
  inv,
  lstsq,
  lu,
  matrixRank,
  norm,
  pinv,
  qr,
  slogdet,
  solve,
  solveTriangular,
  svd,
  trace,
} from "deepbox/linalg";
import { dot, tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("linalg");
header("Benchmark 03 — Linear Algebra");

// ── Matrix generators ───────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 2 ** 32) * 2 - 1;
  };
}

function randMatrix(rows: number, cols: number, seed: number) {
  const rand = seededRng(seed);
  const data: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) row.push(rand());
    data.push(row);
  }
  return tensor(data);
}

function symPosDefMatrix(n: number, seed: number) {
  const rand = seededRng(seed);
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) row.push(rand());
    A.push(row);
  }
  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[i][k] * A[j][k];
      row.push(s + (i === j ? n : 0));
    }
    result.push(row);
  }
  return tensor(result);
}

function upperTriangular(n: number, seed: number) {
  const rand = seededRng(seed);
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) row.push(j >= i ? rand() + (i === j ? 1 : 0) : 0);
    data.push(row);
  }
  return tensor(data);
}

// Matrices
const m10 = randMatrix(10, 10, 42);
const m25 = randMatrix(25, 25, 42);
const m50 = randMatrix(50, 50, 42);
const m100 = randMatrix(100, 100, 42);
const m200 = randMatrix(200, 200, 42);
const sym25 = symPosDefMatrix(25, 42);
const sym50 = symPosDefMatrix(50, 42);
const sym100 = symPosDefMatrix(100, 42);
const _sym200 = symPosDefMatrix(200, 42);
const tri25 = upperTriangular(25, 42);
const tri50 = upperTriangular(50, 42);
const tri100 = upperTriangular(100, 42);
const b25 = randMatrix(25, 1, 99);
const b50 = randMatrix(50, 1, 99);
const b100 = randMatrix(100, 1, 99);
const _b200 = randMatrix(200, 1, 99);
const rect50x30 = randMatrix(50, 30, 42);
const rect100x50 = randMatrix(100, 50, 42);

// ── Determinant ─────────────────────────────────────────

run(suite, "det", "10x10", () => det(m10));
run(suite, "det", "25x25", () => det(m25));
run(suite, "det", "50x50", () => det(m50));
run(suite, "det", "100x100", () => det(m100));

// ── Trace ───────────────────────────────────────────────

run(suite, "trace", "25x25", () => trace(m25));
run(suite, "trace", "100x100", () => trace(m100));
run(suite, "trace", "200x200", () => trace(m200));

// ── Norm ────────────────────────────────────────────────

run(suite, "norm (fro)", "25x25", () => norm(m25));
run(suite, "norm (fro)", "50x50", () => norm(m50));
run(suite, "norm (fro)", "100x100", () => norm(m100));

// ── Condition Number ────────────────────────────────────

run(suite, "cond", "25x25", () => cond(m25));
run(suite, "cond", "50x50", () => cond(m50));

// ── Matrix Rank ─────────────────────────────────────────

run(suite, "matrixRank", "25x25", () => matrixRank(m25));
run(suite, "matrixRank", "50x50", () => matrixRank(m50));

// ── Slogdet ─────────────────────────────────────────────

run(suite, "slogdet", "25x25", () => slogdet(m25));
run(suite, "slogdet", "50x50", () => slogdet(m50));
run(suite, "slogdet", "100x100", () => slogdet(m100));

// ── Inverse ─────────────────────────────────────────────

run(suite, "inv", "10x10", () => inv(m10));
run(suite, "inv", "25x25", () => inv(m25));
run(suite, "inv", "50x50", () => inv(m50));
run(suite, "inv", "100x100", () => inv(m100));

// ── Pseudo-Inverse ──────────────────────────────────────

run(suite, "pinv", "25x25", () => pinv(m25));
run(suite, "pinv", "50x50", () => pinv(m50));
run(suite, "pinv", "50x30", () => pinv(rect50x30));

// ── SVD ─────────────────────────────────────────────────

run(suite, "svd", "10x10", () => svd(m10));
run(suite, "svd", "25x25", () => svd(m25));
run(suite, "svd", "50x50", () => svd(m50));
run(suite, "svd", "100x100", () => svd(m100), { iterations: 10 });

// ── QR ──────────────────────────────────────────────────

run(suite, "qr", "10x10", () => qr(m10));
run(suite, "qr", "25x25", () => qr(m25));
run(suite, "qr", "50x50", () => qr(m50));
run(suite, "qr", "100x100", () => qr(m100), { iterations: 10 });

// ── LU ──────────────────────────────────────────────────

run(suite, "lu", "10x10", () => lu(m10));
run(suite, "lu", "25x25", () => lu(m25));
run(suite, "lu", "50x50", () => lu(m50));
run(suite, "lu", "100x100", () => lu(m100));

// ── Cholesky ────────────────────────────────────────────

run(suite, "cholesky", "25x25", () => cholesky(sym25));
run(suite, "cholesky", "50x50", () => cholesky(sym50));
run(suite, "cholesky", "100x100", () => cholesky(sym100));

// ── Eigenvalues (symmetric) ─────────────────────────────

run(suite, "eigvalsh", "25x25", () => eigvalsh(sym25));
run(suite, "eigvalsh", "50x50", () => eigvalsh(sym50));
run(suite, "eigvalsh", "100x100", () => eigvalsh(sym100));

// ── Solve ───────────────────────────────────────────────

run(suite, "solve", "25x25", () => solve(sym25, b25));
run(suite, "solve", "50x50", () => solve(sym50, b50));
run(suite, "solve", "100x100", () => solve(sym100, b100));

// ── Solve Triangular ────────────────────────────────────

run(suite, "solveTriangular", "25x25", () => solveTriangular(tri25, b25));
run(suite, "solveTriangular", "50x50", () => solveTriangular(tri50, b50));
run(suite, "solveTriangular", "100x100", () => solveTriangular(tri100, b100));

// ── Least Squares ───────────────────────────────────────

run(suite, "lstsq", "50x30", () => lstsq(rect50x30, b50));
run(suite, "lstsq", "100x50", () => lstsq(rect100x50, b100));

// ── Matmul (dot) ────────────────────────────────────────

run(suite, "matmul", "25x25", () => dot(m25, m25));
run(suite, "matmul", "50x50", () => dot(m50, m50));
run(suite, "matmul", "100x100", () => dot(m100, m100));
run(suite, "matmul", "200x200", () => dot(m200, m200), { iterations: 10 });

footer(suite, "deepbox-linalg.json");
