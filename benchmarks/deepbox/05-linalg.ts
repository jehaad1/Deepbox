/**
 * Benchmark 05 — Linear Algebra
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
  norm,
  pinv,
  qr,
  solve,
  svd,
  trace,
} from "deepbox/linalg";
import { dot, tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("linalg");
header("Benchmark 05 — Linear Algebra");

// ── Well-conditioned matrices ─────────────────────────────

function makeMatrix(n: number, seed: number): number[][] {
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) row.push(rand() * 2 - 1);
    row[i] = (row[i] ?? 0) + n;
    m.push(row);
  }
  return m;
}

function makeSymPD(n: number, seed: number): number[][] {
  const A = makeMatrix(n, seed);
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      let v = 0;
      for (let k = 0; k < n; k++) v += (A[k]?.[i] ?? 0) * (A[k]?.[j] ?? 0);
      row.push(i === j ? v + n : v);
    }
    m.push(row);
  }
  return m;
}

const mat50 = tensor(makeMatrix(50, 42));
const mat100 = tensor(makeMatrix(100, 123));
const mat200 = tensor(makeMatrix(200, 456));
const sym50 = tensor(makeSymPD(50, 42));
const sym100 = tensor(makeSymPD(100, 123));

// ── Properties ────────────────────────────────────────────

run(suite, "determinant", "50x50", () => det(mat50));
run(suite, "determinant", "100x100", () => det(mat100));
run(suite, "determinant", "200x200", () => det(mat200), { iterations: 10 });
run(suite, "trace", "50x50", () => trace(mat50));
run(suite, "trace", "200x200", () => trace(mat200));
run(suite, "frobenius norm", "50x50", () => norm(mat50));
run(suite, "frobenius norm", "100x100", () => norm(mat100));
run(suite, "frobenius norm", "200x200", () => norm(mat200));
run(suite, "condition number", "50x50", () => cond(mat50));
run(suite, "condition number", "100x100", () => cond(mat100));

// ── Inverse ───────────────────────────────────────────────

run(suite, "inverse", "50x50", () => inv(mat50));
run(suite, "inverse", "100x100", () => inv(mat100));
run(suite, "inverse", "200x200", () => inv(mat200), { iterations: 10 });
run(suite, "pseudo-inverse", "50x50", () => pinv(mat50));
run(suite, "pseudo-inverse", "100x100", () => pinv(mat100));

// ── Decompositions ────────────────────────────────────────

run(suite, "SVD", "50x50", () => svd(mat50));
run(suite, "SVD", "100x100", () => svd(mat100), { iterations: 10 });
run(suite, "QR", "50x50", () => qr(mat50));
run(suite, "QR", "100x100", () => qr(mat100));
run(suite, "QR", "200x200", () => qr(mat200), { iterations: 10 });
run(suite, "LU", "50x50", () => lu(mat50));
run(suite, "LU", "100x100", () => lu(mat100));
run(suite, "LU", "200x200", () => lu(mat200), { iterations: 10 });
run(suite, "Cholesky", "50x50", () => cholesky(sym50));
run(suite, "Cholesky", "100x100", () => cholesky(sym100));
run(suite, "eigenvalues (sym)", "50x50", () => eigvalsh(sym50));
run(suite, "eigenvalues (sym)", "100x100", () => eigvalsh(sym100));

// ── Solve ─────────────────────────────────────────────────

const b50 = tensor(Array.from({ length: 50 }, (_, i) => [i + 1]));
const b100 = tensor(Array.from({ length: 100 }, (_, i) => [i + 1]));
const b200 = tensor(Array.from({ length: 200 }, (_, i) => [i + 1]));

run(suite, "solve (Ax=b)", "50x50", () => solve(mat50, b50));
run(suite, "solve (Ax=b)", "100x100", () => solve(mat100, b100));
run(suite, "solve (Ax=b)", "200x200", () => solve(mat200, b200), {
  iterations: 10,
});
run(suite, "lstsq", "50x50", () => lstsq(mat50, b50));
run(suite, "lstsq", "100x100", () => lstsq(mat100, b100));

// ── Matrix Multiply ───────────────────────────────────────

run(suite, "matmul", "50x50", () => dot(mat50, mat50));
run(suite, "matmul", "100x100", () => dot(mat100, mat100));
run(suite, "matmul", "200x200", () => dot(mat200, mat200), { iterations: 10 });

footer(suite, "deepbox-linalg.json");
