/**
 * Benchmark 05: Linear Algebra — Deepbox vs NumPy/SciPy
 *
 * Compares matrix decompositions, solvers, and properties:
 * SVD, QR, LU, Cholesky, determinant, inverse, solve.
 */

import { det, inv, lu, norm, qr, solve, svd, trace } from "deepbox/linalg";
import { dot, tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("linear-algebra");
header("Benchmark 05: Linear Algebra");

// ── Helper: create a well-conditioned matrix ──────────────

function makeMatrix(n: number, seed: number): number[][] {
  let state = seed >>> 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(rand() * 2 - 1);
    }
    row[i] = (row[i] ?? 0) + n; // Make diagonally dominant
    m.push(row);
  }
  return m;
}

const mat50 = tensor(makeMatrix(50, 42));
const mat100 = tensor(makeMatrix(100, 123));
const mat200 = tensor(makeMatrix(200, 456));

// ── Determinant ───────────────────────────────────────────

run(suite, "determinant", "50x50", () => {
  det(mat50);
});
run(suite, "determinant", "100x100", () => {
  det(mat100);
});
run(
  suite,
  "determinant",
  "200x200",
  () => {
    det(mat200);
  },
  { iterations: 10 }
);

// ── Trace ─────────────────────────────────────────────────

run(suite, "trace", "50x50", () => {
  trace(mat50);
});
run(suite, "trace", "200x200", () => {
  trace(mat200);
});

// ── Matrix Inverse ────────────────────────────────────────

run(suite, "inverse", "50x50", () => {
  inv(mat50);
});
run(suite, "inverse", "100x100", () => {
  inv(mat100);
});
run(
  suite,
  "inverse",
  "200x200",
  () => {
    inv(mat200);
  },
  { iterations: 10 }
);

// ── Matrix Norm ───────────────────────────────────────────

run(suite, "frobenius norm", "50x50", () => {
  norm(mat50);
});
run(suite, "frobenius norm", "100x100", () => {
  norm(mat100);
});
run(suite, "frobenius norm", "200x200", () => {
  norm(mat200);
});

// ── SVD ───────────────────────────────────────────────────

run(suite, "SVD", "50x50", () => {
  svd(mat50);
});
run(
  suite,
  "SVD",
  "100x100",
  () => {
    svd(mat100);
  },
  { iterations: 10 }
);

// ── QR Decomposition ──────────────────────────────────────

run(suite, "QR decomposition", "50x50", () => {
  qr(mat50);
});
run(suite, "QR decomposition", "100x100", () => {
  qr(mat100);
});
run(
  suite,
  "QR decomposition",
  "200x200",
  () => {
    qr(mat200);
  },
  { iterations: 10 }
);

// ── LU Decomposition ──────────────────────────────────────

run(suite, "LU decomposition", "50x50", () => {
  lu(mat50);
});
run(suite, "LU decomposition", "100x100", () => {
  lu(mat100);
});
run(
  suite,
  "LU decomposition",
  "200x200",
  () => {
    lu(mat200);
  },
  { iterations: 10 }
);

// ── Solve Linear System ───────────────────────────────────

const b50 = tensor(Array.from({ length: 50 }, (_, i) => [i + 1]));
const b100 = tensor(Array.from({ length: 100 }, (_, i) => [i + 1]));
const b200 = tensor(Array.from({ length: 200 }, (_, i) => [i + 1]));

run(suite, "solve (Ax=b)", "50x50", () => {
  solve(mat50, b50);
});
run(suite, "solve (Ax=b)", "100x100", () => {
  solve(mat100, b100);
});
run(
  suite,
  "solve (Ax=b)",
  "200x200",
  () => {
    solve(mat200, b200);
  },
  { iterations: 10 }
);

// ── Matrix Multiply ───────────────────────────────────────

run(suite, "matmul", "50x50", () => {
  dot(mat50, mat50);
});
run(suite, "matmul", "100x100", () => {
  dot(mat100, mat100);
});
run(
  suite,
  "matmul",
  "200x200",
  () => {
    dot(mat200, mat200);
  },
  { iterations: 10 }
);

footer(suite, "deepbox-linalg-ops.json");
