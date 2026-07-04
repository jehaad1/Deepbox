/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { at, fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";
import { schur } from "../decomposition/schur";

/**
 * Solve the continuous Sylvester equation AX + XB = C.
 *
 * Uses the Bartels–Stewart algorithm:
 * 1. Compute real Schur decompositions A = U*Ta*U^T and B = V*Tb*V^T
 * 2. Transform to U^T*C*V = Ta*Y + Y*Tb where Y = U^T*X*V
 * 3. Solve the quasi-triangular system column by column
 * 4. Recover X = U*Y*V^T
 *
 * **Time Complexity**: O(M³ + N³ + M²N) where A is M×M and B is N×N
 *
 * @param a - Square matrix A of shape (M, M)
 * @param b - Square matrix B of shape (N, N)
 * @param c - Matrix C of shape (M, N)
 * @returns Solution matrix X of shape (M, N)
 *
 * @example
 * ```ts
 * import { sylvester } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 0], [0, 2]]);
 * const B = tensor([[3, 0], [0, 4]]);
 * const C = tensor([[4, 5], [6, 12]]);
 * const X = sylvester(A, B, C);
 * // A*X + X*B ≈ C
 * ```
 */
export function sylvester(a: Tensor, b: Tensor, c: Tensor): Tensor {
  if (a.ndim !== 2 || b.ndim !== 2 || c.ndim !== 2) {
    throw new ShapeError("sylvester() requires 2D matrices");
  }

  const m = getDim(a, 0, "sylvester()");
  const m2 = getDim(a, 1, "sylvester()");
  const n = getDim(b, 0, "sylvester()");
  const n2 = getDim(b, 1, "sylvester()");
  const cm = getDim(c, 0, "sylvester()");
  const cn = getDim(c, 1, "sylvester()");

  if (m !== m2) {
    throw new ShapeError(`sylvester(): A must be square; got (${m}, ${m2})`);
  }
  if (n !== n2) {
    throw new ShapeError(`sylvester(): B must be square; got (${n}, ${n2})`);
  }
  if (cm !== m || cn !== n) {
    throw new ShapeError(`sylvester(): C must be (${m}, ${n}); got (${cm}, ${cn})`);
  }

  // Schur decompositions
  const [Ta, U] = schur(a);
  const [Tb, V] = schur(b);

  const tA = toDenseMatrix2D(Ta);
  const tB = toDenseMatrix2D(Tb);
  const uMat = toDenseMatrix2D(U);
  const vMat = toDenseMatrix2D(V);
  const cMat = toDenseMatrix2D(c);

  // F = U^T * C * V
  const F = new Float64Array(m * n);
  const temp = new Float64Array(m * n);

  // temp = U^T * C
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) {
        s += at(uMat.data, k * m + i) * at(cMat.data, k * n + j);
      }
      temp[i * n + j] = s;
    }
  }

  // F = temp * V
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += at(temp, i * n + k) * at(vMat.data, k * n + j);
      }
      F[i * n + j] = s;
    }
  }

  // Solve Ta*Y + Y*Tb = F column by column (forward substitution)
  // For upper quasi-triangular Tb, (Y*Tb)[:,j] = sum_{k<=j} Y[:,k]*Tb[k,j]
  // So process columns left-to-right: columns k < j are already solved.
  const Y = new Float64Array(m * n);

  let j = 0;
  while (j < n) {
    // Check for 2x2 block in Tb (subdiagonal entry at [j+1, j])
    const is2x2 = j + 1 < n && Math.abs(at(tB.data, (j + 1) * n + j)) > 1e-14;

    if (!is2x2) {
      // 1x1 block: solve (Ta + Tb[j,j]*I) * y_j = rhs_j
      const bjj = at(tB.data, j * n + j);
      // Build RHS: F[:,j] - sum_{k<j} Y[:,k] * Tb[k,j]
      const rhs = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        let s = at(F, i * n + j);
        for (let k = 0; k < j; k++) {
          s -= at(Y, i * n + k) * at(tB.data, k * n + j);
        }
        rhs[i] = s;
      }

      solveShiftedQuasiTriangular(tA.data, m, bjj, rhs);

      for (let i = 0; i < m; i++) {
        Y[i * n + j] = at(rhs, i);
      }
      j++;
    } else {
      // 2x2 block: columns j, j+1
      const j1 = j;
      const j2 = j + 1;
      const b11 = at(tB.data, j1 * n + j1);
      const b12 = at(tB.data, j1 * n + j2);
      const b21 = at(tB.data, j2 * n + j1);
      const b22 = at(tB.data, j2 * n + j2);

      const rhs1 = new Float64Array(m);
      const rhs2 = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        let s1 = at(F, i * n + j1);
        let s2 = at(F, i * n + j2);
        for (let k = 0; k < j1; k++) {
          s1 -= at(Y, i * n + k) * at(tB.data, k * n + j1);
          s2 -= at(Y, i * n + k) * at(tB.data, k * n + j2);
        }
        rhs1[i] = s1;
        rhs2[i] = s2;
      }

      solve2x2CoupledSystem(tA.data, m, b11, b12, b21, b22, rhs1, rhs2);

      for (let i = 0; i < m; i++) {
        Y[i * n + j1] = at(rhs1, i);
        Y[i * n + j2] = at(rhs2, i);
      }
      j += 2;
    }
  }

  // X = U * Y * V^T
  const temp2 = new Float64Array(m * n);

  // temp2 = U * Y
  for (let i = 0; i < m; i++) {
    for (let jj = 0; jj < n; jj++) {
      let s = 0;
      for (let k = 0; k < m; k++) {
        s += at(uMat.data, i * m + k) * at(Y, k * n + jj);
      }
      temp2[i * n + jj] = s;
    }
  }

  // X = temp2 * V^T
  const X = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let jj = 0; jj < n; jj++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += at(temp2, i * n + k) * at(vMat.data, jj * n + k);
      }
      X[i * n + jj] = s;
    }
  }

  return fromDenseMatrix2D(m, n, X);
}

/**
 * Solve the continuous Lyapunov equation AX + XAᵀ = Q.
 *
 * This is a special case of the Sylvester equation with B = Aᵀ.
 *
 * @param a - Square matrix A of shape (N, N)
 * @param q - Symmetric matrix Q of shape (N, N)
 * @returns Solution matrix X of shape (N, N)
 *
 * @example
 * ```ts
 * import { lyapunov } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[-1, 0], [0, -2]]);
 * const Q = tensor([[2, 0], [0, 8]]);
 * const X = lyapunov(A, Q);
 * // A*X + X*A^T ≈ Q
 * ```
 */
export function lyapunov(a: Tensor, q: Tensor): Tensor {
  if (a.ndim !== 2 || q.ndim !== 2) {
    throw new ShapeError("lyapunov() requires 2D matrices");
  }

  const n = getDim(a, 0, "lyapunov()");
  const n2 = getDim(a, 1, "lyapunov()");
  const qn = getDim(q, 0, "lyapunov()");
  const qn2 = getDim(q, 1, "lyapunov()");

  if (n !== n2) {
    throw new ShapeError(`lyapunov(): A must be square; got (${n}, ${n2})`);
  }
  if (qn !== n || qn2 !== n) {
    throw new ShapeError(`lyapunov(): Q must be (${n}, ${n}); got (${qn}, ${qn2})`);
  }

  // Build A^T
  const aMat = toDenseMatrix2D(a);
  const aT = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aT[i * n + j] = at(aMat.data, j * n + i);
    }
  }

  const aTTensor = fromDenseMatrix2D(n, n, aT);
  return sylvester(a, aTTensor, q);
}

// ---- Internal helpers ----

/**
 * Solve (T + sigma*I) * x = rhs where T is quasi-upper triangular (Schur form).
 * Back-substitution from bottom. Modifies rhs in-place.
 */
function solveShiftedQuasiTriangular(
  T: Float64Array,
  n: number,
  sigma: number,
  rhs: Float64Array
): void {
  let i = n - 1;
  while (i >= 0) {
    // Check for 2x2 block
    const is2x2 = i > 0 && Math.abs(at(T, i * n + (i - 1))) > 1e-14;

    if (!is2x2) {
      // 1x1 block
      let s = at(rhs, i);
      for (let k = i + 1; k < n; k++) {
        s -= at(T, i * n + k) * at(rhs, k);
      }
      const diag = at(T, i * n + i) + sigma;
      if (Math.abs(diag) <= 1e-30)
        throw new DataValidationError(
          "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique"
        );
      rhs[i] = s / diag;
      i--;
    } else {
      // 2x2 block at rows i-1, i
      const i1 = i - 1;
      let s1 = at(rhs, i1);
      let s2 = at(rhs, i);
      for (let k = i + 1; k < n; k++) {
        s1 -= at(T, i1 * n + k) * at(rhs, k);
        s2 -= at(T, i * n + k) * at(rhs, k);
      }

      const a11 = at(T, i1 * n + i1) + sigma;
      const a12 = at(T, i1 * n + i);
      const a21 = at(T, i * n + i1);
      const a22 = at(T, i * n + i) + sigma;

      const det = a11 * a22 - a12 * a21;
      if (Math.abs(det) <= 1e-30)
        throw new DataValidationError(
          "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique"
        );
      rhs[i1] = (a22 * s1 - a12 * s2) / det;
      rhs[i] = (a11 * s2 - a21 * s1) / det;
      i -= 2;
    }
  }
}

/**
 * Solve coupled 2x2 Sylvester system for 2x2 block in Tb.
 * Modifies rhs1 and rhs2 in-place.
 */
function solve2x2CoupledSystem(
  T: Float64Array,
  n: number,
  b11: number,
  b12: number,
  b21: number,
  b22: number,
  rhs1: Float64Array,
  rhs2: Float64Array
): void {
  // Solve row by row from bottom using back-substitution
  let i = n - 1;
  while (i >= 0) {
    const is2x2 = i > 0 && Math.abs(at(T, i * n + (i - 1))) > 1e-14;

    if (!is2x2) {
      let s1 = at(rhs1, i);
      let s2 = at(rhs2, i);
      for (let k = i + 1; k < n; k++) {
        const tik = at(T, i * n + k);
        s1 -= tik * at(rhs1, k);
        s2 -= tik * at(rhs2, k);
      }

      const tii = at(T, i * n + i);
      // System: (tii + b11)*x1 + b21*x2 = s1
      //          b12*x1 + (tii + b22)*x2 = s2
      const a11 = tii + b11;
      const a22 = tii + b22;
      const det = a11 * a22 - b12 * b21;
      if (Math.abs(det) <= 1e-30)
        throw new DataValidationError(
          "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique"
        );
      rhs1[i] = (a22 * s1 - b21 * s2) / det;
      rhs2[i] = (a11 * s2 - b12 * s1) / det;
      i--;
    } else {
      // 4x4 system — use direct formula for small system
      const i1 = i - 1;
      let s11 = at(rhs1, i1);
      let s12 = at(rhs2, i1);
      let s21 = at(rhs1, i);
      let s22 = at(rhs2, i);
      for (let k = i + 1; k < n; k++) {
        const ti1k = at(T, i1 * n + k);
        const tik = at(T, i * n + k);
        s11 -= ti1k * at(rhs1, k);
        s12 -= ti1k * at(rhs2, k);
        s21 -= tik * at(rhs1, k);
        s22 -= tik * at(rhs2, k);
      }

      // Simple iterative solve for small 4x4 system
      const t11 = at(T, i1 * n + i1);
      const t12 = at(T, i1 * n + i);
      const t21 = at(T, i * n + i1);
      const t22 = at(T, i * n + i);

      // Build 4x4 system M*x = rhs
      // x = [rhs1[i1], rhs1[i], rhs2[i1], rhs2[i]]
      const M = [
        t11 + b11,
        t12,
        b21,
        0,
        t21,
        t22 + b11,
        0,
        b21,
        b12,
        0,
        t11 + b22,
        t12,
        0,
        b12,
        t21,
        t22 + b22,
      ];
      const r = [s11, s21, s12, s22];
      solve4x4(M, r);

      rhs1[i1] = r[0] ?? 0;
      rhs1[i] = r[1] ?? 0;
      rhs2[i1] = r[2] ?? 0;
      rhs2[i] = r[3] ?? 0;
      i -= 2;
    }
  }
}

/** Solve 4x4 system in-place via Gaussian elimination with partial pivoting. */
function solve4x4(M: number[], r: number[]): void {
  for (let col = 0; col < 4; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col * 4 + col] ?? 0);
    for (let row = col + 1; row < 4; row++) {
      const v = Math.abs(M[row * 4 + col] ?? 0);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal < 1e-30) {
      throw new DataValidationError(
        "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique"
      );
    }
    if (maxRow !== col) {
      for (let j = 0; j < 4; j++) {
        const tmp = M[col * 4 + j] ?? 0;
        M[col * 4 + j] = M[maxRow * 4 + j] ?? 0;
        M[maxRow * 4 + j] = tmp;
      }
      const tmp = r[col] ?? 0;
      r[col] = r[maxRow] ?? 0;
      r[maxRow] = tmp;
    }
    const pivot = M[col * 4 + col] ?? 0;
    for (let row = col + 1; row < 4; row++) {
      const factor = (M[row * 4 + col] ?? 0) / pivot;
      for (let j = col; j < 4; j++) {
        M[row * 4 + j] = (M[row * 4 + j] ?? 0) - factor * (M[col * 4 + j] ?? 0);
      }
      r[row] = (r[row] ?? 0) - factor * (r[col] ?? 0);
    }
  }
  // Back-substitution
  for (let i = 3; i >= 0; i--) {
    let s = r[i] ?? 0;
    for (let j = i + 1; j < 4; j++) {
      s -= (M[i * 4 + j] ?? 0) * (r[j] ?? 0);
    }
    const diag = M[i * 4 + i] ?? 0;
    if (Math.abs(diag) <= 1e-30)
      throw new DataValidationError(
        "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique"
      );
    r[i] = s / diag;
  }
}
