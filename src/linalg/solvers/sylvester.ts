/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";
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
 * The equation has a unique solution only when A and -B share no eigenvalue.
 * When they do, or come within rounding error of doing so (relative to the
 * largest entry of the Schur forms), a DataValidationError is thrown instead
 * of returning a meaningless huge solution.
 *
 * **Time Complexity**: O(M³ + N³ + M²N) where A is M×M and B is N×N
 *
 * @param a - Square matrix A of shape (M, M)
 * @param b - Square matrix B of shape (N, N)
 * @param c - Matrix C of shape (M, N)
 * @returns Solution matrix X of shape (M, N) (float64)
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
 *
 * @throws {ShapeError} If a, b or c is not 2-D, a or b is not square, or c is not (M, N)
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {DataValidationError} If an input contains NaN or Infinity, or if A and -B share an
 *   eigenvalue (the solution is not unique)
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

  if (m === 0 || n === 0) {
    return fromDenseMatrix2D(m, n, new Float64Array(m * n));
  }

  // Validate the inputs here so that errors name sylvester() rather than schur().
  toDenseMatrix2D(a, "sylvester()");
  toDenseMatrix2D(b, "sylvester()");

  // Schur decompositions
  const [Ta, U] = schur(a);
  const [Tb, V] = schur(b);

  const tA = toDenseMatrix2D(Ta).data;
  const tB = toDenseMatrix2D(Tb).data;
  const uMat = toDenseMatrix2D(U).data;
  const vMat = toDenseMatrix2D(V).data;
  const cMat = toDenseMatrix2D(c, "sylvester()").data;

  // Smallest pivot treated as non-zero: eps times the largest Schur entry, the
  // threshold LAPACK's trsyl uses to flag (nearly) common eigenvalues.
  let scale = 0;
  for (let i = 0; i < tA.length; i++) scale = Math.max(scale, Math.abs(tA[i] as number));
  for (let i = 0; i < tB.length; i++) scale = Math.max(scale, Math.abs(tB[i] as number));
  const smin = Number.EPSILON * scale;

  // F = U^T * C * V
  const F = matMul(matMul(transpose(uMat, m, m), cMat, m, m, n), vMat, m, n, n);

  // Solve Ta*Y + Y*Tb = F column by column (forward substitution)
  // For upper quasi-triangular Tb, (Y*Tb)[:,j] = sum_{k<=j} Y[:,k]*Tb[k,j]
  // So process columns left-to-right: columns k < j are already solved.
  const Y = new Float64Array(m * n);

  let j = 0;
  while (j < n) {
    // A non-zero subdiagonal entry at [j+1, j] marks a 2x2 block in Tb
    // (the Schur routine sets negligible subdiagonal entries to exactly zero).
    const is2x2 = j + 1 < n && (tB[(j + 1) * n + j] as number) !== 0;

    if (!is2x2) {
      // 1x1 block: solve (Ta + Tb[j,j]*I) * y_j = rhs_j
      const bjj = tB[j * n + j] as number;
      // Build RHS: F[:,j] - sum_{k<j} Y[:,k] * Tb[k,j]
      const rhs = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        let s = F[i * n + j] as number;
        for (let k = 0; k < j; k++) {
          s -= (Y[i * n + k] as number) * (tB[k * n + j] as number);
        }
        rhs[i] = s;
      }

      solveShiftedQuasiTriangular(tA, m, bjj, rhs, smin);

      for (let i = 0; i < m; i++) {
        Y[i * n + j] = rhs[i] as number;
      }
      j++;
    } else {
      // 2x2 block: columns j, j+1
      const j1 = j;
      const j2 = j + 1;
      const b11 = tB[j1 * n + j1] as number;
      const b12 = tB[j1 * n + j2] as number;
      const b21 = tB[j2 * n + j1] as number;
      const b22 = tB[j2 * n + j2] as number;

      const rhs1 = new Float64Array(m);
      const rhs2 = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        let s1 = F[i * n + j1] as number;
        let s2 = F[i * n + j2] as number;
        for (let k = 0; k < j1; k++) {
          const yik = Y[i * n + k] as number;
          s1 -= yik * (tB[k * n + j1] as number);
          s2 -= yik * (tB[k * n + j2] as number);
        }
        rhs1[i] = s1;
        rhs2[i] = s2;
      }

      solve2x2CoupledSystem(tA, m, b11, b12, b21, b22, rhs1, rhs2, smin);

      for (let i = 0; i < m; i++) {
        Y[i * n + j1] = rhs1[i] as number;
        Y[i * n + j2] = rhs2[i] as number;
      }
      j += 2;
    }
  }

  // X = U * Y * V^T
  const X = matMul(matMul(uMat, Y, m, m, n), transpose(vMat, n, n), m, n, n);

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
  const aMat = toDenseMatrix2D(a, "lyapunov()");
  const aT = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aT[i * n + j] = aMat.data[j * n + i] as number;
    }
  }

  const aTTensor = fromDenseMatrix2D(n, n, aT);
  return sylvester(a, aTTensor, q);
}

// ---- Internal helpers ----

const SINGULAR_MESSAGE =
  "sylvester(): equation is singular (A and -B share an eigenvalue); the solution is not unique";

/** C = A B for row-major A (r x k) and B (k x c); each entry sums over k in ascending order. */
function matMul(a: Float64Array, b: Float64Array, r: number, k: number, c: number): Float64Array {
  const out = new Float64Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p] as number;
      for (let j = 0; j < c; j++) {
        out[i * c + j] = (out[i * c + j] as number) + aip * (b[p * c + j] as number);
      }
    }
  }
  return out;
}

function transpose(a: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j * rows + i] = a[i * cols + j] as number;
    }
  }
  return out;
}

/**
 * Solve (T + sigma*I) * x = rhs where T is quasi-upper triangular (Schur form).
 * Back-substitution from bottom. Modifies rhs in-place.
 */
function solveShiftedQuasiTriangular(
  T: Float64Array,
  n: number,
  sigma: number,
  rhs: Float64Array,
  smin: number
): void {
  let i = n - 1;
  while (i >= 0) {
    // A non-zero subdiagonal entry T[i, i-1] marks a 2x2 block at rows i-1, i
    const is2x2 = i > 0 && (T[i * n + (i - 1)] as number) !== 0;

    if (!is2x2) {
      // 1x1 block
      let s = rhs[i] as number;
      for (let k = i + 1; k < n; k++) {
        s -= (T[i * n + k] as number) * (rhs[k] as number);
      }
      const diag = (T[i * n + i] as number) + sigma;
      if (Math.abs(diag) <= smin) throw new DataValidationError(SINGULAR_MESSAGE);
      rhs[i] = s / diag;
      i--;
    } else {
      const i1 = i - 1;
      let s1 = rhs[i1] as number;
      let s2 = rhs[i] as number;
      for (let k = i + 1; k < n; k++) {
        const rk = rhs[k] as number;
        s1 -= (T[i1 * n + k] as number) * rk;
        s2 -= (T[i * n + k] as number) * rk;
      }

      const a11 = (T[i1 * n + i1] as number) + sigma;
      const a12 = T[i1 * n + i] as number;
      const a21 = T[i * n + i1] as number;
      const a22 = (T[i * n + i] as number) + sigma;

      const det = a11 * a22 - a12 * a21;
      const magnitude = Math.abs(a11) + Math.abs(a12) + Math.abs(a21) + Math.abs(a22);
      if (Math.abs(det) <= smin * magnitude) throw new DataValidationError(SINGULAR_MESSAGE);
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
  rhs2: Float64Array,
  smin: number
): void {
  // Solve row by row from bottom using back-substitution
  let i = n - 1;
  while (i >= 0) {
    const is2x2 = i > 0 && (T[i * n + (i - 1)] as number) !== 0;

    if (!is2x2) {
      let s1 = rhs1[i] as number;
      let s2 = rhs2[i] as number;
      for (let k = i + 1; k < n; k++) {
        const tik = T[i * n + k] as number;
        s1 -= tik * (rhs1[k] as number);
        s2 -= tik * (rhs2[k] as number);
      }

      const tii = T[i * n + i] as number;
      // System: (tii + b11)*x1 + b21*x2 = s1
      //          b12*x1 + (tii + b22)*x2 = s2
      const a11 = tii + b11;
      const a22 = tii + b22;
      const det = a11 * a22 - b12 * b21;
      const magnitude = Math.abs(a11) + Math.abs(a22) + Math.abs(b12) + Math.abs(b21);
      if (Math.abs(det) <= smin * magnitude) throw new DataValidationError(SINGULAR_MESSAGE);
      rhs1[i] = (a22 * s1 - b21 * s2) / det;
      rhs2[i] = (a11 * s2 - b12 * s1) / det;
      i--;
    } else {
      // 2x2 block of Ta against a 2x2 block of Tb: a 4x4 coupled system.
      const i1 = i - 1;
      let s11 = rhs1[i1] as number;
      let s12 = rhs2[i1] as number;
      let s21 = rhs1[i] as number;
      let s22 = rhs2[i] as number;
      for (let k = i + 1; k < n; k++) {
        const ti1k = T[i1 * n + k] as number;
        const tik = T[i * n + k] as number;
        s11 -= ti1k * (rhs1[k] as number);
        s12 -= ti1k * (rhs2[k] as number);
        s21 -= tik * (rhs1[k] as number);
        s22 -= tik * (rhs2[k] as number);
      }

      const t11 = T[i1 * n + i1] as number;
      const t12 = T[i1 * n + i] as number;
      const t21 = T[i * n + i1] as number;
      const t22 = T[i * n + i] as number;

      // Build 4x4 system M*x = r with x = [rhs1[i1], rhs1[i], rhs2[i1], rhs2[i]]
      const M = new Float64Array([
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
      ]);
      const r = new Float64Array([s11, s21, s12, s22]);
      solve4x4(M, r, smin);

      rhs1[i1] = r[0] as number;
      rhs1[i] = r[1] as number;
      rhs2[i1] = r[2] as number;
      rhs2[i] = r[3] as number;
      i -= 2;
    }
  }
}

/** Solve 4x4 system in-place via Gaussian elimination with partial pivoting. */
function solve4x4(M: Float64Array, r: Float64Array, smin: number): void {
  for (let col = 0; col < 4; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col * 4 + col] as number);
    for (let row = col + 1; row < 4; row++) {
      const v = Math.abs(M[row * 4 + col] as number);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal <= smin) {
      throw new DataValidationError(SINGULAR_MESSAGE);
    }
    if (maxRow !== col) {
      for (let j = 0; j < 4; j++) {
        const tmp = M[col * 4 + j] as number;
        M[col * 4 + j] = M[maxRow * 4 + j] as number;
        M[maxRow * 4 + j] = tmp;
      }
      const tmp = r[col] as number;
      r[col] = r[maxRow] as number;
      r[maxRow] = tmp;
    }
    const pivot = M[col * 4 + col] as number;
    for (let row = col + 1; row < 4; row++) {
      const factor = (M[row * 4 + col] as number) / pivot;
      for (let j = col; j < 4; j++) {
        M[row * 4 + j] = (M[row * 4 + j] as number) - factor * (M[col * 4 + j] as number);
      }
      r[row] = (r[row] as number) - factor * (r[col] as number);
    }
  }
  // Back-substitution (every pivot was checked against smin above)
  for (let i = 3; i >= 0; i--) {
    let s = r[i] as number;
    for (let j = i + 1; j < 4; j++) {
      s -= (M[i * 4 + j] as number) * (r[j] as number);
    }
    r[i] = s / (M[i * 4 + i] as number);
  }
}
