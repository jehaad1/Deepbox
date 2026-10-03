/**
 * Matrix operations: matrix_power, kron, block_diag, the matrix functions
 * expm / logm / sqrtm, and constructors for special matrices.
 *
 * @module linalg/matrix_ops
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */

import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  getDevice,
  getDtype,
  InvalidParameterError,
  ShapeError,
} from "../core";
import { resolveLosslessDType } from "../core/utils/dtype_utils";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";
import { fromDenseMatrix2D, luFactorSquare, luSolveInPlace, toDenseVector1D } from "./_internal";
import { eig, eigh } from "./decomposition/index";
import { inv } from "./inverse";

// ---------------------------------------------------------------------------
// Dense helpers (row-major Float64Array, square n x n unless noted)
// ---------------------------------------------------------------------------

function assertRealNumeric(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} does not support string dtype`);
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${name} does not support complex dtype`);
  }
}

function assertSquare(A: Tensor, name: string): number {
  if (A.ndim !== 2) {
    throw new ShapeError(`${name} requires a 2-D matrix; got ${A.ndim}-D`);
  }
  const m = A.shape[0] ?? 0;
  if (m !== (A.shape[1] ?? 0)) {
    throw new ShapeError(`${name} requires a square matrix; got shape [${A.shape.join(", ")}]`);
  }
  return m;
}

/** Copy a 2-D tensor into a dense row-major Float64Array, honouring strides and offset. */
function readDense(A: Tensor): Float64Array {
  const rows = A.shape[0] ?? 0;
  const cols = A.shape[1] ?? 0;
  const s0 = A.strides[0] ?? cols;
  const s1 = A.strides[1] ?? 1;
  const data = A.data;
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const base = A.offset + i * s0;
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] = Number(data[base + j * s1]);
    }
  }
  return out;
}

function readFiniteDense(A: Tensor, name: string): Float64Array {
  const out = readDense(A);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i] as number)) {
      throw new DataValidationError(`${name} requires a matrix with only finite values`);
    }
  }
  return out;
}

function identityDense(n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

/** C = A * B for n x n matrices. Each C[i,j] sums over k in ascending order. */
function matMulDense(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const c = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const iRow = i * n;
    for (let k = 0; k < n; k++) {
      const aik = a[iRow + k] as number;
      const kRow = k * n;
      for (let j = 0; j < n; j++) {
        c[iRow + j] = (c[iRow + j] as number) + aik * (b[kRow + j] as number);
      }
    }
  }
  return c;
}

/** Maximum absolute column sum. */
function norm1Dense(a: Float64Array, n: number): number {
  let best = 0;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(a[i * n + j] as number);
    if (s > best || Number.isNaN(s)) best = s;
  }
  return best;
}

function normFroDense(a: Float64Array): number {
  let scale = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i] as number);
    if (v > scale) scale = v;
  }
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const v = (a[i] as number) / scale;
    sum += v * v;
  }
  return scale * Math.sqrt(sum);
}

/** Inverse via LU with partial pivoting. Throws DataValidationError on an exactly singular matrix. */
function invDense(a: Float64Array, n: number): Float64Array {
  const { lu, piv } = luFactorSquare(a, n);
  const rhs = identityDense(n);
  luSolveInPlace(lu, piv, n, rhs, n);
  return rhs;
}

/** a * 2^e, applied in two steps so that neither factor over- or underflows. */
function scaleByPow2(a: Float64Array, e: number): Float64Array {
  const half = Math.trunc(e / 2);
  const f1 = 2 ** half;
  const f2 = 2 ** (e - half);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] as number) * f1 * f2;
  return out;
}

/**
 * Even exponent k such that 2^-k * ||a||_F is close to one, or 0 when the
 * matrix is already well scaled. Used so the iterations below do not need
 * hundreds of steps just to bring a very large or very small matrix to unit size.
 */
function evenScaleExponent(a: Float64Array): number {
  const nrm = normFroDense(a);
  if (!(nrm > 0) || !Number.isFinite(nrm)) return 0;
  if (nrm >= 0.25 && nrm <= 4) return 0;
  return 2 * Math.round(Math.log2(nrm) / 2);
}

// ---------------------------------------------------------------------------
// matrix_power
// ---------------------------------------------------------------------------

/**
 * Raise a square matrix to an integer power.
 *
 * For positive n: computes A^n via repeated squaring.
 * For n=0: returns the identity matrix.
 * For negative n: computes (A^{-1})^{|n|}.
 *
 * The result is always float64, whatever the dtype of `A`.
 *
 * @param A - Square matrix of shape (m, m)
 * @param n - Integer exponent
 * @returns A^n of shape (m, m)
 *
 * @example
 * ```ts
 * import { matrixPower } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const A3 = matrixPower(A, 3); // A * A * A
 * ```
 *
 * @throws {ShapeError} If A is not a square 2-D matrix
 * @throws {InvalidParameterError} If n is not an integer
 * @throws {DTypeError} If A has string or complex dtype
 * @throws {DataValidationError} If n is negative and A is singular
 *
 * @deprecated Prefer {@link matrixPower}.
 */
export function matrix_power(A: Tensor, n: number): Tensor {
  if (A.ndim !== 2) {
    throw new ShapeError(`matrix_power requires 2D matrix; got ndim=${A.ndim}`);
  }
  const m = A.shape[0] ?? 0;
  if (m !== (A.shape[1] ?? 0)) {
    throw new ShapeError(`matrix_power requires square matrix; got shape [${A.shape.join(", ")}]`);
  }
  if (!Number.isInteger(n)) {
    throw new InvalidParameterError(`n must be an integer; received ${n}`, "n", n);
  }
  assertRealNumeric(A, "matrix_power");

  if (n === 0) {
    return fromDenseMatrix2D(m, m, identityDense(m));
  }

  let base: Float64Array;
  let exp: number;
  if (n < 0) {
    base = readDense(inv(A));
    exp = -n;
  } else {
    base = readDense(A);
    exp = n;
  }

  // Exponentiation by squaring. Intermediate arrays are never mutated, so
  // `result` may safely alias `current`.
  let result: Float64Array | undefined;
  let current = base;
  while (exp > 0) {
    if (exp % 2 === 1) {
      result = result ? matMulDense(result, current, m) : current;
    }
    exp = Math.floor(exp / 2);
    if (exp > 0) current = matMulDense(current, current, m);
  }

  return fromDenseMatrix2D(m, m, result ?? base);
}

/**
 * Raise a square matrix to an integer power.
 *
 * For positive n: computes A^n via repeated squaring.
 * For n=0: returns the identity matrix.
 * For negative n: computes (A^{-1})^{|n|}.
 *
 * The result is always float64, whatever the dtype of `A`.
 *
 * @param A - Square matrix of shape (m, m)
 * @param n - Integer exponent
 * @returns A^n of shape (m, m)
 *
 * @example
 * ```ts
 * import { matrixPower } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const A3 = matrixPower(A, 3); // A * A * A
 * ```
 *
 * @throws {ShapeError} If A is not a square 2-D matrix
 * @throws {InvalidParameterError} If n is not an integer
 * @throws {DTypeError} If A has string or complex dtype
 * @throws {DataValidationError} If n is negative and A is singular
 */
export const matrixPower = matrix_power;

// ---------------------------------------------------------------------------
// kron / block_diag
// ---------------------------------------------------------------------------

/**
 * Compute the Kronecker product of two matrices.
 *
 * If A is (m, n) and B is (p, q), the result is (m*p, n*q).
 * The result is always float64.
 *
 * @param A - First matrix of shape (m, n)
 * @param B - Second matrix of shape (p, q)
 * @returns Kronecker product of shape (m*p, n*q)
 *
 * @example
 * ```ts
 * import { kron } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const B = tensor([[0, 5], [6, 7]]);
 * const K = kron(A, B); // shape [4, 4]
 * ```
 *
 * @throws {ShapeError} If A or B is not 2-D
 * @throws {DTypeError} If A or B has string or complex dtype
 */
export function kron(A: Tensor, B: Tensor): Tensor {
  if (A.ndim !== 2) {
    throw new ShapeError(`kron requires 2D matrices; A has ndim=${A.ndim}`);
  }
  if (B.ndim !== 2) {
    throw new ShapeError(`kron requires 2D matrices; B has ndim=${B.ndim}`);
  }
  assertRealNumeric(A, "kron");
  assertRealNumeric(B, "kron");

  const m = A.shape[0] ?? 0;
  const n = A.shape[1] ?? 0;
  const p = B.shape[0] ?? 0;
  const q = B.shape[1] ?? 0;
  const rows = m * p;
  const cols = n * q;

  // Densify both operands once (honouring strides/offset), then run the
  // quadruple loop over monomorphic Float64Arrays into a typed output. The old
  // per-element Number() reads plus a number[] → tensor() re-validation
  // made this about 20x slower.
  const aDense = readDense(A);
  const bDense = readDense(B);

  const out = new Float64Array(rows * cols);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aVal = aDense[i * n + j] as number;
      for (let k = 0; k < p; k++) {
        const outBase = (i * p + k) * cols + j * q;
        const bBase = k * q;
        for (let l = 0; l < q; l++) {
          out[outBase + l] = aVal * (bDense[bBase + l] as number);
        }
      }
    }
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: [rows, cols],
    dtype: "float64",
    device: A.device,
  });
}

/**
 * Construct a block-diagonal matrix from the provided 2-D matrices.
 *
 * The blocks need not be square: for blocks of shape (r_i, c_i) the result
 * has shape (sum r_i, sum c_i) and zeros outside the blocks. With no
 * arguments an empty (0, 0) matrix is returned. The result is float64.
 *
 * @param matrices - One or more 2-D tensors to place along the diagonal
 * @returns Block-diagonal matrix
 *
 * @example
 * ```ts
 * import { blockDiag } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const B = tensor([[5]]);
 * const D = blockDiag(A, B); // shape [3, 3]
 * ```
 *
 * @throws {ShapeError} If any input is not 2-D
 * @throws {DTypeError} If any input has string or complex dtype
 *
 * @deprecated Prefer {@link blockDiag}.
 */
export function block_diag(...matrices: Tensor[]): Tensor {
  if (matrices.length === 0) {
    return fromDenseMatrix2D(0, 0, new Float64Array(0));
  }
  let totalRows = 0;
  let totalCols = 0;
  for (const M of matrices) {
    if (M.ndim !== 2) {
      throw new ShapeError(`block_diag requires 2D matrices; got ndim=${M.ndim}`);
    }
    assertRealNumeric(M, "block_diag");
    totalRows += M.shape[0] ?? 0;
    totalCols += M.shape[1] ?? 0;
  }

  const data = new Float64Array(totalRows * totalCols); // zero-filled
  let rowOff = 0;
  let colOff = 0;
  for (const M of matrices) {
    const mr = M.shape[0] ?? 0;
    const mc = M.shape[1] ?? 0;
    const dense = readDense(M);
    for (let i = 0; i < mr; i++) {
      const outBase = (rowOff + i) * totalCols + colOff;
      for (let j = 0; j < mc; j++) {
        data[outBase + j] = dense[i * mc + j] as number;
      }
    }
    rowOff += mr;
    colOff += mc;
  }

  return TensorClass.fromTypedArray({
    data,
    shape: [totalRows, totalCols],
    dtype: "float64",
    device: matrices[0]?.device ?? "cpu",
  });
}

/**
 * Construct a block-diagonal matrix from the provided 2-D matrices.
 *
 * The blocks need not be square: for blocks of shape (r_i, c_i) the result
 * has shape (sum r_i, sum c_i) and zeros outside the blocks. With no
 * arguments an empty (0, 0) matrix is returned. The result is float64.
 *
 * @param matrices - One or more 2-D tensors to place along the diagonal
 * @returns Block-diagonal matrix
 *
 * @example
 * ```ts
 * import { blockDiag } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const B = tensor([[5]]);
 * const D = blockDiag(A, B); // shape [3, 3]
 * ```
 *
 * @throws {ShapeError} If any input is not 2-D
 * @throws {DTypeError} If any input has string or complex dtype
 */
export const blockDiag = block_diag;

// ---------------------------------------------------------------------------
// Matrix functions: expm, sqrtm, logm
// ---------------------------------------------------------------------------

/**
 * Dense exp(A) by scaling and squaring with a degree-6 Padé approximant.
 * Well defined for defective matrices and matrices with complex eigenvalues.
 */
function expmDense(a: Float64Array, m: number): Float64Array {
  const norm1 = norm1Dense(a, m);
  // Scale so that ||A / 2^s||_1 <= 1/2.
  const s = norm1 > 0 ? Math.max(0, Math.ceil(Math.log2(norm1)) + 1) : 0;
  const invScale = 2 ** -s;

  const B = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) B[i] = (a[i] as number) * invScale;

  // Degree-6 Padé: N = sum c_k B^k, D = sum (-1)^k c_k B^k; expm ≈ D^{-1} N.
  // Coefficients c_k = (2q-k)! q! / ((2q)! k! (q-k)!) for q = 6.
  const cPade = [1, 0.5, 5 / 44, 1 / 66, 1 / 792, 1 / 15840, 1 / 665280];

  const N = new Float64Array(a.length);
  const D = new Float64Array(a.length);
  let power = identityDense(m);
  for (let k = 0; k <= 6; k++) {
    if (k > 0) power = matMulDense(power, B, m);
    const ck = cPade[k] as number;
    const sign = k % 2 === 0 ? 1 : -1;
    for (let i = 0; i < a.length; i++) {
      const term = ck * (power[i] as number);
      N[i] = (N[i] as number) + term;
      D[i] = (D[i] as number) + sign * term;
    }
  }

  // R = D^{-1} N, solved with one LU factorisation of D.
  const { lu, piv } = luFactorSquare(D, m);
  luSolveInPlace(lu, piv, m, N, m);
  let R: Float64Array = N;

  for (let i = 0; i < s; i++) R = matMulDense(R, R, m);
  return R;
}

/**
 * Compute the matrix exponential exp(A).
 *
 * Uses scaling and squaring with a degree-6 Padé approximant, so it also works
 * for defective matrices and matrices with complex eigenvalues (for example a
 * rotation generator). The result is float64.
 *
 * @param A - Square matrix
 * @returns exp(A)
 *
 * @example
 * ```ts
 * import { expm } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[0, 1], [-1, 0]]);
 * const E = expm(A); // [[cos 1, sin 1], [-sin 1, cos 1]]
 * ```
 *
 * @throws {ShapeError} If A is not a square 2-D matrix
 * @throws {DTypeError} If A has string or complex dtype
 * @throws {DataValidationError} If A contains NaN or Infinity
 */
export function expm(A: Tensor): Tensor {
  const m = assertSquare(A, "expm");
  assertRealNumeric(A, "expm");
  if (m === 0) return fromDenseMatrix2D(0, 0, new Float64Array(0));
  const a = readFiniteDense(A, "expm");
  return fromDenseMatrix2D(m, m, expmDense(a, m));
}

/**
 * True when |a[i,j] - a[j,i]| is within rounding of zero for every pair,
 * measured against the largest entry so that the test does not depend on the
 * absolute scale of the matrix.
 */
function isSymmetricDense(a: Float64Array, n: number): boolean {
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] as number));
  const tol = 1e-12 * maxAbs;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs((a[i * n + j] as number) - (a[j * n + i] as number)) > tol) return false;
    }
  }
  return true;
}

/**
 * f(A) = V diag(f(lambda)) V^T for a symmetric matrix. `fn` receives each
 * eigenvalue and returns f(lambda).
 */
function symmetricMatrixFunction(
  a: Float64Array,
  n: number,
  fn: (lambda: number, maxAbs: number) => number
): Float64Array {
  // Symmetrise so eigh() sees an exactly symmetric input.
  const sym = new Float64Array(a.length);
  for (let i = 0; i < n; i++) {
    sym[i * n + i] = a[i * n + i] as number;
    for (let j = i + 1; j < n; j++) {
      const v = 0.5 * ((a[i * n + j] as number) + (a[j * n + i] as number));
      sym[i * n + j] = v;
      sym[j * n + i] = v;
    }
  }
  const [vals, vecs] = eigh(fromDenseMatrix2D(n, n, sym));
  const lambda = toDenseVector1D(vals);
  const V = readDense(vecs);

  let maxAbs = 0;
  for (let k = 0; k < n; k++) maxAbs = Math.max(maxAbs, Math.abs(lambda[k] as number));
  const f = new Float64Array(n);
  for (let k = 0; k < n; k++) f[k] = fn(lambda[k] as number, maxAbs);

  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += (V[i * n + k] as number) * (f[k] as number) * (V[j * n + k] as number);
      }
      out[i * n + j] = sum;
      out[j * n + i] = sum;
    }
  }
  return out;
}

/** Relative Frobenius residual ||S*S - A|| / ||A||. */
function sqrtResidual(s: Float64Array, a: Float64Array, n: number): number {
  const sq = matMulDense(s, s, n);
  for (let i = 0; i < sq.length; i++) sq[i] = (sq[i] as number) - (a[i] as number);
  const na = normFroDense(a);
  return normFroDense(sq) / (na > 0 ? na : 1);
}

const SQRT_RESIDUAL_TOL = Math.sqrt(Number.EPSILON);

/**
 * Denman-Beavers iteration for the principal square root. Returns null when
 * the iteration breaks down (singular iterate) or does not reach a root with a
 * small residual, which happens for matrices with eigenvalues on the closed
 * negative real axis.
 */
function sqrtmDenmanBeavers(a: Float64Array, n: number): Float64Array | null {
  let Y = new Float64Array(a);
  let Z = identityDense(n);
  const tol = 4 * n * Number.EPSILON;
  let prevDiff = Number.POSITIVE_INFINITY;
  for (let iter = 0; iter < 100; iter++) {
    let Yinv: Float64Array;
    let Zinv: Float64Array;
    try {
      Yinv = invDense(Y, n);
      Zinv = invDense(Z, n);
    } catch (err) {
      if (err instanceof DeepboxError) return null;
      throw err;
    }
    const Ynew = new Float64Array(Y.length);
    const Znew = new Float64Array(Z.length);
    const diff = new Float64Array(Y.length);
    for (let i = 0; i < Y.length; i++) {
      const y = 0.5 * ((Y[i] as number) + (Zinv[i] as number));
      Ynew[i] = y;
      Znew[i] = 0.5 * ((Z[i] as number) + (Yinv[i] as number));
      diff[i] = y - (Y[i] as number);
    }
    Y = Ynew;
    Z = Znew;
    const nY = normFroDense(Y);
    if (!Number.isFinite(nY)) return null;
    const d = normFroDense(diff);
    if (d <= tol * nY) break;
    // Rounding noise keeps the update from reaching `tol`: stop once it stalls.
    if (iter >= 3 && d >= prevDiff && d <= 1e-8 * nY) break;
    prevDiff = d;
  }
  return sqrtResidual(Y, a, n) <= SQRT_RESIDUAL_TOL ? Y : null;
}

/**
 * Principal square root through a general eigendecomposition. Handles
 * diagonalisable matrices with real, non-negative eigenvalues (including
 * singular ones, which the Denman-Beavers iteration cannot).
 */
function sqrtmViaEig(a: Float64Array, n: number): Float64Array {
  const fail = (reason: string, cause?: unknown): never => {
    throw new DataValidationError(
      `sqrtm: could not compute a real principal square root (${reason})`,
      cause === undefined ? undefined : { cause }
    );
  };
  let vals: Float64Array;
  let V: Float64Array;
  let Vinv: Float64Array;
  try {
    const [evals, evecs] = eig(fromDenseMatrix2D(n, n, new Float64Array(a)));
    vals = toDenseVector1D(evals);
    V = readDense(evecs);
    Vinv = invDense(new Float64Array(V), n);
  } catch (err) {
    if (err instanceof DeepboxError) {
      return fail("complex or defective spectrum with a non-positive eigenvalue", err);
    }
    throw err;
  }

  let maxAbs = 0;
  for (let k = 0; k < n; k++) maxAbs = Math.max(maxAbs, Math.abs(vals[k] as number));
  const tol = n * Number.EPSILON * maxAbs;
  const f = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const lam = vals[k] as number;
    if (lam < -tol) return fail(`eigenvalue ${lam} is negative`);
    f[k] = Math.sqrt(Math.max(lam, 0));
  }

  const scaled = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      scaled[i * n + k] = (V[i * n + k] as number) * (f[k] as number);
    }
  }
  const S = matMulDense(scaled, Vinv, n);
  if (!(sqrtResidual(S, a, n) <= SQRT_RESIDUAL_TOL)) {
    return fail("the matrix is defective");
  }
  return S;
}

function sqrtmDense(a: Float64Array, n: number): Float64Array {
  if (isSymmetricDense(a, n)) {
    return symmetricMatrixFunction(a, n, (lam, maxAbs) => {
      // Eigenvalues within rounding of zero (a PSD matrix) are clamped to zero.
      if (lam < -n * Number.EPSILON * maxAbs) {
        throw new DataValidationError(
          `sqrtm: matrix has a negative eigenvalue (${lam}); the principal square root is not real`
        );
      }
      return Math.sqrt(Math.max(lam, 0));
    });
  }
  // sqrt(2^k A') = 2^(k/2) sqrt(A'), so the iteration runs on a unit-size matrix.
  const k = evenScaleExponent(a);
  const root = sqrtmDenmanBeavers(k === 0 ? a : scaleByPow2(a, -k), n);
  if (root !== null) return k === 0 ? root : scaleByPow2(root, k / 2);
  return sqrtmViaEig(a, n);
}

/**
 * Compute the principal matrix square root sqrt(A), the solution S of
 * S * S = A whose eigenvalues have non-negative real part.
 *
 * - Symmetric input is handled through the symmetric eigendecomposition.
 *   Eigenvalues that are negative only within rounding error (a positive
 *   semi-definite matrix) are treated as zero.
 * - Other input uses the Denman-Beavers iteration, so defective matrices and
 *   matrices with complex eigenvalues work as long as the result is real.
 *   Diagonalisable singular matrices fall back to the eigendecomposition.
 *
 * The result is float64.
 *
 * @param A - Square matrix with no eigenvalues on the negative real axis
 * @returns sqrt(A)
 *
 * @example
 * ```ts
 * import { sqrtm } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const S = sqrtm(tensor([[4, 0], [0, 9]])); // [[2, 0], [0, 3]]
 * ```
 *
 * @throws {ShapeError} If A is not a square 2-D matrix
 * @throws {DTypeError} If A has string or complex dtype
 * @throws {DataValidationError} If A contains NaN or Infinity, or has no real principal
 *   square root (for example a negative eigenvalue)
 */
export function sqrtm(A: Tensor): Tensor {
  const n = assertSquare(A, "sqrtm");
  assertRealNumeric(A, "sqrtm");
  if (n === 0) return fromDenseMatrix2D(0, 0, new Float64Array(0));
  const a = readFiniteDense(A, "sqrtm");
  return fromDenseMatrix2D(n, n, sqrtmDense(a, n));
}

/** log(I + E) for ||E||_1 small, by its Taylor series. */
function logSeries(E: Float64Array, n: number): Float64Array {
  const out = new Float64Array(E);
  let power = E;
  for (let j = 2; j <= 200; j++) {
    power = matMulDense(power, E, n);
    const sign = j % 2 === 0 ? -1 : 1;
    let termNorm = 0;
    for (let i = 0; i < out.length; i++) {
      const t = (sign * (power[i] as number)) / j;
      out[i] = (out[i] as number) + t;
      termNorm += Math.abs(t);
    }
    if (termNorm <= Number.EPSILON * normFroDense(out)) break;
  }
  return out;
}

function logmDense(a: Float64Array, n: number): Float64Array {
  if (isSymmetricDense(a, n)) {
    return symmetricMatrixFunction(a, n, (lam) => {
      if (!(lam > 0)) {
        throw new DataValidationError(
          `logm: matrix has a non-positive eigenvalue (${lam}); the principal logarithm is not real`
        );
      }
      return Math.log(lam);
    });
  }

  // Inverse scaling and squaring: take square roots until X is close to the
  // identity, use the log series there, and undo the roots with a factor 2^k.
  const fail = (): never => {
    throw new DataValidationError(
      "logm: could not compute a real principal logarithm; " +
        "eigenvalues must not lie on the closed negative real axis"
    );
  };
  // log(2^j A') = j ln(2) I + log(A') for the principal logarithm.
  const shift = evenScaleExponent(a);
  let X: Float64Array = shift === 0 ? new Float64Array(a) : scaleByPow2(a, -shift);
  let k = 0;
  const E = new Float64Array(a.length);
  for (;;) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        E[i * n + j] = (X[i * n + j] as number) - (i === j ? 1 : 0);
      }
    }
    if (norm1Dense(E, n) <= 0.25) break;
    if (k >= 60) return fail();
    const root = sqrtmDenmanBeavers(X, n);
    if (root === null) return fail();
    X = root;
    k++;
  }
  const L = logSeries(E, n);
  const factor = 2 ** k;
  for (let i = 0; i < L.length; i++) L[i] = (L[i] as number) * factor;
  if (shift !== 0) {
    for (let i = 0; i < n; i++) L[i * n + i] = (L[i * n + i] as number) + shift * Math.LN2;
  }
  return L;
}

/**
 * Compute the principal matrix logarithm log(A), the solution L of
 * expm(L) = A whose eigenvalues have imaginary part in (-pi, pi).
 *
 * Symmetric input is handled through the symmetric eigendecomposition and
 * requires strictly positive eigenvalues. Other input uses inverse scaling and
 * squaring, so defective matrices and matrices with complex eigenvalues work
 * as long as no eigenvalue lies on the closed negative real axis. The result
 * is float64.
 *
 * @param A - Square matrix with no eigenvalues on the closed negative real axis
 * @returns log(A)
 *
 * @example
 * ```ts
 * import { logm, expm } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[2, 1], [1, 2]]);
 * const L = logm(A);
 * // expm(L) is A up to rounding
 * ```
 *
 * @throws {ShapeError} If A is not a square 2-D matrix
 * @throws {DTypeError} If A has string or complex dtype
 * @throws {DataValidationError} If A contains NaN or Infinity, or has no real principal
 *   logarithm (a zero, negative or negative-real-axis eigenvalue)
 */
export function logm(A: Tensor): Tensor {
  const n = assertSquare(A, "logm");
  assertRealNumeric(A, "logm");
  if (n === 0) return fromDenseMatrix2D(0, 0, new Float64Array(0));
  const a = readFiniteDense(A, "logm");
  return fromDenseMatrix2D(n, n, logmDense(a, n));
}

// ---------------------------------------------------------------------------
// Special matrix constructors
// ---------------------------------------------------------------------------

/**
 * Wrap a row-major Float64Array as a tensor in the configured default dtype
 * and device. Float dtypes are filled directly. Other default dtypes are used only when
 * every value fits them exactly; otherwise the result is `float32`, so values are never
 * truncated (for example a Hilbert matrix under an integer default).
 */
function constructorResult(data: Float64Array, rows: number, cols: number): Tensor {
  const dtype = resolveLosslessDType(getDtype(), data);
  if (dtype === "float64") {
    return TensorClass.fromTypedArray({
      data,
      shape: [rows, cols],
      dtype,
      device: getDevice(),
    });
  }
  if (dtype === "float32") {
    return TensorClass.fromTypedArray({
      data: Float32Array.from(data),
      shape: [rows, cols],
      dtype,
      device: getDevice(),
    });
  }
  const nested: number[][] = [];
  for (let i = 0; i < rows; i++) {
    nested.push(Array.from(data.subarray(i * cols, (i + 1) * cols)));
  }
  return tensor(nested, { dtype, device: getDevice() });
}

/**
 * Create a Hilbert matrix of size n.
 *
 * H[i,j] = 1 / (i + j + 1)
 *
 * @param n - Size of the matrix
 * @returns Hilbert matrix of shape (n, n)
 *
 * @throws {InvalidParameterError} If n is not a positive integer
 */
export function hilbert(n: number): Tensor {
  if (!Number.isInteger(n) || n <= 0) {
    throw new InvalidParameterError("hilbert: n must be a positive integer", "n", n);
  }
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data[i * n + j] = 1 / (i + j + 1);
    }
  }
  return constructorResult(data, n, n);
}

/**
 * Create a Toeplitz matrix from first column and optional first row.
 *
 * A Toeplitz matrix has constant diagonals. If only c is given, the
 * result is symmetric. If both c and r are given, c defines the first
 * column and r defines the first row (r[0] is ignored in favor of c[0]).
 *
 * @param c - First column
 * @param r - First row (optional; defaults to c for symmetric Toeplitz)
 * @returns Toeplitz matrix of shape (c.length, r.length)
 *
 * @example
 * ```ts
 * import { toeplitz } from 'deepbox/linalg';
 *
 * toeplitz([1, 2, 3]);              // [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
 * toeplitz([1, 2, 3], [1, 5, 6]);   // [[1, 5, 6], [2, 1, 5], [3, 2, 1]]
 * ```
 *
 * @throws {InvalidParameterError} If c is empty
 */
export function toeplitz(c: number[], r?: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("toeplitz: c must have at least one element", "c", c);
  }
  const row = r ?? c;
  const n = c.length;
  const m = row.length;
  // Row i is the window buf[n-1-i .. n-1-i+m) of the combined sequence
  // [reversed c[1..], row], so each row is one memcpy instead of a
  // per-element diagonal branch. The diagonal always comes from c[0].
  const buf = new Float64Array(n - 1 + m);
  for (let i = 1; i < n; i++) buf[n - 1 - i] = c[i] ?? 0;
  for (let j = 0; j < m; j++) buf[n - 1 + j] = (j === 0 ? c[0] : row[j]) ?? 0;
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    out.set(buf.subarray(n - 1 - i, n - 1 - i + m), i * m);
  }
  return constructorResult(out, n, m);
}

/**
 * Create a Vandermonde matrix.
 *
 * V[i,j] = x[i]^(N-1-j) by default, or V[i,j] = x[i]^j when `increasing`
 * is true (same convention as `numpy.vander`).
 *
 * @param x - Input vector of length n
 * @param N - Number of columns (default: n)
 * @param increasing - If true, powers increase left to right
 * @returns Vandermonde matrix of shape (n, N)
 *
 * @example
 * ```ts
 * import { vandermonde } from 'deepbox/linalg';
 *
 * vandermonde([1, 2, 3]);              // [[1, 1, 1], [4, 2, 1], [9, 3, 1]]
 * vandermonde([1, 2, 3], 3, true);     // [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
 * ```
 *
 * @throws {InvalidParameterError} If x is empty or N is not a positive integer
 */
export function vandermonde(x: number[], N?: number, increasing?: boolean): Tensor {
  if (x.length === 0) {
    throw new InvalidParameterError("vandermonde: x must have at least one element", "x", x);
  }
  const n = x.length;
  const cols = N ?? n;
  if (!Number.isInteger(cols) || cols <= 0) {
    throw new InvalidParameterError("vandermonde: N must be a positive integer", "N", cols);
  }
  // Powers are built by repeated multiplication (as numpy does), which is
  // cheaper than pow() and exact for small integer inputs.
  const data = new Float64Array(n * cols);
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? 0;
    const base = i * cols;
    if (increasing) {
      data[base] = 1;
      for (let j = 1; j < cols; j++) data[base + j] = (data[base + j - 1] as number) * xi;
    } else {
      data[base + cols - 1] = 1;
      for (let j = cols - 2; j >= 0; j--) data[base + j] = (data[base + j + 1] as number) * xi;
    }
  }
  return constructorResult(data, n, cols);
}

/**
 * Create a Hadamard matrix of order n.
 *
 * n must be a power of 2. Uses Sylvester's construction, so
 * H[i,j] = (-1)^popcount(i & j).
 *
 * @param n - Order of the matrix (must be power of 2)
 * @returns Hadamard matrix of shape (n, n) with entries +1/-1
 *
 * @throws {InvalidParameterError} If n is not a positive power of 2, or is too large to allocate
 */
export function hadamard(n: number): Tensor {
  let isPow2 = Number.isInteger(n) && n > 0;
  if (isPow2) {
    let p = 1;
    while (p < n) p *= 2;
    isPow2 = p === n;
  }
  if (!isPow2) {
    throw new InvalidParameterError("hadamard: n must be a positive power of 2", "n", n);
  }
  if (n * n > 2 ** 28) {
    throw new InvalidParameterError("hadamard: n is too large to allocate", "n", n);
  }
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      // Parity of the number of set bits in i & j.
      let bits = i & j;
      bits ^= bits >>> 16;
      bits ^= bits >>> 8;
      bits ^= bits >>> 4;
      bits ^= bits >>> 2;
      bits ^= bits >>> 1;
      data[i * n + j] = (bits & 1) === 0 ? 1 : -1;
    }
  }
  return constructorResult(data, n, n);
}

/**
 * Create a companion matrix from polynomial coefficients.
 *
 * The companion matrix of the polynomial
 *   p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n]
 * is the n×n matrix with -c[1..n]/c[0] in the first row and ones on the
 * sub-diagonal (the convention of `scipy.linalg.companion`). Its eigenvalues
 * are the roots of p.
 *
 * @param c - Polynomial coefficients (leading coefficient first), length >= 2
 * @returns Companion matrix of shape (c.length - 1, c.length - 1)
 *
 * @example
 * ```ts
 * import { companion } from 'deepbox/linalg';
 *
 * companion([1, -10, 31, -30]); // [[10, -31, 30], [1, 0, 0], [0, 1, 0]]
 * ```
 *
 * @throws {InvalidParameterError} If c has fewer than 2 elements or c[0] is zero or not finite
 */
export function companion(c: number[]): Tensor {
  if (c.length < 2) {
    throw new InvalidParameterError("companion: c must have at least 2 elements", "c", c);
  }
  const leading = c[0] ?? 1;
  if (leading === 0 || !Number.isFinite(leading)) {
    throw new InvalidParameterError(
      "companion: leading coefficient must be non-zero and finite",
      "c[0]",
      leading
    );
  }
  const n = c.length - 1;
  const data = new Float64Array(n * n);
  // First row: -c[1..n] / c[0]
  for (let j = 0; j < n; j++) {
    data[j] = -(c[j + 1] ?? 0) / leading;
  }
  // Remaining rows: ones on the sub-diagonal
  for (let i = 1; i < n; i++) {
    data[i * n + (i - 1)] = 1;
  }
  return constructorResult(data, n, n);
}

/**
 * Create a circulant matrix from a vector.
 *
 * C[i,j] = c[(i - j) mod n], so c is the first column and every column is a
 * cyclic shift of the previous one.
 *
 * @param c - First column of the circulant matrix
 * @returns Circulant matrix of shape (n, n)
 *
 * @throws {InvalidParameterError} If c is empty
 */
export function circulant(c: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("circulant: c must have at least one element", "c", c);
  }
  const n = c.length;
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data[i * n + j] = c[(i - j + n) % n] ?? 0;
    }
  }
  return constructorResult(data, n, n);
}

/**
 * Create a Hankel matrix from first column and optional last row.
 *
 * H[i,j] = c[i+j] for i+j < len(c), otherwise r[i+j-len(c)+1]. The first
 * element of r is ignored (the last element of c is used instead), as in
 * `scipy.linalg.hankel`.
 *
 * @param c - First column
 * @param r - Last row (optional; defaults to zeros of length c.length)
 * @returns Hankel matrix of shape (c.length, r.length)
 *
 * @example
 * ```ts
 * import { hankel } from 'deepbox/linalg';
 *
 * hankel([1, 2, 3]);                // [[1, 2, 3], [2, 3, 0], [3, 0, 0]]
 * hankel([1, 2, 3], [3, 4, 5]);     // [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
 * ```
 *
 * @throws {InvalidParameterError} If c is empty
 */
export function hankel(c: number[], r?: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("hankel: c must have at least one element", "c", c);
  }
  const n = c.length;
  const m = r ? r.length : n;
  // Anti-diagonal values: c followed by r[1..] (or zeros when r is omitted).
  const vals = new Float64Array(n + Math.max(m, 1) - 1);
  for (let i = 0; i < n; i++) vals[i] = c[i] ?? 0;
  if (r) {
    for (let i = 1; i < r.length; i++) vals[n + i - 1] = r[i] ?? 0;
  }
  const data = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      data[i * m + j] = vals[i + j] as number;
    }
  }
  return constructorResult(data, n, m);
}
