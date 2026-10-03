import {
  type Axis,
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import {
  fromDenseVector1D,
  getDim,
  getStride,
  luFactorSquare,
  toDenseMatrix2D,
  toDenseVector1D,
} from "./_internal";
import { svdvals } from "./decomposition/svd";

/** True for the error `luFactorSquare` raises on an exactly zero pivot. */
function isSingularError(err: unknown): boolean {
  return err instanceof DataValidationError && /singular/i.test(err.message);
}

/** True for the error `luFactorSquare` raises when elimination overflows. */
function isOverflowError(err: unknown): boolean {
  return err instanceof DataValidationError && /overflowed/i.test(err.message);
}

/**
 * Scale the n x n matrix A by 2^-e so that LU elimination cannot overflow, and return the
 * scaled copy with e. With partial pivoting no entry grows by more than 2^(n - 1), so the
 * largest entry is brought down to about 2^(1020 - n) and not further: a larger shift would
 * push small entries into the subnormal range and cost them digits. Powers of two keep the
 * scaling exact.
 */
function scaleToAvoidOverflow(
  A: Float64Array,
  n: number
): { readonly scaled: Float64Array; readonly e: number } {
  let maxAbs = 0;
  for (let i = 0; i < A.length; i++) maxAbs = Math.max(maxAbs, Math.abs(A[i] as number));
  const e = Math.max(0, frexp(maxAbs)[1] - Math.max(0, 1020 - n));
  const scaled = new Float64Array(A.length);
  for (let i = 0; i < A.length; i++) scaled[i] = ldexp(A[i] as number, -e);
  return { scaled, e };
}

/** Smallest positive normal float64. */
const MIN_NORMAL = 2.2250738585072014e-308;

/** Split a finite non-zero x into m * 2^e with 0.5 <= |m| < 1. */
function frexp(x: number): [number, number] {
  if (x === 0 || !Number.isFinite(x)) return [x, 0];
  let e = Math.max(-1022, Math.floor(Math.log2(Math.abs(x))) + 1);
  let m = x * 2 ** -e;
  while (Math.abs(m) < 0.5) {
    m *= 2;
    e--;
  }
  while (Math.abs(m) >= 1) {
    m /= 2;
    e++;
  }
  return [m, e];
}

/** m * 2^e without the intermediate 2^e overflowing or underflowing too early. */
function ldexp(m: number, e: number): number {
  const half = Math.trunc(e / 2);
  return m * 2 ** half * 2 ** (e - half);
}

/**
 * Machine epsilon of the working precision implied by the dtypes of the given
 * tensors, used for default tolerances the way NumPy uses `finfo(dtype).eps`:
 * float32 gives 2^-23, float16 2^-10, bfloat16 2^-7, and float64, integer and
 * bool tensors float64 epsilon. With several tensors the most precise dtype
 * wins, as in NumPy's type promotion.
 */
function workingEpsilon(...tensors: readonly Tensor[]): number {
  let eps = 0;
  for (const t of tensors) {
    let e: number;
    switch (t.dtype) {
      case "float32":
      case "complex64":
        e = 2 ** -23;
        break;
      case "float16":
        e = 2 ** -10;
        break;
      case "bfloat16":
        e = 2 ** -7;
        break;
      default:
        e = Number.EPSILON;
    }
    // Promotion to the more precise type means the smaller epsilon wins.
    eps = eps === 0 ? e : Math.min(eps, e);
  }
  return eps === 0 ? Number.EPSILON : eps;
}

/**
 * Compute the determinant of a matrix.
 *
 * Uses LU decomposition with partial pivoting for numerical stability.
 * The determinant is computed as: det(A) = pivSign * product(diag(U))
 *
 * **Algorithm**: LU decomposition
 * **Time Complexity**: O(N³) for N×N matrix
 * **Space Complexity**: O(N²) for LU factorization
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 *
 * **Returns**: Determinant value (scalar)
 *
 * **Properties**:
 * - det(A) = 0 if and only if A is singular (non-invertible)
 * - det(AB) = det(A) * det(B)
 * - det(A^T) = det(A)
 * - det(cA) = c^N * det(A) for scalar c and N×N matrix A
 * - det(I) = 1 for identity matrix
 *
 * @example
 * ```ts
 * import { det } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * console.log(det(A));  // -2
 * ```
 *
 * @throws {ShapeError} If input is not a 2D square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */
export function det(a: Tensor): number {
  if (a.ndim !== 2) throw new ShapeError("det requires a 2-D matrix");
  const rows = getDim(a, 0, "det()");
  const cols = getDim(a, 1, "det()");
  if (rows !== cols) throw new ShapeError("det requires a square matrix");

  const n = rows;
  if (n === 0) return 1;

  const { data: A } = toDenseMatrix2D(a, "det()");

  try {
    const { lu, pivSign } = luFactorSquare(A, n);
    let detVal = pivSign;
    let inRange = true;
    for (let i = 0; i < n; i++) {
      detVal *= lu[i * n + i] as number;
      // A partial product that is subnormal has already lost digits, and one
      // that is infinite or zero has lost the value, even if later factors
      // would bring it back into range.
      const mag = Math.abs(detVal);
      if (!(mag >= MIN_NORMAL && mag <= Number.MAX_VALUE)) inRange = false;
    }
    if (inRange) return detVal;
    // A partial product left the normal float64 range. Redo the product with
    // the powers of two kept in a separate exponent, so that a determinant
    // inside the float64 range survives transient overflow or underflow (e.g.
    // diag(1e200, 1e200, 1e-200, 1e-200) is 1, not Infinity).
    let mantissa = pivSign;
    let exponent = 0;
    for (let i = 0; i < n; i++) {
      const [m, e] = frexp(lu[i * n + i] as number);
      mantissa *= m;
      exponent += e;
      if (Math.abs(mantissa) < 2 ** -500) {
        const [m2, e2] = frexp(mantissa);
        mantissa = m2;
        exponent += e2;
      }
    }
    return ldexp(mantissa, exponent);
  } catch (err) {
    if (isSingularError(err)) return 0;
    if (!isOverflowError(err)) throw err;
    // Elimination overflowed although the input is finite (for example
    // [[1e308, -1e308], [1e308, 1e308]]). det(A) = det(A / 2^e) * 2^(e*n), so
    // factor the rescaled matrix and apply the power of two at the end. The
    // result is +/-Infinity when the determinant itself exceeds the float64 range.
    const { scaled, e } = scaleToAvoidOverflow(A, n);
    let factors: { readonly lu: Float64Array; readonly pivSign: number };
    try {
      factors = luFactorSquare(scaled, n);
    } catch (err2) {
      // The rescaled matrix can underflow to an exactly singular one.
      if (isSingularError(err2)) return 0;
      throw err2;
    }
    const { lu, pivSign } = factors;
    let mantissa = pivSign;
    let exponent = e * n;
    for (let i = 0; i < n; i++) {
      const [m, ex] = frexp(lu[i * n + i] as number);
      mantissa *= m;
      exponent += ex;
      if (Math.abs(mantissa) < 2 ** -500) {
        const [m2, e2] = frexp(mantissa);
        mantissa = m2;
        exponent += e2;
      }
    }
    return ldexp(mantissa, exponent);
  }
}

/**
 * Compute sign and natural logarithm of the determinant.
 *
 * More numerically stable than det() for large matrices or matrices with
 * very large/small determinants that would overflow/underflow.
 *
 * **Algorithm**: LU decomposition
 * **Time Complexity**: O(N³)
 * **Space Complexity**: O(N²)
 *
 * **Parameters**:
 * @param a - Square matrix
 *
 * **Returns**: [sign, logdet], both 0-D float64 tensors
 * - sign: +1, -1, or 0 for a singular matrix
 * - logdet: Natural log of |det(A)|, -Infinity if singular (an empty matrix gives [1, 0])
 *
 * **Mathematical Relation**:
 * det(A) = sign * exp(logdet)
 *
 * **Advantages over det()**:
 * - Avoids overflow for large determinants
 * - Avoids underflow for small determinants
 * - More numerically stable for ill-conditioned matrices
 *
 * @example
 * ```ts
 * import { slogdet } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const [sign, logdet] = slogdet(A);
 * // det(A) = sign * exp(logdet)
 * ```
 *
 * @throws {ShapeError} If input is not a 2D square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function slogdet(a: Tensor): [Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("slogdet requires a 2-D matrix");
  const rows = getDim(a, 0, "slogdet()");
  const cols = getDim(a, 1, "slogdet()");
  if (rows !== cols) throw new ShapeError("slogdet requires a square matrix");

  const n = rows;
  if (n === 0) return [tensor(1, { dtype: "float64" }), tensor(0, { dtype: "float64" })];

  const { data: A } = toDenseMatrix2D(a, "slogdet()");

  try {
    const { lu, pivSign } = luFactorSquare(A, n);
    let sign = pivSign;
    let logAbsDet = 0;

    for (let i = 0; i < n; i++) {
      // luFactorSquare throws on a zero pivot, so every diagonal entry is non-zero here.
      const d = lu[i * n + i] as number;
      sign *= Math.sign(d);
      logAbsDet += Math.log(Math.abs(d));
    }

    return [tensor(sign, { dtype: "float64" }), tensor(logAbsDet, { dtype: "float64" })];
  } catch (err) {
    // An exactly singular matrix has sign 0 and log-determinant -Infinity.
    if (isSingularError(err)) {
      return [tensor(0, { dtype: "float64" }), tensor(-Infinity, { dtype: "float64" })];
    }
    if (!isOverflowError(err)) throw err;
    // Elimination overflowed although the input is finite. Factor A / 2^e
    // instead: log|det(A)| = log|det(A / 2^e)| + e * n * ln(2).
    const { scaled, e } = scaleToAvoidOverflow(A, n);
    let factors: { readonly lu: Float64Array; readonly pivSign: number };
    try {
      factors = luFactorSquare(scaled, n);
    } catch (err2) {
      // The rescaled matrix can underflow to an exactly singular one.
      if (isSingularError(err2)) {
        return [tensor(0, { dtype: "float64" }), tensor(-Infinity, { dtype: "float64" })];
      }
      throw err2;
    }
    const { lu, pivSign } = factors;
    let sign = pivSign;
    let logAbsDet = e * n * Math.LN2;
    for (let i = 0; i < n; i++) {
      const d = lu[i * n + i] as number;
      sign *= Math.sign(d);
      logAbsDet += Math.log(Math.abs(d));
    }
    return [tensor(sign, { dtype: "float64" }), tensor(logAbsDet, { dtype: "float64" })];
  }
}

/**
 * Compute the trace of a matrix.
 *
 * Sum of diagonal elements. Supports offset diagonals.
 *
 * **Algorithm**: Direct summation
 * **Time Complexity**: O(min(M, N)) where M×N is matrix size
 * **Space Complexity**: O(1)
 *
 * **Parameters**:
 * @param a - Input matrix (at least 2D)
 * @param offset - Integer offset from main diagonal:
 *   - 0: main diagonal (default)
 *   - >0: upper diagonal (k-th diagonal above main)
 *   - <0: lower diagonal (k-th diagonal below main)
 * @param axis1 - First axis to take the diagonal from (default: 0)
 * @param axis2 - Second axis to take the diagonal from (default: 1)
 *
 * **Returns**: Trace values as a float64 tensor: shape `[1]` for a 2-D input, otherwise one
 * value per remaining index (the input shape without `axis1` and `axis2`)
 *
 * **Properties**:
 * - trace(A) = sum of eigenvalues (for square matrices)
 * - trace(AB) = trace(BA) (cyclic property)
 * - trace(A + B) = trace(A) + trace(B) (linearity)
 * - trace(cA) = c * trace(A) for scalar c
 * - trace(A^T) = trace(A)
 *
 * NaN and Infinity entries on the summed diagonal propagate into the result.
 *
 * @example
 * ```ts
 * import { trace } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * console.log(trace(A).toArray());  // [15] (1 + 5 + 9)
 * console.log(trace(A, 1).toArray());  // [8] (2 + 6, upper diagonal)
 * console.log(trace(A, -1).toArray());  // [12] (4 + 8, lower diagonal)
 * ```
 *
 * @throws {ShapeError} If input is not at least 2D
 * @throws {InvalidParameterError} If axis values are invalid/identical or offset is non-integer
 * @throws {DTypeError} If input has string or complex dtype
 */
export function trace(a: Tensor, offset = 0, axis1: Axis = 0, axis2: Axis = 1): Tensor {
  if (a.ndim < 2) {
    throw new ShapeError("Input must be at least 2D");
  }

  if (!Number.isInteger(offset)) {
    throw new InvalidParameterError("offset must be an integer", "offset", offset);
  }

  if (a.dtype === "string") {
    throw new DTypeError("trace() does not support string dtype");
  }
  if (a.dtype === "complex64" || a.dtype === "complex128") {
    throw new DTypeError("trace() does not support complex dtype");
  }

  const ndim = a.ndim;

  const ax1 = normalizeAxis(axis1, ndim);
  const ax2 = normalizeAxis(axis2, ndim);
  if (ax1 === ax2) {
    throw new InvalidParameterError("axis1 and axis2 must be different", "axis", [axis1, axis2]);
  }

  const dim1 = getDim(a, ax1, "trace()");
  const dim2 = getDim(a, ax2, "trace()");
  const stride1 = getStride(a, ax1, "trace()");
  const stride2 = getStride(a, ax2, "trace()");

  const outerShape: number[] = [];
  const outerStrides: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (i !== ax1 && i !== ax2) {
      outerShape.push(getDim(a, i, "trace()"));
      outerStrides.push(getStride(a, i, "trace()"));
    }
  }

  const outerSize = outerShape.length === 0 ? 1 : outerShape.reduce((acc, v) => acc * v, 1);
  const out = new Float64Array(outerSize);

  for (let outer = 0; outer < outerSize; outer++) {
    let baseOffset = a.offset;
    let rem = outer;

    for (let d = outerShape.length - 1; d >= 0; d--) {
      const dim = outerShape[d];
      if (dim === undefined) throw new ShapeError("trace(): outer shape is out of bounds");

      const idx = rem % dim;
      rem = Math.floor(rem / dim);

      const stride = outerStrides[d];
      if (stride === undefined) throw new ShapeError("trace(): outer stride is out of bounds");

      baseOffset += idx * stride;
    }

    let sum = 0;
    if (offset >= 0) {
      const n = Math.min(dim1, Math.max(0, dim2 - offset));
      for (let i = 0; i < n; i++) {
        sum += Number(a.data[baseOffset + i * stride1 + (i + offset) * stride2]);
      }
    } else {
      const absOffset = -offset;
      const n = Math.min(Math.max(0, dim1 - absOffset), dim2);
      for (let i = 0; i < n; i++) {
        sum += Number(a.data[baseOffset + (i + absOffset) * stride1 + i * stride2]);
      }
    }

    out[outer] = sum;
  }

  // Keep float64: a plain number[] would be rounded to the default dtype (float32).
  if (outerShape.length === 0) {
    return fromDenseVector1D(out);
  }
  return tensor(out).view(outerShape);
}

/**
 * Compute the rank of a matrix.
 *
 * Number of linearly independent rows/columns.
 * Uses SVD to count singular values above a threshold.
 *
 * **Algorithm**: SVD-based rank computation
 * **Time Complexity**: O(min(M,N) * M * N) for M×N matrix
 * **Space Complexity**: O(M*N) for SVD computation
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param tol - Threshold for small singular values (optional)
 *   - Default: max(M,N) * largest_singular_value * machine_epsilon, where the epsilon follows
 *     the dtype of `a` as in NumPy (about 1.2e-7 for float32, 2.2e-16 for float64 and integers)
 *   - Singular values > tol are counted as non-zero
 *
 * **Returns**: Rank (integer between 0 and min(M, N))
 *
 * **Properties**:
 * - 0 ≤ rank(A) ≤ min(M, N)
 * - rank(A) = rank(A^T)
 * - rank(A) = number of non-zero singular values
 * - Full rank: rank(A) = min(M, N)
 * - Rank deficient: rank(A) < min(M, N)
 *
 * @example
 * ```ts
 * import { matrixRank } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 4]]);  // Rank 1 (linearly dependent rows)
 * console.log(matrixRank(A));  // 1
 *
 * const B = tensor([[1, 0], [0, 1]]);  // Full rank
 * console.log(matrixRank(B));  // 2
 * ```
 *
 * @throws {ShapeError} If input is not a 2D matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {InvalidParameterError} If tol is negative or non-finite
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function matrixRank(a: Tensor, tol?: number): number {
  if (a.ndim !== 2) throw new ShapeError("matrixRank requires a 2-D matrix");
  if (tol !== undefined && (!Number.isFinite(tol) || tol < 0)) {
    throw new InvalidParameterError("tol must be a non-negative finite number", "tol", tol);
  }

  const rows = getDim(a, 0, "matrixRank()");
  const cols = getDim(a, 1, "matrixRank()");
  const k = Math.min(rows, cols);
  if (k === 0) return 0;

  // Only singular values are needed, so skip U/V accumulation.
  const sDense = toDenseVector1D(svdvals(a));

  const defaultTol = (sDense[0] as number) * workingEpsilon(a) * Math.max(rows, cols);
  const threshold = tol ?? defaultTol;

  let rank = 0;
  for (let i = 0; i < k; i++) {
    if ((sDense[i] as number) > threshold) rank++;
  }
  return rank;
}
