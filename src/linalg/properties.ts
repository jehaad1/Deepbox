import {
  type Axis,
  DataValidationError,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import {
  at,
  getDim,
  getStride,
  luFactorSquare,
  toDenseMatrix2D,
  toDenseVector1D,
} from "./_internal";
import { svd } from "./decomposition/svd";

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
 * @throws {DTypeError} If input has string dtype
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

  const { data: A } = toDenseMatrix2D(a);

  try {
    const { lu, pivSign } = luFactorSquare(A, n);
    let detVal = pivSign;
    for (let i = 0; i < n; i++) detVal *= at(lu, i * n + i);
    return detVal;
  } catch (err) {
    // (Fix) Do NOT swallow all errors. Only treat the specific "singular matrix" case as det=0.
    if (err instanceof DataValidationError && err.message === "Matrix is singular") {
      return 0;
    }
    throw err;
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
 * **Returns**: [sign, logdet]
 * - sign: +1, -1, or 0 (as 0D tensor)
 * - logdet: Natural log of |det(A)| (as 0D tensor), -Infinity if singular
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
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function slogdet(a: Tensor): [Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("slogdet requires a 2-D matrix");
  const rows = getDim(a, 0, "slogdet()");
  const cols = getDim(a, 1, "slogdet()");
  if (rows !== cols) throw new ShapeError("slogdet requires a square matrix");

  const n = rows;
  if (n === 0) return [tensor([1]), tensor([0])];

  const { data: A } = toDenseMatrix2D(a);

  try {
    const { lu, pivSign } = luFactorSquare(A, n);
    let sign = pivSign;
    let logAbsDet = 0;

    for (let i = 0; i < n; i++) {
      const d = at(lu, i * n + i);
      if (d === 0) {
        return [tensor([0]), tensor([-Infinity])];
      }
      sign *= Math.sign(d);
      logAbsDet += Math.log(Math.abs(d));
    }

    return [tensor([sign]), tensor([logAbsDet])];
  } catch (err) {
    // (Fix) Do NOT swallow all errors. Only treat the specific "singular matrix" case specially.
    if (err instanceof DataValidationError && err.message === "Matrix is singular") {
      return [tensor([0]), tensor([-Infinity])];
    }
    throw err;
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
 * **Returns**: Trace values as tensor (one per slice if input is batched)
 *
 * **Properties**:
 * - trace(A) = sum of eigenvalues (for square matrices)
 * - trace(AB) = trace(BA) (cyclic property)
 * - trace(A + B) = trace(A) + trace(B) (linearity)
 * - trace(cA) = c * trace(A) for scalar c
 * - trace(A^T) = trace(A)
 *
 * @example
 * ```ts
 * import { trace } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * console.log(trace(A));  // 1 + 5 + 9 = 15
 * console.log(trace(A, 1));  // 2 + 6 = 8 (upper diagonal)
 * console.log(trace(A, -1));  // 4 + 8 = 12 (lower diagonal)
 * ```
 *
 * @throws {ShapeError} If input is not at least 2D
 * @throws {InvalidParameterError} If axis values are invalid/identical or offset is non-integer
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function trace(a: Tensor, offset = 0, axis1: Axis = 0, axis2: Axis = 1): Tensor {
  if (a.ndim < 2) {
    throw new ShapeError("Input must be at least 2D");
  }

  if (!Number.isInteger(offset)) {
    throw new InvalidParameterError("offset must be an integer", "offset", offset);
  }

  if (a.dtype === "string") {
    throw new DataValidationError("trace() does not support string dtype");
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

  if (outerShape.length === 0) {
    return tensor([at(out, 0)]);
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
 *   - Default: max(M,N) * largest_singular_value * machine_epsilon
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
 * @throws {DTypeError} If input has string dtype
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

  const [_U, s, _Vt] = svd(a);
  const sDense = toDenseVector1D(s);

  const defaultTol = at(sDense, 0) * Number.EPSILON * Math.max(rows, cols);
  const threshold = tol ?? defaultTol;

  let rank = 0;
  for (let i = 0; i < k; i++) {
    if (at(sDense, i) > threshold) rank++;
  }
  return rank;
}
