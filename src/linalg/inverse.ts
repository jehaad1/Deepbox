import { DeepboxError, InvalidParameterError, ShapeError } from "../core";
import type { Tensor } from "../ndarray";
import {
  fromDenseMatrix2D,
  getDim,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
  toDenseVector1D,
} from "./_internal";
import { svd } from "./decomposition/svd";

/**
 * Compute the inverse of a matrix.
 *
 * Finds matrix A^(-1) such that A * A^(-1) = I.
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 *
 * **Returns**: Inverse matrix
 *
 * **Requirements**:
 * - Matrix must be square
 * - Matrix must be non-singular (det(A) ≠ 0)
 *
 * **Numerical singularity**: as in NumPy, only an exactly zero pivot is reported as
 * singular. A matrix that is singular up to rounding error, such as
 * `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, returns very large entries instead of throwing.
 * Check `cond(a)` or `matrixRank(a)` first when the input may be rank deficient, or use
 * {@link pinv}.
 *
 * **Properties**:
 * - A * inv(A) = I
 * - inv(inv(A)) = A
 * - inv(AB) = inv(B) * inv(A)
 *
 * @example
 * ```ts
 * import { inv } from 'deepbox/linalg';
 * import { tensor, matmul } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const Ainv = inv(A);
 *
 * // Verify: A * A^(-1) ≈ I
 * const I = matmul(A, Ainv);
 * ```
 *
 * @throws {ShapeError} If matrix is not square or not 2D
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If matrix is singular or contains non-finite values
 *
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */
export function inv(a: Tensor): Tensor {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  const rows = getDim(a, 0, "inv()");
  const cols = getDim(a, 1, "inv()");
  if (rows !== cols) throw new ShapeError("inv requires a square matrix");

  const n = rows;

  if (n === 0) {
    return fromDenseMatrix2D(0, 0, new Float64Array(0));
  }

  const { data: A } = toDenseMatrix2D(a, "inv()");
  const { lu, piv } = luFactorSquare(A, n);

  const rhs = new Float64Array(n * n);
  for (let i = 0; i < n; i++) rhs[i * n + i] = 1;

  luSolveInPlace(lu, piv, n, rhs, n);
  return fromDenseMatrix2D(n, n, rhs);
}

/**
 * Compute the Moore-Penrose pseudo-inverse.
 *
 * Generalization of matrix inverse for non-square or singular matrices.
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param rcond - Cutoff for small singular values, relative to the largest one.
 *               Defaults to max(M, N) * machine epsilon, as in NumPy 2.x.
 *
 * **Returns**: Pseudo-inverse of shape (N, M)
 *
 * **Properties**:
 * - A * pinv(A) * A = A
 * - pinv(A) * A * pinv(A) = pinv(A)
 * - For full rank square matrix: pinv(A) = inv(A)
 * - For overdetermined system: pinv(A) gives least squares solution
 *
 * **Algorithm**: Using SVD
 * A = U * Σ * V^T
 * pinv(A) = V * Σ^+ * U^T
 * where Σ^+ is pseudo-inverse of Σ (1/s_i for s_i > rcond * s_max, else 0)
 *
 * @example
 * ```ts
 * import { pinv } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Non-square matrix
 * const A = tensor([[1, 2], [3, 4], [5, 6]]);
 * const Apinv = pinv(A);
 *
 * console.log(Apinv.shape);  // [2, 3]
 * ```
 *
 * @throws {ShapeError} If input is not 2D matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {InvalidParameterError} If rcond is negative or non-finite
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */
export function pinv(a: Tensor, rcond?: number): Tensor {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  if (rcond !== undefined && (!Number.isFinite(rcond) || rcond < 0)) {
    throw new InvalidParameterError("rcond must be a non-negative finite number", "rcond", rcond);
  }

  const m = getDim(a, 0, "pinv()");
  const n = getDim(a, 1, "pinv()");
  const k = Math.min(m, n);

  if (k === 0) {
    return fromDenseMatrix2D(n, m, new Float64Array(n * m));
  }

  const [U_t, s_t, Vt_t] = svd(a, false);
  const { data: U, cols: uCols, rows: uRows } = toDenseMatrix2D(U_t, "pinv()");
  const s = toDenseVector1D(s_t);
  const { data: Vt, cols: vtCols, rows: vtRows } = toDenseMatrix2D(Vt_t);

  // U: (m, k), Vt: (k, n)
  if (uRows !== m || uCols !== k) throw new DeepboxError("Internal error: unexpected U shape");
  if (vtRows !== k || vtCols !== n) throw new DeepboxError("Internal error: unexpected Vt shape");

  const s0 = s[0] as number;
  const rcondVal = rcond ?? Number.EPSILON * Math.max(m, n);
  const cutoff = rcondVal * s0;

  const sInv = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    const si = s[i] as number;
    sInv[i] = si > cutoff ? 1 / si : 0;
  }

  // Compute pinv(A) = V * diag(sInv) * U^T. Scale the rows of Vt once, then
  // accumulate rank-one terms so that U and the output are read row by row.
  const out = new Float64Array(n * m);
  for (let r = 0; r < k; r++) {
    const si = sInv[r] as number;
    if (si === 0) continue;
    for (let i = 0; i < n; i++) {
      const vs = (Vt[r * n + i] as number) * si; // V[i,r] / s_r
      if (vs === 0) continue;
      const outRow = i * m;
      for (let j = 0; j < m; j++) {
        out[outRow + j] = (out[outRow + j] as number) + vs * (U[j * k + r] as number);
      }
    }
  }

  return fromDenseMatrix2D(n, m, out);
}
