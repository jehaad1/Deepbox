import { DeepboxError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
  fromDenseMatrix2D,
  fromDenseVector1D,
  getDim,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../_internal";
import { svd } from "../decomposition/svd";

/**
 * Least squares solution to A * x = b.
 *
 * Finds x that minimizes ||A*x - b||^2 (Euclidean norm).
 * Works for overdetermined (M > N), underdetermined (M < N), and square systems.
 * For rank-deficient A the minimum-norm solution is returned.
 *
 * **Algorithm**: SVD-based least squares
 *
 * **Parameters**:
 * @param a - Coefficient matrix of shape (M, N)
 * @param b - Target values of shape (M,) or (M, K)
 * @param rcond - Cutoff for small singular values: singular values `<= rcond * s_max` are
 *   treated as zero (default: float64 machine epsilon * max(M,N), as in NumPy, which always
 *   solves in double precision)
 *
 * **Returns**: Object with:
 * - x: Least squares solution of shape (N,) or (N, K)
 * - residuals: Sum of squared residuals ||b - A*x||^2, shape (1,) for 1-D b or (K,) for 2-D b.
 *   Unlike NumPy, it is always filled in, also for rank-deficient or underdetermined systems.
 * - rank: Effective rank of A
 * - s: Singular values of A
 *
 * **Requirements**:
 * - A can be any M x N matrix
 * - First dimension of A must match first dimension of b
 *
 * @example
 * ```ts
 * import { lstsq } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Overdetermined system (more equations than unknowns)
 * const A = tensor([[1, 1], [1, 2], [1, 3]]);
 * const b = tensor([2, 3, 5]);
 * const result = lstsq(A, b);
 *
 * console.log(result.x);  // Best fit solution
 * console.log(result.residuals);  // Residual error
 * ```
 *
 * @throws {ShapeError} If A is not 2D matrix
 * @throws {ShapeError} If b is not 1D or 2D tensor
 * @throws {ShapeError} If dimensions don't match
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {InvalidParameterError} If rcond is negative or non-finite
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra Solvers}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 5.5.4
 */
export function lstsq(
  a: Tensor,
  b: Tensor,
  rcond?: number
): {
  readonly x: Tensor;
  readonly residuals: Tensor;
  readonly rank: number;
  readonly s: Tensor;
} {
  if (a.ndim !== 2) throw new ShapeError("A must be a 2D matrix");
  if (b.ndim !== 1 && b.ndim !== 2) throw new ShapeError("b must be a 1D or 2D tensor");

  if (rcond !== undefined && (!Number.isFinite(rcond) || rcond < 0)) {
    throw new InvalidParameterError("rcond must be a non-negative finite number", "rcond", rcond);
  }

  const m = getDim(a, 0, "lstsq()");
  const n = getDim(a, 1, "lstsq()");
  const bRows = getDim(b, 0, "lstsq()");
  if (bRows !== m) throw new ShapeError("A and b dimensions do not match");

  const nrhs = b.ndim === 1 ? 1 : getDim(b, 1, "lstsq()");
  const k = Math.min(m, n);

  // Right-hand side as a dense (m, nrhs) row-major block (a vector is one column).
  const B = b.ndim === 1 ? toDenseVector1D(b, "lstsq()") : toDenseMatrix2D(b, "lstsq()").data;

  const pack = (x: Float64Array, residuals: Float64Array, rank: number, s: Tensor) => ({
    x: b.ndim === 1 ? fromDenseVector1D(x) : fromDenseMatrix2D(n, nrhs, x),
    // fromDense* keeps float64; a bare tensor([residual]) would downcast to the
    // default float32 dtype
    residuals: fromDenseVector1D(residuals),
    rank,
    s,
  });

  // ||b_c - A x_c||^2 for every right-hand-side column c.
  const residualsOf = (A: Float64Array | null, X: Float64Array): Float64Array => {
    const out = new Float64Array(nrhs);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < nrhs; c++) {
        let pred = 0;
        if (A !== null) {
          for (let j = 0; j < n; j++) {
            pred += (A[i * n + j] as number) * (X[j * nrhs + c] as number);
          }
        }
        const err = (B[i * nrhs + c] as number) - pred;
        out[c] = (out[c] as number) + err * err;
      }
    }
    return out;
  };

  if (k === 0) {
    // No unknowns (n = 0): x is empty and the residual is ||b||^2.
    // No equations (m = 0): x is zero and the residual is 0.
    const X = new Float64Array(n * nrhs);
    return pack(X, residualsOf(null, X), 0, fromDenseVector1D(new Float64Array(0)));
  }

  const rcondVal = rcond ?? Number.EPSILON * Math.max(m, n);

  // Validates A under the name lstsq(), and gives the residuals their dense copy.
  const { data: A } = toDenseMatrix2D(a, "lstsq()");

  // Compute SVD once: A = U diag(s) V^T with U (m, k) and Vt (k, n).
  const [U_t, s, Vt_t] = svd(a, false);
  const { data: U, cols: uCols, rows: uRows } = toDenseMatrix2D(U_t);
  const sDense = toDenseVector1D(s);
  const { data: Vt, cols: vtCols, rows: vtRows } = toDenseMatrix2D(Vt_t);

  // Validate SVD shapes
  if (uRows !== m || uCols !== k) {
    throw new DeepboxError("Internal error: unexpected U shape");
  }
  if (vtRows !== k || vtCols !== n) {
    throw new DeepboxError("Internal error: unexpected Vt shape");
  }

  // Rank and inverse singular values
  const cutoff = (sDense[0] as number) * rcondVal;
  let rank = 0;
  const sInv = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    const si = sDense[i] as number;
    if (si > cutoff) {
      rank++;
      sInv[i] = 1 / si;
    }
  }

  // x = V diag(sInv) U^T b, evaluated right to left so the pseudo-inverse
  // (n x m) is never formed: cost O((m + n) k K) instead of O(m n k).
  // C = diag(sInv) U^T B, shape (k, nrhs)
  const C = new Float64Array(k * nrhs);
  for (let j = 0; j < m; j++) {
    for (let r = 0; r < k; r++) {
      const u = U[j * k + r] as number;
      if (u === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        C[r * nrhs + c] = (C[r * nrhs + c] as number) + u * (B[j * nrhs + c] as number);
      }
    }
  }
  for (let r = 0; r < k; r++) {
    const inv = sInv[r] as number;
    for (let c = 0; c < nrhs; c++) C[r * nrhs + c] = (C[r * nrhs + c] as number) * inv;
  }
  // X = V C, with V[i, r] = Vt[r, i]
  const X = new Float64Array(n * nrhs);
  for (let r = 0; r < k; r++) {
    if (sInv[r] === 0) continue;
    for (let i = 0; i < n; i++) {
      const v = Vt[r * n + i] as number;
      if (v === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        X[i * nrhs + c] = (X[i * nrhs + c] as number) + v * (C[r * nrhs + c] as number);
      }
    }
  }

  return pack(X, residualsOf(A, X), rank, s);
}
