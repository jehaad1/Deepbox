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
 *
 * **Algorithm**: SVD-based least squares
 *
 * **Parameters**:
 * @param a - Coefficient matrix of shape (M, N)
 * @param b - Target values of shape (M,) or (M, K)
 * @param rcond - Cutoff for small singular values (default: machine epsilon * max(M,N))
 *
 * **Returns**: Object with:
 * - x: Least squares solution of shape (N,) or (N, K)
 * - residuals: Sum of squared residuals ||b - A*x||^2
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
 * @throws {DTypeError} If input has string dtype
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

  const k = Math.min(m, n);

  if (k === 0) {
    const zeroX =
      b.ndim === 1
        ? fromDenseVector1D(new Float64Array(n))
        : fromDenseMatrix2D(
            n,
            getDim(b, 1, "lstsq()"),
            new Float64Array(n * getDim(b, 1, "lstsq()"))
          );
    const zeroResiduals =
      b.ndim === 1
        ? fromDenseVector1D(new Float64Array([0]))
        : fromDenseVector1D(new Float64Array(getDim(b, 1, "lstsq()")));
    return {
      x: zeroX,
      residuals: zeroResiduals,
      rank: 0,
      s: fromDenseVector1D(new Float64Array(0)),
    };
  }

  const rcondVal = rcond ?? Number.EPSILON * Math.max(m, n);

  // Compute SVD once and reuse for both rank and pseudo-inverse
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

  // Compute rank and inverse singular values
  const cutoff = (sDense[0] as number) * rcondVal;
  let rank = 0;
  const sInv = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    const si = sDense[i] as number;
    if (si > cutoff) {
      rank++;
      sInv[i] = 1 / si;
    } else {
      sInv[i] = 0;
    }
  }

  // Compute pinv(A) = V * diag(sInv) * U^T manually (n x m)
  const P = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let r = 0; r < k; r++) {
        const v_ir = Vt[r * n + i] as number; // Vt[r,i] = V[i,r]
        const u_jr = U[j * k + r] as number; // U[j,r]
        sum += v_ir * (sInv[r] as number) * u_jr;
      }
      P[i * m + j] = sum;
    }
  }

  if (b.ndim === 1) {
    const bv = toDenseVector1D(b);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < m; j++) {
        sum += (P[i * m + j] as number) * (bv[j] as number);
      }
      x[i] = sum;
    }

    // residual = ||b - A x||^2
    const { data: A } = toDenseMatrix2D(a);
    let residual = 0;
    for (let i = 0; i < m; i++) {
      let pred = 0;
      for (let j = 0; j < n; j++) {
        pred += (A[i * n + j] as number) * (x[j] as number);
      }
      const err = (bv[i] as number) - pred;
      residual += err * err;
    }

    return {
      x: fromDenseVector1D(x),
      // fromDenseVector1D keeps float64; a bare tensor([residual]) would
      // downcast to the default float32 dtype, inconsistent with the 2-D-b path
      residuals: fromDenseVector1D(new Float64Array([residual])),
      rank,
      s,
    };
  }

  const nrhs = getDim(b, 1, "lstsq()");
  const { data: B, rows: bM, cols: bK } = toDenseMatrix2D(b);
  if (bM !== m || bK !== nrhs) throw new DeepboxError("Internal error: unexpected b shape");

  const X = new Float64Array(n * nrhs);
  for (let i = 0; i < n; i++) {
    for (let k2 = 0; k2 < nrhs; k2++) {
      let sum = 0;
      for (let j = 0; j < m; j++) {
        sum += (P[i * m + j] as number) * (B[j * nrhs + k2] as number);
      }
      X[i * nrhs + k2] = sum;
    }
  }

  const { data: A } = toDenseMatrix2D(a);
  const residuals = new Float64Array(nrhs);
  for (let k2 = 0; k2 < nrhs; k2++) {
    let rsum = 0;
    for (let i = 0; i < m; i++) {
      let pred = 0;
      for (let j = 0; j < n; j++) {
        pred += (A[i * n + j] as number) * (X[j * nrhs + k2] as number);
      }
      const err = (B[i * nrhs + k2] as number) - pred;
      rsum += err * err;
    }
    residuals[k2] = rsum;
  }

  return {
    x: fromDenseMatrix2D(n, nrhs, X),
    residuals: fromDenseVector1D(residuals),
    rank,
    s,
  };
}
