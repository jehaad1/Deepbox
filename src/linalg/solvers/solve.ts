import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
  fromDenseMatrix2D,
  fromDenseVector1D,
  getDim,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../_internal";

/**
 * Solve linear system A * x = b.
 *
 * Finds vector x that satisfies the equation A * x = b.
 *
 * **Algorithm**: LU decomposition with partial pivoting
 *
 * **Parameters**:
 * @param a - Coefficient matrix of shape (N, N)
 * @param b - Right-hand side of shape (N,) or (N, K) for multiple RHS
 *
 * **Returns**: Solution x of same shape as b
 *
 * **Requirements**:
 * - A must be square matrix
 * - A must be non-singular (invertible)
 * - Number of rows in A must equal length of b
 *
 * **Numerical singularity**: as in NumPy, only an exactly zero pivot is reported as
 * singular. A matrix that is singular up to rounding error returns a solution with very
 * large entries instead of throwing. Check `cond(a)` or `matrixRank(a)` first when the
 * input may be rank deficient, or use `lstsq()`.
 *
 * @example
 * ```ts
 * import { solve } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[3, 1], [1, 2]]);
 * const b = tensor([9, 8]);
 * const x = solve(A, b);
 *
 * // Verify: A @ x ≈ b
 * console.log(x);  // [2, 3]
 * ```
 *
 * @throws {ShapeError} If A is not square
 * @throws {ShapeError} If dimensions don't match
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If A is singular
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra Solvers}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 3.4.1
 */
export function solve(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 2) throw new ShapeError("A must be a 2D matrix");
  const n = getDim(a, 0, "solve()");
  const n2 = getDim(a, 1, "solve()");
  if (n !== n2) throw new ShapeError("A must be square");

  if (b.ndim !== 1 && b.ndim !== 2) {
    throw new ShapeError("b must be a 1D or 2D tensor");
  }

  const bRows = getDim(b, 0, "solve()");
  if (bRows !== n) {
    throw new ShapeError("A and b dimensions do not match");
  }

  const { data: A } = toDenseMatrix2D(a, "solve()");
  const { lu, piv } = luFactorSquare(A, n);

  if (b.ndim === 1) {
    // toDenseVector1D returns a fresh copy, so it can be solved in place.
    const rhs = toDenseVector1D(b, "solve()");
    luSolveInPlace(lu, piv, n, rhs, 1);
    return fromDenseVector1D(rhs);
  }

  const nrhs = getDim(b, 1, "solve()");
  const { data: B } = toDenseMatrix2D(b, "solve()");
  luSolveInPlace(lu, piv, n, B, nrhs);
  return fromDenseMatrix2D(n, nrhs, B);
}

/**
 * Options for {@link solveTriangular}.
 */
export type SolveTriangularOptions = {
  /** Solve `A^T x = b` instead of `A x = b` (default: false). */
  readonly trans?: boolean;
  /**
   * Treat the diagonal of A as all ones and do not read it (default: false).
   * Useful for the unit-lower factor of an LU decomposition.
   */
  readonly unitDiagonal?: boolean;
};

/**
 * Solve triangular system.
 *
 * More efficient than solve() when A is already triangular. Only the selected
 * triangle of A is used; the entries of the other triangle are ignored (they must
 * still be finite).
 *
 * **Parameters**:
 * @param a - Triangular matrix of shape (N, N)
 * @param b - Right-hand side of shape (N,) or (N, K)
 * @param lower - If true (default), A is lower triangular; if false, upper triangular.
 *   Note that `scipy.linalg.solve_triangular` defaults to upper.
 * @param options - `trans` solves with the transpose of A (which swaps the
 *   triangle that is read), `unitDiagonal` assumes a unit diagonal
 *
 * **Returns**: Solution x of the same shape as b (float64)
 *
 * @example
 * ```ts
 * import { solveTriangular } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const L = tensor([[2, 0], [3, 4]]);  // Lower triangular
 * const b = tensor([6, 18]);
 * const x = solveTriangular(L, b, true);               // [3, 2.25]
 * const y = solveTriangular(L, b, true, { trans: true }); // solves L^T y = b
 * ```
 *
 * @throws {ShapeError} If A is not square
 * @throws {ShapeError} If dimensions don't match
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If A is singular (zero diagonal; ignored with `unitDiagonal`)
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra Solvers}
 */
export function solveTriangular(
  a: Tensor,
  b: Tensor,
  lower = true,
  options: SolveTriangularOptions = {}
): Tensor {
  if (a.ndim !== 2) throw new ShapeError("A must be a 2D matrix");
  const n = getDim(a, 0, "solveTriangular()");
  const n2 = getDim(a, 1, "solveTriangular()");
  if (n !== n2) throw new ShapeError("A must be square");
  if (b.ndim !== 1 && b.ndim !== 2) {
    throw new ShapeError("b must be a 1D or 2D tensor");
  }
  const bN = getDim(b, 0, "solveTriangular()");
  if (bN !== n) throw new ShapeError("A and b dimensions do not match");

  const trans = options.trans === true;
  const unit = options.unitDiagonal === true;

  const { data: A } = toDenseMatrix2D(a, "solveTriangular()");
  const nrhs = b.ndim === 1 ? 1 : getDim(b, 1, "solveTriangular()");
  const X =
    b.ndim === 1
      ? toDenseVector1D(b, "solveTriangular()")
      : toDenseMatrix2D(b, "solveTriangular()").data;

  // Effective matrix T = trans ? A^T : A, read through strides. Transposing a
  // lower-triangular matrix gives an upper-triangular one and vice versa.
  const rowStride = trans ? 1 : n;
  const colStride = trans ? n : 1;
  const effLower = trans ? !lower : lower;

  // Substitution is done one right-hand-side row at a time (an axpy over the K
  // columns), so both X rows are walked contiguously. For every entry the
  // operations are the same, in the same order, as the textbook column loop.
  const solveRow = (i: number): void => {
    const iRow = i * nrhs;
    const from = effLower ? 0 : i + 1;
    const to = effLower ? i : n;
    for (let j = from; j < to; j++) {
      const t = A[i * rowStride + j * colStride] as number;
      if (t === 0) continue;
      const jRow = j * nrhs;
      for (let c = 0; c < nrhs; c++) {
        X[iRow + c] = (X[iRow + c] as number) - t * (X[jRow + c] as number);
      }
    }
    if (!unit) {
      const diag = A[i * (n + 1)] as number;
      if (diag === 0) throw new DataValidationError("Matrix is singular");
      for (let c = 0; c < nrhs; c++) X[iRow + c] = (X[iRow + c] as number) / diag;
    }
  };

  if (effLower) {
    for (let i = 0; i < n; i++) solveRow(i);
  } else {
    for (let i = n - 1; i >= 0; i--) solveRow(i);
  }

  return b.ndim === 1 ? fromDenseVector1D(X) : fromDenseMatrix2D(n, nrhs, X);
}
