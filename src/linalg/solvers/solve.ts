import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
  at,
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
 * @throws {DTypeError} If input has string dtype
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

  const { data: A } = toDenseMatrix2D(a);
  const { lu, piv } = luFactorSquare(A, n);

  if (b.ndim === 1) {
    const rhs = toDenseVector1D(b);
    const rhsMat = new Float64Array(n * 1);
    for (let i = 0; i < n; i++) rhsMat[i] = at(rhs, i);
    luSolveInPlace(lu, piv, n, rhsMat, 1);
    return fromDenseVector1D(rhsMat);
  }

  const nrhs = getDim(b, 1, "solve()");
  const { data: B } = toDenseMatrix2D(b);
  luSolveInPlace(lu, piv, n, B, nrhs);
  return fromDenseMatrix2D(n, nrhs, B);
}

/**
 * Solve triangular system.
 *
 * More efficient than solve() when A is already triangular.
 *
 * **Parameters**:
 * @param a - Triangular matrix of shape (N, N)
 * @param b - Right-hand side of shape (N,) or (N, K)
 * @param lower - If true, A is lower triangular; if false, upper triangular
 *
 * **Returns**: Solution x
 *
 * @example
 * ```ts
 * import { solveTriangular } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const L = tensor([[2, 0], [3, 4]]);  // Lower triangular
 * const b = tensor([6, 18]);
 * const x = solveTriangular(L, b, true);
 * ```
 *
 * @throws {ShapeError} If A is not square
 * @throws {ShapeError} If dimensions don't match
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If A is singular (zero diagonal)
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra Solvers}
 */
export function solveTriangular(a: Tensor, b: Tensor, lower = true): Tensor {
  if (a.ndim !== 2) throw new ShapeError("A must be a 2D matrix");
  const n = getDim(a, 0, "solveTriangular()");
  const n2 = getDim(a, 1, "solveTriangular()");
  if (n !== n2) throw new ShapeError("A must be square");
  if (b.ndim !== 1 && b.ndim !== 2) {
    throw new ShapeError("b must be a 1D or 2D tensor");
  }
  const bN = getDim(b, 0, "solveTriangular()");
  if (bN !== n) throw new ShapeError("A and b dimensions do not match");

  const { data: A } = toDenseMatrix2D(a);
  if (b.ndim === 1) {
    const rhs = toDenseVector1D(b);
    const x = new Float64Array(n);

    if (lower) {
      for (let i = 0; i < n; i++) {
        let sum = at(rhs, i);
        for (let j = 0; j < i; j++) {
          sum -= at(A, i * n + j) * at(x, j);
        }
        const diag = at(A, i * n + i);
        if (diag === 0) throw new DataValidationError("Matrix is singular");
        x[i] = sum / diag;
      }
    } else {
      for (let i = n - 1; i >= 0; i--) {
        let sum = at(rhs, i);
        for (let j = i + 1; j < n; j++) {
          sum -= at(A, i * n + j) * at(x, j);
        }
        const diag = at(A, i * n + i);
        if (diag === 0) throw new DataValidationError("Matrix is singular");
        x[i] = sum / diag;
      }
    }

    return fromDenseVector1D(x);
  }

  const nrhs = getDim(b, 1, "solveTriangular()");
  const { data: B } = toDenseMatrix2D(b);
  const X = new Float64Array(n * nrhs);

  if (lower) {
    for (let k = 0; k < nrhs; k++) {
      for (let i = 0; i < n; i++) {
        let sum = at(B, i * nrhs + k);
        for (let j = 0; j < i; j++) {
          sum -= at(A, i * n + j) * at(X, j * nrhs + k);
        }
        const diag = at(A, i * n + i);
        if (diag === 0) throw new DataValidationError("Matrix is singular");
        X[i * nrhs + k] = sum / diag;
      }
    }
  } else {
    for (let k = 0; k < nrhs; k++) {
      for (let i = n - 1; i >= 0; i--) {
        let sum = at(B, i * nrhs + k);
        for (let j = i + 1; j < n; j++) {
          sum -= at(A, i * n + j) * at(X, j * nrhs + k);
        }
        const diag = at(A, i * n + i);
        if (diag === 0) throw new DataValidationError("Matrix is singular");
        X[i * nrhs + k] = sum / diag;
      }
    }
  }

  return fromDenseMatrix2D(n, nrhs, X);
}
