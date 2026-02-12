import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { at, fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * Cholesky decomposition.
 *
 * Factorizes symmetric positive definite matrix A into L * L^T where L is lower triangular.
 *
 * **Algorithm**: Cholesky-Banachiewicz algorithm
 * **Time Complexity**: O(N³/3) - approximately half the cost of LU decomposition
 * **Space Complexity**: O(N²) for output matrix
 *
 * **Mathematical Background**:
 * For a symmetric positive definite matrix A, the Cholesky decomposition is:
 * A = L * L^T
 * where L is lower triangular with positive diagonal elements.
 *
 * **Numerical Stability**:
 * - Only works for positive definite matrices (all eigenvalues > 0)
 * - More stable than LU for this class of matrices
 * - No pivoting required due to positive definiteness
 *
 * **Parameters**:
 * @param a - Symmetric positive definite matrix of shape (N, N)
 *
 * **Returns**: L - Lower triangular matrix where A = L * L^T
 *
 * **Requirements**:
 * - Input must be 2D square matrix
 * - Matrix must be symmetric: A = A^T
 * - Matrix must be positive definite: x^T * A * x > 0 for all x ≠ 0
 *
 * **Properties**:
 * - A = L @ L^T
 * - L is lower triangular (L[i,j] = 0 for j > i)
 * - L has positive diagonal elements
 * - Much faster than LU decomposition for positive definite matrices
 * - Determinant: det(A) = (product of L[i,i])²
 *
 * @example
 * ```ts
 * import { cholesky } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Positive definite matrix
 * const A = tensor([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]);
 * const L = cholesky(A);
 *
 * // Verify: A ≈ L @ L^T
 * ```
 *
 * @throws {ShapeError} If input is not 2D square matrix
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If matrix is not symmetric
 * @throws {DataValidationError} If matrix is not positive definite
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://en.wikipedia.org/wiki/Cholesky_decomposition | Wikipedia: Cholesky decomposition}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 4.2.1
 */
export function cholesky(a: Tensor): Tensor {
  // Validate input dimensions
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }
  const rows = getDim(a, 0, "cholesky()");
  const cols = getDim(a, 1, "cholesky()");
  if (rows !== cols) {
    throw new ShapeError("cholesky requires a square matrix");
  }

  const n = rows;

  // Handle empty matrix edge case
  if (n === 0) {
    return fromDenseMatrix2D(0, 0, new Float64Array(0));
  }

  // Convert input to dense Float64Array for efficient access
  // Note: toDenseMatrix2D already validates non-finite values
  const { data: A } = toDenseMatrix2D(a);

  // Validate symmetry (using tolerance for floating-point comparison)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const aij = at(A, i * n + j);
      const aji = at(A, j * n + i);
      if (Math.abs(aij - aji) > 1e-10) {
        throw new DataValidationError("Matrix must be symmetric");
      }
    }
  }

  // Allocate output matrix L (initialized to zeros)
  const L = new Float64Array(n * n);

  // Cholesky-Banachiewicz algorithm
  // Computes L row by row from top to bottom
  for (let i = 0; i < n; i++) {
    // Process elements in row i up to and including diagonal
    for (let j = 0; j <= i; j++) {
      let sum = 0;

      if (j === i) {
        // Diagonal element: L[i,i] = sqrt(A[i,i] - sum(L[i,k]² for k < i))
        // Sum of squares of elements in row i before diagonal
        for (let k = 0; k < j; k++) {
          const Lik = at(L, i * n + k);
          sum += Lik * Lik;
        }
        // Compute diagonal element
        const val = at(A, i * n + i) - sum;
        // Check positive definiteness
        if (val <= 0) {
          throw new DataValidationError("Matrix is not positive definite");
        }
        // Store square root of diagonal element
        L[i * n + i] = Math.sqrt(val);
      } else {
        // Off-diagonal element: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k < j)) / L[j,j]
        // Dot product of partial rows i and j
        for (let k = 0; k < j; k++) {
          sum += at(L, i * n + k) * at(L, j * n + k);
        }
        // Compute off-diagonal element
        const Ljj = at(L, j * n + j);
        // Ljj should never be 0 here since we check positive definiteness on diagonal
        // If it is 0, the matrix is not positive definite (caught earlier)
        L[i * n + j] = (at(A, i * n + j) - sum) / Ljj;
      }
    }
  }

  // Convert result back to tensor
  return fromDenseMatrix2D(n, n, L);
}
