import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * Options for {@link cholesky}.
 *
 * - `upper`: if true, return the upper triangular factor U with A = Uᵀ * U
 *   instead of the lower triangular factor L with A = L * Lᵀ (default: false).
 */
export type CholeskyOptions = {
  readonly upper?: boolean;
};

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
 * @param options - Optional settings (see {@link CholeskyOptions})
 * @param options.upper - Return the upper factor U (A = Uᵀ U) instead of L (default: false)
 *
 * **Returns**: L - Lower triangular matrix where A = L * L^T (or U when `upper` is true)
 *
 * **Requirements**:
 * - Input must be 2D square matrix
 * - Matrix must be symmetric: A = A^T. The check is relative: entries a[i][j] and
 *   a[j][i] may differ by up to 1e-10 * max|A|. Only the lower triangle is read for
 *   the factorization.
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
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If matrix is not symmetric
 * @throws {DataValidationError} If matrix is not positive definite
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 4.2.1
 */
export function cholesky(a: Tensor, options: CholeskyOptions = {}): Tensor {
  // Validate input dimensions
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }
  const rows = getDim(a, 0, "cholesky()");
  const cols = getDim(a, 1, "cholesky()");
  if (rows !== cols) {
    throw new ShapeError("cholesky requires a square matrix");
  }
  const upper = options.upper ?? false;
  if (typeof upper !== "boolean") {
    throw new InvalidParameterError("upper must be a boolean", "upper", upper);
  }

  const n = rows;

  // Handle empty matrix edge case
  if (n === 0) {
    return fromDenseMatrix2D(0, 0, new Float64Array(0));
  }

  // Convert input to dense Float64Array for efficient access
  // Note: toDenseMatrix2D already validates non-finite values
  const { data: A } = toDenseMatrix2D(a, "cholesky()");

  // Validate symmetry. The tolerance is relative to the largest entry so that matrices
  // with large entries (e.g. products like XᵀX) are not rejected for rounding-level
  // asymmetry, and matrices with tiny entries are not accepted when clearly asymmetric.
  let maxAbs = 0;
  for (let i = 0; i < A.length; i++) {
    const v = Math.abs(A[i] as number);
    if (v > maxAbs) maxAbs = v;
  }
  const symTol = 1e-10 * maxAbs;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs((A[i * n + j] as number) - (A[j * n + i] as number)) > symTol) {
        throw new DataValidationError("Matrix must be symmetric");
      }
    }
  }

  // Allocate output matrix L (initialized to zeros)
  const L = new Float64Array(n * n);

  // Cholesky-Banachiewicz algorithm
  // Computes L row by row from top to bottom
  for (let i = 0; i < n; i++) {
    const iRow = i * n;
    // Process elements in row i up to and including diagonal
    for (let j = 0; j < i; j++) {
      // Off-diagonal element: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k < j)) / L[j,j]
      const jRow = j * n;
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += (L[iRow + k] as number) * (L[jRow + k] as number);
      }
      // L[j,j] > 0 here: every earlier diagonal pivot passed the check below.
      L[iRow + j] = ((A[iRow + j] as number) - sum) / (L[jRow + j] as number);
    }

    // Diagonal element: L[i,i] = sqrt(A[i,i] - sum(L[i,k]² for k < i))
    let sum = 0;
    for (let k = 0; k < i; k++) {
      const Lik = L[iRow + k] as number;
      sum += Lik * Lik;
    }
    const val = (A[iRow + i] as number) - sum;
    // Also rejects NaN, which would indicate overflow in the sums above.
    if (!(val > 0)) {
      throw new DataValidationError(
        `Matrix is not positive definite (leading minor of order ${i + 1} is not positive)`
      );
    }
    L[iRow + i] = Math.sqrt(val);
  }

  if (upper) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) U[j * n + i] = L[i * n + j] as number;
    }
    return fromDenseMatrix2D(n, n, U);
  }

  // Convert result back to tensor
  return fromDenseMatrix2D(n, n, L);
}
