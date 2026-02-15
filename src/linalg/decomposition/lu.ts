import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { at, atInt, fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * LU decomposition with partial pivoting.
 *
 * Factorizes matrix A into P * A = L * U where:
 * - P is a permutation matrix
 * - L is lower triangular with unit diagonal
 * - U is upper triangular
 *
 * **Algorithm**: Gaussian elimination with partial pivoting
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 *
 * **Returns**: [P, L, U]
 * - P: Permutation matrix of shape (M, M)
 * - L: Lower triangular matrix of shape (M, K) where K = min(M, N)
 * - U: Upper triangular matrix of shape (K, N)
 *
 * **Requirements**:
 * - Input must be 2D matrix
 *
 * **Properties**:
 * - P @ A = L @ U
 * - L has unit diagonal (L[i,i] = 1)
 * - Partial pivoting ensures numerical stability
 *
 * @example
 * ```ts
 * import { lu } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[2, 1, 1], [4, 3, 3], [8, 7, 9]]);
 * const [P, L, U] = lu(A);
 *
 * // Verify: P @ A ≈ L @ U
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * **Note**: Rank-deficient matrices (with zero pivots) are handled gracefully.
 * The factorization will still produce valid P, L, U factors.
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 3.4.1
 */
export function lu(a: Tensor): [Tensor, Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "lu()");
  const n = getDim(a, 1, "lu()");
  const k = Math.min(m, n);

  const { data: A0 } = toDenseMatrix2D(a);
  const Ufull = new Float64Array(A0);

  const Lfull = new Float64Array(m * k);
  for (let i = 0; i < Math.min(m, k); i++) Lfull[i * k + i] = 1;

  const piv = new Int32Array(m);
  for (let i = 0; i < m; i++) piv[i] = i;

  for (let col = 0; col < k; col++) {
    // Find pivot row
    let maxRow = col;
    let maxVal = Math.abs(at(Ufull, col * n + col));
    for (let i = col + 1; i < m; i++) {
      const v = Math.abs(at(Ufull, i * n + col));
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }

    if (maxRow !== col) {
      // Swap rows in U
      for (let j = 0; j < n; j++) {
        const tmp = at(Ufull, col * n + j);
        Ufull[col * n + j] = at(Ufull, maxRow * n + j);
        Ufull[maxRow * n + j] = tmp;
      }

      // Swap already computed entries of L (columns < col)
      for (let j = 0; j < col; j++) {
        const tmp = at(Lfull, col * k + j);
        Lfull[col * k + j] = at(Lfull, maxRow * k + j);
        Lfull[maxRow * k + j] = tmp;
      }

      const tp = atInt(piv, col);
      piv[col] = atInt(piv, maxRow);
      piv[maxRow] = tp;
    }

    const pivot = at(Ufull, col * n + col);

    // Skip elimination for zero-pivot columns (rank-deficient matrices).
    // L column stays zero (except the unit diagonal already set), U retains the row as-is.
    if (pivot === 0) continue;

    for (let i = col + 1; i < m; i++) {
      const factor = at(Ufull, i * n + col) / pivot;
      Lfull[i * k + col] = factor;
      Ufull[i * n + col] = factor;
      for (let j = col + 1; j < n; j++) {
        Ufull[i * n + j] = at(Ufull, i * n + j) - factor * at(Ufull, col * n + j);
      }
    }
  }

  // Build permutation matrix P from piv.
  const Pfull = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    const pi = atInt(piv, i);
    Pfull[i * m + pi] = 1;
  }

  // Extract U (k x n) from Ufull.
  const U = new Float64Array(k * n);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < n; j++) {
      U[i * n + j] = i <= j ? at(Ufull, i * n + j) : 0;
    }
  }

  return [
    fromDenseMatrix2D(m, m, Pfull),
    fromDenseMatrix2D(m, k, Lfull),
    fromDenseMatrix2D(k, n, U),
  ];
}
