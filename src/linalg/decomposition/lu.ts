import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

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
 * **Convention**: P is the row permutation applied to A (`P @ A = L @ U`). This
 * is the transpose of the `p` returned by `scipy.linalg.lu`, which satisfies
 * `A = p @ l @ u`.
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
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * Rank-deficient matrices (with zero pivots) are accepted.
 * The factorization still produces valid P, L, U factors.
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 3.4.1
 */
export function lu(a: Tensor): [Tensor, Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "lu()");
  const n = getDim(a, 1, "lu()");
  const k = Math.min(m, n);

  // Ufull holds the multipliers below the diagonal while eliminating; they are
  // copied into L at the end, and row swaps move both parts together.
  const { data: A0 } = toDenseMatrix2D(a, "lu()");
  const Ufull = new Float64Array(A0);

  const piv = new Int32Array(m);
  for (let i = 0; i < m; i++) piv[i] = i;

  for (let col = 0; col < k; col++) {
    // Find pivot row
    let maxRow = col;
    let maxVal = Math.abs(Ufull[col * n + col] as number);
    for (let i = col + 1; i < m; i++) {
      const v = Math.abs(Ufull[i * n + col] as number);
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }

    // A zero pivot means the whole column below the diagonal is zero too
    // (rank-deficient input): there is nothing to eliminate, and the multipliers
    // for this column are zero.
    if (maxVal === 0) continue;

    if (maxRow !== col) {
      const cRow = col * n;
      const mRow = maxRow * n;
      for (let j = 0; j < n; j++) {
        const tmp = Ufull[cRow + j] as number;
        Ufull[cRow + j] = Ufull[mRow + j] as number;
        Ufull[mRow + j] = tmp;
      }
      const tp = piv[col] as number;
      piv[col] = piv[maxRow] as number;
      piv[maxRow] = tp;
    }

    const cRow = col * n;
    const pivot = Ufull[cRow + col] as number;
    for (let i = col + 1; i < m; i++) {
      const iRow = i * n;
      const factor = (Ufull[iRow + col] as number) / pivot;
      Ufull[iRow + col] = factor;
      if (factor === 0) continue;
      for (let j = col + 1; j < n; j++) {
        Ufull[iRow + j] = (Ufull[iRow + j] as number) - factor * (Ufull[cRow + j] as number);
      }
    }
  }

  // Build permutation matrix P from piv.
  const Pfull = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    Pfull[i * m + (piv[i] as number)] = 1;
  }

  // Split the packed factors: L (m x k, unit diagonal) and U (k x n).
  const Lfull = new Float64Array(m * k);
  for (let i = 0; i < m; i++) {
    const lim = Math.min(i, k);
    for (let j = 0; j < lim; j++) Lfull[i * k + j] = Ufull[i * n + j] as number;
    if (i < k) Lfull[i * k + i] = 1;
  }
  const U = new Float64Array(k * n);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < n; j++) U[i * n + j] = Ufull[i * n + j] as number;
  }

  return [
    fromDenseMatrix2D(m, m, Pfull),
    fromDenseMatrix2D(m, k, Lfull),
    fromDenseMatrix2D(k, n, U),
  ];
}
