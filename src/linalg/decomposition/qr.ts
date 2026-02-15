import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { at, fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * QR decomposition.
 *
 * Factorizes matrix A into Q * R where Q is orthogonal and R is upper triangular.
 *
 * **Algorithm**: Householder reflections
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param mode - Decomposition mode:
 *   - 'reduced': Q has shape (M, K), R has shape (K, N) where K = min(M, N)
 *   - 'complete': Q has shape (M, M), R has shape (M, N)
 *
 * **Returns**: [Q, R]
 * - Q: Orthogonal matrix (Q^T * Q = I)
 * - R: Upper triangular matrix
 *
 * **Requirements**:
 * - Input must be 2D matrix
 *
 * **Properties**:
 * - A = Q @ R
 * - Q is orthogonal: Q^T @ Q = I
 * - R is upper triangular
 *
 * @example
 * ```ts
 * import { qr } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]);
 * const [Q, R] = qr(A);
 *
 * // Verify: A ≈ Q @ R
 * // Verify: Q is orthogonal (Q^T @ Q ≈ I)
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 5.2.1
 */
export function qr(a: Tensor, mode: "reduced" | "complete" = "reduced"): [Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "qr()");
  const n = getDim(a, 1, "qr()");
  const minDim = Math.min(m, n);

  // Convert to dense arrays for processing
  const { data: A_data } = toDenseMatrix2D(a);

  // R starts as a copy of A
  // We use Float64Array for precision
  const R = new Float64Array(A_data);

  // Q starts as Identity (m x m)
  const Q = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    Q[i * m + i] = 1.0;
  }

  // Work vectors
  const v = new Float64Array(m);
  const w = new Float64Array(m);

  // Householder reflections
  // Iterate over columns to zero out elements below diagonal
  for (let k = 0; k < minDim; k++) {
    // 1. Compute Householder vector v for column k of R
    // x = R[k:m, k]
    let normXSq = 0;
    for (let i = k; i < m; i++) {
      const val = at(R, i * n + k);
      normXSq += val * val;
    }

    // If column is already close to zero, skip
    // This handles rank-deficient matrices gracefully
    if (normXSq < 1e-15) continue;

    const normX = Math.sqrt(normXSq);
    const x0 = at(R, k * n + k);

    // Choose sign to avoid catastrophic cancellation: alpha = -sign(x0) * ||x||
    const sign = x0 >= 0 ? 1 : -1;
    const alpha = -sign * normX;

    // Construct v:
    // v = x - alpha * e1
    // v[0] = x0 - alpha
    // v[1..] = x[1..]
    // Then normalize v

    // We compute v directly in the work array
    // Reset v to 0 first (important for i < k)
    v.fill(0);

    v[k] = x0 - alpha;
    for (let i = k + 1; i < m; i++) {
      v[i] = at(R, i * n + k);
    }

    // Normalize v
    let normVSq = 0;
    for (let i = k; i < m; i++) {
      const vi = at(v, i);
      normVSq += vi * vi;
    }
    const normV = Math.sqrt(normVSq);

    if (normV < 1e-15) continue;

    const invNormV = 1.0 / normV;
    for (let i = k; i < m; i++) {
      v[i] = at(v, i) * invNormV;
    }

    // 2. Apply H to R: R = H * R = (I - 2vv^T) R = R - 2 v (v^T R)
    // Update submatrix R[k:m, k:n]
    for (let j = k; j < n; j++) {
      // Compute dot product: v^T * R[:, j]
      let dot = 0;
      for (let i = k; i < m; i++) {
        dot += at(v, i) * at(R, i * n + j);
      }

      // R[:, j] -= 2 * dot * v
      for (let i = k; i < m; i++) {
        R[i * n + j] = at(R, i * n + j) - 2 * dot * at(v, i);
      }
    }

    // Force sub-diagonal elements in column k to zero (clean up numerical noise)
    R[k * n + k] = alpha;
    for (let i = k + 1; i < m; i++) {
      R[i * n + k] = 0;
    }

    // 3. Apply H to Q: Q = Q * H^T = Q * H (since H is symmetric)
    // Q = Q (I - 2vv^T) = Q - 2 (Q v) v^T
    // Update Q[:, k:m]

    // Compute w = Q[:, k:m] * v[k:m]
    // This is a matrix-vector multiplication: w = Q * v
    // But v is only non-zero from k to m, so we optimize loops
    w.fill(0);
    for (let i = 0; i < m; i++) {
      let dot = 0;
      for (let j = k; j < m; j++) {
        dot += at(Q, i * m + j) * at(v, j);
      }
      w[i] = dot;
    }

    // Q[:, j] -= 2 * w * v[j] for each column j in k..m-1
    for (let j = k; j < m; j++) {
      const vj = at(v, j);
      for (let i = 0; i < m; i++) {
        Q[i * m + j] = at(Q, i * m + j) - 2 * at(w, i) * vj;
      }
    }
  }

  // Return requested mode
  if (mode === "reduced") {
    const k = minDim;

    // Extract first k columns of Q
    const Q_reduced = new Float64Array(m * k);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        Q_reduced[i * k + j] = at(Q, i * m + j);
      }
    }

    // Extract first k rows of R
    const R_reduced = new Float64Array(k * n);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n; j++) {
        R_reduced[i * n + j] = at(R, i * n + j);
      }
    }

    return [fromDenseMatrix2D(m, k, Q_reduced), fromDenseMatrix2D(k, n, R_reduced)];
  }

  // Complete mode: return full Q (m x m) and R (m x n)
  return [fromDenseMatrix2D(m, m, Q), fromDenseMatrix2D(m, n, R)];
}
