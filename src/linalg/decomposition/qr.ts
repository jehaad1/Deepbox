import { InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";
import { extremeScaleFactor, normalizeScaled, scaledNorm } from "./hessenberg";

/**
 * QR decomposition.
 *
 * Factorizes matrix A into Q * R where Q is orthogonal and R is upper triangular.
 *
 * **Algorithm**: Householder reflections, with the sign convention of LAPACK
 * (and therefore NumPy): the reflection for column k maps it to
 * `-sign(x₀) * ‖x‖ * e₁`, and a column that is already zero below the diagonal
 * is left alone. As a result the diagonal of R is not necessarily positive.
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param mode - Decomposition mode:
 *   - 'reduced' (default): Q has shape (M, K), R has shape (K, N) where K = min(M, N)
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
 * - In 'reduced' mode only the first K columns of Q are formed, so the cost is
 *   O(M * N * K) rather than O(M²) memory for tall inputs.
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
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If mode is not 'reduced' or 'complete'
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 5.2.1
 */
export function qr(a: Tensor, mode: "reduced" | "complete" = "reduced"): [Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  if (mode !== "reduced" && mode !== "complete") {
    throw new InvalidParameterError("mode must be 'reduced' or 'complete'", "mode", mode);
  }

  const m = getDim(a, 0, "qr()");
  const n = getDim(a, 1, "qr()");
  const minDim = Math.min(m, n);

  // R starts as a copy of A and is reduced in place.
  const { data: A_data } = toDenseMatrix2D(a, "qr()");
  const R = new Float64Array(A_data);
  // The reflector updates sum products of entries, which can overflow for entries near the
  // largest double. Q does not depend on a power-of-two scale, and R is rescaled at the end.
  const factor = extremeScaleFactor(R);
  if (factor !== 1) {
    for (let i = 0; i < R.length; i++) R[i] = (R[i] as number) / factor;
  }

  // Householder vectors (unit norm, H = I - 2 v vᵀ), stored column k at V[i * minDim + k] for
  // rows i >= k. A column that needed no reflection keeps an all-zero vector.
  const V = new Float64Array(m * minDim);
  const v = new Float64Array(m);
  const w = new Float64Array(Math.max(n, m));
  const reflected = new Uint8Array(minDim);

  for (let k = 0; k < minDim; k++) {
    const len = m - k;
    let tailMax = 0;
    for (let i = 0; i < len; i++) {
      const val = R[(k + i) * n + k] as number;
      v[i] = val;
      if (i > 0 && Math.abs(val) > tailMax) tailMax = Math.abs(val);
    }
    // Entries below the diagonal are already zero: the reflector is the identity.
    if (tailMax === 0) continue;

    const x0 = v[0] as number;
    const normX = scaledNorm(v, len);
    // alpha = -sign(x0) * ||x|| avoids cancellation in x0 - alpha.
    const alpha = x0 >= 0 ? -normX : normX;
    v[0] = x0 - alpha;
    normalizeScaled(v, len);
    for (let i = 0; i < len; i++) V[(k + i) * minDim + k] = v[i] as number;
    reflected[k] = 1;

    // R[k:, k+1:] -= 2 v (vᵀ R[k:, k+1:])
    for (let j = k + 1; j < n; j++) w[j] = 0;
    for (let i = 0; i < len; i++) {
      const vi = v[i] as number;
      const row = (k + i) * n;
      for (let j = k + 1; j < n; j++) w[j] = (w[j] as number) + vi * (R[row + j] as number);
    }
    for (let i = 0; i < len; i++) {
      const tvi = 2 * (v[i] as number);
      const row = (k + i) * n;
      for (let j = k + 1; j < n; j++) R[row + j] = (R[row + j] as number) - tvi * (w[j] as number);
    }

    // Column k is (alpha, 0, ..., 0) after the reflection.
    R[k * n + k] = alpha;
    for (let i = k + 1; i < m; i++) R[i * n + k] = 0;
  }

  // Q = H_0 H_1 ... H_{K-1} applied to the first qCols columns of the identity.
  // Accumulating backwards touches only columns k.. at step k, so reduced mode
  // never forms the full M x M matrix.
  const qCols = mode === "complete" ? m : minDim;
  const Q = new Float64Array(m * qCols);
  for (let i = 0; i < qCols; i++) Q[i * qCols + i] = 1;

  for (let k = minDim - 1; k >= 0; k--) {
    if (reflected[k] === 0) continue;
    for (let j = k; j < qCols; j++) w[j] = 0;
    for (let i = k; i < m; i++) {
      const vi = V[i * minDim + k] as number;
      if (vi === 0) continue;
      const row = i * qCols;
      for (let j = k; j < qCols; j++) w[j] = (w[j] as number) + vi * (Q[row + j] as number);
    }
    for (let i = k; i < m; i++) {
      const tvi = 2 * (V[i * minDim + k] as number);
      if (tvi === 0) continue;
      const row = i * qCols;
      for (let j = k; j < qCols; j++) Q[row + j] = (Q[row + j] as number) - tvi * (w[j] as number);
    }
  }

  if (factor !== 1) {
    for (let i = 0; i < R.length; i++) R[i] = (R[i] as number) * factor;
  }

  if (mode === "reduced") {
    // Extract first minDim rows of R.
    const R_reduced = R.slice(0, minDim * n);
    return [fromDenseMatrix2D(m, minDim, Q), fromDenseMatrix2D(minDim, n, R_reduced)];
  }

  // Complete mode: return full Q (m x m) and R (m x n)
  return [fromDenseMatrix2D(m, m, Q), fromDenseMatrix2D(m, n, R)];
}
