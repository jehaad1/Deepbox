import { InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D, toDenseVector1D } from "../_internal";
import { svd } from "./svd";

/**
 * Polar decomposition of a matrix.
 *
 * Decomposes A into A = U * P where:
 * - U has orthonormal columns (a unitary matrix when A is square, M >= N) or
 *   orthonormal rows (when M < N)
 * - P is a symmetric positive semi-definite matrix (right polar factor)
 *
 * Or A = S * U where:
 * - S is a symmetric positive semi-definite matrix (left polar factor)
 * - U is as above
 *
 * Uses the SVD-based algorithm: given the reduced SVD A = W Σ Vᵀ,
 *   U = W Vᵀ (orthogonal factor)
 *   P = V Σ Vᵀ (right polar factor)
 *   S = W Σ Wᵀ (left polar factor)
 *
 * For a rank-deficient A the orthogonal factor U is not unique; the one returned
 * follows from the singular vectors that belong to the zero singular values.
 *
 * **Time Complexity**: O(M * N * min(M,N)) dominated by SVD
 *
 * @param a - Input matrix of shape (M, N)
 * @param side - 'right' (default) returns [U, P] where A = U*P,
 *               'left' returns [S, U] where A = S*U
 * @returns [U, P] of shapes (M, N) and (N, N), or [S, U] of shapes (M, M) and (M, N)
 *
 * @example
 * ```ts
 * import { polar } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const [U, P] = polar(A);
 * // U is unitary: U * U^T ≈ I
 * // P is positive semi-definite
 * // U * P ≈ A
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If side is not 'right' or 'left'
 * @throws {ConvergenceError} If the underlying SVD does not converge
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 */
export function polar(a: Tensor, side: "right" | "left" = "right"): [Tensor, Tensor] {
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }
  if (side !== "right" && side !== "left") {
    throw new InvalidParameterError("side must be 'right' or 'left'", "side", side);
  }

  const m = getDim(a, 0, "polar()");
  const n = getDim(a, 1, "polar()");

  // Reduced SVD: A = W * diag(sigma) * Vt, with W (m x k) and Vt (k x n).
  const [W, sigma, Vt] = svd(a, false);
  const wMat = toDenseMatrix2D(W).data;
  const vtMat = toDenseMatrix2D(Vt).data;
  const sigmaArr = toDenseVector1D(sigma);
  const k = Math.min(m, n);

  // U = W * Vt  (m x n)
  const uData = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const wip = wMat[i * k + p] as number;
      if (wip === 0) continue;
      for (let j = 0; j < n; j++) {
        uData[i * n + j] = (uData[i * n + j] as number) + wip * (vtMat[p * n + j] as number);
      }
    }
  }

  if (side === "right") {
    // P = V * diag(sigma) * Vt  (n x n), symmetric by construction.
    const pData = new Float64Array(n * n);
    for (let p = 0; p < k; p++) {
      const sp = sigmaArr[p] as number;
      for (let i = 0; i < n; i++) {
        const vi = (vtMat[p * n + i] as number) * sp;
        if (vi === 0) continue;
        for (let j = i; j < n; j++) {
          pData[i * n + j] = (pData[i * n + j] as number) + vi * (vtMat[p * n + j] as number);
        }
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) pData[j * n + i] = pData[i * n + j] as number;
    }
    return [fromDenseMatrix2D(m, n, uData), fromDenseMatrix2D(n, n, pData)];
  }

  // Left: A = S * U, with S = W * diag(sigma) * Wt  (m x m), symmetric by construction.
  const sData = new Float64Array(m * m);
  for (let p = 0; p < k; p++) {
    const sp = sigmaArr[p] as number;
    for (let i = 0; i < m; i++) {
      const wi = (wMat[i * k + p] as number) * sp;
      if (wi === 0) continue;
      for (let j = i; j < m; j++) {
        sData[i * m + j] = (sData[i * m + j] as number) + wi * (wMat[j * k + p] as number);
      }
    }
  }
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) sData[j * m + i] = sData[i * m + j] as number;
  }
  return [fromDenseMatrix2D(m, m, sData), fromDenseMatrix2D(m, n, uData)];
}
