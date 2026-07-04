import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";
import { svd } from "./svd";

/**
 * Polar decomposition of a matrix.
 *
 * Decomposes A into A = U * P where:
 * - U is a unitary (orthogonal) matrix
 * - P is a symmetric positive semi-definite matrix (right polar factor)
 *
 * Or A = S * U where:
 * - S is a symmetric positive semi-definite matrix (left polar factor)
 * - U is a unitary (orthogonal) matrix
 *
 * Uses the SVD-based algorithm: given A = W Σ Vᵀ,
 *   U = W Vᵀ (unitary factor)
 *   P = V Σ Vᵀ (right polar factor)
 *   S = W Σ Wᵀ (left polar factor)
 *
 * **Time Complexity**: O(M * N * min(M,N)) dominated by SVD
 *
 * @param a - Input matrix of shape (M, N)
 * @param side - 'right' (default) returns [U, P] where A = U*P,
 *               'left' returns [S, U] where A = S*U
 * @returns [U, P] or [S, U] depending on side parameter
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
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 */
export function polar(a: Tensor, side: "right" | "left" = "right"): [Tensor, Tensor] {
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }

  const m = getDim(a, 0, "polar()");
  const n = getDim(a, 1, "polar()");

  // Compute full SVD: A = W * diag(sigma) * Vt
  const [W, sigma, Vt] = svd(a, false);
  const wMat = toDenseMatrix2D(W);
  const vtMat = toDenseMatrix2D(Vt);
  const k = Math.min(m, n);

  // Read singular values
  const sigmaArr = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    sigmaArr[i] = Number(sigma.data[sigma.offset + i]);
  }

  if (side === "right") {
    // U = W * Vt  (m x n)
    const uData = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum +=
            (wMat.data[i * wMat.cols + p] as number) * (vtMat.data[p * vtMat.cols + j] as number);
        }
        uData[i * n + j] = sum;
      }
    }

    // P = V * diag(sigma) * Vt  (n x n)
    // V = Vt^T
    const pData = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          // V[i,p] = Vt[p,i]
          sum +=
            (vtMat.data[p * vtMat.cols + i] as number) *
            (sigmaArr[p] as number) *
            (vtMat.data[p * vtMat.cols + j] as number);
        }
        pData[i * n + j] = sum;
      }
    }

    return [fromDenseMatrix2D(m, n, uData), fromDenseMatrix2D(n, n, pData)];
  } else {
    // Left: A = S * U
    // U = W * Vt  (m x n)
    const uData = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum +=
            (wMat.data[i * wMat.cols + p] as number) * (vtMat.data[p * vtMat.cols + j] as number);
        }
        uData[i * n + j] = sum;
      }
    }

    // S = W * diag(sigma) * Wt  (m x m)
    const sData = new Float64Array(m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum +=
            (wMat.data[i * wMat.cols + p] as number) *
            (sigmaArr[p] as number) *
            (wMat.data[j * wMat.cols + p] as number);
        }
        sData[i * m + j] = sum;
      }
    }

    return [fromDenseMatrix2D(m, m, sData), fromDenseMatrix2D(m, n, uData)];
  }
}
