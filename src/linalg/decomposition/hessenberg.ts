/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * Upper Hessenberg decomposition of a square matrix.
 *
 * Decomposes A into A = Q * H * Qᵀ where:
 * - Q is an orthogonal matrix
 * - H is an upper Hessenberg matrix (zero below the first sub-diagonal)
 *
 * Uses Householder reflections.
 *
 * **Time Complexity**: O(N³)
 *
 * @param a - Input square matrix of shape (N, N)
 * @returns [H, Q] where H is upper Hessenberg and Q is orthogonal
 *
 * @example
 * ```ts
 * import { hessenberg } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * const [H, Q] = hessenberg(A);
 * // Q * H * Q^T ≈ A
 * // H is upper Hessenberg (zero below first sub-diagonal)
 * ```
 */
export function hessenberg(a: Tensor): [Tensor, Tensor] {
  if (a.ndim !== 2) {
    throw new ShapeError("hessenberg() requires a 2D matrix");
  }

  const n = getDim(a, 0, "hessenberg()");
  const n2 = getDim(a, 1, "hessenberg()");
  if (n !== n2) {
    throw new ShapeError(`hessenberg() requires a square matrix; got (${n}, ${n2})`);
  }

  if (n === 0) {
    return [
      fromDenseMatrix2D(0, 0, new Float64Array(0)),
      fromDenseMatrix2D(0, 0, new Float64Array(0)),
    ];
  }

  if (n === 1) {
    const mat = toDenseMatrix2D(a);
    return [
      fromDenseMatrix2D(1, 1, new Float64Array([mat.data[0] as number])),
      fromDenseMatrix2D(1, 1, new Float64Array([1])),
    ];
  }

  const mat = toDenseMatrix2D(a);
  const H = new Float64Array(mat.data);
  const Q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;

  const v = new Float64Array(n);

  for (let k = 0; k < n - 2; k++) {
    const len = n - k - 1;
    let norm = 0;
    for (let i = 0; i < len; i++) {
      const val = H[(k + 1 + i) * n + k] as number;
      v[i] = val;
      norm += val * val;
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-15) continue;

    const sign = (v[0] as number) >= 0 ? 1 : -1;
    v[0] = (v[0] as number) + sign * norm;

    let vNorm = 0;
    for (let i = 0; i < len; i++) {
      vNorm += (v[i] as number) * (v[i] as number);
    }
    if (vNorm < 1e-30) continue;
    const invVNorm = Math.sqrt(2 / vNorm);
    for (let i = 0; i < len; i++) {
      v[i] = (v[i] as number) * invVNorm;
    }

    // Apply H_k from left: H = (I - v*v^T) * H
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < len; i++) {
        dot += (v[i] as number) * (H[(k + 1 + i) * n + j] as number);
      }
      for (let i = 0; i < len; i++) {
        H[(k + 1 + i) * n + j] = (H[(k + 1 + i) * n + j] as number) - (v[i] as number) * dot;
      }
    }

    // Apply H_k from right: H = H * (I - v*v^T)
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < len; j++) {
        dot += (H[i * n + (k + 1 + j)] as number) * (v[j] as number);
      }
      for (let j = 0; j < len; j++) {
        H[i * n + (k + 1 + j)] = (H[i * n + (k + 1 + j)] as number) - dot * (v[j] as number);
      }
    }

    // Accumulate in Q
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < len; j++) {
        dot += (Q[i * n + (k + 1 + j)] as number) * (v[j] as number);
      }
      for (let j = 0; j < len; j++) {
        Q[i * n + (k + 1 + j)] = (Q[i * n + (k + 1 + j)] as number) - dot * (v[j] as number);
      }
    }
  }

  // Clean up sub-sub-diagonal entries
  for (let i = 2; i < n; i++) {
    for (let j = 0; j < i - 1; j++) {
      H[i * n + j] = 0;
    }
  }

  return [fromDenseMatrix2D(n, n, H), fromDenseMatrix2D(n, n, Q)];
}
