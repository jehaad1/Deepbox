/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * Euclidean norm of `v[0..len)` computed with scaling, so it neither overflows
 * for entries near 1e200 nor underflows for entries near 1e-200.
 *
 * @internal
 */
export function scaledNorm(v: Float64Array, len: number): number {
  let max = 0;
  for (let i = 0; i < len; i++) {
    const a = Math.abs(v[i] as number);
    if (a > max) max = a;
  }
  if (max === 0) return 0;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const t = (v[i] as number) / max;
    sum += t * t;
  }
  return max * Math.sqrt(sum);
}

/**
 * Scales `v[0..len)` to unit Euclidean norm in place. The entries are first divided by the
 * largest magnitude, so the result stays orthonormal to full precision even when the
 * norm itself is a denormal number (where dividing by it would lose most digits).
 * Does nothing when all entries are zero.
 *
 * @internal
 */
export function normalizeScaled(v: Float64Array, len: number): void {
  let max = 0;
  for (let i = 0; i < len; i++) {
    const a = Math.abs(v[i] as number);
    if (a > max) max = a;
  }
  if (max === 0) return;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const t = (v[i] as number) / max;
    v[i] = t;
    sum += t * t;
  }
  const norm = Math.sqrt(sum);
  for (let i = 0; i < len; i++) v[i] = (v[i] as number) / norm;
}

/**
 * Returns a power of two `f` such that dividing the entries of `A` by `f` brings
 * the largest magnitude close to 1, or 1 when the entries are already in a safe
 * range (1e-100 to 1e100). Iterative eigenvalue algorithms form products and
 * squares of entries, which underflow to zero for entries around 1e-200 and
 * overflow for entries around 1e200; scaling by a power of two is exact.
 *
 * @internal
 */
export function extremeScaleFactor(A: Float64Array): number {
  let maxAbs = 0;
  for (let i = 0; i < A.length; i++) {
    const v = Math.abs(A[i] as number);
    if (v > maxAbs) maxAbs = v;
  }
  if (maxAbs > 0 && (maxAbs > 1e100 || maxAbs < 1e-100)) {
    // floor keeps the factor finite for maxAbs near the largest double (2 ** 1024 overflows).
    return 2 ** Math.floor(Math.log2(maxAbs));
  }
  return 1;
}

/**
 * Reduces the row-major n x n matrix `H` to upper Hessenberg form in place with
 * Householder reflections, H <- Q^T H Q. When `Q` is given it must hold an
 * n x n matrix (usually the identity) and is overwritten with `Q * H_0 * H_1 ...`.
 *
 * A column whose entries below the sub-diagonal are already zero is left
 * untouched, which makes the reduction scale-invariant (no absolute thresholds).
 * Entries below the first sub-diagonal are set to exact zeros on exit.
 *
 * @internal
 */
export function hessenbergReduceInPlace(H: Float64Array, Q: Float64Array | null, n: number): void {
  const v = new Float64Array(n);
  const w = new Float64Array(n);

  for (let k = 0; k < n - 2; k++) {
    const len = n - k - 1;
    let tailMax = 0;
    for (let i = 0; i < len; i++) {
      const val = H[(k + 1 + i) * n + k] as number;
      v[i] = val;
      if (i > 0 && Math.abs(val) > tailMax) tailMax = Math.abs(val);
    }
    // Nothing below the sub-diagonal to eliminate: the reflector would be the identity.
    if (tailMax === 0) continue;

    const alpha = v[0] as number;
    const norm = scaledNorm(v, len);
    const beta = alpha >= 0 ? -norm : norm;
    v[0] = alpha - beta;
    normalizeScaled(v, len);

    // Left update H[k+1:, k+1:] -= 2 v (v^T H[k+1:, k+1:]); column k is set below.
    const c0 = k + 1;
    for (let j = c0; j < n; j++) w[j] = 0;
    for (let i = 0; i < len; i++) {
      const vi = v[i] as number;
      const row = (k + 1 + i) * n;
      for (let j = c0; j < n; j++) w[j] = (w[j] as number) + vi * (H[row + j] as number);
    }
    for (let i = 0; i < len; i++) {
      const tvi = 2 * (v[i] as number);
      const row = (k + 1 + i) * n;
      for (let j = c0; j < n; j++) {
        H[row + j] = (H[row + j] as number) - tvi * (w[j] as number);
      }
    }

    // Right update H[:, k+1:] -= 2 (H[:, k+1:] v) v^T.
    for (let i = 0; i < n; i++) {
      const row = i * n;
      let dot = 0;
      for (let j = 0; j < len; j++) dot += (H[row + c0 + j] as number) * (v[j] as number);
      dot *= 2;
      for (let j = 0; j < len; j++) {
        H[row + c0 + j] = (H[row + c0 + j] as number) - dot * (v[j] as number);
      }
    }

    // Column k after the reflection is (beta, 0, ..., 0) below the diagonal.
    H[(k + 1) * n + k] = beta;
    for (let i = k + 2; i < n; i++) H[i * n + k] = 0;

    if (Q !== null) {
      for (let i = 0; i < n; i++) {
        const row = i * n;
        let dot = 0;
        for (let j = 0; j < len; j++) dot += (Q[row + c0 + j] as number) * (v[j] as number);
        dot *= 2;
        for (let j = 0; j < len; j++) {
          Q[row + c0 + j] = (Q[row + c0 + j] as number) - dot * (v[j] as number);
        }
      }
    }
  }
}

/**
 * Upper Hessenberg decomposition of a square matrix.
 *
 * Decomposes A into A = Q * H * Qᵀ where:
 * - Q is an orthogonal matrix
 * - H is an upper Hessenberg matrix (zero below the first sub-diagonal)
 *
 * Uses Householder reflections. Unlike `scipy.linalg.hessenberg`, both factors
 * are always returned.
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
 *
 * @throws {ShapeError} If input is not a square 2D matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
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

  const mat = toDenseMatrix2D(a, "hessenberg()");
  const H = new Float64Array(mat.data);
  const Q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;

  // Reflector updates sum products of entries; a power-of-two scale keeps them finite for
  // entries near the largest double. Q is unaffected and H is rescaled afterwards.
  const factor = extremeScaleFactor(H);
  if (factor !== 1) {
    for (let i = 0; i < H.length; i++) H[i] = (H[i] as number) / factor;
  }
  hessenbergReduceInPlace(H, Q, n);
  if (factor !== 1) {
    for (let i = 0; i < H.length; i++) H[i] = (H[i] as number) * factor;
  }

  return [fromDenseMatrix2D(n, n, H), fromDenseMatrix2D(n, n, Q)];
}
