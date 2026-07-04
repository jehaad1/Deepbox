import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";

/**
 * Real Schur decomposition of a square matrix.
 *
 * Decomposes A into A = Q * T * Qᵀ where:
 * - Q is an orthogonal matrix (Schur vectors)
 * - T is a quasi-upper triangular matrix (Schur form)
 *   - Real eigenvalues appear on the diagonal
 *   - Complex conjugate pairs appear in 2×2 blocks on the diagonal
 *
 * Uses the QR algorithm with implicit double shifts (Francis iteration).
 *
 * **Time Complexity**: O(N³) typical, O(N⁴) worst case
 *
 * @param a - Input square matrix of shape (N, N)
 * @returns [T, Q] where T is the Schur form and Q contains the Schur vectors
 *
 * @example
 * ```ts
 * import { schur } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const [T, Q] = schur(A);
 * // Q * T * Q^T ≈ A
 * // T is quasi-upper triangular
 * ```
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 */
export function schur(a: Tensor): [Tensor, Tensor] {
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }

  const n = getDim(a, 0, "schur()");
  const n2 = getDim(a, 1, "schur()");
  if (n !== n2) {
    throw new ShapeError(`schur() requires a square matrix; got (${n}, ${n2})`);
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
  const T = new Float64Array(mat.data);
  // Q starts as identity
  const Q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;

  // Step 1: Reduce to upper Hessenberg form via Householder reflections
  reduceToHessenberg(T, Q, n);

  // Step 2: QR iteration with implicit double shifts
  qrIteration(T, Q, n);

  return [fromDenseMatrix2D(n, n, T), fromDenseMatrix2D(n, n, Q)];
}

/**
 * Reduce a matrix to upper Hessenberg form using Householder reflections.
 * Modifies T in-place and accumulates transformations in Q.
 */
function reduceToHessenberg(T: Float64Array, Q: Float64Array, n: number): void {
  const v = new Float64Array(n);

  for (let k = 0; k < n - 2; k++) {
    // Build Householder vector for column k, rows k+1..n-1
    const len = n - k - 1;
    let norm = 0;
    for (let i = 0; i < len; i++) {
      const val = T[(k + 1 + i) * n + k] as number;
      v[i] = val;
      norm += val * val;
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-15) continue;

    const sign = (v[0] as number) >= 0 ? 1 : -1;
    v[0] = (v[0] as number) + sign * norm;

    // Normalize v
    let vNorm = 0;
    for (let i = 0; i < len; i++) {
      vNorm += (v[i] as number) * (v[i] as number);
    }
    if (vNorm < 1e-30) continue;
    const invVNorm = Math.sqrt(2 / vNorm);
    for (let i = 0; i < len; i++) {
      v[i] = (v[i] as number) * invVNorm;
    }

    // Apply H from left: T = (I - v*v^T) * T
    // For rows k+1..n-1, columns 0..n-1
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < len; i++) {
        dot += (v[i] as number) * (T[(k + 1 + i) * n + j] as number);
      }
      for (let i = 0; i < len; i++) {
        T[(k + 1 + i) * n + j] = (T[(k + 1 + i) * n + j] as number) - (v[i] as number) * dot;
      }
    }

    // Apply H from right: T = T * (I - v*v^T)
    // For rows 0..n-1, columns k+1..n-1
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < len; j++) {
        dot += (T[i * n + (k + 1 + j)] as number) * (v[j] as number);
      }
      for (let j = 0; j < len; j++) {
        T[i * n + (k + 1 + j)] = (T[i * n + (k + 1 + j)] as number) - dot * (v[j] as number);
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
}

/**
 * QR iteration with implicit single/double shifts on upper Hessenberg matrix.
 * Modifies T and Q in-place.
 */
function qrIteration(T: Float64Array, Q: Float64Array, n: number): void {
  const maxIter = 100 * n;
  let p = n - 1; // Working submatrix is T[0..p, 0..p]
  let iterSinceDeflation = 0;

  for (let iter = 0; iter < maxIter && p > 0; iter++) {
    // Check for convergence: is T[p, p-1] ≈ 0?
    const tol =
      1e-14 * (Math.abs(T[(p - 1) * n + (p - 1)] as number) + Math.abs(T[p * n + p] as number));
    if (Math.abs(T[p * n + (p - 1)] as number) <= Math.max(tol, 1e-15)) {
      T[p * n + (p - 1)] = 0;
      p--;
      iterSinceDeflation = 0;
      continue;
    }

    // Check for 2x2 block deflation
    if (p > 1) {
      const tol2 =
        1e-14 *
        (Math.abs(T[(p - 2) * n + (p - 2)] as number) +
          Math.abs(T[(p - 1) * n + (p - 1)] as number));
      if (Math.abs(T[(p - 1) * n + (p - 2)] as number) <= Math.max(tol2, 1e-15)) {
        T[(p - 1) * n + (p - 2)] = 0;
        p -= 2;
        iterSinceDeflation = 0;
        continue;
      }
    }

    // Wilkinson shift: eigenvalue of bottom-right 2x2 block closest to T[p,p]
    const a11 = T[(p - 1) * n + (p - 1)] as number;
    const a12 = T[(p - 1) * n + p] as number;
    const a21 = T[p * n + (p - 1)] as number;
    const a22 = T[p * n + p] as number;

    const tr = a11 + a22;
    const det = a11 * a22 - a12 * a21;
    const disc = tr * tr - 4 * det;

    let shift: number;
    if (disc >= 0) {
      const sqrtDisc = Math.sqrt(disc);
      const e1 = (tr + sqrtDisc) / 2;
      const e2 = (tr - sqrtDisc) / 2;
      shift = Math.abs(e1 - a22) < Math.abs(e2 - a22) ? e1 : e2;
    } else {
      shift = a22;
    }

    // Exceptional shift: the Wilkinson shift stalls on matrices whose active
    // block has purely imaginary or symmetric spectra (e.g. cyclic
    // permutations), leaving two nonzero subdiagonals forever. Every 10
    // iterations without deflation, perturb the shift by the local
    // subdiagonal magnitude (Francis/LAPACK strategy) to break the cycle.
    iterSinceDeflation++;
    if (iterSinceDeflation % 10 === 0) {
      const lowerSub = p >= 2 ? Math.abs(T[(p - 1) * n + (p - 2)] as number) : 0;
      shift = a22 + Math.abs(T[p * n + (p - 1)] as number) + lowerSub;
    }

    // Single-shift QR step on the active submatrix
    qrStep(T, Q, n, 0, p, shift);
  }

  // Clean up small subdiagonal entries
  for (let i = 1; i < n; i++) {
    const tol =
      1e-14 * (Math.abs(T[(i - 1) * n + (i - 1)] as number) + Math.abs(T[i * n + i] as number));
    if (Math.abs(T[i * n + (i - 1)] as number) <= Math.max(tol, 1e-15)) {
      T[i * n + (i - 1)] = 0;
    }
  }
}

/**
 * Perform one implicit single-shift QR step with Givens rotations.
 */
function qrStep(
  T: Float64Array,
  Q: Float64Array,
  n: number,
  lo: number,
  hi: number,
  shift: number
): void {
  // Bulge chase using Givens rotations
  let x = (T[lo * n + lo] as number) - shift;
  let y = T[(lo + 1) * n + lo] as number;

  for (let k = lo; k < hi; k++) {
    // Compute Givens rotation to zero out y
    const r = Math.sqrt(x * x + y * y);
    if (r < 1e-30) break;
    const c = x / r;
    const s = -y / r;

    // Apply rotation from left: rows k, k+1
    for (let j = 0; j < n; j++) {
      const tk = T[k * n + j] as number;
      const tk1 = T[(k + 1) * n + j] as number;
      T[k * n + j] = c * tk - s * tk1;
      T[(k + 1) * n + j] = s * tk + c * tk1;
    }

    // Apply rotation from right: columns k, k+1
    for (let i = 0; i < n; i++) {
      const tk = T[i * n + k] as number;
      const tk1 = T[i * n + (k + 1)] as number;
      T[i * n + k] = c * tk - s * tk1;
      T[i * n + (k + 1)] = s * tk + c * tk1;
    }

    // Accumulate in Q
    for (let i = 0; i < n; i++) {
      const qk = Q[i * n + k] as number;
      const qk1 = Q[i * n + (k + 1)] as number;
      Q[i * n + k] = c * qk - s * qk1;
      Q[i * n + (k + 1)] = s * qk + c * qk1;
    }

    if (k + 2 <= hi) {
      x = T[(k + 1) * n + k] as number;
      y = T[(k + 2) * n + k] as number;
    }
  }
}
