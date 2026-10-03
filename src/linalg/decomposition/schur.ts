import { ConvergenceError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, getDim, toDenseMatrix2D } from "../_internal";
import { extremeScaleFactor, hessenbergReduceInPlace } from "./hessenberg";

/** Fortran SIGN(a, b): |a| with the sign of b. */
function sign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

type Block2x2 = {
  a: number;
  b: number;
  c: number;
  d: number;
  cs: number;
  sn: number;
  rt1r: number;
  rt1i: number;
  rt2r: number;
  rt2i: number;
};

/**
 * Schur factorization of a real 2x2 block in standardized form (LAPACK dlanv2).
 *
 * Returns the rotation (cs, sn) such that
 * [[a, b], [c, d]] = [[cs, -sn], [sn, cs]] * [[aa, bb], [cc, dd]] * [[cs, sn], [-sn, cs]]
 * where the middle block is upper triangular (real eigenvalues) or has equal
 * diagonal entries and off-diagonal entries of opposite sign (complex pair).
 *
 * @internal
 */
function standardizeBlock(a0: number, b0: number, c0: number, d0: number): Block2x2 {
  const eps = Number.EPSILON;
  let a = a0;
  let b = b0;
  let c = c0;
  let d = d0;
  let cs: number;
  let sn: number;

  if (c === 0) {
    cs = 1;
    sn = 0;
  } else if (b === 0) {
    // Swap rows and columns.
    cs = 0;
    sn = 1;
    const tmp = d;
    d = a;
    a = tmp;
    b = -c;
    c = 0;
  } else if (a - d === 0 && sign(1, b) !== sign(1, c)) {
    cs = 1;
    sn = 0;
  } else {
    let temp = a - d;
    let p = 0.5 * temp;
    const bcmax = Math.max(Math.abs(b), Math.abs(c));
    const bcmis = Math.min(Math.abs(b), Math.abs(c)) * sign(1, b) * sign(1, c);
    const scale = Math.max(Math.abs(p), bcmax);
    let z = (p / scale) * p + (bcmax / scale) * bcmis;

    if (z >= 4 * eps) {
      // Real eigenvalues.
      z = p + sign(Math.sqrt(scale) * Math.sqrt(z), p);
      a = d + z;
      d -= (bcmax / z) * bcmis;
      const tau = Math.hypot(c, z);
      cs = z / tau;
      sn = c / tau;
      b -= c;
      c = 0;
    } else {
      // Complex eigenvalues, or real (almost) equal eigenvalues: make the
      // diagonal entries equal.
      const sigma = b + c;
      const tau = Math.hypot(sigma, temp);
      cs = Math.sqrt(0.5 * (1 + Math.abs(sigma) / tau));
      sn = -(p / (tau * cs)) * sign(1, sigma);
      const aa = a * cs + b * sn;
      const bb = -a * sn + b * cs;
      const cc = c * cs + d * sn;
      const dd = -c * sn + d * cs;
      a = aa * cs + cc * sn;
      b = bb * cs + dd * sn;
      c = -aa * sn + cc * cs;
      d = -bb * sn + dd * cs;
      temp = 0.5 * (a + d);
      a = temp;
      d = temp;
      if (c !== 0) {
        if (b !== 0) {
          if (sign(1, b) === sign(1, c)) {
            // Real eigenvalues: reduce to upper triangular form.
            const sab = Math.sqrt(Math.abs(b));
            const sac = Math.sqrt(Math.abs(c));
            p = sign(sab * sac, c);
            const tau2 = 1 / Math.sqrt(Math.abs(b + c));
            a = temp + p;
            d = temp - p;
            b -= c;
            c = 0;
            const cs1 = sab * tau2;
            const sn1 = sac * tau2;
            const t = cs * cs1 - sn * sn1;
            sn = cs * sn1 + sn * cs1;
            cs = t;
          }
        } else {
          b = -c;
          c = 0;
          const t = cs;
          cs = -sn;
          sn = t;
        }
      }
    }
  }

  const rt1i = c === 0 ? 0 : Math.sqrt(Math.abs(b)) * Math.sqrt(Math.abs(c));
  return { a, b, c, d, cs, sn, rt1r: a, rt1i, rt2r: d, rt2i: -rt1i };
}

/**
 * Francis double-shift QR iteration on an upper Hessenberg matrix (EISPACK
 * hqr2 without the eigenvector phase).
 *
 * `H` (row-major n x n, upper Hessenberg on entry) is overwritten with the real
 * Schur form T: upper triangular except for 2x2 blocks on the diagonal that
 * hold complex conjugate pairs. 2x2 blocks with real eigenvalues are split,
 * and complex blocks are standardized (equal diagonal entries). When `Z` is
 * given (n x n) the orthogonal transformations are accumulated into it.
 *
 * @param eps - Relative threshold for treating a sub-diagonal entry as zero
 * @param maxIter - Maximum QR sweeps spent on one eigenvalue (or 2x2 block)
 * @returns Real and imaginary parts of the eigenvalues, in Schur-form order
 * @throws {ConvergenceError} If a sub-diagonal entry does not become negligible
 *
 * @internal
 */
export function realSchurIteration(
  H: Float64Array,
  Z: Float64Array | null,
  n: number,
  maxIter: number,
  eps: number,
  context: string
): { readonly wr: Float64Array; readonly wi: Float64Array } {
  const wr = new Float64Array(n);
  const wi = new Float64Array(n);

  let norm = 0;
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(i - 1, 0); j < n; j++) norm += Math.abs(H[i * n + j] as number);
  }
  if (norm === 0) return { wr, wi };
  // Sub-diagonal entries below this are treated as zero (LAPACK's smlnum).
  const smallNumber = 2.2250738585072014e-308 * (n / Number.EPSILON);

  let exshift = 0;
  let p = 0;
  let q = 0;
  let r = 0;
  let s = 0;
  let z = 0;
  let x = 0;
  let y = 0;
  let w = 0;
  let iter = 0;
  let hi = n - 1;

  while (hi >= 0) {
    // Look for a single small sub-diagonal element.
    let l = hi;
    while (l > 0) {
      const sub = Math.abs(H[l * n + (l - 1)] as number);
      // Entries at the underflow limit carry no information and would stall the
      // iteration (graded matrices such as a Hessenberg form of the all-ones matrix).
      if (sub <= smallNumber) break;
      s = Math.abs(H[(l - 1) * n + (l - 1)] as number) + Math.abs(H[l * n + l] as number);
      if (s === 0) s = norm;
      if (sub < eps * s) break;
      l--;
    }
    if (l > 0) H[l * n + (l - 1)] = 0;

    if (l === hi) {
      // One root found.
      H[hi * n + hi] = (H[hi * n + hi] as number) + exshift;
      wr[hi] = H[hi * n + hi] as number;
      wi[hi] = 0;
      hi--;
      iter = 0;
    } else if (l === hi - 1) {
      // Two roots found: split or standardize the 2x2 block.
      const i0 = hi - 1;
      const i1 = hi;
      const blk = standardizeBlock(
        (H[i0 * n + i0] as number) + exshift,
        H[i0 * n + i1] as number,
        H[i1 * n + i0] as number,
        (H[i1 * n + i1] as number) + exshift
      );
      H[i0 * n + i0] = blk.a;
      H[i0 * n + i1] = blk.b;
      H[i1 * n + i0] = blk.c;
      H[i1 * n + i1] = blk.d;
      wr[i0] = blk.rt1r;
      wi[i0] = blk.rt1i;
      wr[i1] = blk.rt2r;
      wi[i1] = blk.rt2i;
      const { cs, sn } = blk;
      if (cs !== 1 || sn !== 0) {
        // Rows i0, i1 to the right of the block.
        for (let j = i1 + 1; j < n; j++) {
          const t0 = H[i0 * n + j] as number;
          const t1 = H[i1 * n + j] as number;
          H[i0 * n + j] = cs * t0 + sn * t1;
          H[i1 * n + j] = cs * t1 - sn * t0;
        }
        // Columns i0, i1 above the block.
        for (let i = 0; i < i0; i++) {
          const t0 = H[i * n + i0] as number;
          const t1 = H[i * n + i1] as number;
          H[i * n + i0] = cs * t0 + sn * t1;
          H[i * n + i1] = cs * t1 - sn * t0;
        }
        if (Z !== null) {
          for (let i = 0; i < n; i++) {
            const t0 = Z[i * n + i0] as number;
            const t1 = Z[i * n + i1] as number;
            Z[i * n + i0] = cs * t0 + sn * t1;
            Z[i * n + i1] = cs * t1 - sn * t0;
          }
        }
      }
      if (i0 > 0) H[i0 * n + (i0 - 1)] = 0;
      hi -= 2;
      iter = 0;
    } else {
      // No convergence yet: form the shift.
      x = H[hi * n + hi] as number;
      y = 0;
      w = 0;
      if (l < hi) {
        y = H[(hi - 1) * n + (hi - 1)] as number;
        w = (H[hi * n + (hi - 1)] as number) * (H[(hi - 1) * n + hi] as number);
      }

      // Exceptional shift (Wilkinson's ad hoc shift).
      if (iter === 10) {
        exshift += x;
        for (let i = 0; i <= hi; i++) H[i * n + i] = (H[i * n + i] as number) - x;
        s =
          Math.abs(H[hi * n + (hi - 1)] as number) + Math.abs(H[(hi - 1) * n + (hi - 2)] as number);
        x = 0.75 * s;
        y = x;
        w = -0.4375 * s * s;
      }

      // Second exceptional shift (the one used by MATLAB).
      if (iter === 30) {
        s = (y - x) / 2;
        s = s * s + w;
        if (s > 0) {
          s = Math.sqrt(s);
          if (y < x) s = -s;
          s = x - w / ((y - x) / 2 + s);
          for (let i = 0; i <= hi; i++) H[i * n + i] = (H[i * n + i] as number) - s;
          exshift += s;
          x = 0.964;
          y = x;
          w = x;
        }
      }

      iter++;
      if (iter > maxIter) {
        throw new ConvergenceError(`${context} failed to converge after ${maxIter} iterations`, {
          iterations: maxIter,
          tolerance: eps,
        });
      }

      // Shifts: the eigenvalues of the trailing 2x2 block [[y, *], [*, x]] whose
      // off-diagonal product is w (a real pair, or a complex conjugate pair).
      const halfTrace = 0.5 * (x + y);
      const disc = (0.5 * (y - x)) ** 2 + w;
      let rt1r = halfTrace;
      let rt2r = halfTrace;
      let rt1i = 0;
      let rt2i = 0;
      if (disc >= 0) {
        const root = Math.sqrt(disc);
        rt1r = halfTrace + root;
        rt2r = halfTrace - root;
      } else {
        rt1i = Math.sqrt(-disc);
        rt2i = -rt1i;
      }

      // Look for two consecutive small sub-diagonal elements. The first column of
      // (H - rt1)(H - rt2) is divided by the sub-diagonal entry before it is formed
      // (as in LAPACK dlahqr), so graded matrices with tiny sub-diagonal entries
      // cannot overflow.
      let m = hi - 2;
      while (m >= l) {
        z = H[m * n + m] as number;
        let h21s = H[(m + 1) * n + m] as number;
        s = Math.abs(z - rt2r) + Math.abs(rt2i) + Math.abs(h21s);
        h21s /= s;
        p =
          h21s * (H[m * n + (m + 1)] as number) + (z - rt1r) * ((z - rt2r) / s) - rt1i * (rt2i / s);
        q = h21s * (z + (H[(m + 1) * n + (m + 1)] as number) - rt1r - rt2r);
        r = h21s * (H[(m + 2) * n + (m + 1)] as number);
        s = Math.abs(p) + Math.abs(q) + Math.abs(r);
        if (s !== 0) {
          p /= s;
          q /= s;
          r /= s;
        }
        if (m === l) break;
        if (
          Math.abs(H[m * n + (m - 1)] as number) * (Math.abs(q) + Math.abs(r)) <
          eps *
            (Math.abs(p) *
              (Math.abs(H[(m - 1) * n + (m - 1)] as number) +
                Math.abs(z) +
                Math.abs(H[(m + 1) * n + (m + 1)] as number)))
        ) {
          break;
        }
        m--;
      }

      for (let i = m + 2; i <= hi; i++) {
        H[i * n + (i - 2)] = 0;
        if (i > m + 2) H[i * n + (i - 3)] = 0;
      }

      // Double QR step involving rows l..hi and columns m..hi.
      for (let k = m; k <= hi - 1; k++) {
        const notLast = k !== hi - 1;
        if (k !== m) {
          p = H[k * n + (k - 1)] as number;
          q = H[(k + 1) * n + (k - 1)] as number;
          r = notLast ? (H[(k + 2) * n + (k - 1)] as number) : 0;
          x = Math.abs(p) + Math.abs(q) + Math.abs(r);
          if (x === 0) continue;
          p /= x;
          q /= x;
          r /= x;
        }
        s = Math.sqrt(p * p + q * q + r * r);
        if (p < 0) s = -s;
        if (s !== 0) {
          if (k !== m) {
            H[k * n + (k - 1)] = -s * x;
          } else if (l !== m) {
            H[k * n + (k - 1)] = -(H[k * n + (k - 1)] as number);
          }
          p += s;
          x = p / s;
          y = q / s;
          z = r / s;
          q /= p;
          r /= p;

          // Row modification.
          for (let j = k; j < n; j++) {
            p = (H[k * n + j] as number) + q * (H[(k + 1) * n + j] as number);
            if (notLast) {
              p += r * (H[(k + 2) * n + j] as number);
              H[(k + 2) * n + j] = (H[(k + 2) * n + j] as number) - p * z;
            }
            H[k * n + j] = (H[k * n + j] as number) - p * x;
            H[(k + 1) * n + j] = (H[(k + 1) * n + j] as number) - p * y;
          }

          // Column modification.
          const iMax = Math.min(hi, k + 3);
          for (let i = 0; i <= iMax; i++) {
            p = x * (H[i * n + k] as number) + y * (H[i * n + (k + 1)] as number);
            if (notLast) {
              p += z * (H[i * n + (k + 2)] as number);
              H[i * n + (k + 2)] = (H[i * n + (k + 2)] as number) - p * r;
            }
            H[i * n + k] = (H[i * n + k] as number) - p;
            H[i * n + (k + 1)] = (H[i * n + (k + 1)] as number) - p * q;
          }

          // Accumulate transformations.
          if (Z !== null) {
            for (let i = 0; i < n; i++) {
              p = x * (Z[i * n + k] as number) + y * (Z[i * n + (k + 1)] as number);
              if (notLast) {
                p += z * (Z[i * n + (k + 2)] as number);
                Z[i * n + (k + 2)] = (Z[i * n + (k + 2)] as number) - p * r;
              }
              Z[i * n + k] = (Z[i * n + k] as number) - p;
              Z[i * n + (k + 1)] = (Z[i * n + (k + 1)] as number) - p * q;
            }
          }
        }
      }
    }
  }

  // Entries below the first sub-diagonal only hold stale bulge values.
  for (let i = 2; i < n; i++) {
    for (let j = 0; j < i - 1; j++) H[i * n + j] = 0;
  }

  return { wr, wi };
}

/**
 * Real Schur decomposition of a square matrix.
 *
 * Decomposes A into A = Q * T * Qᵀ where:
 * - Q is an orthogonal matrix (Schur vectors)
 * - T is a quasi-upper triangular matrix (Schur form)
 *   - Real eigenvalues appear on the diagonal
 *   - Complex conjugate pairs appear in 2×2 blocks on the diagonal; such a
 *     block has equal diagonal entries and off-diagonal entries of opposite sign
 *
 * Reduces A to Hessenberg form with Householder reflections, then runs the
 * QR algorithm with implicit double shifts (Francis iteration) and deflation.
 * The result has the same structure as `scipy.linalg.schur(a, output="real")`;
 * the Schur vectors are unique only up to sign.
 *
 * **Time Complexity**: O(N³)
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
 * @throws {ShapeError} If input is not a square 2D matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {ConvergenceError} If the QR iteration does not converge
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

  const mat = toDenseMatrix2D(a, "schur()");
  const T = new Float64Array(mat.data);
  const Q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;

  // Entries near 1e-200 or 1e200 would underflow/overflow in the shift computation,
  // so work on a power-of-two rescaled copy: A = f * (A / f) and T = f * T'.
  const factor = extremeScaleFactor(T);
  if (factor !== 1) {
    for (let i = 0; i < T.length; i++) T[i] = (T[i] as number) / factor;
  }
  hessenbergReduceInPlace(T, Q, n);
  realSchurIteration(T, Q, n, 300, Number.EPSILON, "schur()");
  if (factor !== 1) {
    for (let i = 0; i < T.length; i++) T[i] = (T[i] as number) * factor;
  }

  return [fromDenseMatrix2D(n, n, T), fromDenseMatrix2D(n, n, Q)];
}
