import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import type { Tensor } from "../../ndarray";
import { atArr, fromDenseMatrix2D, fromDenseVector1D, toDenseMatrix2D } from "../_internal";
import { extremeScaleFactor, hessenbergReduceInPlace, scaledNorm } from "./hessenberg";
import { realSchurIteration } from "./schur";

/**
 * Relative tolerance for accepting a matrix as symmetric in `eigh` and `eigvalsh`: entries
 * a[i][j] and a[j][i] must differ by at most `SYMMETRY_TOL * max|A|`.
 */
const SYMMETRY_TOL = 1e-10;

/**
 * Tolerance of `eig` and `eigvals` for taking the symmetric shortcut. It is only a few
 * rounding errors wide: the shortcut replaces A by (A + A^T) / 2, which moves the
 * eigenvectors (not the eigenvalues) by about the asymmetry, so a looser threshold would
 * cap the accuracy of the eigenpairs of a nearly symmetric matrix far above machine epsilon.
 */
function shortcutSymmetryTol(n: number): number {
  return 32 * Math.max(n, 1) * Number.EPSILON;
}

function getSquareMatrixSize(a: Tensor, context: string): number {
  if (a.ndim !== 2) {
    throw new ShapeError(`${context}: input must be 2D matrix`);
  }
  const rows = a.shape[0];
  const cols = a.shape[1];
  if (rows === undefined || cols === undefined || rows !== cols) {
    throw new ShapeError(`${context}: input must be square matrix`);
  }
  return rows;
}

function isSymmetric(A: Float64Array, n: number, relTol: number = SYMMETRY_TOL): boolean {
  let maxAbs = 0;
  for (let i = 0; i < A.length; i++) {
    const v = Math.abs(A[i] as number);
    if (v > maxAbs) maxAbs = v;
  }
  const tol = relTol * maxAbs;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs((A[i * n + j] as number) - (A[j * n + i] as number)) > tol) return false;
    }
  }
  return true;
}

/** Returns (A + Aᵀ) / 2 so that rounding-level asymmetry does not bias the result. */
function symmetrizedCopy(A: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    out[i * n + i] = A[i * n + i] as number;
    for (let j = i + 1; j < n; j++) {
      const v = 0.5 * ((A[i * n + j] as number) + (A[j * n + i] as number));
      out[i * n + j] = v;
      out[j * n + i] = v;
    }
  }
  return out;
}

/**
 * Divides `A` in place by a power of two when its entries are extreme (see
 * {@link extremeScaleFactor}) and returns that factor, 1 when nothing was scaled. The QL
 * sweep multiplies several off-diagonal entries, which underflows near 1e-154 and
 * overflows near 1e154; eigenvalues are multiplied by the factor afterwards and
 * eigenvectors are unaffected.
 */
function scaleToSafeRange(A: Float64Array): number {
  const factor = extremeScaleFactor(A);
  if (factor !== 1) {
    for (let i = 0; i < A.length; i++) A[i] = (A[i] as number) / factor;
  }
  return factor;
}

/**
 * Householder tridiagonalization of a symmetric matrix (EISPACK tred2).
 *
 * Reduces the symmetric matrix in `z` (row-major n×n, overwritten with the
 * accumulated orthogonal transform) to tridiagonal form with diagonal `d`
 * and off-diagonal `e` (e[0] unused). Costs about 4n³/3 flops.
 */
function tred2(n: number, z: Float64Array, d: Float64Array, e: Float64Array): void {
  for (let i = 0; i < n; i++) {
    d[i] = z[(n - 1) * n + i] as number;
  }

  for (let i = n - 1; i > 0; i--) {
    const l = i - 1;
    let h = 0;
    let scale = 0;
    if (l > 0) {
      for (let k = 0; k <= l; k++) scale += Math.abs(d[k] as number);
      if (scale === 0) {
        e[i] = d[l] as number;
        for (let j = 0; j <= l; j++) {
          d[j] = z[l * n + j] as number;
          z[i * n + j] = 0;
          z[j * n + i] = 0;
        }
      } else {
        const invScale = 1 / scale;
        for (let k = 0; k <= l; k++) {
          const v = (d[k] as number) * invScale;
          d[k] = v;
          h += v * v;
        }
        let f = d[l] as number;
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        d[l] = f - g;
        for (let j = 0; j <= l; j++) e[j] = 0;

        for (let j = 0; j <= l; j++) {
          f = d[j] as number;
          z[j * n + i] = f;
          g = (e[j] as number) + (z[j * n + j] as number) * f;
          for (let k = j + 1; k <= l; k++) {
            g += (z[k * n + j] as number) * (d[k] as number);
            e[k] = (e[k] as number) + (z[k * n + j] as number) * f;
          }
          e[j] = g;
        }
        f = 0;
        const invH = 1 / h;
        for (let j = 0; j <= l; j++) {
          const ej = (e[j] as number) * invH;
          e[j] = ej;
          f += ej * (d[j] as number);
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          e[j] = (e[j] as number) - hh * (d[j] as number);
        }
        for (let j = 0; j <= l; j++) {
          f = d[j] as number;
          g = e[j] as number;
          for (let k = j; k <= l; k++) {
            z[k * n + j] = (z[k * n + j] as number) - f * (e[k] as number) - g * (d[k] as number);
          }
          d[j] = z[l * n + j] as number;
          z[i * n + j] = 0;
        }
      }
    } else {
      e[i] = d[l] as number;
      for (let j = 0; j <= l; j++) {
        d[j] = z[l * n + j] as number;
        z[i * n + j] = 0;
        z[j * n + i] = 0;
      }
      h = 0;
    }
    d[i] = h;
  }

  // Accumulate transformations.
  for (let i = 1; i < n; i++) {
    const l = i - 1;
    z[(n - 1) * n + l] = z[l * n + l] as number;
    z[l * n + l] = 1;
    const h = d[i] as number;
    if (h !== 0) {
      const invH = 1 / h;
      for (let k = 0; k <= l; k++) d[k] = (z[k * n + i] as number) * invH;
      for (let j = 0; j <= l; j++) {
        let g = 0;
        for (let k = 0; k <= l; k++) g += (z[k * n + i] as number) * (z[k * n + j] as number);
        for (let k = 0; k <= l; k++) {
          z[k * n + j] = (z[k * n + j] as number) - g * (d[k] as number);
        }
      }
    }
    for (let k = 0; k <= l; k++) z[k * n + i] = 0;
  }
  for (let j = 0; j < n; j++) {
    d[j] = z[(n - 1) * n + j] as number;
    z[(n - 1) * n + j] = 0;
  }
  z[(n - 1) * n + (n - 1)] = 1;
  e[0] = 0;
}

/**
 * QL algorithm with implicit shifts for a symmetric tridiagonal matrix
 * (EISPACK tql2). Consumes `d`/`e` from {@link tred2}. The eigenvalues are left
 * in `d` in no particular order; callers sort. When `z` is given, the
 * eigenvectors are accumulated into it, otherwise only eigenvalues are computed.
 */
function tql2(
  n: number,
  d: Float64Array,
  e: Float64Array,
  z: Float64Array | null,
  context: string
): void {
  for (let i = 1; i < n; i++) e[i - 1] = e[i] as number;
  e[n - 1] = 0;

  let f = 0;
  let tst1 = 0;
  const eps = Number.EPSILON;
  const maxIter = 60;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l] as number) + Math.abs(e[l] as number));
    let m = l;
    while (m < n) {
      if (Math.abs(e[m] as number) <= eps * tst1) break;
      m++;
    }
    if (m > l) {
      let iter = 0;
      do {
        if (iter++ === maxIter) {
          throw new ConvergenceError(`${context}: QL iteration failed to converge`, {
            iterations: maxIter,
          });
        }
        // Compute implicit shift.
        let g = d[l] as number;
        let p = ((d[l + 1] as number) - g) / (2 * (e[l] as number));
        let r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = (e[l] as number) / (p + r);
        d[l + 1] = (e[l] as number) * (p + r);
        const dl1 = d[l + 1] as number;
        let h = g - (d[l] as number);
        for (let i = l + 2; i < n; i++) d[i] = (d[i] as number) - h;
        f += h;

        // Implicit QL transformation.
        p = d[m] as number;
        let c = 1;
        let c2 = c;
        let c3 = c;
        const el1 = e[l + 1] as number;
        let s = 0;
        let s2 = 0;
        for (let i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * (e[i] as number);
          h = c * p;
          r = Math.hypot(p, e[i] as number);
          e[i + 1] = s * r;
          s = (e[i] as number) / r;
          c = p / r;
          p = c * (d[i] as number) - s * g;
          d[i + 1] = h + s * (c * g + s * (d[i] as number));
          if (z !== null) {
            for (let k = 0; k < n; k++) {
              h = z[k * n + i + 1] as number;
              const zki = z[k * n + i] as number;
              z[k * n + i + 1] = s * zki + c * h;
              z[k * n + i] = c * zki - s * h;
            }
          }
        }
        p = (-s * s2 * c3 * el1 * (e[l] as number)) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l] as number) > eps * tst1);
    }
    d[l] = (d[l] as number) + f;
    e[l] = 0;
  }
}

/**
 * Symmetric eigendecomposition via Householder tridiagonalization + QL with
 * implicit shifts. Returns eigenvalues in ascending order and the matching
 * eigenvectors as columns of a row-major n×n matrix.
 */
function symmetricEigen(
  a: Float64Array,
  n: number,
  context: string
): { readonly values: Float64Array; readonly vectors: Float64Array } {
  const z = symmetrizedCopy(a, n);
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  if (n === 1) {
    d[0] = z[0] as number;
    z[0] = 1;
    return { values: d, vectors: z };
  }
  const factor = scaleToSafeRange(z);
  tred2(n, z, d, e);
  tql2(n, d, e, z, context);
  if (factor !== 1) for (let i = 0; i < n; i++) d[i] = (d[i] as number) * factor;

  // Sort ascending (stable for equal eigenvalues) and permute the columns.
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((i, j) => (d[i] as number) - (d[j] as number) || i - j);

  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = atArr(idx, col);
    values[col] = d[src] as number;
    for (let row = 0; row < n; row++) {
      vectors[row * n + col] = z[row * n + src] as number;
    }
  }
  return { values, vectors };
}

/**
 * Eigenvalues only of a symmetric matrix: tridiagonalize, then run the QL sweep
 * without eigenvector updates (~2x less work than {@link symmetricEigen}).
 * Returns the eigenvalues in ascending order.
 */
function symmetricEigenvalues(a: Float64Array, n: number, context: string): Float64Array {
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  const A = symmetrizedCopy(a, n);
  if (n === 1) {
    d[0] = A[0] as number;
    return d;
  }
  const factor = scaleToSafeRange(A);
  tred2(n, A, d, e);
  tql2(n, d, e, null, context);
  if (factor !== 1) for (let i = 0; i < n; i++) d[i] = (d[i] as number) * factor;
  d.sort();
  return d;
}

/**
 * Parlett-Reinsch balancing: finds a diagonal similarity D⁻¹ A D with power-of-two
 * entries that brings the row and column norms of `B` closer together. This
 * improves the accuracy of eigenvalues of badly scaled matrices. `B` is
 * overwritten with the balanced matrix and the diagonal of D is returned.
 */
function balanceInPlace(B: Float64Array, n: number): Float64Array {
  const scale = new Float64Array(n).fill(1);
  const radix = 2;
  const sqrdx = radix * radix;
  for (let sweep = 0; sweep < 100; sweep++) {
    let done = true;
    for (let i = 0; i < n; i++) {
      let c = 0;
      let r = 0;
      for (let j = 0; j < n; j++) {
        if (j !== i) {
          c += Math.abs(B[j * n + i] as number);
          r += Math.abs(B[i * n + j] as number);
        }
      }
      if (c === 0 || r === 0) continue;
      let g = r / radix;
      let f = 1;
      const s = c + r;
      while (c < g) {
        f *= radix;
        c *= sqrdx;
      }
      g = r * radix;
      while (c > g) {
        f /= radix;
        c /= sqrdx;
      }
      if ((c + r) / f < 0.95 * s) {
        done = false;
        const gInv = 1 / f;
        scale[i] = (scale[i] as number) * f;
        for (let j = 0; j < n; j++) B[i * n + j] = (B[i * n + j] as number) * gInv;
        for (let j = 0; j < n; j++) B[j * n + i] = (B[j * n + i] as number) * f;
      }
    }
    if (done) break;
  }
  return scale;
}

/**
 * Makes the 2x2 Schur block at rows/columns (i, i + 1) upper triangular. The block has
 * equal diagonal entries a and off-diagonal entries b, c with b * c < 0 and a negligible
 * product, so dropping the smaller of |b|, |c| changes the matrix by less than the
 * imaginary part of the pair. When |c| is the larger one, rows and columns i and i + 1
 * are swapped first (an orthogonal similarity) so that the entry that is dropped is
 * below the diagonal.
 */
function realifyBlock(T: Float64Array, Q: Float64Array | null, n: number, i: number): void {
  const j = i + 1;
  const b = T[i * n + j] as number;
  const c = T[j * n + i] as number;
  if (Math.abs(c) > Math.abs(b)) {
    for (let col = 0; col < n; col++) {
      const t = T[i * n + col] as number;
      T[i * n + col] = T[j * n + col] as number;
      T[j * n + col] = t;
    }
    for (let row = 0; row < n; row++) {
      const t = T[row * n + i] as number;
      T[row * n + i] = T[row * n + j] as number;
      T[row * n + j] = t;
    }
    if (Q !== null) {
      for (let row = 0; row < n; row++) {
        const t = Q[row * n + i] as number;
        Q[row * n + i] = Q[row * n + j] as number;
        Q[row * n + j] = t;
      }
    }
  }
  T[j * n + i] = 0;
}

type GeneralEigenResult = {
  readonly values: Float64Array;
  readonly imag: Float64Array;
  readonly vectors: Float64Array | null;
  readonly vectorsImag: Float64Array | null;
};

type ComplexMatrix = { readonly re: Float64Array; readonly im: Float64Array };

/**
 * Eigenvectors of the quasi upper triangular real Schur form `T` (row-major n x n) in the Schur
 * basis, as the columns of a complex matrix. `wi[p] > 0` marks a 2 x 2 block at rows (p, p + 1)
 * with equal diagonal entries and off-diagonal entries of opposite sign; column p holds the
 * eigenvector of a + i w and column p + 1 its conjugate. Every other column k solves
 * (T - lambda_k I) x = 0 with x_k = 1 by block back-substitution (as LAPACK dtrevc does).
 */
function quasiTriangularEigenvectors(T: Float64Array, n: number, wi: Float64Array): ComplexMatrix {
  let tnorm = 0;
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(i - 1, 0); j < n; j++) tnorm += Math.abs(T[i * n + j] as number);
  }
  const eps = Number.EPSILON;
  const smin = Math.max(eps * tnorm, Number.MIN_VALUE);
  const Xr = new Float64Array(n * n);
  const Xi = new Float64Array(n * n);
  const xr = new Float64Array(n);
  const xi = new Float64Array(n);

  for (let k = n - 1; k >= 0; k--) {
    const isPairSecond = k > 0 && (wi[k - 1] as number) > 0;
    const first = isPairSecond ? k - 1 : k;
    // Vector of the eigenvalue with non-negative imaginary part.
    const lr = T[first * n + first] as number;
    const li = isPairSecond ? (wi[first] as number) : 0;
    xr.fill(0);
    xi.fill(0);
    if (isPairSecond) {
      const b = T[first * n + k] as number;
      const c = T[k * n + first] as number;
      if (Math.abs(b) >= Math.abs(c)) {
        xr[first] = 1;
        xi[k] = li / b;
      } else {
        xr[k] = 1;
        xi[first] = li / c;
      }
    } else {
      xr[k] = 1;
    }

    for (let i = first - 1; i >= 0; i--) {
      const blockStart = i > 0 && (wi[i - 1] as number) > 0 ? i - 1 : i;
      let rr = 0;
      let ri = 0;
      for (let j = i + 1; j <= k; j++) {
        const t = T[i * n + j] as number;
        rr += t * (xr[j] as number);
        ri += t * (xi[j] as number);
      }
      if (blockStart === i) {
        // (T_ii - lambda) x_i = -r
        let dr = (T[i * n + i] as number) - lr;
        let di = -li;
        if (Math.hypot(dr, di) < smin) {
          dr = smin;
          di = 0;
        }
        const den = dr * dr + di * di;
        xr[i] = -(rr * dr + ri * di) / den;
        xi[i] = -(ri * dr - rr * di) / den;
      } else {
        // 2 x 2 block at rows (i - 1, i): [[m11, m12], [m21, m22]] [x_a, x_b] = -[r_a, r_b].
        const a = i - 1;
        let sr = 0;
        let si = 0;
        for (let j = i + 1; j <= k; j++) {
          const t = T[a * n + j] as number;
          sr += t * (xr[j] as number);
          si += t * (xi[j] as number);
        }
        const m11r = (T[a * n + a] as number) - lr;
        const m22r = (T[i * n + i] as number) - lr;
        const m12 = T[a * n + i] as number;
        const m21 = T[i * n + a] as number;
        // det = m11 m22 - m12 m21 with m11 = m11r - i li, m22 = m22r - i li.
        let detr = m11r * m22r - li * li - m12 * m21;
        let deti = -li * (m11r + m22r);
        const cmax = Math.max(
          Math.abs(m11r) + Math.abs(li),
          Math.abs(m22r) + Math.abs(li),
          Math.abs(m12),
          Math.abs(m21)
        );
        if (Math.hypot(detr, deti) < smin * cmax) {
          detr = smin * cmax;
          deti = 0;
        }
        const den = detr * detr + deti * deti;
        // x_a = (-s_a m22 + m12 r_i) / det, x_b = (-m11 r_i + m21 s_a) / det with s_a the
        // row-a sum, r_i the row-i sum (both of the already known entries).
        const nar = -(sr * m22r + si * li) + m12 * rr;
        const nai = -(si * m22r - sr * li) + m12 * ri;
        const nbr = -(rr * m11r + ri * li) + m21 * sr;
        const nbi = -(ri * m11r - rr * li) + m21 * si;
        xr[a] = (nar * detr + nai * deti) / den;
        xi[a] = (nai * detr - nar * deti) / den;
        xr[i] = (nbr * detr + nbi * deti) / den;
        xi[i] = (nbi * detr - nbr * deti) / den;
        i--;
      }
      // Rescale the partial solution before it can overflow.
      const top = blockStart;
      let mag = 0;
      for (let j = top; j <= k; j++)
        mag = Math.max(mag, Math.hypot(xr[j] as number, xi[j] as number));
      if (eps * mag * mag > 1) {
        for (let j = top; j <= k; j++) {
          xr[j] = (xr[j] as number) / mag;
          xi[j] = (xi[j] as number) / mag;
        }
      }
    }

    for (let i = 0; i <= k; i++) {
      Xr[i * n + first] = xr[i] as number;
      Xi[i * n + first] = xi[i] as number;
      if (isPairSecond) {
        Xr[i * n + k] = xr[i] as number;
        Xi[i * n + k] = -(xi[i] as number);
      }
    }
    if (isPairSecond) k--;
  }
  return { re: Xr, im: Xi };
}

/**
 * Maps the Schur-basis eigenvectors `X` back to the (balanced) input basis, undoes the
 * balancing and normalizes each column to unit Euclidean norm. As in LAPACK dgeev, the
 * component of largest magnitude of a complex eigenvector is made real. The column of the
 * second member of a conjugate pair is the conjugate of the first.
 */
function backTransformComplex(
  Q: Float64Array,
  X: ComplexMatrix,
  dscale: Float64Array,
  n: number,
  wi: Float64Array
): { readonly vectors: Float64Array; readonly vectorsImag: Float64Array } {
  const Vr = new Float64Array(n * n);
  const Vi = new Float64Array(n * n);
  const col = new Float64Array(2 * n);
  for (let k = 0; k < n; k++) {
    const imagK = wi[k] as number;
    if (imagK < 0) {
      for (let i = 0; i < n; i++) {
        Vr[i * n + k] = Vr[i * n + k - 1] as number;
        Vi[i * n + k] = -(Vi[i * n + k - 1] as number);
      }
      continue;
    }
    for (let i = 0; i < n; i++) {
      let sr = 0;
      let si = 0;
      for (let j = 0; j <= k + (imagK > 0 ? 1 : 0); j++) {
        const q = Q[i * n + j] as number;
        sr += q * (X.re[j * n + k] as number);
        si += q * (X.im[j * n + k] as number);
      }
      const d = dscale[i] as number;
      col[i] = sr * d;
      col[n + i] = si * d;
    }
    const nrm = scaledNorm(col, 2 * n);
    if (nrm > 0) {
      const inv = 1 / nrm;
      for (let i = 0; i < 2 * n; i++) col[i] = (col[i] as number) * inv;
    }
    if (imagK > 0) {
      let best = 0;
      let bestAbs = -1;
      for (let i = 0; i < n; i++) {
        const m = Math.hypot(col[i] as number, col[n + i] as number);
        if (m > bestAbs) {
          bestAbs = m;
          best = i;
        }
      }
      if (bestAbs > 0) {
        const pr = (col[best] as number) / bestAbs;
        const pi = -(col[n + best] as number) / bestAbs;
        for (let i = 0; i < n; i++) {
          const re = col[i] as number;
          const im = col[n + i] as number;
          col[i] = re * pr - im * pi;
          col[n + i] = re * pi + im * pr;
        }
        col[n + best] = 0;
      }
    }
    for (let i = 0; i < n; i++) {
      Vr[i * n + k] = col[i] as number;
      Vi[i * n + k] = col[n + i] as number;
    }
  }
  return { vectors: Vr, vectorsImag: Vi };
}

/**
 * Eigenvalues (and optionally unit-norm eigenvectors) of a general real matrix
 * whose spectrum is real: balance, reduce to Hessenberg form, run Francis QR to
 * the real Schur form, then back-substitute for the eigenvectors of the
 * triangular factor.
 */
function generalEigen(
  A: Float64Array,
  n: number,
  maxIter: number,
  tol: number,
  wantVectors: boolean,
  allowComplex: boolean
): GeneralEigenResult {
  const T = new Float64Array(A);
  // Entries near 1e-200 or 1e200 would underflow/overflow inside the QR iteration;
  // eigenvectors are unaffected by a uniform scale, eigenvalues are rescaled below.
  const factor = extremeScaleFactor(T);
  if (factor !== 1) {
    for (let i = 0; i < T.length; i++) T[i] = (T[i] as number) / factor;
  }
  const dscale = balanceInPlace(T, n);
  let maxAbs = 0;
  for (let i = 0; i < T.length; i++) {
    const v = Math.abs(T[i] as number);
    if (v > maxAbs) maxAbs = v;
  }
  let Q: Float64Array | null = null;
  if (wantVectors) {
    Q = new Float64Array(n * n);
    for (let i = 0; i < n; i++) Q[i * n + i] = 1;
  }
  hessenbergReduceInPlace(T, Q, n);
  const { wr, wi } = realSchurIteration(T, Q, n, maxIter, tol, "eig()");

  // A conjugate pair whose imaginary part is at rounding level (typical for the repeated zero
  // eigenvalue of a low-rank matrix) is indistinguishable from a repeated real eigenvalue, so it
  // is reported as one (the backward error is of the same size as the rounding error).
  const imagTol = 10 * n * Number.EPSILON * maxAbs;
  let hasComplex = false;
  for (let i = 0; i < n; i++) {
    const im = wi[i] as number;
    if (im === 0) continue;
    if (!(im > 0 && im <= imagTol)) {
      if (allowComplex) {
        hasComplex = true;
        i++;
        continue;
      }
      // Report the first offending eigenvalue, in the units of the input.
      const real = (wr[i] as number) * factor;
      const imag = Math.abs(im) * factor;
      throw new InvalidParameterError(
        "Matrix has complex eigenvalues. Pass { complex: true } to eig() or eigvals() " +
          "to get the real and imaginary parts, or use a symmetric matrix, whose " +
          `eigenvalues are always real. Found the eigenvalue pair ${real} +/- ${imag}i.`,
        "a",
        { real, imag }
      );
    }
    realifyBlock(T, Q, n, i);
    wi[i] = 0;
    wi[i + 1] = 0;
    i++;
  }
  // Eigenvectors of a quasi-triangular T (complex pairs present) are computed in the scaled
  // units of T, so they come before the eigenvalues are rescaled.
  const complexVectors =
    Q !== null && hasComplex
      ? backTransformComplex(Q, quasiTriangularEigenvectors(T, n, wi), dscale, n, wi)
      : null;
  if (factor !== 1) {
    for (let i = 0; i < n; i++) {
      wr[i] = (wr[i] as number) * factor;
      wi[i] = (wi[i] as number) * factor;
    }
  }
  if (Q === null) return { values: wr, imag: wi, vectors: null, vectorsImag: null };
  if (complexVectors !== null) return { values: wr, imag: wi, ...complexVectors };

  // T is upper triangular. Solve (T - λ_k I) x = 0 with x_k = 1 by back-substitution.
  let tnorm = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) tnorm += Math.abs(T[i * n + j] as number);
  }
  const eps = Number.EPSILON;
  const smin = eps * tnorm;
  const X = new Float64Array(n * n);
  const x = new Float64Array(n);
  for (let k = n - 1; k >= 0; k--) {
    const lambda = T[k * n + k] as number;
    x.fill(0);
    x[k] = 1;
    for (let i = k - 1; i >= 0; i--) {
      let w = (T[i * n + i] as number) - lambda;
      let r = 0;
      for (let j = i + 1; j <= k; j++) r += (T[i * n + j] as number) * (x[j] as number);
      // Repeated eigenvalue: perturb the pivot, as LAPACK's dtrevc does.
      if (Math.abs(w) < smin) w = smin;
      const xi = -r / w;
      x[i] = xi;
      // Rescale the partial solution before it can overflow.
      const t = Math.abs(xi);
      if (eps * t * t > 1) {
        for (let j = i; j <= k; j++) x[j] = (x[j] as number) / t;
      }
    }
    for (let i = 0; i <= k; i++) X[i * n + k] = x[i] as number;
  }

  // V = Q X, then undo the balancing and normalize the columns.
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      let sum = 0;
      for (let j = 0; j <= k; j++) sum += (Q[i * n + j] as number) * (X[j * n + k] as number);
      V[i * n + k] = sum * (dscale[i] as number);
    }
  }
  const col = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) col[i] = V[i * n + k] as number;
    const nrm = scaledNorm(col, n);
    if (nrm > 0) {
      const inv = 1 / nrm;
      for (let i = 0; i < n; i++) V[i * n + k] = (V[i * n + k] as number) * inv;
    }
  }
  return { values: wr, imag: wi, vectors: V, vectorsImag: new Float64Array(n * n) };
}

/**
 * Options for {@link eig} and {@link eigvals}.
 *
 * - `maxIter`: maximum number of QR sweeps spent on a single eigenvalue before a
 *   {@link ConvergenceError} is thrown (default: 300). A trailing 2×2 block is
 *   solved in closed form and needs no sweeps.
 * - `tol`: relative threshold below which a sub-diagonal entry is treated as zero
 *   (default: machine epsilon, about 2.2e-16). Larger values deflate earlier at
 *   the cost of accuracy.
 * - `complex`: when `true`, matrices with complex eigenvalues are accepted and the
 *   result holds real and imaginary parts (see {@link ComplexEig} and
 *   {@link ComplexEigvals}) instead of the default tuple or tensor (default: false).
 */
export type EigOptions = {
  readonly maxIter?: number;
  readonly tol?: number;
  readonly complex?: boolean;
};

/**
 * Result of {@link eigvals} with `{ complex: true }`. `real[i] + imag[i] * i` is the
 * i-th eigenvalue. Complex conjugate pairs are adjacent, the member with positive
 * imaginary part first.
 */
export type ComplexEigvals = {
  readonly real: Tensor;
  readonly imag: Tensor;
};

/**
 * Result of {@link eig} with `{ complex: true }`. Column `k` of `vectorsReal + vectorsImag * i`
 * is the unit-norm eigenvector of the eigenvalue `real[k] + imag[k] * i`.
 */
export type ComplexEig = {
  readonly real: Tensor;
  readonly imag: Tensor;
  readonly vectorsReal: Tensor;
  readonly vectorsImag: Tensor;
};

function resolveEigOptions(options: EigOptions): { maxIter: number; tol: number } {
  const maxIter = options.maxIter ?? 300;
  const tol = options.tol ?? Number.EPSILON;
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new InvalidParameterError("maxIter must be a positive integer", "maxIter", maxIter);
  }
  if (!Number.isFinite(tol) || tol <= 0) {
    throw new InvalidParameterError("tol must be a positive finite number", "tol", tol);
  }
  return { maxIter, tol };
}

/**
 * Compute eigenvalues and eigenvectors of a square matrix.
 *
 * Solves A * v = λ * v where λ are eigenvalues and v are eigenvectors.
 *
 * **Algorithm**:
 * - Symmetric matrices: Householder tridiagonalization + implicit QL (same as {@link eigh}).
 *   Eigenvalues are returned in ascending order.
 * - General matrices: balancing, Hessenberg reduction, Francis double-shift QR to the
 *   real Schur form, then back-substitution for the eigenvectors. Eigenvalues are
 *   returned in the order they appear on the diagonal of the Schur form.
 *
 * **Limitations**:
 * - By default only real eigenvalues are supported. A non-symmetric matrix whose
 *   spectrum includes complex eigenvalues causes an {@link InvalidParameterError}.
 *   Pass `{ complex: true }` to get the real and imaginary parts instead (see below).
 * - A conjugate pair whose imaginary part is at rounding level (at most
 *   10 * N * eps * max|A| after balancing, as for the repeated zero eigenvalue of a
 *   low-rank matrix) is reported as a repeated real eigenvalue.
 * - For symmetric/Hermitian matrices, use `eigh()` for better performance
 * - Eigenvectors of a defective matrix (fewer independent eigenvectors than
 *   eigenvalues) are nearly parallel, as with `numpy.linalg.eig`.
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 * @param options - Optional configuration overrides (see {@link EigOptions})
 * @param options.maxIter - Maximum QR sweeps per eigenvalue (default: 300)
 * @param options.tol - Relative deflation threshold (default: machine epsilon)
 *
 * **Returns**: [eigenvalues, eigenvectors]
 * - eigenvalues: Real values of shape (N,)
 * - eigenvectors: Column vectors of shape (N, N) where eigenvectors[:,i] corresponds to eigenvalues[i]
 *
 * With `{ complex: true }` the return value is a {@link ComplexEig} object
 * `{ real, imag, vectorsReal, vectorsImag }` and complex eigenvalues are accepted.
 * Conjugate pairs are adjacent (positive imaginary part first); as in `numpy.linalg.eig`
 * the largest component of a complex eigenvector is real.
 *
 * **Properties**:
 * - A @ v[:,i] = λ[i] * v[:,i]
 * - Each eigenvector has unit Euclidean norm (its sign is arbitrary)
 *
 * @example
 * ```ts
 * import { eig } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 1]]);
 * const [eigenvalues, eigenvectors] = eig(A);
 *
 * // Verify: A @ eigenvectors[:,i] ≈ eigenvalues[i] * eigenvectors[:,i]
 *
 * // Complex eigenvalues: a rotation by 90 degrees has eigenvalues +i and -i.
 * const R = tensor([[0, -1], [1, 0]]);
 * const { real, imag, vectorsReal, vectorsImag } = eig(R, { complex: true });
 * ```
 *
 * @throws {ShapeError} If input is not square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If matrix has complex eigenvalues (and `complex` is not set), or an option is invalid
 * @throws {ConvergenceError} If the QR iteration does not converge within `maxIter` sweeps
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 7.5.2
 */
export function eig(
  a: Tensor,
  options?: EigOptions & { readonly complex?: false }
): [Tensor, Tensor];
export function eig(a: Tensor, options: EigOptions & { readonly complex: true }): ComplexEig;
export function eig(a: Tensor, options: EigOptions): [Tensor, Tensor] | ComplexEig;
export function eig(a: Tensor, options: EigOptions = {}): [Tensor, Tensor] | ComplexEig {
  const n = getSquareMatrixSize(a, "eig");
  const { maxIter, tol } = resolveEigOptions(options);
  const complex = options.complex === true;
  if (n === 0) {
    const empty = fromDenseVector1D(new Float64Array(0));
    const emptyMatrix = fromDenseMatrix2D(0, 0, new Float64Array(0));
    return complex
      ? { real: empty, imag: empty, vectorsReal: emptyMatrix, vectorsImag: emptyMatrix }
      : [empty, emptyMatrix];
  }

  const { data: A } = toDenseMatrix2D(a, "eig()");

  if (isSymmetric(A, n, shortcutSymmetryTol(n))) {
    const { values, vectors } = symmetricEigen(A, n, "eig()");
    if (complex) {
      return {
        real: fromDenseVector1D(values),
        imag: fromDenseVector1D(new Float64Array(n)),
        vectorsReal: fromDenseMatrix2D(n, n, vectors),
        vectorsImag: fromDenseMatrix2D(n, n, new Float64Array(n * n)),
      };
    }
    return [fromDenseVector1D(values), fromDenseMatrix2D(n, n, vectors)];
  }

  const { values, imag, vectors, vectorsImag } = generalEigen(A, n, maxIter, tol, true, complex);
  if (vectors === null || vectorsImag === null) {
    throw new ShapeError("eig(): eigenvectors were not computed");
  }
  if (complex) {
    return {
      real: fromDenseVector1D(values),
      imag: fromDenseVector1D(imag),
      vectorsReal: fromDenseMatrix2D(n, n, vectors),
      vectorsImag: fromDenseMatrix2D(n, n, vectorsImag),
    };
  }
  return [fromDenseVector1D(values), fromDenseMatrix2D(n, n, vectors)];
}

/**
 * Compute eigenvalues only (faster than eig).
 *
 * Skips the eigenvector computation. Eigenvalues of symmetric input are returned
 * in ascending order; for general input they follow the order of the real Schur form.
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 * @param options - Optional configuration overrides (see {@link EigOptions})
 *
 * **Returns**: eigenvalues - Array of real eigenvalues
 *
 * **Returns** with `{ complex: true }`: a {@link ComplexEigvals} object `{ real, imag }`.
 * Matrices with complex eigenvalues are accepted; conjugate pairs are adjacent with the
 * positive imaginary part first.
 *
 * **Limitations**:
 * - Without `complex: true`, matrices with complex eigenvalues throw
 *   {@link InvalidParameterError}
 *
 * @example
 * ```ts
 * import { eigvals } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 1]]);
 * const eigenvalues = eigvals(A);
 * console.log(eigenvalues);  // [-1, 3]
 *
 * const { real, imag } = eigvals(tensor([[0, -1], [1, 0]]), { complex: true });
 * // real = [0, 0], imag = [1, -1]
 * ```
 *
 * @throws {ShapeError} If input is not square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If matrix has complex eigenvalues (and `complex` is not set), or an option is invalid
 * @throws {ConvergenceError} If the QR iteration does not converge within `maxIter` sweeps
 */
export function eigvals(a: Tensor, options?: EigOptions & { readonly complex?: false }): Tensor;
export function eigvals(
  a: Tensor,
  options: EigOptions & { readonly complex: true }
): ComplexEigvals;
export function eigvals(a: Tensor, options: EigOptions): Tensor | ComplexEigvals;
export function eigvals(a: Tensor, options: EigOptions = {}): Tensor | ComplexEigvals {
  const n = getSquareMatrixSize(a, "eigvals");
  const { maxIter, tol } = resolveEigOptions(options);
  const complex = options.complex === true;
  if (n === 0) {
    const empty = fromDenseVector1D(new Float64Array(0));
    return complex ? { real: empty, imag: empty } : empty;
  }

  const { data: A } = toDenseMatrix2D(a, "eigvals()");
  if (isSymmetric(A, n, shortcutSymmetryTol(n))) {
    const values = fromDenseVector1D(symmetricEigenvalues(A, n, "eigvals()"));
    return complex ? { real: values, imag: fromDenseVector1D(new Float64Array(n)) } : values;
  }
  const result = generalEigen(A, n, maxIter, tol, false, complex);
  if (complex) {
    return { real: fromDenseVector1D(result.values), imag: fromDenseVector1D(result.imag) };
  }
  return fromDenseVector1D(result.values);
}

/**
 * Compute eigenvalues only of symmetric matrix (faster than eigh).
 *
 * **Parameters**:
 * @param a - Symmetric square matrix of shape (N, N)
 *
 * **Returns**: eigenvalues - Array of real eigenvalues in ascending order
 *
 * @example
 * ```ts
 * import { eigvalsh } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 1]]);
 * const eigenvalues = eigvalsh(A);
 * console.log(eigenvalues);  // [-1, 3]
 * ```
 *
 * @throws {ShapeError} If input is not square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input is not symmetric
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {ConvergenceError} If the QL iteration does not converge
 */
export function eigvalsh(a: Tensor): Tensor {
  const n = getSquareMatrixSize(a, "eigvalsh");
  const { data: A } = toDenseMatrix2D(a, "eigvalsh()");

  if (!isSymmetric(A, n)) {
    throw new DataValidationError("Input must be symmetric for eigvalsh");
  }
  if (n === 0) return fromDenseVector1D(new Float64Array(0));

  // Eigenvalues only: skip the O(n³) eigenvector accumulation entirely.
  return fromDenseVector1D(symmetricEigenvalues(A, n, "eigvalsh()"));
}

/**
 * Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix.
 *
 * More efficient than eig() for symmetric matrices.
 *
 * **Parameters**:
 * @param a - Symmetric matrix of shape (N, N)
 *
 * **Returns**: [eigenvalues, eigenvectors]
 * - Eigenvalues are real and in ascending order
 * - Eigenvectors are orthonormal columns
 *
 * @example
 * ```ts
 * import { eigh } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 1]]);  // Symmetric
 * const [eigenvalues, eigenvectors] = eigh(A);
 * ```
 *
 * @throws {ShapeError} If input is not square matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input is not symmetric
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {ConvergenceError} If the QL iteration does not converge
 */
export function eigh(a: Tensor): [Tensor, Tensor] {
  const n = getSquareMatrixSize(a, "eigh");
  const { data: A } = toDenseMatrix2D(a, "eigh()");

  if (!isSymmetric(A, n)) {
    throw new DataValidationError("Input must be symmetric for eigh");
  }
  if (n === 0) {
    return [fromDenseVector1D(new Float64Array(0)), fromDenseMatrix2D(0, 0, new Float64Array(0))];
  }

  const { values, vectors } = symmetricEigen(A, n, "eigh()");
  return [fromDenseVector1D(values), fromDenseMatrix2D(n, n, vectors)];
}
