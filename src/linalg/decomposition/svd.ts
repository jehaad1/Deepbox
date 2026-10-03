import { ConvergenceError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { fromDenseMatrix2D, fromDenseVector1D, getDim, toDenseMatrix2D } from "../_internal";

/** Maximum implicit-shift QR sweeps spent on one singular value. */
const MAX_SVD_SWEEPS = 75;

function identityMatrix(n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

function pythag(x: number, y: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  if (ax > ay) return ax * Math.sqrt(1 + (ay / ax) ** 2);
  return ay === 0 ? 0 : ay * Math.sqrt(1 + (ax / ay) ** 2);
}

/** Fortran SIGN(a, b): |a| with the sign of b. */
function sign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

/**
 * Extends the first `startCol` orthonormal columns of the row-major
 * `rows x cols` matrix `mat` to a full orthonormal basis of R^rows.
 *
 * The existing columns are factored with Householder reflections (they are
 * orthonormal, so the triangular factor is diagonal with entries +-1). The
 * trailing columns of the corresponding Q span the orthogonal complement and
 * are formed by applying the reflections to the matching identity columns. This
 * costs O(rows * startCol * cols) and is orthonormal to machine precision.
 */
function completeOrthonormalColumns(
  mat: Float64Array,
  rows: number,
  cols: number,
  startCol: number
): void {
  if (startCol >= cols) return;

  // Work copy of the leading columns, reduced column by column.
  const W = new Float64Array(rows * startCol);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < startCol; j++) W[i * startCol + j] = mat[i * cols + j] as number;
  }
  const V = new Float64Array(rows * startCol); // unit Householder vectors, column j from row j
  const used = new Uint8Array(startCol);
  const w = new Float64Array(Math.max(startCol, cols));
  for (let j = 0; j < startCol; j++) {
    let tailSq = 0;
    for (let i = j + 1; i < rows; i++) tailSq += (W[i * startCol + j] as number) ** 2;
    if (tailSq === 0) continue;
    const x0 = W[j * startCol + j] as number;
    const normX = Math.sqrt(x0 * x0 + tailSq);
    const alpha = x0 >= 0 ? -normX : normX;
    const v0 = x0 - alpha;
    const inv = 1 / Math.sqrt(v0 * v0 + tailSq);
    V[j * startCol + j] = v0 * inv;
    for (let i = j + 1; i < rows; i++) {
      V[i * startCol + j] = (W[i * startCol + j] as number) * inv;
    }
    used[j] = 1;
    // W[j:, j+1:] -= 2 v (vᵀ W[j:, j+1:])
    for (let c = j + 1; c < startCol; c++) w[c] = 0;
    for (let i = j; i < rows; i++) {
      const vi = V[i * startCol + j] as number;
      for (let c = j + 1; c < startCol; c++) {
        w[c] = (w[c] as number) + vi * (W[i * startCol + c] as number);
      }
    }
    for (let i = j; i < rows; i++) {
      const tvi = 2 * (V[i * startCol + j] as number);
      for (let c = j + 1; c < startCol; c++) {
        W[i * startCol + c] = (W[i * startCol + c] as number) - tvi * (w[c] as number);
      }
    }
  }

  // Complement = H_0 ... H_{k-1} applied to the identity columns startCol..cols-1.
  const extra = cols - startCol;
  const C = new Float64Array(rows * extra);
  for (let c = 0; c < extra; c++) C[(startCol + c) * extra + c] = 1;
  for (let j = startCol - 1; j >= 0; j--) {
    if (used[j] === 0) continue;
    for (let c = 0; c < extra; c++) w[c] = 0;
    for (let i = j; i < rows; i++) {
      const vi = V[i * startCol + j] as number;
      if (vi === 0) continue;
      for (let c = 0; c < extra; c++) w[c] = (w[c] as number) + vi * (C[i * extra + c] as number);
    }
    for (let i = j; i < rows; i++) {
      const tvi = 2 * (V[i * startCol + j] as number);
      if (tvi === 0) continue;
      for (let c = 0; c < extra; c++)
        C[i * extra + c] = (C[i * extra + c] as number) - tvi * (w[c] as number);
    }
  }
  for (let i = 0; i < rows; i++) {
    for (let c = 0; c < extra; c++) mat[i * cols + startCol + c] = C[i * extra + c] as number;
  }
}

/**
 * Golub–Reinsch core: Householder bidiagonalization followed by implicit-shift
 * QR on the bidiagonal. Requires `m >= n`.
 *
 * On exit `w` holds the (unsorted, non-negative) singular values. With
 * `wantVectors`, `a` holds the reduced left vectors (m×n) and `vmat` the right
 * vectors (n×n); otherwise the O(n³) accumulation of the transformations is
 * skipped and both are `null`.
 *
 * @internal
 */
function golubReinschCore(
  Ain: Float64Array,
  m: number,
  n: number,
  wantVectors: boolean
): { readonly a: Float64Array; readonly w: Float64Array; readonly vmat: Float64Array | null } {
  const a = new Float64Array(Ain); // working copy; holds the reduced U on exit
  const w = new Float64Array(n);
  const rv1 = new Float64Array(n);
  const vmat = wantVectors ? new Float64Array(n * n) : null;

  let g = 0;
  let scale = 0;
  let anorm = 0;
  let l = 0;

  // Householder reduction to bidiagonal form.
  for (let i = 0; i < n; i++) {
    l = i + 1;
    rv1[i] = scale * g;
    g = 0;
    let s = 0;
    scale = 0;
    if (i < m) {
      for (let k = i; k < m; k++) scale += Math.abs(a[k * n + i] as number);
      if (scale !== 0) {
        for (let k = i; k < m; k++) {
          const val = (a[k * n + i] as number) / scale;
          a[k * n + i] = val;
          s += val * val;
        }
        let f = a[i * n + i] as number;
        g = -sign(Math.sqrt(s), f);
        const h = f * g - s;
        a[i * n + i] = f - g;
        for (let j = l; j < n; j++) {
          let sum = 0;
          for (let k = i; k < m; k++) sum += (a[k * n + i] as number) * (a[k * n + j] as number);
          f = sum / h;
          for (let k = i; k < m; k++)
            a[k * n + j] = (a[k * n + j] as number) + f * (a[k * n + i] as number);
        }
        for (let k = i; k < m; k++) a[k * n + i] = (a[k * n + i] as number) * scale;
      }
    }
    w[i] = scale * g;
    g = 0;
    s = 0;
    scale = 0;
    if (i < m && i !== n - 1) {
      for (let k = l; k < n; k++) scale += Math.abs(a[i * n + k] as number);
      if (scale !== 0) {
        for (let k = l; k < n; k++) {
          const val = (a[i * n + k] as number) / scale;
          a[i * n + k] = val;
          s += val * val;
        }
        const f = a[i * n + l] as number;
        g = -sign(Math.sqrt(s), f);
        const h = f * g - s;
        a[i * n + l] = f - g;
        for (let k = l; k < n; k++) rv1[k] = (a[i * n + k] as number) / h;
        for (let j = l; j < m; j++) {
          let sum = 0;
          for (let k = l; k < n; k++) sum += (a[j * n + k] as number) * (a[i * n + k] as number);
          for (let k = l; k < n; k++)
            a[j * n + k] = (a[j * n + k] as number) + sum * (rv1[k] as number);
        }
        for (let k = l; k < n; k++) a[i * n + k] = (a[i * n + k] as number) * scale;
      }
    }
    anorm = Math.max(anorm, Math.abs(w[i] as number) + Math.abs(rv1[i] as number));
  }

  if (vmat !== null) {
    // Accumulation of right-hand transformations (V).
    for (let i = n - 1; i >= 0; i--) {
      if (i < n - 1) {
        if (g !== 0) {
          for (let j = l; j < n; j++) {
            vmat[j * n + i] = (a[i * n + j] as number) / (a[i * n + l] as number) / g;
          }
          for (let j = l; j < n; j++) {
            let sum = 0;
            for (let k = l; k < n; k++)
              sum += (a[i * n + k] as number) * (vmat[k * n + j] as number);
            for (let k = l; k < n; k++)
              vmat[k * n + j] = (vmat[k * n + j] as number) + sum * (vmat[k * n + i] as number);
          }
        }
        for (let j = l; j < n; j++) {
          vmat[i * n + j] = 0;
          vmat[j * n + i] = 0;
        }
      }
      vmat[i * n + i] = 1;
      g = rv1[i] as number;
      l = i;
    }

    // Accumulation of left-hand transformations (U, stored back into a).
    for (let i = Math.min(m, n) - 1; i >= 0; i--) {
      l = i + 1;
      g = w[i] as number;
      for (let j = l; j < n; j++) a[i * n + j] = 0;
      if (g !== 0) {
        g = 1 / g;
        for (let j = l; j < n; j++) {
          let sum = 0;
          for (let k = l; k < m; k++) sum += (a[k * n + i] as number) * (a[k * n + j] as number);
          const f = (sum / (a[i * n + i] as number)) * g;
          for (let k = i; k < m; k++)
            a[k * n + j] = (a[k * n + j] as number) + f * (a[k * n + i] as number);
        }
        for (let j = i; j < m; j++) a[j * n + i] = (a[j * n + i] as number) * g;
      } else {
        for (let j = i; j < m; j++) a[j * n + i] = 0;
      }
      a[i * n + i] = (a[i * n + i] as number) + 1;
    }
  }

  // Diagonalization of the bidiagonal form: QR with implicit Wilkinson shifts.
  const tiny = Number.EPSILON * anorm;
  for (let k = n - 1; k >= 0; k--) {
    for (let its = 0; its < MAX_SVD_SWEEPS; its++) {
      let flag = true;
      let nm = 0;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if (l === 0 || Math.abs(rv1[l] as number) <= tiny) {
          flag = false;
          break;
        }
        if (Math.abs(w[nm] as number) <= tiny) break;
      }
      if (flag) {
        // Cancellation of rv1[l] when w[l - 1] is negligible.
        let c = 0;
        let s = 1;
        for (let i = l; i <= k; i++) {
          const f = s * (rv1[i] as number);
          rv1[i] = c * (rv1[i] as number);
          if (Math.abs(f) <= tiny) break;
          g = w[i] as number;
          let h = pythag(f, g);
          w[i] = h;
          h = 1 / h;
          c = g * h;
          s = -f * h;
          if (wantVectors) {
            for (let j = 0; j < m; j++) {
              const y = a[j * n + nm] as number;
              const z = a[j * n + i] as number;
              a[j * n + nm] = y * c + z * s;
              a[j * n + i] = z * c - y * s;
            }
          }
        }
      }
      const z0 = w[k] as number;
      if (l === k) {
        // Convergence: make the singular value non-negative.
        if (z0 < 0) {
          w[k] = -z0;
          if (vmat !== null) {
            for (let j = 0; j < n; j++) vmat[j * n + k] = -(vmat[j * n + k] as number);
          }
        }
        break;
      }
      if (its === MAX_SVD_SWEEPS - 1) {
        throw new ConvergenceError(
          `svd() failed to converge after ${MAX_SVD_SWEEPS} QR sweeps for singular value ${k}`,
          { iterations: MAX_SVD_SWEEPS, tolerance: Number.EPSILON }
        );
      }
      let x = w[l] as number;
      nm = k - 1;
      let y = w[nm] as number;
      g = rv1[nm] as number;
      let h = rv1[k] as number;
      let f = ((y - z0) * (y + z0) + (g - h) * (g + h)) / (2 * h * y);
      g = pythag(f, 1);
      f = ((x - z0) * (x + z0) + h * (y / (f + sign(g, f)) - h)) / x;
      let c = 1;
      let s = 1;
      for (let j = l; j <= nm; j++) {
        const i = j + 1;
        g = rv1[i] as number;
        y = w[i] as number;
        h = s * g;
        g = c * g;
        let z = pythag(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y *= c;
        if (vmat !== null) {
          for (let jj = 0; jj < n; jj++) {
            const xx = vmat[jj * n + j] as number;
            const zz = vmat[jj * n + i] as number;
            vmat[jj * n + j] = xx * c + zz * s;
            vmat[jj * n + i] = zz * c - xx * s;
          }
        }
        z = pythag(f, h);
        w[j] = z;
        if (z !== 0) {
          z = 1 / z;
          c = f * z;
          s = h * z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
        if (wantVectors) {
          for (let jj = 0; jj < m; jj++) {
            const yy = a[jj * n + j] as number;
            const zz = a[jj * n + i] as number;
            a[jj * n + j] = yy * c + zz * s;
            a[jj * n + i] = zz * c - yy * s;
          }
        }
      }
      rv1[l] = 0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  return { a, w, vmat };
}

/**
 * Golub–Reinsch SVD (Householder bidiagonalization + implicit-shift QR).
 *
 * Runs in O(n³) with a small constant. Requires `m >= n`.
 *
 * @param Ain - Input matrix, m×n row-major (not modified)
 * @returns U (m×n, reduced), s (n, descending), V (n×n); A = U·diag(s)·Vᵀ
 * @internal
 */
function golubReinschSVD(
  Ain: Float64Array,
  m: number,
  n: number
): { readonly U: Float64Array; readonly s: Float64Array; readonly V: Float64Array } {
  const { a, w, vmat } = golubReinschCore(Ain, m, n, true);
  if (vmat === null) throw new ShapeError("svd(): right singular vectors were not computed");

  // Sort singular values descending and permute the matching columns of U, V.
  const idx = Array.from({ length: n }, (_, i) => i).sort(
    (p, q) => (w[q] as number) - (w[p] as number) || p - q
  );
  const s = new Float64Array(n);
  const U = new Float64Array(m * n);
  const V = new Float64Array(n * n);
  for (let jj = 0; jj < n; jj++) {
    const col = idx[jj] ?? 0;
    s[jj] = Math.max(0, w[col] as number);
    for (let i = 0; i < m; i++) U[i * n + jj] = a[i * n + col] as number;
    for (let i = 0; i < n; i++) V[i * n + jj] = vmat[i * n + col] as number;
  }
  return { U, s, V };
}

/**
 * Singular values only (no U/V) via Golub–Reinsch, skipping the O(n³)
 * left/right transformation accumulation. Requires `m >= n`. Returns n values in
 * descending order.
 *
 * @internal
 */
function golubReinschSingularValues(a0: Float64Array, m: number, n: number): Float64Array {
  const { w } = golubReinschCore(a0, m, n, false);
  for (let i = 0; i < n; i++) w[i] = Math.max(0, w[i] as number);
  w.sort();
  w.reverse();
  return w;
}

/**
 * Scales `A` by a power of two when its entries are extreme, so the
 * norm sums inside the algorithm neither overflow near 1e154 nor underflow
 * near 1e-154. Power-of-two scaling is exact, and singular values scale
 * linearly, so the caller multiplies them by `scale` afterwards.
 *
 * @internal
 */
function prescale(A0: Float64Array): { readonly A: Float64Array; readonly scale: number } {
  let maxAbs = 0;
  for (let i = 0; i < A0.length; i++) {
    const v = Math.abs(A0[i] as number);
    if (v > maxAbs) maxAbs = v;
  }
  if (maxAbs > 0 && (maxAbs > 1e100 || maxAbs < 1e-100)) {
    // floor keeps the factor finite for maxAbs near the largest double (2 ** 1024 overflows).
    const scale = 2 ** Math.floor(Math.log2(maxAbs));
    const A = new Float64Array(A0.length);
    for (let i = 0; i < A0.length; i++) A[i] = (A0[i] as number) / scale;
    return { A, scale };
  }
  return { A: A0, scale: 1 };
}

function transposeDense(A: Float64Array, rows: number, cols: number): Float64Array {
  const At = new Float64Array(cols * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) At[j * rows + i] = A[i * cols + j] as number;
  }
  return At;
}

/**
 * Singular values of `a` only (no U/V).
 *
 * Skips the accumulation of the singular vectors, so it is considerably cheaper
 * than {@link svd}. Entries of extreme magnitude are prescaled by a power of
 * two, as in {@link svd}.
 *
 * @param a - Input matrix of shape (M, N)
 * @returns Singular values of shape (min(M, N),) in descending order
 *
 * @example
 * ```ts
 * import { svdvals } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const s = svdvals(tensor([[3, 0], [0, -2]]));
 * // [3, 2]
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {ConvergenceError} If the QR iteration does not converge
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 */
export function svdvals(a: Tensor): Tensor {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  const m = getDim(a, 0, "svdvals()");
  const n = getDim(a, 1, "svdvals()");
  const k = Math.min(m, n);

  const { data: A0 } = toDenseMatrix2D(a, "svdvals()");
  if (k === 0) return fromDenseVector1D(new Float64Array(0));

  const { A, scale } = prescale(A0);
  const values =
    m >= n
      ? golubReinschSingularValues(A, m, n)
      : golubReinschSingularValues(transposeDense(A, m, n), n, m);
  if (scale !== 1) {
    for (let i = 0; i < values.length; i++) values[i] = (values[i] as number) * scale;
  }
  return fromDenseVector1D(values);
}

function emptySvd(m: number, n: number, fullMatrices: boolean): [Tensor, Tensor, Tensor] {
  const k = Math.min(m, n);
  const s = new Float64Array(0);
  if (!fullMatrices) {
    return [
      fromDenseMatrix2D(m, k, new Float64Array(m * k)),
      fromDenseVector1D(s),
      fromDenseMatrix2D(k, n, new Float64Array(k * n)),
    ];
  }
  const Ufull = identityMatrix(m);
  const VtFull = identityMatrix(n);
  return [fromDenseMatrix2D(m, m, Ufull), fromDenseVector1D(s), fromDenseMatrix2D(n, n, VtFull)];
}

/**
 * Assemble the [U, s, Vᵀ] tensors from a Golub–Reinsch result.
 *
 * When `transposed` is true, the decomposition was computed on Aᵀ (tall) for a
 * wide input A, so the roles of the left and right factors are swapped.
 *
 * @internal
 */
function buildSvdFromGR(
  U: Float64Array,
  s: Float64Array,
  V: Float64Array,
  m: number,
  n: number,
  transposed: boolean,
  fullMatrices: boolean
): [Tensor, Tensor, Tensor] {
  const k = Math.min(m, n);
  const sTensor = fromDenseVector1D(s.slice(0, k));

  if (!transposed) {
    // A = U(m×n) diag(s) V(n×n)ᵀ, with m >= n = k.
    const Vt = transposeDense(V, n, n);
    if (!fullMatrices) {
      return [fromDenseMatrix2D(m, k, U), sTensor, fromDenseMatrix2D(k, n, Vt)];
    }
    // Extend the reduced U (m×n) to a full orthonormal U (m×m).
    let Ufull = U;
    if (m !== n) {
      Ufull = new Float64Array(m * m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) Ufull[i * m + j] = U[i * n + j] as number;
      }
      completeOrthonormalColumns(Ufull, m, m, n);
    }
    return [fromDenseMatrix2D(m, m, Ufull), sTensor, fromDenseMatrix2D(n, n, Vt)];
  }

  // Wide A (m < n): GR ran on Aᵀ (n×m tall) → U is n×m, s is m, V is m×m.
  // A = V(m×m) diag(s) U(n×m)ᵀ, so U_A = V, Vt_A = Uᵀ (m×n), k = m.
  const UA = V; // m×m
  if (!fullMatrices) {
    return [fromDenseMatrix2D(m, m, UA), sTensor, fromDenseMatrix2D(m, n, transposeDense(U, n, m))];
  }
  // Full Vᵀ is n×n: extend U (n×m) to n×n orthonormal, then transpose.
  let Ufull = U;
  if (n !== m) {
    Ufull = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) Ufull[i * n + j] = U[i * m + j] as number;
    }
    completeOrthonormalColumns(Ufull, n, n, m);
  }
  return [
    fromDenseMatrix2D(m, m, UA),
    sTensor,
    fromDenseMatrix2D(n, n, transposeDense(Ufull, n, n)),
  ];
}

/**
 * Singular Value Decomposition.
 *
 * Factorizes matrix A into three matrices: A = U * Σ * V^T
 *
 * **Algorithm**: Golub–Reinsch. Householder reflections reduce A to bidiagonal form,
 * then implicit-shift QR iteration diagonalizes the bidiagonal matrix. Wide
 * matrices (M < N) are handled by decomposing Aᵀ and swapping the factors.
 *
 * **Numerical Stability**: Works on A directly and never forms AᵀA, so small
 * singular values keep their relative accuracy far better than a normal-equations
 * approach. Matrices with extreme entry magnitudes are prescaled by a power of
 * two (an exact operation). When `fullMatrices` is true, the extra columns of U
 * (or rows of Vᵀ) are an orthonormal completion and are not unique.
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param fullMatrices - If true (default), U has shape (M, M) and Vt has shape (N, N).
 *                       If false, U has shape (M, K) and Vt has shape (K, N) where K = min(M, N)
 *
 * **Returns**: [U, s, Vt]
 * - U: Left singular vectors of shape (M, M) or (M, K)
 * - s: Singular values of shape (K,) in descending order
 * - Vt: Right singular vectors transposed, of shape (N, N) or (K, N)
 *
 * **Requirements**:
 * - Input must be 2D matrix
 * - Works with any M x N matrix, including rank-deficient and empty ones
 *
 * **Properties**:
 * - A = U @ diag(s) @ Vt (with the first K columns of U and rows of Vt)
 * - U and Vt have orthonormal columns and rows respectively
 * - Singular values are non-negative and sorted in descending order
 * - Singular vectors are unique only up to a common sign flip per pair
 *
 * @example
 * ```ts
 * import { svd } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4], [5, 6]]);
 * const [U, s, Vt] = svd(A);
 *
 * console.log(U.shape);   // [3, 3] if fullMatrices=true, [3, 2] if false
 * console.log(s.shape);   // [2]
 * console.log(Vt.shape);  // [2, 2]
 *
 * // Reconstruction: A ≈ U[:, :2] @ diag(s) @ Vt
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {ConvergenceError} If the QR iteration does not converge
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 8.6.2
 */
export function svd(a: Tensor, fullMatrices: boolean = true): [Tensor, Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "svd()");
  const n = getDim(a, 1, "svd()");

  // Validate (and densify) before the empty shortcut so non-finite input is always rejected.
  const { data: A0 } = toDenseMatrix2D(a, "svd()");

  if (m === 0 || n === 0) {
    return emptySvd(m, n, fullMatrices);
  }

  const { A, scale } = prescale(A0);

  const rescale = (result: [Tensor, Tensor, Tensor]): [Tensor, Tensor, Tensor] => {
    if (scale === 1) return result;
    const sData = (result[1].data as Float64Array).slice();
    for (let i = 0; i < sData.length; i++) sData[i] = (sData[i] as number) * scale;
    return [result[0], fromDenseVector1D(sData), result[2]];
  };

  if (m >= n) {
    const { U, s, V } = golubReinschSVD(A, m, n);
    return rescale(buildSvdFromGR(U, s, V, m, n, false, fullMatrices));
  }

  // For wide matrices, compute the SVD of A^T (which is tall) and swap the
  // roles of U and V.
  const { U, s, V } = golubReinschSVD(transposeDense(A, m, n), n, m);
  return rescale(buildSvdFromGR(U, s, V, m, n, true, fullMatrices));
}
