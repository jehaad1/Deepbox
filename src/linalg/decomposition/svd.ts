import { DataValidationError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
  at,
  fromDenseMatrix2D,
  fromDenseVector1D,
  getDim,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../_internal";

function identityMatrix(n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

function fillOrthonormalColumn(
  mat: Float64Array,
  rows: number,
  cols: number,
  col: number
): boolean {
  for (let basis = 0; basis < rows; basis++) {
    const vec = new Float64Array(rows);
    vec[basis] = 1;
    for (let j = 0; j < col; j++) {
      let dot = 0;
      for (let i = 0; i < rows; i++) {
        dot += at(mat, i * cols + j) * at(vec, i);
      }
      for (let i = 0; i < rows; i++) {
        vec[i] = at(vec, i) - dot * at(mat, i * cols + j);
      }
    }
    let norm = 0;
    for (let i = 0; i < rows; i++) {
      const v = at(vec, i);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      const inv = 1 / norm;
      for (let i = 0; i < rows; i++) {
        mat[i * cols + col] = at(vec, i) * inv;
      }
      return true;
    }
  }
  return false;
}

function completeOrthonormalColumns(
  mat: Float64Array,
  rows: number,
  cols: number,
  startCol: number
): void {
  for (let col = startCol; col < cols; col++) {
    fillOrthonormalColumn(mat, rows, cols, col);
  }
}

/**
 * Singular values only (no U/V) via Golub–Reinsch: Householder bidiagonalization
 * followed by implicit-shift QR on the bidiagonal, skipping the O(n³) left/right
 * transformation accumulation. Requires `m >= n`. Returns k = min(m, n) values
 * in descending order.
 *
 * @internal
 */
function golubReinschSingularValues(a0: Float64Array, m: number, n: number): Float64Array {
  const a = new Float64Array(a0);
  const w = new Float64Array(n);
  const rv1 = new Float64Array(n);

  const pythag = (x: number, y: number): number => {
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    if (ax > ay) return ax * Math.sqrt(1 + (ay / ax) ** 2);
    return ay === 0 ? 0 : ay * Math.sqrt(1 + (ax / ay) ** 2);
  };
  const sign = (x: number, y: number): number => (y >= 0 ? Math.abs(x) : -Math.abs(x));

  let g = 0;
  let scale = 0;
  let anorm = 0;
  let l = 0;

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

  // QR diagonalization of the bidiagonal (w, rv1) — values only.
  for (let k = n - 1; k >= 0; k--) {
    for (let its = 0; its < 30; its++) {
      let flag = true;
      let nm = 0;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if (l === 0 || Math.abs(rv1[l] as number) <= Number.EPSILON * anorm) {
          flag = false;
          break;
        }
        if (Math.abs(w[nm] as number) <= Number.EPSILON * anorm) break;
      }
      if (flag) {
        let c = 0;
        let s = 1;
        for (let i = l; i <= k; i++) {
          const f = s * (rv1[i] as number);
          rv1[i] = c * (rv1[i] as number);
          if (Math.abs(f) <= Number.EPSILON * anorm) break;
          g = w[i] as number;
          let h = pythag(f, g);
          w[i] = h;
          h = 1 / h;
          c = g * h;
          s = -f * h;
        }
      }
      const z0 = w[k] as number;
      if (l === k) {
        if (z0 < 0) w[k] = -z0;
        break;
      }
      if (its === 29) throw new DataValidationError("SVD did not converge");
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
        z = pythag(f, h);
        w[j] = z;
        if (z !== 0) {
          z = 1 / z;
          c = f * z;
          s = h * z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
      }
      rv1[l] = 0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  const kk = Math.min(m, n);
  for (let i = 0; i < n; i++) w[i] = Math.max(0, w[i] as number);
  w.sort();
  const out = new Float64Array(kk);
  for (let i = 0; i < kk; i++) out[i] = w[n - 1 - i] as number;
  return out;
}
/**
 * Singular values of `a` only (no U/V). Uses the values-only Jacobi path on
 * the taller orientation, with the same power-of-two prescaling as {@link svd}.
 */
export function svdvals(a: Tensor): Tensor {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  const m = getDim(a, 0, "svdvals()");
  const n = getDim(a, 1, "svdvals()");
  const k = Math.min(m, n);
  if (k === 0) return fromDenseVector1D(new Float64Array(0));

  const { data: A0 } = toDenseMatrix2D(a);
  let maxAbs = 0;
  for (let i = 0; i < A0.length; i++) {
    const v = Math.abs(at(A0, i));
    if (v > maxAbs && Number.isFinite(v)) maxAbs = v;
  }
  let scale = 1;
  let A = A0;
  if (maxAbs > 0 && (maxAbs > 1e100 || maxAbs < 1e-100)) {
    const exp = Math.round(Math.log2(maxAbs));
    scale = 2 ** exp;
    A = new Float64Array(A0.length);
    for (let i = 0; i < A0.length; i++) A[i] = at(A0, i) / scale;
  }

  let values: Float64Array;
  if (m >= n) {
    values = golubReinschSingularValues(A, m, n);
  } else {
    const At = new Float64Array(n * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) At[j * m + i] = at(A, i * n + j);
    }
    values = golubReinschSingularValues(At, n, m);
  }
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
 * Singular Value Decomposition.
 *
 * Factorizes matrix A into three matrices: A = U * Σ * V^T
 *
 * **Algorithm**: One-sided Jacobi (Hestenes) SVD
 *
 * **Numerical Stability**: Uses orthogonal rotations on A's columns to
 * directly compute singular values and right singular vectors without forming
 * A^T A. This is substantially more stable for ill-conditioned matrices than
 * the normal-equations approach.
 *
 * **Implementation Details**:
 * 1. Apply cyclic Jacobi rotations to orthogonalize columns of A
 * 2. Column norms converge to singular values
 * 3. Right singular vectors are accumulated as the product of rotations
 * 4. Left singular vectors are normalized columns of the rotated matrix
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param fullMatrices - If true, U has shape (M, M) and V has shape (N, N).
 *                       If false, U has shape (M, K) and V has shape (N, K) where K = min(M, N)
 *
 * **Returns**: [U, s, Vt]
 * - U: Left singular vectors of shape (M, M) or (M, K)
 * - s: Singular values of shape (K,) in descending order
 * - Vt: Right singular vectors transposed of shape (N, N) or (K, N)
 *
 * **Requirements**:
 * - Input must be 2D matrix
 * - Works with any M x N matrix
 *
 * **Properties**:
 * - A = U @ diag(s) @ Vt
 * - U and V are orthogonal matrices
 * - Singular values are non-negative and sorted in descending order
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
 * console.log(Vt.shape);  // [2, 2] if fullMatrices=true, [2, 2] if false
 *
 * // Reconstruction: A ≈ U @ diag(s) @ Vt
 * ```
 *
 * @throws {ShapeError} If input is not 2D
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 *
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 8.6.2
 */
/**
 * Golub–Reinsch SVD (Householder bidiagonalization + implicit-shift QR).
 *
 * Runs in O(n³) with a small constant, replacing the one-sided Jacobi method
 * whose ~10–30 full sweeps each cost O(n³). Requires `m >= n`.
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
  const a = new Float64Array(Ain); // working copy; holds the reduced U on exit
  const w = new Float64Array(n);
  const vmat = new Float64Array(n * n);
  const rv1 = new Float64Array(n);

  const pythag = (x: number, y: number): number => {
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    if (ax > ay) return ax * Math.sqrt(1 + (ay / ax) ** 2);
    return ay === 0 ? 0 : ay * Math.sqrt(1 + (ax / ay) ** 2);
  };
  const sign = (x: number, y: number): number => (y >= 0 ? Math.abs(x) : -Math.abs(x));

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

  // Accumulation of right-hand transformations (V).
  for (let i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g !== 0) {
        for (let j = l; j < n; j++) {
          vmat[j * n + i] = (a[i * n + j] as number) / (a[i * n + l] as number) / g;
        }
        for (let j = l; j < n; j++) {
          let sum = 0;
          for (let k = l; k < n; k++) sum += (a[i * n + k] as number) * (vmat[k * n + j] as number);
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

  // Diagonalization of the bidiagonal form: QR with implicit Wilkinson shifts.
  for (let k = n - 1; k >= 0; k--) {
    for (let its = 0; its < 30; its++) {
      let flag = true;
      let nm = 0;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if (l === 0 || Math.abs(rv1[l] as number) <= Number.EPSILON * anorm) {
          flag = false;
          break;
        }
        if (Math.abs(w[nm] as number) <= Number.EPSILON * anorm) break;
      }
      if (flag) {
        let c = 0;
        let s = 1;
        for (let i = l; i <= k; i++) {
          const f = s * (rv1[i] as number);
          rv1[i] = c * (rv1[i] as number);
          if (Math.abs(f) <= Number.EPSILON * anorm) break;
          g = w[i] as number;
          let h = pythag(f, g);
          w[i] = h;
          h = 1 / h;
          c = g * h;
          s = -f * h;
          for (let j = 0; j < m; j++) {
            const y = a[j * n + nm] as number;
            const z = a[j * n + i] as number;
            a[j * n + nm] = y * c + z * s;
            a[j * n + i] = z * c - y * s;
          }
        }
      }
      const z0 = w[k] as number;
      if (l === k) {
        if (z0 < 0) {
          w[k] = -z0;
          for (let j = 0; j < n; j++) vmat[j * n + k] = -(vmat[j * n + k] as number);
        }
        break;
      }
      if (its === 29) {
        throw new DataValidationError("SVD did not converge");
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
        for (let jj = 0; jj < n; jj++) {
          const xx = vmat[jj * n + j] as number;
          const zz = vmat[jj * n + i] as number;
          vmat[jj * n + j] = xx * c + zz * s;
          vmat[jj * n + i] = zz * c - xx * s;
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
        for (let jj = 0; jj < m; jj++) {
          const yy = a[jj * n + j] as number;
          const zz = a[jj * n + i] as number;
          a[jj * n + j] = yy * c + zz * s;
          a[jj * n + i] = zz * c - yy * s;
        }
      }
      rv1[l] = 0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // Sort singular values descending and permute the matching columns of U, V.
  const idx = Array.from({ length: n }, (_, i) => i).sort(
    (p, q) => (w[q] as number) - (w[p] as number)
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
    const Vt = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) Vt[i * n + j] = V[j * n + i] as number;
    }
    if (!fullMatrices) {
      return [fromDenseMatrix2D(m, k, U), sTensor, fromDenseMatrix2D(k, n, Vt)];
    }
    // Extend the reduced U (m×n) to a full orthonormal U (m×m).
    let Ufull = U;
    if (m !== n) {
      Ufull = new Float64Array(m * m);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) Ufull[i * m + j] = U[i * n + j] as number;
      }
      completeOrthonormalColumns(Ufull, m, m, n);
    }
    return [fromDenseMatrix2D(m, m, Ufull), sTensor, fromDenseMatrix2D(n, n, Vt)];
  }

  // Wide A (m < n): GR ran on Aᵀ (n×m tall) → U is n×m, s is m, V is m×m.
  // A = V(m×m) diag(s) U(n×m)ᵀ, so U_A = V, Vt_A = Uᵀ (m×n), k = m.
  const UA = V; // m×m
  const VtReduced = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) VtReduced[i * n + j] = U[j * m + i] as number;
  }
  if (!fullMatrices) {
    return [fromDenseMatrix2D(m, m, UA), sTensor, fromDenseMatrix2D(m, n, VtReduced)];
  }
  // Full Vᵀ is n×n: extend U (n×m) to n×n orthonormal, then transpose.
  let Ufull = U;
  if (n !== m) {
    Ufull = new Float64Array(n * n);
    for (let j = 0; j < m; j++) {
      for (let i = 0; i < n; i++) Ufull[i * n + j] = U[i * m + j] as number;
    }
    completeOrthonormalColumns(Ufull, n, n, m);
  }
  const VtFull = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) VtFull[i * n + j] = Ufull[j * n + i] as number;
  }
  return [fromDenseMatrix2D(m, m, UA), sTensor, fromDenseMatrix2D(n, n, VtFull)];
}

export function svd(a: Tensor, fullMatrices: boolean = true): [Tensor, Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "svd()");
  const n = getDim(a, 1, "svd()");

  if (m === 0 || n === 0) {
    return emptySvd(m, n, fullMatrices);
  }

  const { data: A0 } = toDenseMatrix2D(a);

  // Pre-scale by a power of two so extreme magnitudes don't over/underflow the
  // Jacobi column-norm sums (sqrt(alpha*beta) overflows near 1e154, and ip*ip
  // underflows near 1e-200). Power-of-two scaling is exact in floating point;
  // singular values scale linearly, so we rescale s afterward.
  let maxAbs = 0;
  for (let i = 0; i < A0.length; i++) {
    const v = Math.abs(at(A0, i));
    if (v > maxAbs && Number.isFinite(v)) maxAbs = v;
  }
  let scale = 1;
  let A = A0;
  if (maxAbs > 0 && (maxAbs > 1e100 || maxAbs < 1e-100)) {
    const exp = Math.round(Math.log2(maxAbs));
    scale = 2 ** exp;
    A = new Float64Array(A0.length);
    for (let i = 0; i < A0.length; i++) A[i] = at(A0, i) / scale;
  }

  const rescale = (result: [Tensor, Tensor, Tensor]): [Tensor, Tensor, Tensor] => {
    if (scale === 1) return result;
    const s = result[1];
    const sData = toDenseVector1D(s);
    const scaled = new Float64Array(sData.length);
    for (let i = 0; i < sData.length; i++) scaled[i] = at(sData, i) * scale;
    return [result[0], fromDenseVector1D(scaled), result[2]];
  };

  if (m >= n) {
    const { U, s, V } = golubReinschSVD(A, m, n);
    return rescale(buildSvdFromGR(U, s, V, m, n, false, fullMatrices));
  }

  // For wide matrices, compute the SVD of A^T (which is tall) and swap the
  // roles of U and V.
  const At = new Float64Array(n * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      At[j * m + i] = A[i * n + j] as number;
    }
  }
  const { U, s, V } = golubReinschSVD(At, n, m);
  return rescale(buildSvdFromGR(U, s, V, m, n, true, fullMatrices));
}
