import { ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { at, fromDenseMatrix2D, fromDenseVector1D, getDim, toDenseMatrix2D } from "../_internal";

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

function jacobiSVDCols(
  a: Float64Array,
  m: number,
  n: number,
  maxSweeps = 100,
  tol = 1e-12
): { readonly B: Float64Array; readonly V: Float64Array } {
  const B = new Float64Array(a);
  const V = identityMatrix(n);

  if (m === 0 || n === 0) return { B, V };

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxCorr = 0;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;

        for (let i = 0; i < m; i++) {
          const ip = at(B, i * n + p);
          const iq = at(B, i * n + q);
          alpha += ip * ip;
          beta += iq * iq;
          gamma += ip * iq;
        }

        if (alpha === 0 || beta === 0) continue;
        const denom = Math.sqrt(alpha * beta);
        if (denom === 0) continue;

        const corr = Math.abs(gamma) / denom;
        if (corr > maxCorr) maxCorr = corr;
        if (corr <= tol) continue;

        const tau = (beta - alpha) / (2 * gamma);
        const sign = tau >= 0 ? 1 : -1;
        const t = sign / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;

        for (let i = 0; i < m; i++) {
          const ip = at(B, i * n + p);
          const iq = at(B, i * n + q);
          B[i * n + p] = c * ip - s * iq;
          B[i * n + q] = s * ip + c * iq;
        }

        for (let i = 0; i < n; i++) {
          const vip = at(V, i * n + p);
          const viq = at(V, i * n + q);
          V[i * n + p] = c * vip - s * viq;
          V[i * n + q] = s * vip + c * viq;
        }
      }
    }

    if (maxCorr <= tol) break;
  }

  return { B, V };
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

function buildSvdFromJacobi(
  B: Float64Array,
  V: Float64Array,
  m: number,
  n: number,
  fullMatrices: boolean
): [Tensor, Tensor, Tensor] {
  const k = Math.min(m, n);
  const norms = new Float64Array(n);
  let maxSigma = 0;

  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i < m; i++) {
      const v = at(B, i * n + j);
      sum += v * v;
    }
    const sigma = Math.sqrt(sum);
    norms[j] = sigma;
    if (sigma > maxSigma) maxSigma = sigma;
  }

  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((i, j) => at(norms, j) - at(norms, i));

  const sigmaTol = Number.EPSILON * Math.max(m, n) * (maxSigma || 1);

  const s = new Float64Array(k);
  const U = new Float64Array(m * k);
  const filled = new Array<boolean>(k).fill(false);

  for (let j = 0; j < k; j++) {
    const col = idx[j] ?? 0;
    const sigma = norms[col] ?? 0;
    s[j] = sigma;
    if (sigma > sigmaTol) {
      const inv = 1 / sigma;
      for (let i = 0; i < m; i++) {
        U[i * k + j] = at(B, i * n + col) * inv;
      }
      filled[j] = true;
    }
  }

  for (let j = 0; j < k; j++) {
    if (!filled[j]) {
      fillOrthonormalColumn(U, m, k, j);
    }
  }

  const Vsorted = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const col = idx[j] ?? 0;
    for (let i = 0; i < n; i++) {
      Vsorted[i * n + j] = at(V, i * n + col);
    }
  }

  if (!fullMatrices) {
    const Vt = new Float64Array(k * n);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n; j++) {
        Vt[i * n + j] = at(Vsorted, j * n + i);
      }
    }
    return [fromDenseMatrix2D(m, k, U), fromDenseVector1D(s), fromDenseMatrix2D(k, n, Vt)];
  }

  let Ufull = U;
  if (m !== k) {
    Ufull = new Float64Array(m * m);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < m; i++) {
        Ufull[i * m + j] = at(U, i * k + j);
      }
    }
    completeOrthonormalColumns(Ufull, m, m, k);
  }

  const VtFull = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      VtFull[i * n + j] = at(Vsorted, j * n + i);
    }
  }

  return [fromDenseMatrix2D(m, m, Ufull), fromDenseVector1D(s), fromDenseMatrix2D(n, n, VtFull)];
}

function buildSvdFromJacobiTransposed(
  B: Float64Array,
  V: Float64Array,
  m: number,
  n: number,
  fullMatrices: boolean
): [Tensor, Tensor, Tensor] {
  const k = Math.min(m, n);
  const norms = new Float64Array(k);
  let maxSigma = 0;

  for (let j = 0; j < k; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = at(B, i * k + j);
      sum += v * v;
    }
    const sigma = Math.sqrt(sum);
    norms[j] = sigma;
    if (sigma > maxSigma) maxSigma = sigma;
  }

  const idx = new Array<number>(k);
  for (let i = 0; i < k; i++) idx[i] = i;
  idx.sort((i, j) => at(norms, j) - at(norms, i));

  const sigmaTol = Number.EPSILON * Math.max(m, n) * (maxSigma || 1);

  const s = new Float64Array(k);
  const U = new Float64Array(m * k);
  const Vcols = new Float64Array(n * k);
  const filledV = new Array<boolean>(k).fill(false);

  for (let j = 0; j < k; j++) {
    const col = idx[j] ?? 0;
    const sigma = norms[col] ?? 0;
    s[j] = sigma;

    for (let i = 0; i < m; i++) {
      U[i * k + j] = at(V, i * k + col);
    }

    if (sigma > sigmaTol) {
      const inv = 1 / sigma;
      for (let i = 0; i < n; i++) {
        Vcols[i * k + j] = at(B, i * k + col) * inv;
      }
      filledV[j] = true;
    }
  }

  for (let j = 0; j < k; j++) {
    if (!filledV[j]) {
      fillOrthonormalColumn(Vcols, n, k, j);
    }
  }

  if (!fullMatrices) {
    const Vt = new Float64Array(k * n);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n; j++) {
        Vt[i * n + j] = at(Vcols, j * k + i);
      }
    }
    return [fromDenseMatrix2D(m, k, U), fromDenseVector1D(s), fromDenseMatrix2D(k, n, Vt)];
  }

  let Vfull = Vcols;
  if (n !== k) {
    Vfull = new Float64Array(n * n);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < n; i++) {
        Vfull[i * n + j] = at(Vcols, i * k + j);
      }
    }
    completeOrthonormalColumns(Vfull, n, n, k);
  }

  const VtFull = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      VtFull[i * n + j] = at(Vfull, j * n + i);
    }
  }

  return [fromDenseMatrix2D(m, m, U), fromDenseVector1D(s), fromDenseMatrix2D(n, n, VtFull)];
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
export function svd(a: Tensor, fullMatrices: boolean = true): [Tensor, Tensor, Tensor] {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");

  const m = getDim(a, 0, "svd()");
  const n = getDim(a, 1, "svd()");

  if (m === 0 || n === 0) {
    return emptySvd(m, n, fullMatrices);
  }

  const { data: A } = toDenseMatrix2D(a);
  const maxSweeps = 100;
  const tol = 1e-12;

  if (m >= n) {
    const { B, V } = jacobiSVDCols(A, m, n, maxSweeps, tol);
    return buildSvdFromJacobi(B, V, m, n, fullMatrices);
  }

  // For wide matrices, compute SVD of A^T and swap roles of U and V.
  const At = new Float64Array(n * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      At[j * m + i] = at(A, i * n + j);
    }
  }

  const { B, V } = jacobiSVDCols(At, n, m, maxSweeps, tol);
  return buildSvdFromJacobiTransposed(B, V, m, n, fullMatrices);
}
