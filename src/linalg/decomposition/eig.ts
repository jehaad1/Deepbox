import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import type { Tensor } from "../../ndarray";
import {
  at,
  atArr,
  fromDenseMatrix2D,
  fromDenseVector1D,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../_internal";
import { svd } from "./svd";

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

function jacobiEigenSymmetric(
  a: Float64Array,
  n: number,
  maxSweeps = 100,
  tol = 1e-12
): { readonly values: Float64Array; readonly vectors: Float64Array } {
  const A = new Float64Array(a);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let p = 0;
    let q = 1;
    let max = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(at(A, i * n + j));
        if (v > max) {
          max = v;
          p = i;
          q = j;
        }
      }
    }

    if (max < tol) break;

    const app = at(A, p * n + p);
    const aqq = at(A, q * n + q);
    const apq = at(A, p * n + q);
    if (apq === 0) continue;

    const tau = (aqq - app) / (2 * apq);
    const sign = tau >= 0 ? 1 : -1;
    const t = sign / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const aik = at(A, k * n + p);
      const akq = at(A, k * n + q);
      const newAik = c * aik - s * akq;
      const newAkq = s * aik + c * akq;
      A[k * n + p] = newAik;
      A[p * n + k] = newAik;
      A[k * n + q] = newAkq;
      A[q * n + k] = newAkq;
    }

    const newApp = c * c * app - 2 * s * c * apq + s * s * aqq;
    const newAqq = s * s * app + 2 * s * c * apq + c * c * aqq;

    A[p * n + p] = newApp;
    A[q * n + q] = newAqq;
    A[p * n + q] = 0;
    A[q * n + p] = 0;

    for (let k = 0; k < n; k++) {
      const vkp = at(V, k * n + p);
      const vkq = at(V, k * n + q);
      V[k * n + p] = c * vkp - s * vkq;
      V[k * n + q] = s * vkp + c * vkq;
    }
  }

  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = at(A, i * n + i);
  return { values, vectors: V };
}

function matmulSquare(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += at(a, i * n + k) * at(b, k * n + j);
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

function qrFactorSquare(
  a: Float64Array,
  n: number
): { readonly Q: Float64Array; readonly R: Float64Array } {
  // Modified Gram-Schmidt is sufficient here for QR iteration on small n.
  const Q = new Float64Array(n * n);
  const R = new Float64Array(n * n);

  const v = new Float64Array(n * n);
  v.set(a);

  const fillOrthonormalColumn = (col: number): void => {
    for (let basis = 0; basis < n; basis++) {
      const vec = new Float64Array(n);
      vec[basis] = 1;
      for (let j = 0; j < col; j++) {
        let dot = 0;
        for (let k = 0; k < n; k++) {
          dot += at(Q, k * n + j) * at(vec, k);
        }
        for (let k = 0; k < n; k++) {
          vec[k] = at(vec, k) - dot * at(Q, k * n + j);
        }
      }
      let norm = 0;
      for (let k = 0; k < n; k++) {
        const val = at(vec, k);
        norm += val * val;
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-12) {
        const inv = 1 / norm;
        for (let k = 0; k < n; k++) {
          Q[k * n + col] = at(vec, k) * inv;
        }
        return;
      }
    }
  };

  for (let j = 0; j < n; j++) {
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let k = 0; k < n; k++) {
        dot += at(Q, k * n + i) * at(v, k * n + j);
      }
      R[i * n + j] = dot;
      for (let k = 0; k < n; k++) {
        v[k * n + j] = at(v, k * n + j) - dot * at(Q, k * n + i);
      }
    }

    let norm = 0;
    for (let k = 0; k < n; k++) {
      const x = at(v, k * n + j);
      norm += x * x;
    }
    norm = Math.sqrt(norm);
    R[j * n + j] = norm;
    if (norm > 1e-12) {
      const inv = 1 / norm;
      for (let k = 0; k < n; k++) {
        Q[k * n + j] = at(v, k * n + j) * inv;
      }
    } else {
      fillOrthonormalColumn(j);
    }
  }

  return { Q, R };
}

/**
 * Reduce matrix to upper Hessenberg form using Householder reflections.
 * Returns H and Q such that A = Q * H * Q^T where H is upper Hessenberg.
 *
 * Upper Hessenberg form has zeros below the first subdiagonal, which
 * significantly speeds up QR iteration convergence.
 *
 * @internal
 */
function hessenbergReduce(
  a: Float64Array,
  n: number
): { readonly H: Float64Array; readonly Q: Float64Array } {
  const H = new Float64Array(a);
  const Q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;

  const v = new Float64Array(n);

  for (let col = 0; col < n - 2; col++) {
    // Extract column col below diagonal
    let norm = 0;
    for (let i = col + 1; i < n; i++) {
      const val = at(H, i * n + col);
      v[i] = val;
      norm += val * val;
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-14) continue;

    // Choose sign to avoid cancellation
    const vkp1 = at(v, col + 1);
    const sign = vkp1 >= 0 ? 1 : -1;
    v[col + 1] = vkp1 + sign * norm;

    // Normalize v
    let vnorm = 0;
    for (let i = col + 1; i < n; i++) {
      const val = at(v, i);
      vnorm += val * val;
    }
    vnorm = Math.sqrt(vnorm);
    if (vnorm < 1e-14) continue;

    for (let i = col + 1; i < n; i++) {
      v[i] = at(v, i) / vnorm;
    }

    // Apply H = (I - 2*v*v^T) * H from left
    for (let j = col; j < n; j++) {
      let dot = 0;
      for (let i = col + 1; i < n; i++) {
        dot += at(v, i) * at(H, i * n + j);
      }
      dot *= 2;
      for (let i = col + 1; i < n; i++) {
        H[i * n + j] = at(H, i * n + j) - dot * at(v, i);
      }
    }

    // Apply H = H * (I - 2*v*v^T) from right
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = col + 1; j < n; j++) {
        dot += at(H, i * n + j) * at(v, j);
      }
      dot *= 2;
      for (let j = col + 1; j < n; j++) {
        H[i * n + j] = at(H, i * n + j) - dot * at(v, j);
      }
    }

    // Accumulate Q = Q * (I - 2*v*v^T)
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = col + 1; j < n; j++) {
        dot += at(Q, i * n + j) * at(v, j);
      }
      dot *= 2;
      for (let j = col + 1; j < n; j++) {
        Q[i * n + j] = at(Q, i * n + j) - dot * at(v, j);
      }
    }
  }

  return { H, Q };
}

/**
 * Compute eigenvalues and eigenvectors of a square matrix.
 *
 * Solves A * v = λ * v where λ are eigenvalues and v are eigenvectors.
 *
 * **Algorithm**:
 * - Symmetric matrices: Jacobi iteration (stable, accurate)
 * - General matrices: QR iteration with Hessenberg reduction
 *
 * **Limitations**:
 * - Only real eigenvalues are supported. Non-symmetric matrices whose
 *   spectrum includes complex eigenvalues will cause an
 *   {@link InvalidParameterError} to be thrown.
 * - For symmetric/Hermitian matrices, use `eigh()` for better performance
 * - May not converge for some matrices (bounded QR iterations; see options)
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 * @param options - Optional configuration overrides (see {@link EigOptions})
 * @param options.maxIter - Maximum QR iterations (default: 300)
 * @param options.tol - Convergence tolerance for subdiagonal norm (default: 1e-10)
 *
 * **Returns**: [eigenvalues, eigenvectors]
 * - eigenvalues: Real values of shape (N,)
 * - eigenvectors: Column vectors of shape (N, N) where eigenvectors[:,i] corresponds to eigenvalues[i]
 *
 * **Requirements**:
 * - Input must be square matrix
 * - Matrix must have only real eigenvalues
 * - For symmetric matrices, use eigh() for better performance
 *
 * **Properties**:
 * - A @ v[:,i] = λ[i] * v[:,i]
 * - Eigenvectors are normalized
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
 * ```
 *
 * @throws {ShapeError} If input is not square matrix
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If matrix has complex eigenvalues
 *
 * @see {@link https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix | Wikipedia: Eigendecomposition}
 * @see Golub & Van Loan, "Matrix Computations", Algorithm 7.5.2
 */
export type EigOptions = {
  readonly maxIter?: number;
  readonly tol?: number;
};

export function eig(a: Tensor, options: EigOptions = {}): [Tensor, Tensor] {
  const n = getSquareMatrixSize(a, "eig");
  if (n === 0) {
    return [fromDenseVector1D(new Float64Array(0)), fromDenseMatrix2D(0, 0, new Float64Array(0))];
  }

  // If symmetric, use Jacobi (real eigenvalues, orthonormal eigenvectors).
  const { data: A0 } = toDenseMatrix2D(a);

  let symmetric = true;
  for (let i = 0; i < n && symmetric; i++) {
    for (let j = i + 1; j < n; j++) {
      const aij = at(A0, i * n + j);
      const aji = at(A0, j * n + i);
      if (Math.abs(aij - aji) > 1e-10) {
        symmetric = false;
        break;
      }
    }
  }

  if (symmetric) {
    return eigh(a);
  }

  // Reduce to Hessenberg form first for faster QR iteration convergence
  const { H } = hessenbergReduce(A0, n);

  // Shifted QR iteration on Hessenberg matrix (converges faster and more reliably)
  const maxIter = options.maxIter ?? 300;
  const convTol = options.tol ?? 1e-10;
  let Ak = H;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    let off = 0;
    for (let i = 1; i < n; i++) {
      const v = at(Ak, i * n + (i - 1));
      off += v * v;
    }
    if (Math.sqrt(off) < convTol) {
      converged = true;
      break;
    }

    const mu = at(Ak, (n - 1) * n + (n - 1));
    const shifted = new Float64Array(Ak);
    for (let i = 0; i < n; i++) {
      shifted[i * n + i] = at(shifted, i * n + i) - mu;
    }

    const { Q, R } = qrFactorSquare(shifted, n);
    Ak = matmulSquare(R, Q, n);
    for (let i = 0; i < n; i++) {
      Ak[i * n + i] = at(Ak, i * n + i) + mu;
    }
  }

  // Clean small subdiagonal entries to stabilize eigenvalue detection
  for (let i = 1; i < n; i++) {
    const v = at(Ak, i * n + (i - 1));
    if (Math.abs(v) < convTol) {
      Ak[i * n + (i - 1)] = 0;
    }
  }

  // Detect complex eigenvalues by checking for 2x2 blocks in the quasi-upper
  // triangular (real Schur) form. A 2x2 diagonal block [[a, b], [c, d]] has
  // complex conjugate eigenvalues when its discriminant (a-d)^2 + 4*b*c < 0.
  // This check runs before the convergence check because matrices with complex
  // eigenvalues will never converge (2x2 blocks persist), and the complex
  // eigenvalue error is more informative than a generic convergence failure.
  const complexTol = 1e-8;
  let hasComplex = false;
  for (let i = 0; i < n - 1; i++) {
    const subdiag = Math.abs(at(Ak, (i + 1) * n + i));
    if (subdiag > complexTol) {
      // Non-negligible subdiagonal element indicates a 2x2 block
      const a11 = at(Ak, i * n + i);
      const a12 = at(Ak, i * n + (i + 1));
      const a21 = at(Ak, (i + 1) * n + i);
      const a22 = at(Ak, (i + 1) * n + (i + 1));
      const discriminant = (a11 - a22) * (a11 - a22) + 4 * a12 * a21;
      if (discriminant < -complexTol) {
        hasComplex = true;
        break;
      }
    }
  }

  if (hasComplex) {
    throw new InvalidParameterError(
      "Matrix has complex eigenvalues, which are not supported. " +
        "Only matrices with real eigenvalues can be decomposed. " +
        "Symmetric matrices always have real eigenvalues.",
      "a"
    );
  }

  if (!converged) {
    throw new ConvergenceError(`eig() failed to converge after ${maxIter} iterations`, {
      iterations: maxIter,
      tolerance: convTol,
    });
  }

  const evals = new Float64Array(n);
  for (let i = 0; i < n; i++) evals[i] = at(Ak, i * n + i);

  // Compute eigenvectors by finding nullspace of (A - λI)
  const vectors = new Float64Array(n * n);
  const used = new Array<boolean>(n).fill(false);
  const clusterTol = 1e-8;

  for (let i = 0; i < n; i++) {
    if (used[i]) continue;
    const lambda = at(evals, i);
    const cluster = [i];
    used[i] = true;
    for (let j = i + 1; j < n; j++) {
      if (used[j]) continue;
      const diff = Math.abs(at(evals, j) - lambda);
      const scale = Math.max(1, Math.abs(lambda));
      if (diff <= clusterTol * scale) {
        used[j] = true;
        cluster.push(j);
      }
    }

    const basis = (() => {
      const M = new Float64Array(A0);
      for (let d = 0; d < n; d++) {
        M[d * n + d] = at(M, d * n + d) - lambda;
      }
      const [_, s, Vt] = svd(fromDenseMatrix2D(n, n, M), true);
      const sDense = toDenseVector1D(s);
      const { data: VtData } = toDenseMatrix2D(Vt);
      const vData = new Float64Array(n * n);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          vData[r * n + c] = at(VtData, c * n + r);
        }
      }
      const sMax = sDense.length === 0 ? 0 : at(sDense, 0);
      const tol = Number.EPSILON * n * sMax;
      const basisVecs: Float64Array[] = [];
      for (let r = sDense.length - 1; r >= 0; r--) {
        if (at(sDense, r) <= tol) {
          const vec = new Float64Array(n);
          for (let c = 0; c < n; c++) {
            vec[c] = at(vData, c * n + r);
          }
          basisVecs.push(vec);
        }
      }
      if (basisVecs.length === 0 && sDense.length > 0) {
        const r = sDense.length - 1;
        const vec = new Float64Array(n);
        for (let c = 0; c < n; c++) {
          vec[c] = at(vData, c * n + r);
        }
        basisVecs.push(vec);
      }
      return basisVecs.length === 0 ? [new Float64Array(n)] : basisVecs;
    })();

    for (let b = 0; b < cluster.length; b++) {
      const colIndex = cluster[b];
      if (colIndex === undefined) {
        throw new ShapeError("eig(): eigenvector index is missing");
      }
      const vec = basis[b % basis.length];
      if (vec === undefined) {
        throw new ShapeError("eig(): eigenvector basis is missing");
      }
      let norm = 0;
      for (let r = 0; r < n; r++) {
        const v = at(vec, r);
        norm += v * v;
      }
      if (norm === 0) {
        for (let r = 0; r < n; r++) {
          vectors[r * n + colIndex] = r === colIndex ? 1 : 0;
        }
        continue;
      }
      const inv = 1 / Math.sqrt(norm);
      for (let r = 0; r < n; r++) {
        vectors[r * n + colIndex] = at(vec, r) * inv;
      }
    }
  }

  return [fromDenseVector1D(evals), fromDenseMatrix2D(n, n, vectors)];
}

/**
 * Compute eigenvalues only (faster than eig).
 *
 * **Parameters**:
 * @param a - Square matrix of shape (N, N)
 *
 * **Returns**: eigenvalues - Array of real eigenvalues
 *
 * **Limitations**:
 * - Matrices with complex eigenvalues are not supported and will throw
 *   {@link InvalidParameterError}
 *
 * @example
 * ```ts
 * import { eigvals } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [2, 1]]);
 * const eigenvalues = eigvals(A);
 * console.log(eigenvalues);  // [3, -1]
 * ```
 */
export function eigvals(a: Tensor, options?: EigOptions): Tensor {
  const [eigenvalues] = eig(a, options);
  return eigenvalues;
}

/**
 * Compute eigenvalues only of symmetric matrix (faster than eigh).
 *
 * **Parameters**:
 * @param a - Symmetric square matrix of shape (N, N)
 *
 * **Returns**: eigenvalues - Array of real eigenvalues
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
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input is not symmetric
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function eigvalsh(a: Tensor): Tensor {
  const n = getSquareMatrixSize(a, "eigvalsh");
  const { data: A } = toDenseMatrix2D(a);

  // Validate symmetry
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const aij = at(A, i * n + j);
      const aji = at(A, j * n + i);
      if (Math.abs(aij - aji) > 1e-10) {
        throw new DataValidationError("Input must be symmetric for eigvalsh");
      }
    }
  }

  const [eigenvalues] = eigh(a);
  return eigenvalues;
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
 * - All eigenvalues are real
 * - Eigenvectors are orthonormal
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
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input is not symmetric
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function eigh(a: Tensor): [Tensor, Tensor] {
  const n = getSquareMatrixSize(a, "eigh");
  if (n === 0) {
    return [fromDenseVector1D(new Float64Array(0)), fromDenseMatrix2D(0, 0, new Float64Array(0))];
  }

  const { data: A } = toDenseMatrix2D(a);

  // Validate symmetry
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const aij = at(A, i * n + j);
      const aji = at(A, j * n + i);
      if (Math.abs(aij - aji) > 1e-10) {
        throw new DataValidationError("Input must be symmetric for eigh");
      }
    }
  }

  const { values, vectors } = jacobiEigenSymmetric(A, n);

  // Sort ascending like NumPy eigh
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((i, j) => at(values, i) - at(values, j));

  const outVals = new Float64Array(n);
  const outVecs = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = atArr(idx, col);
    outVals[col] = at(values, src);
    for (let row = 0; row < n; row++) {
      outVecs[row * n + col] = at(vectors, row * n + src);
    }
  }

  return [fromDenseVector1D(outVals), fromDenseMatrix2D(n, n, outVecs)];
}
