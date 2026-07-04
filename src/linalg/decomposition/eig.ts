import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import type { Tensor } from "../../ndarray";
import {
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

/**
 * Householder tridiagonalization of a symmetric matrix (EISPACK tred2).
 *
 * Reduces the symmetric matrix in `z` (row-major n×n, overwritten with the
 * accumulated orthogonal transform) to tridiagonal form with diagonal `d`
 * and off-diagonal `e` (e[0] unused). O(4n³/3) — the cyclic-Jacobi sweep
 * this replaces cost ~10x more FLOPs at 100×100.
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
 * (EISPACK tql2). Consumes `d`/`e` from {@link tred2}, leaves ascending is
 * NOT guaranteed — callers sort. Eigenvectors are accumulated into `z`.
 */
function tql2(n: number, d: Float64Array, e: Float64Array, z: Float64Array): void {
  for (let i = 1; i < n; i++) e[i - 1] = e[i] as number;
  e[n - 1] = 0;

  let f = 0;
  let tst1 = 0;
  const eps = Number.EPSILON;
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
        if (iter++ === 60) {
          throw new ConvergenceError("eigh: QL iteration failed to converge", { iterations: 60 });
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
          for (let k = 0; k < n; k++) {
            h = z[k * n + i + 1] as number;
            const zki = z[k * n + i] as number;
            z[k * n + i + 1] = s * zki + c * h;
            z[k * n + i] = c * zki - s * h;
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
 * implicit shifts. Same interface as the Jacobi routine it replaces.
 */
function symmetricEigen(
  a: Float64Array,
  n: number
): { readonly values: Float64Array; readonly vectors: Float64Array } {
  const z = new Float64Array(a);
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  if (n === 0) return { values: d, vectors: z };
  if (n === 1) {
    d[0] = a[0] as number;
    z[0] = 1;
    return { values: d, vectors: z };
  }
  tred2(n, z, d, e);
  tql2(n, d, e, z);
  return { values: d, vectors: z };
}

/**
 * Tridiagonalize (accumulating no transform) then run the QL sweep without
 * eigenvector updates — eigenvalues only, ~2x less work than
 * {@link symmetricEigen}. Returns unsorted eigenvalues.
 */
function symmetricEigenvalues(a: Float64Array, n: number): Float64Array {
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  if (n === 0) return d;
  if (n === 1) {
    d[0] = a[0] as number;
    return d;
  }
  const A = new Float64Array(a);
  tred2(n, A, d, e);
  // QL without eigenvector accumulation (structure mirrors tql2).
  for (let i = 1; i < n; i++) e[i - 1] = e[i] as number;
  e[n - 1] = 0;
  let f = 0;
  let tst1 = 0;
  const eps = Number.EPSILON;
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
        if (iter++ === 60) {
          throw new ConvergenceError("eigvalsh: QL iteration failed to converge", {
            iterations: 60,
          });
        }
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
        }
        p = (-s * s2 * c3 * el1 * (e[l] as number)) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l] as number) > eps * tst1);
    }
    d[l] = (d[l] as number) + f;
    e[l] = 0;
  }
  return d;
}

function matmulSquare(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += (a[i * n + k] as number) * (b[k * n + j] as number);
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
          dot += (Q[k * n + j] as number) * (vec[k] as number);
        }
        for (let k = 0; k < n; k++) {
          vec[k] = (vec[k] as number) - dot * (Q[k * n + j] as number);
        }
      }
      let norm = 0;
      for (let k = 0; k < n; k++) {
        const val = vec[k] as number;
        norm += val * val;
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-12) {
        const inv = 1 / norm;
        for (let k = 0; k < n; k++) {
          Q[k * n + col] = (vec[k] as number) * inv;
        }
        return;
      }
    }
  };

  for (let j = 0; j < n; j++) {
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let k = 0; k < n; k++) {
        dot += (Q[k * n + i] as number) * (v[k * n + j] as number);
      }
      R[i * n + j] = dot;
      for (let k = 0; k < n; k++) {
        v[k * n + j] = (v[k * n + j] as number) - dot * (Q[k * n + i] as number);
      }
    }

    let norm = 0;
    for (let k = 0; k < n; k++) {
      const x = v[k * n + j] as number;
      norm += x * x;
    }
    norm = Math.sqrt(norm);
    R[j * n + j] = norm;
    if (norm > 1e-12) {
      const inv = 1 / norm;
      for (let k = 0; k < n; k++) {
        Q[k * n + j] = (v[k * n + j] as number) * inv;
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
      const val = H[i * n + col] as number;
      v[i] = val;
      norm += val * val;
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-14) continue;

    // Choose sign to avoid cancellation
    const vkp1 = v[col + 1] as number;
    const sign = vkp1 >= 0 ? 1 : -1;
    v[col + 1] = vkp1 + sign * norm;

    // Normalize v
    let vnorm = 0;
    for (let i = col + 1; i < n; i++) {
      const val = v[i] as number;
      vnorm += val * val;
    }
    vnorm = Math.sqrt(vnorm);
    if (vnorm < 1e-14) continue;

    for (let i = col + 1; i < n; i++) {
      v[i] = (v[i] as number) / vnorm;
    }

    // Apply H = (I - 2*v*v^T) * H from left
    for (let j = col; j < n; j++) {
      let dot = 0;
      for (let i = col + 1; i < n; i++) {
        dot += (v[i] as number) * (H[i * n + j] as number);
      }
      dot *= 2;
      for (let i = col + 1; i < n; i++) {
        H[i * n + j] = (H[i * n + j] as number) - dot * (v[i] as number);
      }
    }

    // Apply H = H * (I - 2*v*v^T) from right
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = col + 1; j < n; j++) {
        dot += (H[i * n + j] as number) * (v[j] as number);
      }
      dot *= 2;
      for (let j = col + 1; j < n; j++) {
        H[i * n + j] = (H[i * n + j] as number) - dot * (v[j] as number);
      }
    }

    // Accumulate Q = Q * (I - 2*v*v^T)
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = col + 1; j < n; j++) {
        dot += (Q[i * n + j] as number) * (v[j] as number);
      }
      dot *= 2;
      for (let j = col + 1; j < n; j++) {
        Q[i * n + j] = (Q[i * n + j] as number) - dot * (v[j] as number);
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
 * @see {@link https://deepbox.dev/docs/linalg-decompositions | Deepbox Linear Algebra}
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
      const aij = A0[i * n + j] as number;
      const aji = A0[j * n + i] as number;
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
      const v = Ak[i * n + (i - 1)] as number;
      off += v * v;
    }
    if (Math.sqrt(off) < convTol) {
      converged = true;
      break;
    }

    // Wilkinson shift from the trailing 2x2 block. A pure Rayleigh shift
    // (mu = A[n-1][n-1]) livelocks on matrices like [[0,2],[0.5,0]] whose
    // trailing diagonal entry never moves. Every 12th iteration applies an
    // exceptional shift to break any remaining symmetric cycling.
    let mu: number;
    const t11 = Ak[(n - 2) * n + (n - 2)] as number;
    const t12 = Ak[(n - 2) * n + (n - 1)] as number;
    const t21 = Ak[(n - 1) * n + (n - 2)] as number;
    const t22 = Ak[(n - 1) * n + (n - 1)] as number;
    const delta = (t11 - t22) / 2;
    const disc = delta * delta + t12 * t21;
    if ((iter + 1) % 12 === 0) {
      mu = t22 + Math.abs(t21) + Math.abs(delta);
    } else if (disc >= 0) {
      const sgn = delta >= 0 ? 1 : -1;
      const denom = delta + sgn * Math.sqrt(disc);
      mu = denom === 0 ? t22 : t22 - (t12 * t21) / denom;
    } else {
      // Complex eigenvalue pair in the trailing block; fall back to the
      // Rayleigh shift (the complex-pair detection below reports it).
      mu = t22;
    }
    const shifted = new Float64Array(Ak);
    for (let i = 0; i < n; i++) {
      shifted[i * n + i] = (shifted[i * n + i] as number) - mu;
    }

    const { Q, R } = qrFactorSquare(shifted, n);
    Ak = matmulSquare(R, Q, n);
    for (let i = 0; i < n; i++) {
      Ak[i * n + i] = (Ak[i * n + i] as number) + mu;
    }
  }

  // Clean small subdiagonal entries to stabilize eigenvalue detection
  for (let i = 1; i < n; i++) {
    const v = Ak[i * n + (i - 1)] as number;
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
    const subdiag = Math.abs(Ak[(i + 1) * n + i] as number);
    if (subdiag > complexTol) {
      // Non-negligible subdiagonal element indicates a 2x2 block
      const a11 = Ak[i * n + i] as number;
      const a12 = Ak[i * n + (i + 1)] as number;
      const a21 = Ak[(i + 1) * n + i] as number;
      const a22 = Ak[(i + 1) * n + (i + 1)] as number;
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
  for (let i = 0; i < n; i++) evals[i] = Ak[i * n + i] as number;

  // Compute eigenvectors by finding nullspace of (A - λI)
  const vectors = new Float64Array(n * n);
  const used = new Array<boolean>(n).fill(false);
  const clusterTol = 1e-8;

  for (let i = 0; i < n; i++) {
    if (used[i]) continue;
    const lambda = evals[i] as number;
    const cluster = [i];
    used[i] = true;
    for (let j = i + 1; j < n; j++) {
      if (used[j]) continue;
      const diff = Math.abs((evals[j] as number) - lambda);
      const scale = Math.max(1, Math.abs(lambda));
      if (diff <= clusterTol * scale) {
        used[j] = true;
        cluster.push(j);
      }
    }

    const basis = (() => {
      const M = new Float64Array(A0);
      for (let d = 0; d < n; d++) {
        M[d * n + d] = (M[d * n + d] as number) - lambda;
      }
      const [_, s, Vt] = svd(fromDenseMatrix2D(n, n, M), true);
      const sDense = toDenseVector1D(s);
      const { data: VtData } = toDenseMatrix2D(Vt);
      const vData = new Float64Array(n * n);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          vData[r * n + c] = VtData[c * n + r] as number;
        }
      }
      const sMax = sDense.length === 0 ? 0 : (sDense[0] as number);
      const tol = Number.EPSILON * n * sMax;
      const basisVecs: Float64Array[] = [];
      for (let r = sDense.length - 1; r >= 0; r--) {
        if ((sDense[r] as number) <= tol) {
          const vec = new Float64Array(n);
          for (let c = 0; c < n; c++) {
            vec[c] = vData[c * n + r] as number;
          }
          basisVecs.push(vec);
        }
      }
      if (basisVecs.length === 0 && sDense.length > 0) {
        const r = sDense.length - 1;
        const vec = new Float64Array(n);
        for (let c = 0; c < n; c++) {
          vec[c] = vData[c * n + r] as number;
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
        const v = vec[r] as number;
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
        vectors[r * n + colIndex] = (vec[r] as number) * inv;
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
      const aij = A[i * n + j] as number;
      const aji = A[j * n + i] as number;
      if (Math.abs(aij - aji) > 1e-10) {
        throw new DataValidationError("Input must be symmetric for eigvalsh");
      }
    }
  }

  // Eigenvalues only — skip the O(n³) eigenvector accumulation entirely.
  const values = symmetricEigenvalues(A, n);
  values.sort();
  return fromDenseVector1D(values);
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
      const aij = A[i * n + j] as number;
      const aji = A[j * n + i] as number;
      if (Math.abs(aij - aji) > 1e-10) {
        throw new DataValidationError("Input must be symmetric for eigh");
      }
    }
  }

  const { values, vectors } = symmetricEigen(A, n);

  // Sort ascending like eigh
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((i, j) => (values[i] as number) - (values[j] as number));

  const outVals = new Float64Array(n);
  const outVecs = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = atArr(idx, col);
    outVals[col] = values[src] as number;
    for (let row = 0; row < n; row++) {
      outVecs[row * n + col] = vectors[row * n + src] as number;
    }
  }

  return [fromDenseVector1D(outVals), fromDenseMatrix2D(n, n, outVecs)];
}
