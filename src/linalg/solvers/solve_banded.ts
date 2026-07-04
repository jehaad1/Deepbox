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

/**
 * Solve a banded linear system A x = b.
 *
 * The banded matrix A is specified by its lower and upper bandwidths
 * and the band data in compact banded storage (same convention as
 * scipy.linalg.solve_banded).
 *
 * Band storage format: `ab` has shape `(l + u + 1, N)` where
 * `ab[u + i - j, j] = A[i, j]` for `max(0, j-u) <= i <= min(N-1, j+l)`.
 *
 * Uses banded LU factorization with partial pivoting (Thomas algorithm
 * for tridiagonal, general banded Gaussian elimination otherwise).
 *
 * **Time Complexity**: O(N * (l + u)²) — much faster than dense O(N³)
 *
 * @param luBands - Tuple [l, u] where l = number of lower diagonals, u = number of upper diagonals
 * @param ab - Band matrix in compact form, shape (l + u + 1, N)
 * @param b - Right-hand side vector of shape (N,) or matrix of shape (N, nrhs)
 * @returns Solution x with same shape as b
 *
 * @example
 * ```ts
 * import { solve_banded } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Tridiagonal system: [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
 * // l=1, u=1, band storage:
 * //   row 0 (upper): [0, -1, -1]
 * //   row 1 (diag):  [2,  2,  2]
 * //   row 2 (lower): [-1, -1, 0]
 * const ab = tensor([[0, -1, -1], [2, 2, 2], [-1, -1, 0]]);
 * const b = tensor([1, 0, 1]);
 * const x = solve_banded([1, 1], ab, b);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra}
 */
export function solve_banded(luBands: readonly [number, number], ab: Tensor, b: Tensor): Tensor {
  const l = luBands[0];
  const u = luBands[1];

  if (l < 0 || u < 0 || !Number.isInteger(l) || !Number.isInteger(u)) {
    throw new ShapeError(`Band widths must be non-negative integers; got l=${l}, u=${u}`);
  }

  if (ab.ndim !== 2) {
    throw new ShapeError("Band matrix ab must be 2D");
  }

  const bandRows = getDim(ab, 0, "solve_banded()");
  const N = getDim(ab, 1, "solve_banded()");

  if (bandRows !== l + u + 1) {
    throw new ShapeError(`ab must have ${l + u + 1} rows for l=${l}, u=${u}; got ${bandRows}`);
  }

  if (N === 0) {
    if (b.ndim === 1) return fromDenseVector1D(new Float64Array(0));
    return fromDenseMatrix2D(0, 0, new Float64Array(0));
  }

  const abMat = toDenseMatrix2D(ab);

  // Helper to get A[i,j] from banded storage
  const getA = (i: number, j: number): number => {
    const row = u + i - j;
    if (row < 0 || row >= bandRows || j < 0 || j >= N) return 0;
    return at(abMat.data, row * N + j);
  };

  if (b.ndim === 1) {
    // Single RHS
    const bSize = getDim(b, 0, "solve_banded()");
    if (bSize !== N) {
      throw new ShapeError(`b length (${bSize}) must match matrix size (${N})`);
    }
    const bVec = toDenseVector1D(b);

    if (l === 1 && u === 1) {
      // Thomas algorithm for tridiagonal; falls back to the pivoting general
      // solver on a near-zero pivot (Thomas has no pivoting and would wrongly
      // reject nonsingular but non-diagonally-dominant systems).
      const tri = solveTridiagonal(abMat.data, N, u, bVec);
      if (tri !== null) return fromDenseVector1D(tri);
      return fromDenseVector1D(solveBandedGeneral(l, u, N, getA, bVec));
    }

    return fromDenseVector1D(solveBandedGeneral(l, u, N, getA, bVec));
  }

  // Multiple RHS: b is (N, nrhs)
  if (b.ndim !== 2) {
    throw new ShapeError("b must be 1D or 2D");
  }
  const bRows = getDim(b, 0, "solve_banded()");
  const nrhs = getDim(b, 1, "solve_banded()");
  if (bRows !== N) {
    throw new ShapeError(`b rows (${bRows}) must match matrix size (${N})`);
  }

  const bMat = toDenseMatrix2D(b);
  const result = new Float64Array(N * nrhs);

  for (let j = 0; j < nrhs; j++) {
    const col = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      col[i] = at(bMat.data, i * nrhs + j);
    }

    let sol: Float64Array | null;
    if (l === 1 && u === 1) {
      sol = solveTridiagonal(abMat.data, N, u, col);
      if (sol === null) sol = solveBandedGeneral(l, u, N, getA, col);
    } else {
      sol = solveBandedGeneral(l, u, N, getA, col);
    }

    for (let i = 0; i < N; i++) {
      result[i * nrhs + j] = at(sol, i);
    }
  }

  return fromDenseMatrix2D(N, nrhs, result);
}

/**
 * Thomas algorithm for tridiagonal systems.
 * Time: O(N), Space: O(N).
 */
function solveTridiagonal(
  abData: Float64Array,
  N: number,
  u: number,
  b: Float64Array
): Float64Array | null {
  // Extract diagonals from band storage
  // Upper diagonal: ab[0, 1..N-1]  (a[i, i+1])
  // Main diagonal:  ab[u, 0..N-1]  (a[i, i])
  // Lower diagonal: ab[u+1, 0..N-2]  (a[i+1, i])
  const x = new Float64Array(b);
  const c = new Float64Array(N); // modified upper diagonal

  // Forward sweep
  const d0 = at(abData, u * N + 0); // main diagonal[0]
  if (Math.abs(d0) < 1e-15) {
    // Zero pivot: Thomas cannot proceed without pivoting — signal the caller
    // to retry with the pivoting general solver rather than declaring the
    // (possibly nonsingular) system singular.
    return null;
  }
  c[0] = at(abData, 0 * N + 1) / d0; // upper[0] / diag[0]
  x[0] = at(x, 0) / d0;

  for (let i = 1; i < N; i++) {
    const lower = at(abData, (u + 1) * N + (i - 1)); // a[i, i-1]
    const diag = at(abData, u * N + i); // a[i, i]
    const m = diag - lower * at(c, i - 1);
    if (Math.abs(m) < 1e-15) {
      return null;
    }
    if (i < N - 1) {
      c[i] = at(abData, 0 * N + (i + 1)) / m; // upper[i] / m
    }
    x[i] = (at(x, i) - lower * at(x, i - 1)) / m;
  }

  // Back substitution
  for (let i = N - 2; i >= 0; i--) {
    x[i] = at(x, i) - at(c, i) * at(x, i + 1);
  }

  return x;
}

/**
 * General banded Gaussian elimination with partial pivoting, using LAPACK
 * dgbtrf-style column-oriented banded storage.
 *
 * Element A[i][j] is stored at row `l + u + i - j`, column `j` in a storage
 * of `2*l + u + 1` rows. The extra `l` rows above the band hold fill-in that
 * partial pivoting introduces. Because storage is keyed by absolute column,
 * a matrix-row swap becomes a per-column swap between two storage rows that
 * shift together with the column — so pivoting never misaligns entries (the
 * previous row-relative layout silently corrupted the result on any swap).
 */
function solveBandedGeneral(
  l: number,
  u: number,
  N: number,
  getA: (i: number, j: number) => number,
  b: Float64Array
): Float64Array {
  const ldab = 2 * l + u + 1;
  const AB = new Float64Array(ldab * N);
  // storage(r, j) helpers: r in [0, ldab), j in [0, N)
  const idx = (r: number, j: number): number => r * N + j;
  const set = (i: number, j: number, v: number): void => {
    AB[idx(l + u + i - j, j)] = v;
  };
  const get = (i: number, j: number): number => {
    const r = l + u + i - j;
    if (r < 0 || r >= ldab) return 0;
    return at(AB, idx(r, j));
  };

  for (let i = 0; i < N; i++) {
    for (let j = Math.max(0, i - l); j <= Math.min(N - 1, i + u); j++) {
      set(i, j, getA(i, j));
    }
  }

  const x = new Float64Array(b);
  const piv = new Int32Array(N);

  // Forward elimination with partial pivoting. After eliminating column k,
  // fill-in can extend a row's upper reach by up to l columns (to k+u+l).
  for (let k = 0; k < N; k++) {
    const iMax = Math.min(N - 1, k + l);

    let pivotRow = k;
    let pivotVal = Math.abs(get(k, k));
    for (let i = k + 1; i <= iMax; i++) {
      const v = Math.abs(get(i, k));
      if (v > pivotVal) {
        pivotVal = v;
        pivotRow = i;
      }
    }

    if (pivotVal < 1e-15) {
      throw new DataValidationError("Singular banded matrix");
    }

    piv[k] = pivotRow;
    if (pivotRow !== k) {
      // Swap matrix rows k and pivotRow across all columns they touch.
      const jMax = Math.min(N - 1, pivotRow + u + l);
      for (let j = k; j <= jMax; j++) {
        const a = get(k, j);
        const bb = get(pivotRow, j);
        set(k, j, bb);
        set(pivotRow, j, a);
      }
      const tmp = at(x, k);
      x[k] = at(x, pivotRow);
      x[pivotRow] = tmp;
    }

    const pivot = get(k, k);
    const jMax = Math.min(N - 1, k + u + l);
    for (let i = k + 1; i <= iMax; i++) {
      const m = get(i, k) / pivot;
      set(i, k, 0);
      if (m === 0) continue;
      for (let j = k + 1; j <= jMax; j++) {
        set(i, j, get(i, j) - m * get(k, j));
      }
      x[i] = at(x, i) - m * at(x, k);
    }
  }

  // Back substitution (fill-in widened the upper band to u + l)
  for (let i = N - 1; i >= 0; i--) {
    let sum = at(x, i);
    const jMax = Math.min(N - 1, i + u + l);
    for (let j = i + 1; j <= jMax; j++) {
      sum -= get(i, j) * at(x, j);
    }
    const diag = get(i, i);
    if (Math.abs(diag) < 1e-15) {
      throw new DataValidationError("Singular banded matrix");
    }
    x[i] = sum / diag;
  }

  return x;
}
