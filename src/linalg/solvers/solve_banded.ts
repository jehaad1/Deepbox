import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
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
 * Entries of `ab` that do not correspond to a matrix element (the corners)
 * are ignored, but must still be finite.
 *
 * Tridiagonal systems that are diagonally dominant use the Thomas algorithm
 * (no pivoting is needed for those); every other system uses banded LU with
 * partial pivoting. The matrix is factored once, however many right-hand
 * sides there are.
 *
 * **Time Complexity**: O(N * (l + u)²) for the factorization plus O(N * (l + u)) per
 * right-hand side, much faster than dense O(N³)
 *
 * @param luBands - Tuple [l, u] where l = number of lower diagonals, u = number of upper diagonals
 * @param ab - Band matrix in compact form, shape (l + u + 1, N)
 * @param b - Right-hand side vector of shape (N,) or matrix of shape (N, nrhs)
 * @returns Solution x with same shape as b (float64)
 *
 * @example
 * ```ts
 * import { solveBanded } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Tridiagonal system: [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
 * // l=1, u=1, band storage:
 * //   row 0 (upper): [0, -1, -1]
 * //   row 1 (diag):  [2,  2,  2]
 * //   row 2 (lower): [-1, -1, 0]
 * const ab = tensor([[0, -1, -1], [2, 2, 2], [-1, -1, 0]]);
 * const b = tensor([1, 0, 1]);
 * const x = solveBanded([1, 1], ab, b); // [1, 1, 1]
 * ```
 *
 * @throws {InvalidParameterError} If l or u is not a non-negative integer
 * @throws {ShapeError} If ab is not 2-D with l + u + 1 rows, or b does not match N
 * @throws {DTypeError} If ab or b has string or complex dtype
 * @throws {DataValidationError} If the matrix is singular, or ab or b contain NaN or Infinity
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra}
 *
 * @deprecated Prefer {@link solveBanded}.
 */
export function solve_banded(luBands: readonly [number, number], ab: Tensor, b: Tensor): Tensor {
  const l = luBands[0];
  const u = luBands[1];

  if (!Number.isInteger(l) || !Number.isInteger(u) || l < 0 || u < 0) {
    throw new InvalidParameterError(
      `Band widths must be non-negative integers; got l=${l}, u=${u}`,
      "luBands",
      luBands
    );
  }

  if (ab.ndim !== 2) {
    throw new ShapeError("Band matrix ab must be 2D");
  }

  const bandRows = getDim(ab, 0, "solve_banded()");
  const N = getDim(ab, 1, "solve_banded()");

  if (bandRows !== l + u + 1) {
    throw new ShapeError(`ab must have ${l + u + 1} rows for l=${l}, u=${u}; got ${bandRows}`);
  }

  if (b.ndim !== 1 && b.ndim !== 2) {
    throw new ShapeError("b must be 1D or 2D");
  }
  const bRows = getDim(b, 0, "solve_banded()");
  if (bRows !== N) {
    throw new ShapeError(
      b.ndim === 1
        ? `b length (${bRows}) must match matrix size (${N})`
        : `b rows (${bRows}) must match matrix size (${N})`
    );
  }
  const nrhs = b.ndim === 1 ? 1 : getDim(b, 1, "solve_banded()");

  // Right-hand sides as one dense (N, nrhs) row-major block (a vector is one column).
  const B =
    b.ndim === 1 ? toDenseVector1D(b, "solve_banded()") : toDenseMatrix2D(b, "solve_banded()").data;

  const finish = (X: Float64Array): Tensor =>
    b.ndim === 1 ? fromDenseVector1D(X) : fromDenseMatrix2D(N, nrhs, X);

  if (N === 0) return finish(new Float64Array(0));

  const abData = toDenseMatrix2D(ab, "solve_banded()").data;

  // Tridiagonal and diagonally dominant: Thomas algorithm, no pivoting required.
  if (l === 1 && u === 1 && isDiagonallyDominantTridiagonal(abData, N)) {
    const thomas = factorTridiagonal(abData, N);
    if (thomas !== null) {
      solveTridiagonalInPlace(thomas, N, B, nrhs);
      return finish(B);
    }
  }

  const factors = factorBanded(l, u, N, abData);
  solveBandedInPlace(factors, l, u, N, B, nrhs);
  return finish(B);
}

/**
 * Solve a banded linear system A x = b.
 *
 * The banded matrix A is specified by its lower and upper bandwidths
 * and the band data in compact banded storage (same convention as
 * scipy.linalg.solve_banded).
 *
 * Band storage format: `ab` has shape `(l + u + 1, N)` where
 * `ab[u + i - j, j] = A[i, j]` for `max(0, j-u) <= i <= min(N-1, j+l)`.
 * Entries of `ab` that do not correspond to a matrix element (the corners)
 * are ignored, but must still be finite.
 *
 * Tridiagonal systems that are diagonally dominant use the Thomas algorithm
 * (no pivoting is needed for those); every other system uses banded LU with
 * partial pivoting. The matrix is factored once, however many right-hand
 * sides there are.
 *
 * **Time Complexity**: O(N * (l + u)²) for the factorization plus O(N * (l + u)) per
 * right-hand side, much faster than dense O(N³)
 *
 * @param luBands - Tuple [l, u] where l = number of lower diagonals, u = number of upper diagonals
 * @param ab - Band matrix in compact form, shape (l + u + 1, N)
 * @param b - Right-hand side vector of shape (N,) or matrix of shape (N, nrhs)
 * @returns Solution x with same shape as b (float64)
 *
 * @example
 * ```ts
 * import { solveBanded } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Tridiagonal system: [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
 * // l=1, u=1, band storage:
 * //   row 0 (upper): [0, -1, -1]
 * //   row 1 (diag):  [2,  2,  2]
 * //   row 2 (lower): [-1, -1, 0]
 * const ab = tensor([[0, -1, -1], [2, 2, 2], [-1, -1, 0]]);
 * const b = tensor([1, 0, 1]);
 * const x = solveBanded([1, 1], ab, b); // [1, 1, 1]
 * ```
 *
 * @throws {InvalidParameterError} If l or u is not a non-negative integer
 * @throws {ShapeError} If ab is not 2-D with l + u + 1 rows, or b does not match N
 * @throws {DTypeError} If ab or b has string or complex dtype
 * @throws {DataValidationError} If the matrix is singular, or ab or b contain NaN or Infinity
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra}
 */
export const solveBanded = solve_banded;

/** Row diagonal dominance |a_ii| >= |a_i,i-1| + |a_i,i+1| of a tridiagonal band (u = 1). */
function isDiagonallyDominantTridiagonal(abData: Float64Array, N: number): boolean {
  for (let i = 0; i < N; i++) {
    const diag = Math.abs(abData[N + i] as number); // a[i, i]
    const lower = i > 0 ? Math.abs(abData[2 * N + (i - 1)] as number) : 0; // a[i, i-1]
    const upper = i < N - 1 ? Math.abs(abData[i + 1] as number) : 0; // a[i, i+1]
    if (diag < lower + upper) return false;
  }
  return true;
}

type TridiagonalFactors = {
  /** Sub-diagonal a[i, i-1] for i = 1..N-1 (index i-1). */
  readonly lower: Float64Array;
  /** Eliminated pivots. */
  readonly pivots: Float64Array;
  /** Scaled super-diagonal c[i] = a[i, i+1] / pivots[i]. */
  readonly upper: Float64Array;
};

/**
 * Thomas factorization of a tridiagonal band (ab rows: upper, diagonal, lower).
 * Returns null on an exactly zero pivot, which a pivoting solver may still
 * get past or report as singular.
 */
function factorTridiagonal(abData: Float64Array, N: number): TridiagonalFactors | null {
  const lower = new Float64Array(Math.max(N - 1, 0));
  const pivots = new Float64Array(N);
  const upper = new Float64Array(Math.max(N - 1, 0));

  for (let i = 0; i < N; i++) {
    let m = abData[N + i] as number; // a[i, i]
    if (i > 0) {
      const sub = abData[2 * N + (i - 1)] as number; // a[i, i-1]
      lower[i - 1] = sub;
      m -= sub * (upper[i - 1] as number);
    }
    if (m === 0) return null;
    pivots[i] = m;
    if (i < N - 1) upper[i] = (abData[i + 1] as number) / m; // a[i, i+1] / m
  }
  return { lower, pivots, upper };
}

function solveTridiagonalInPlace(
  f: TridiagonalFactors,
  N: number,
  B: Float64Array,
  nrhs: number
): void {
  for (let c = 0; c < nrhs; c++) {
    // Forward sweep
    B[c] = (B[c] as number) / (f.pivots[0] as number);
    for (let i = 1; i < N; i++) {
      const prev = B[(i - 1) * nrhs + c] as number;
      B[i * nrhs + c] =
        ((B[i * nrhs + c] as number) - (f.lower[i - 1] as number) * prev) / (f.pivots[i] as number);
    }
    // Back substitution
    for (let i = N - 2; i >= 0; i--) {
      B[i * nrhs + c] =
        (B[i * nrhs + c] as number) - (f.upper[i] as number) * (B[(i + 1) * nrhs + c] as number);
    }
  }
}

type BandedFactors = {
  /** LAPACK dgbtrf-style storage: A[i][j] at row `l + u + i - j`, column `j`, `2l + u + 1` rows. */
  readonly AB: Float64Array;
  /** piv[k] is the row swapped with row k at elimination step k. */
  readonly piv: Int32Array;
};

/**
 * General banded LU with partial pivoting in dgbtrf-style column-oriented
 * storage.
 *
 * Element A[i][j] is stored at row `l + u + i - j`, column `j`, in a storage
 * of `2*l + u + 1` rows. The extra `l` rows above the band hold the fill-in
 * that partial pivoting introduces. Because storage is keyed by absolute
 * column, a matrix-row swap becomes a per-column swap between two storage
 * rows that shift together with the column, so pivoting never misaligns
 * entries. The multipliers of L stay in the sub-diagonal positions they were
 * computed for and are replayed (swap, then eliminate) by the solve.
 */
function factorBanded(l: number, u: number, N: number, abData: Float64Array): BandedFactors {
  const ldab = 2 * l + u + 1;
  const AB = new Float64Array(ldab * N);
  const at = (i: number, j: number): number => {
    const r = l + u + i - j;
    return r < 0 || r >= ldab ? 0 : (AB[r * N + j] as number);
  };
  const put = (i: number, j: number, v: number): void => {
    AB[(l + u + i - j) * N + j] = v;
  };

  // Copy the band of A out of the compact (l + u + 1, N) input.
  for (let j = 0; j < N; j++) {
    const iLo = Math.max(0, j - u);
    const iHi = Math.min(N - 1, j + l);
    for (let i = iLo; i <= iHi; i++) {
      put(i, j, abData[(u + i - j) * N + j] as number);
    }
  }

  const piv = new Int32Array(N);

  // Forward elimination with partial pivoting. After eliminating column k,
  // fill-in can extend a row's upper reach by up to l columns (to k+u+l).
  for (let k = 0; k < N; k++) {
    const iMax = Math.min(N - 1, k + l);

    let pivotRow = k;
    let pivotVal = Math.abs(at(k, k));
    for (let i = k + 1; i <= iMax; i++) {
      const v = Math.abs(at(i, k));
      if (v > pivotVal) {
        pivotVal = v;
        pivotRow = i;
      }
    }

    if (pivotVal === 0) {
      throw new DataValidationError("Singular banded matrix");
    }

    piv[k] = pivotRow;
    if (pivotRow !== k) {
      // Swap matrix rows k and pivotRow in the columns they still share
      // (both rows are zero beyond column k + u + l).
      const jSwap = Math.min(N - 1, k + u + l);
      for (let j = k; j <= jSwap; j++) {
        const a = at(k, j);
        const bb = at(pivotRow, j);
        put(k, j, bb);
        put(pivotRow, j, a);
      }
    }

    const pivot = at(k, k);
    const jMax = Math.min(N - 1, k + u + l);
    for (let i = k + 1; i <= iMax; i++) {
      const m = at(i, k) / pivot;
      put(i, k, m);
      if (m === 0) continue;
      for (let j = k + 1; j <= jMax; j++) {
        put(i, j, at(i, j) - m * at(k, j));
      }
    }
  }

  return { AB, piv };
}

/** Solve L U X = P B in place in B, an (N, nrhs) row-major block. */
function solveBandedInPlace(
  f: BandedFactors,
  l: number,
  u: number,
  N: number,
  B: Float64Array,
  nrhs: number
): void {
  const { AB, piv } = f;
  const ldab = 2 * l + u + 1;
  const at = (i: number, j: number): number => {
    const r = l + u + i - j;
    return r < 0 || r >= ldab ? 0 : (AB[r * N + j] as number);
  };

  // Forward: replay the row swaps and eliminations in factorization order.
  for (let k = 0; k < N; k++) {
    const p = piv[k] as number;
    if (p !== k) {
      for (let c = 0; c < nrhs; c++) {
        const tmp = B[k * nrhs + c] as number;
        B[k * nrhs + c] = B[p * nrhs + c] as number;
        B[p * nrhs + c] = tmp;
      }
    }
    const iMax = Math.min(N - 1, k + l);
    for (let i = k + 1; i <= iMax; i++) {
      const m = at(i, k);
      if (m === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        B[i * nrhs + c] = (B[i * nrhs + c] as number) - m * (B[k * nrhs + c] as number);
      }
    }
  }

  // Back substitution (fill-in widened the upper band to u + l)
  for (let i = N - 1; i >= 0; i--) {
    const jMax = Math.min(N - 1, i + u + l);
    for (let j = i + 1; j <= jMax; j++) {
      const v = at(i, j);
      if (v === 0) continue;
      for (let c = 0; c < nrhs; c++) {
        B[i * nrhs + c] = (B[i * nrhs + c] as number) - v * (B[j * nrhs + c] as number);
      }
    }
    const diag = at(i, i);
    for (let c = 0; c < nrhs; c++) B[i * nrhs + c] = (B[i * nrhs + c] as number) / diag;
  }
}
