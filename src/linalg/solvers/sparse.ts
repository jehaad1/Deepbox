import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";
import { CSRMatrix as NdarrayCSRMatrix, type Tensor } from "../../ndarray";
import {
  fromDenseVector1D,
  getDim,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../_internal";

/**
 * Compressed Sparse Row (CSR) matrix representation.
 *
 * Stores a sparse matrix using three arrays:
 * - `values`: Non-zero values (NNZ elements)
 * - `colIndices`: Column index for each non-zero value (NNZ elements)
 * - `rowPointers`: Start index in values/colIndices for each row (N+1 elements)
 *
 * For row i, the non-zero entries are at indices rowPointers[i]..rowPointers[i+1]-1.
 * Repeated (row, column) entries are summed, as in `scipy.sparse`.
 *
 * This record is not the `CSRMatrix` class of `deepbox/ndarray`, but the solvers accept both
 * (see {@link SparseMatrixInput}).
 */
export type CSRMatrix = {
  readonly n: number;
  readonly values: Float64Array;
  readonly colIndices: Int32Array;
  readonly rowPointers: Int32Array;
};

/**
 * A sparse matrix accepted by {@link sparseSolve} and {@link sparseCholeskySolve}: either the
 * plain {@link CSRMatrix} record made by {@link denseToCSR}, or the `CSRMatrix` class of
 * `deepbox/ndarray` (which must be square).
 */
export type SparseMatrixInput = CSRMatrix | NdarrayCSRMatrix;

/**
 * Normalize a sparse input to the plain CSR record. The arrays of an ndarray `CSRMatrix`
 * are shared, not copied.
 */
function toCSRRecord(m: SparseMatrixInput, fn: string): CSRMatrix {
  if (!(m instanceof NdarrayCSRMatrix)) return m;
  const rows = m.shape[0];
  const cols = m.shape[1];
  if (rows === undefined || cols === undefined || rows !== cols) {
    throw new ShapeError(`${fn}: the sparse matrix must be square; got (${rows}, ${cols})`);
  }
  return { n: rows, values: m.data, colIndices: m.indices, rowPointers: m.indptr };
}

/**
 * Convert a dense tensor to CSR sparse format.
 *
 * @param a - Input 2D tensor (must be square)
 * @returns CSR representation
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 * import { denseToCSR } from 'deepbox/linalg';
 *
 * const A = tensor([[4, 1, 0], [1, 3, 1], [0, 1, 2]]);
 * const csr = denseToCSR(A);
 * ```
 *
 * @throws {ShapeError} If input is not a square 2-D matrix
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 */
export function denseToCSR(a: Tensor): CSRMatrix {
  if (a.ndim !== 2) {
    throw new ShapeError("Input must be a 2D matrix");
  }
  const n = getDim(a, 0, "denseToCSR()");
  const n2 = getDim(a, 1, "denseToCSR()");
  if (n !== n2) {
    throw new ShapeError(`denseToCSR() requires a square matrix; got (${n}, ${n2})`);
  }

  const mat = toDenseMatrix2D(a, "denseToCSR()");
  const vals: number[] = [];
  const cols: number[] = [];
  const rowPtrs = new Int32Array(n + 1);

  for (let i = 0; i < n; i++) {
    rowPtrs[i] = vals.length;
    for (let j = 0; j < n; j++) {
      const v = mat.data[i * n + j] as number;
      if (v !== 0) {
        vals.push(v);
        cols.push(j);
      }
    }
  }
  rowPtrs[n] = vals.length;

  return {
    n,
    values: new Float64Array(vals),
    colIndices: new Int32Array(cols),
    rowPointers: rowPtrs,
  };
}

/**
 * Check that a CSR matrix is internally consistent before reading it.
 * Column indices outside [0, n) would otherwise be dropped or land in the
 * wrong row of the dense workspace without any error.
 */
function assertValidCSR(csr: CSRMatrix, fn: string): void {
  const { n, values, colIndices, rowPointers } = csr;
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError(`${fn}: csr.n must be a non-negative integer`, "csr.n", n);
  }
  if (rowPointers.length !== n + 1) {
    throw new ShapeError(
      `${fn}: rowPointers must have n + 1 = ${n + 1} entries; got ${rowPointers.length}`
    );
  }
  const nnz = rowPointers[n] as number;
  if (rowPointers[0] !== 0 || nnz > values.length || nnz > colIndices.length) {
    throw new DataValidationError(
      `${fn}: rowPointers must start at 0 and end at a count of stored entries that ` +
        `values (${values.length}) and colIndices (${colIndices.length}) can hold; got ${nnz}`
    );
  }
  for (let i = 0; i < n; i++) {
    if ((rowPointers[i] as number) > (rowPointers[i + 1] as number)) {
      throw new DataValidationError(`${fn}: rowPointers must be non-decreasing (row ${i})`);
    }
  }
  // Entries past rowPointers[n] are spare capacity and never read.
  for (let k = 0; k < nnz; k++) {
    const j = colIndices[k] as number;
    if (j < 0 || j >= n) {
      throw new DataValidationError(
        `${fn}: column index ${j} at position ${k} is outside [0, ${n})`
      );
    }
    if (!Number.isFinite(values[k] as number)) {
      throw new DataValidationError(`${fn}: values must be finite`);
    }
  }
}

function assertVectorRhs(csr: CSRMatrix, b: Tensor, fn: string): void {
  if (b.ndim !== 1) {
    throw new ShapeError("b must be a 1D vector");
  }
  const bLen = getDim(b, 0, fn);
  if (bLen !== csr.n) {
    throw new ShapeError(`b length (${bLen}) must match matrix size (${csr.n})`);
  }
}

/**
 * Solve a sparse linear system A x = b by LU factorization with partial pivoting.
 *
 * The CSR matrix is expanded into a dense n x n workspace and factored with
 * the same LU routine as {@link solve}, skipping the elimination of zero
 * entries. This is fast for moderately sized sparse systems, but memory use is
 * O(N²) and time is O(N³) in the worst case: it is not a sparse direct solver
 * with fill-reducing ordering. For symmetric positive-definite systems with a
 * small bandwidth, {@link sparseCholeskySolve} stores only the envelope.
 *
 * **Time Complexity**: O(N²) to scan for pivots plus the cost of the non-zero
 * updates (O(N * bandwidth²) for banded matrices), O(N³) for dense ones
 * **Space Complexity**: O(N²)
 *
 * @param csr - Sparse matrix in CSR format (from {@link denseToCSR} or an ndarray `CSRMatrix`)
 * @param b - Right-hand side vector of shape (N,)
 * @returns Solution vector x of shape (N,)
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 * import { denseToCSR, sparseSolve } from 'deepbox/linalg';
 *
 * const A = tensor([[4, 1, 0], [1, 3, 1], [0, 1, 2]]);
 * const b = tensor([1, 2, 3]);
 * const x = sparseSolve(denseToCSR(A), b);
 * ```
 *
 * @throws {ShapeError} If b is not a vector of length N, or rowPointers does not have N + 1 entries
 * @throws {DataValidationError} If the CSR structure is invalid (index out of range, decreasing
 *   rowPointers, non-finite values), or the matrix is singular
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra}
 */
export function sparseSolve(input: SparseMatrixInput, b: Tensor): Tensor {
  const csr = toCSRRecord(input, "sparseSolve()");
  assertValidCSR(csr, "sparseSolve()");
  assertVectorRhs(csr, b, "sparseSolve()");

  const n = csr.n;
  if (n === 0) {
    return fromDenseVector1D(new Float64Array(0));
  }

  const x = toDenseVector1D(b, "sparseSolve()");

  // Expand CSR into the dense workspace; repeated entries add up.
  const dense = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const start = csr.rowPointers[i] as number;
    const end = csr.rowPointers[i + 1] as number;
    for (let k = start; k < end; k++) {
      const idx = i * n + (csr.colIndices[k] as number);
      dense[idx] = (dense[idx] as number) + (csr.values[k] as number);
    }
  }

  try {
    const { lu, piv } = luFactorSquare(dense, n);
    luSolveInPlace(lu, piv, n, x, 1);
  } catch (err) {
    if (err instanceof DataValidationError && /singular/i.test(err.message)) {
      throw new DataValidationError("Singular sparse matrix", { cause: err });
    }
    throw err;
  }

  return fromDenseVector1D(x);
}

/**
 * Solve a sparse symmetric positive-definite system using sparse Cholesky.
 *
 * Performs Cholesky decomposition A = L Lᵀ where L is lower triangular,
 * then solves via forward/backward substitution.
 *
 * Only the lower triangle of the sparse matrix is read (the upper triangle may
 * be absent or hold anything); symmetry is not checked. L is stored in
 * envelope (skyline) form: row i keeps the columns from its first stored
 * entry to the diagonal, which is where Cholesky fill-in is confined. Work and
 * memory therefore scale with the envelope, not with N².
 *
 * **Time Complexity**: O(N * bandwidth²) for banded SPD matrices
 * **Space Complexity**: O(envelope size), at most O(N²) for dense rows
 *
 * @param csr - Sparse SPD matrix in CSR format (from {@link denseToCSR} or an ndarray `CSRMatrix`)
 * @param b - Right-hand side vector of shape (N,)
 * @returns Solution vector x of shape (N,)
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 * import { denseToCSR, sparseCholeskySolve } from 'deepbox/linalg';
 *
 * const A = tensor([[4, 2, 0], [2, 5, 1], [0, 1, 3]]);
 * const b = tensor([1, 2, 3]);
 * const x = sparseCholeskySolve(denseToCSR(A), b);
 * ```
 *
 * @throws {ShapeError} If b is not a vector of length N, or rowPointers does not have N + 1 entries
 * @throws {DataValidationError} If the CSR structure is invalid, or the matrix is not positive definite
 *
 * @see {@link https://deepbox.dev/docs/linalg-solvers | Deepbox Linear Algebra}
 */
export function sparseCholeskySolve(input: SparseMatrixInput, b: Tensor): Tensor {
  const csr = toCSRRecord(input, "sparseCholeskySolve()");
  assertValidCSR(csr, "sparseCholeskySolve()");
  assertVectorRhs(csr, b, "sparseCholeskySolve()");

  const n = csr.n;
  if (n === 0) {
    return fromDenseVector1D(new Float64Array(0));
  }

  const x = toDenseVector1D(b, "sparseCholeskySolve()");

  // Envelope layout: row i holds columns first[i]..i at L[start[i] + (j - first[i])].
  const first = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let f = i;
    const rs = csr.rowPointers[i] as number;
    const re = csr.rowPointers[i + 1] as number;
    for (let k = rs; k < re; k++) {
      const j = csr.colIndices[k] as number;
      if (j < f) f = j;
    }
    first[i] = f;
  }
  const start = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    start[i + 1] = (start[i] as number) + (i - (first[i] as number) + 1);
  }
  const L = new Float64Array(start[n] as number);

  // Scatter the lower triangle of A into the envelope (repeated entries add up).
  for (let i = 0; i < n; i++) {
    const rs = csr.rowPointers[i] as number;
    const re = csr.rowPointers[i + 1] as number;
    const base = (start[i] as number) - (first[i] as number);
    for (let k = rs; k < re; k++) {
      const j = csr.colIndices[k] as number;
      if (j <= i) L[base + j] = (L[base + j] as number) + (csr.values[k] as number);
    }
  }

  // Cholesky factorization A = L L^T, in place over the envelope.
  for (let i = 0; i < n; i++) {
    const fi = first[i] as number;
    const baseI = (start[i] as number) - fi;
    for (let j = fi; j <= i; j++) {
      const fj = first[j] as number;
      const baseJ = (start[j] as number) - fj;
      let sum = L[baseI + j] as number;
      // L[i,k] and L[j,k] are both zero for k < max(first[i], first[j]).
      for (let k = Math.max(fi, fj); k < j; k++) {
        sum -= (L[baseI + k] as number) * (L[baseJ + k] as number);
      }
      if (i === j) {
        if (!(sum > 0)) {
          throw new DataValidationError("Matrix is not positive definite (Cholesky failed)");
        }
        L[baseI + j] = Math.sqrt(sum);
      } else {
        L[baseI + j] = sum / (L[baseJ + j] as number);
      }
    }
  }

  // Forward substitution: L y = b
  for (let i = 0; i < n; i++) {
    const fi = first[i] as number;
    const baseI = (start[i] as number) - fi;
    let sum = x[i] as number;
    for (let k = fi; k < i; k++) sum -= (L[baseI + k] as number) * (x[k] as number);
    x[i] = sum / (L[baseI + i] as number);
  }

  // Backward substitution: L^T x = y, column-oriented so only stored entries are touched.
  for (let i = n - 1; i >= 0; i--) {
    const fi = first[i] as number;
    const baseI = (start[i] as number) - fi;
    const xi = (x[i] as number) / (L[baseI + i] as number);
    x[i] = xi;
    for (let k = fi; k < i; k++) {
      x[k] = (x[k] as number) - (L[baseI + k] as number) * xi;
    }
  }

  return fromDenseVector1D(x);
}
