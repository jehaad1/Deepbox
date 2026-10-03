/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

import {
  DataValidationError,
  DTypeError,
  getConfig,
  IndexError,
  InvalidParameterError,
  ShapeError,
} from "../core";
import { Tensor } from "../ndarray";

/** Smallest positive normal float64 (2^-1022). Below it, 1/x can overflow. */
const SAFE_MIN = 2.2250738585072014e-308;
/** 2^1022, the largest magnitude whose reciprocal is still a normal float64. */
const SAFE_MAX = 1 / SAFE_MIN;

/**
 * True when 1/x is a finite, normal float64, so multiplying by the reciprocal
 * loses no range. Otherwise callers should divide directly.
 */
function reciprocalIsSafe(x: number): boolean {
  const a = Math.abs(x);
  return a >= SAFE_MIN && a <= SAFE_MAX;
}

function assertDim(n: number, name: string): void {
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError(`${name} must be a non-negative integer, got ${n}`, name, n);
  }
}

/**
 * Rejects dtypes that the dense linear algebra kernels cannot read as real
 * numbers. Strings are not numeric, and complex data uses interleaved storage,
 * so reading it element by element would silently drop the imaginary part.
 */
function assertRealNumericDType(t: Tensor): void {
  if (t.dtype === "string") {
    throw new DTypeError("String tensors are not supported");
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`Complex tensors (${t.dtype}) are not supported by linear algebra`);
  }
}

/** Message for a non-finite input, prefixed with the calling function when known. */
function nonFiniteMessage(context: string | undefined): string {
  return context === undefined
    ? "Input contains non-finite values"
    : `${context}: input contains non-finite values`;
}

/**
 * Represents a 2D matrix view over a tensor's underlying data buffer.
 *
 * This structure provides efficient access to matrix elements without copying data.
 * It tracks strides to handle both contiguous and non-contiguous memory layouts.
 *
 * @property tensor - The underlying tensor object
 * @property rows - Number of rows (M)
 * @property cols - Number of columns (N)
 * @property offset - Starting position in the data buffer
 * @property strideRow - Number of elements to skip between consecutive rows
 * @property strideCol - Number of elements to skip between consecutive columns
 * @property isRowMajorContiguous - True if data is stored in contiguous row-major order
 *
 * @internal
 */
type Matrix2D = {
  readonly tensor: Tensor;
  readonly rows: number;
  readonly cols: number;
  readonly offset: number;
  readonly strideRow: number;
  readonly strideCol: number;
  readonly isRowMajorContiguous: boolean;
};

/**
 * Converts a 2D tensor into a Matrix2D view for efficient element access.
 *
 * **Time Complexity**: O(1) - only metadata extraction, no data copying
 *
 * @param t - Input tensor (must be 2D)
 * @returns Matrix2D view with stride information
 * @throws {ShapeError} If tensor is not 2D
 * @throws {DTypeError} If tensor has string or complex dtype
 *
 * @internal
 */
export function asMatrix2D(t: Tensor): Matrix2D {
  assertRealNumericDType(t);

  // Validate tensor dimensionality
  if (t.ndim !== 2) {
    throw new ShapeError("Expected a 2D tensor");
  }

  // Extract shape dimensions
  const rows = t.shape[0];
  const cols = t.shape[1];
  if (rows === undefined || cols === undefined) {
    throw new ShapeError("Tensor shape metadata is inconsistent");
  }

  // Extract stride information for memory layout
  const strideRow = t.strides[0];
  const strideCol = t.strides[1];
  if (strideRow === undefined || strideCol === undefined) {
    throw new ShapeError("Tensor stride metadata is inconsistent");
  }

  // Check if data is contiguous in row-major (C-order) layout.
  // Row-major: elements in the same row are adjacent (strideCol=1) and rows are
  // `cols` apart. The stride of an axis of length 0 or 1 never moves the pointer,
  // so it does not matter.
  const isRowMajorContiguous = (rows <= 1 || strideRow === cols) && (cols <= 1 || strideCol === 1);

  return {
    tensor: t,
    rows,
    cols,
    offset: t.offset,
    strideRow,
    strideCol,
    isRowMajorContiguous,
  };
}

/**
 * Converts a tensor to a dense Float64Array in row-major order.
 *
 * Handles both contiguous and strided memory layouts efficiently.
 * The result is always a copy, so callers may modify it freely.
 *
 * **Time Complexity**:
 * - O(M*N) where M=rows, N=cols
 * - Optimized path for contiguous data: O(M*N) with better cache locality
 *
 * @param t - Input 2D tensor
 * @param context - Optional name of the calling function, used in error messages
 * @returns Object containing dimensions and dense data array
 * @throws {ShapeError} If tensor is not 2D
 * @throws {DTypeError} If tensor has string or complex dtype
 * @throws {DataValidationError} If tensor contains non-finite values
 *
 * @internal
 */
export function toDenseMatrix2D(
  t: Tensor,
  context?: string
): {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;
} {
  const m = asMatrix2D(t);
  const { rows, cols } = m;

  const out = new Float64Array(rows * cols);

  if (m.isRowMajorContiguous) {
    const start = m.offset;
    const end = start + rows * cols;
    for (let i = 0, j = start; j < end; i++, j++) {
      const val = Number(m.tensor.data[j]);
      if (!Number.isFinite(val)) {
        throw new DataValidationError(nonFiniteMessage(context));
      }
      out[i] = val;
    }
    return { rows, cols, data: out };
  }

  for (let i = 0; i < rows; i++) {
    const base = m.offset + i * m.strideRow;
    for (let j = 0; j < cols; j++) {
      const val = Number(m.tensor.data[base + j * m.strideCol]);
      if (!Number.isFinite(val)) {
        throw new DataValidationError(nonFiniteMessage(context));
      }
      out[i * cols + j] = val;
    }
  }

  return { rows, cols, data: out };
}

/**
 * Converts a 1D tensor to a dense Float64Array.
 *
 * **Time Complexity**: O(N) where N is the vector length
 *
 * @param t - Input 1D tensor
 * @param context - Optional name of the calling function, used in error messages
 * @returns Dense Float64Array copy of the data
 * @throws {ShapeError} If tensor is not 1D
 * @throws {DTypeError} If tensor has string or complex dtype
 * @throws {DataValidationError} If tensor contains non-finite values
 *
 * @internal
 */
export function toDenseVector1D(t: Tensor, context?: string): Float64Array {
  assertRealNumericDType(t);
  if (t.ndim !== 1) {
    throw new ShapeError("Expected a 1D tensor");
  }
  const n = t.shape[0];
  if (n === undefined) {
    throw new ShapeError("Tensor shape metadata is inconsistent");
  }
  const stride = t.strides[0];
  if (stride === undefined) {
    throw new ShapeError("Tensor stride metadata is inconsistent");
  }

  const out = new Float64Array(n);
  const base = t.offset;

  for (let i = 0; i < n; i++) {
    const val = Number(t.data[base + i * stride]);
    if (!Number.isFinite(val)) {
      throw new DataValidationError(nonFiniteMessage(context));
    }
    out[i] = val;
  }
  return out;
}

/**
 * Creates a 2D tensor from a dense Float64Array in row-major order.
 *
 * **Time Complexity**: O(1) - tensor wraps existing array without copying
 *
 * @param rows - Number of rows (M)
 * @param cols - Number of columns (N)
 * @param data - Dense Float64Array of length M*N in row-major order
 * @returns New tensor wrapping the data
 * @throws {ShapeError} If data.length is not rows * cols
 *
 * @internal
 */
export function fromDenseMatrix2D(rows: number, cols: number, data: Float64Array): Tensor {
  if (data.length !== rows * cols) {
    throw new ShapeError(
      `Data length ${data.length} does not match shape [${rows}, ${cols}] (${rows * cols} elements)`
    );
  }
  // Get global configuration for device placement
  const config = getConfig();
  // Create tensor from typed array (zero-copy operation)
  return Tensor.fromTypedArray({
    data,
    shape: [rows, cols],
    dtype: "float64",
    device: config.defaultDevice,
  });
}

/**
 * Creates a 1D tensor from a dense Float64Array.
 *
 * **Time Complexity**: O(1) - tensor wraps existing array without copying
 *
 * @param data - Dense Float64Array
 * @returns New 1D tensor wrapping the data
 *
 * @internal
 */
export function fromDenseVector1D(data: Float64Array): Tensor {
  // Get global configuration for device placement
  const config = getConfig();
  // Create tensor from typed array (zero-copy operation)
  return Tensor.fromTypedArray({
    data,
    shape: [data.length],
    dtype: "float64",
    device: config.defaultDevice,
  });
}

/**
 * Performs LU factorization with partial pivoting on a square matrix.
 *
 * Implements Gaussian elimination with row pivoting for numerical stability.
 * Factorizes A into P*A = L*U where:
 * - P is a permutation matrix (represented by piv array)
 * - L is lower triangular with unit diagonal (stored in lower triangle of lu)
 * - U is upper triangular (stored in upper triangle of lu)
 *
 * **Algorithm**: Gaussian elimination with partial pivoting
 * **Time Complexity**: O(N³) where N is matrix dimension
 * **Space Complexity**: O(N²) for lu array + O(N) for pivot array
 *
 * @param a - Input square matrix as Float64Array in row-major order (N×N)
 * @param n - Matrix dimension (N)
 * @returns Object containing:
 *   - lu: Combined L and U matrices (L below diagonal, U on and above)
 *   - piv: Permutation vector (piv[i] = original row index of current row i)
 *   - pivSign: Sign of permutation (+1 or -1, used for determinant)
 * @throws {InvalidParameterError} If n is not a non-negative integer
 * @throws {ShapeError} If a.length is not n*n
 * @throws {DataValidationError} If a pivot is exactly zero (singular matrix), or the
 *   matrix contains non-finite values or overflows during elimination
 *
 * @internal
 */
export function luFactorSquare(
  a: Float64Array,
  n: number
): {
  readonly lu: Float64Array;
  readonly piv: Int32Array;
  readonly pivSign: number;
} {
  assertDim(n, "n");
  if (a.length !== n * n) {
    throw new ShapeError(`Matrix data length ${a.length} does not match ${n}x${n} (${n * n})`);
  }
  // Create working copy of input matrix
  const lu = new Float64Array(a);
  // Initialize permutation vector to identity
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;

  // Track sign of permutation for determinant calculation
  let pivSign = 1;

  // Main elimination loop over columns. The elimination triple loop is
  // O(n³); its inner accesses use direct typed-array indexing (with row
  // bases hoisted) rather than the bounds-checked `at()` accessor, which
  // V8 cannot keep on its fast path across the undefined-check/throw edge.
  for (let k = 0; k < n; k++) {
    const kRow = k * n;
    // Find pivot: row with largest absolute value in column k
    let maxRow = k;
    let maxVal = Math.abs(lu[kRow + k] as number);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + k] as number);
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }

    // Check for singularity or numerical issues
    if (maxVal === 0) {
      throw new DataValidationError("Matrix is singular");
    }
    if (!Number.isFinite(maxVal)) {
      throw new DataValidationError(
        "Matrix contains non-finite values or overflowed during LU factorization"
      );
    }

    // Perform row swap if needed (partial pivoting)
    if (maxRow !== k) {
      const mRow = maxRow * n;
      for (let j = 0; j < n; j++) {
        const tmp = lu[kRow + j] as number;
        lu[kRow + j] = lu[mRow + j] as number;
        lu[mRow + j] = tmp;
      }
      const tp = atInt(piv, k);
      piv[k] = atInt(piv, maxRow);
      piv[maxRow] = tp;
      pivSign = -pivSign;
    }

    // Perform elimination for rows below pivot
    // Scale by the reciprocal of the pivot (one division per column) when that
    // reciprocal is a normal float64; otherwise divide directly, because 1/pivot
    // would overflow or turn subnormal for very small or very large pivots.
    const pivot = lu[kRow + k] as number;
    const useRecip = reciprocalIsSafe(pivot);
    const invPivot = useRecip ? 1 / pivot : 0;
    for (let i = k + 1; i < n; i++) {
      const iRow = i * n;
      const lik = useRecip ? (lu[iRow + k] as number) * invPivot : (lu[iRow + k] as number) / pivot;
      lu[iRow + k] = lik;
      if (lik === 0) continue;
      // Update row i: row_i = row_i - lik * row_k
      for (let j = k + 1; j < n; j++) {
        lu[iRow + j] = (lu[iRow + j] as number) - lik * (lu[kRow + j] as number);
      }
    }
  }

  return { lu, piv, pivSign };
}

/**
 * Solves linear system(s) A*X = B using precomputed LU factorization.
 *
 * Performs forward substitution (L*Y = P*B) followed by backward substitution (U*X = Y).
 * Modifies b in-place to contain the solution.
 *
 * **Algorithm**: Forward and backward substitution
 * **Time Complexity**: O(N² * K) where N is matrix size, K is number of RHS
 * **Space Complexity**: O(N * K) for temporary permutation copy
 *
 * @param lu - Combined LU matrix from luFactorSquare (N×N)
 * @param piv - Permutation vector from luFactorSquare
 * @param n - Matrix dimension (N)
 * @param b - Right-hand side matrix (N×K), modified in-place to contain solution
 * @param nrhs - Number of right-hand sides (K)
 * @throws {InvalidParameterError} If n or nrhs is not a non-negative integer
 * @throws {ShapeError} If lu, piv or b do not have lengths N*N, N and N*K
 * @throws {DataValidationError} If matrix is singular (zero diagonal in U)
 *
 * @internal
 */
export function luSolveInPlace(
  lu: Float64Array,
  piv: Int32Array,
  n: number,
  b: Float64Array,
  nrhs: number
): void {
  assertDim(n, "n");
  assertDim(nrhs, "nrhs");
  if (lu.length !== n * n || piv.length !== n || b.length !== n * nrhs) {
    throw new ShapeError(
      `luSolveInPlace: expected lu length ${n * n}, piv length ${n} and b length ${n * nrhs}, ` +
        `got ${lu.length}, ${piv.length} and ${b.length}`
    );
  }

  // Step 1: Apply row permutation to RHS
  // Note: piv is a final permutation vector (not swap history)
  // Must use copy to avoid overwriting values needed later
  const b0 = new Float64Array(b);
  for (let i = 0; i < n; i++) {
    const pi = atInt(piv, i);
    for (let j = 0; j < nrhs; j++) {
      b[i * nrhs + j] = at(b0, pi * nrhs + j);
    }
  }

  // Step 2: Forward substitution: solve L*Y = P*B. L has unit diagonal.
  // axpy form (rank-1 update of the whole RHS row per already-solved row)
  // walks both b-rows contiguously in j and avoids the strided per-column
  // dot product of the textbook order.
  for (let i = 0; i < n; i++) {
    const iRow = i * nrhs;
    const luRow = i * n;
    for (let k = 0; k < i; k++) {
      const lik = lu[luRow + k] as number;
      if (lik === 0) continue;
      const kRow = k * nrhs;
      for (let j = 0; j < nrhs; j++) {
        b[iRow + j] = (b[iRow + j] as number) - lik * (b[kRow + j] as number);
      }
    }
  }

  // Step 3: Backward substitution: solve U*X = Y.
  for (let i = n - 1; i >= 0; i--) {
    const iRow = i * nrhs;
    const luRow = i * n;
    const diag = lu[luRow + i] as number;
    if (diag === 0) throw new DataValidationError("Matrix is singular");
    for (let k = i + 1; k < n; k++) {
      const uik = lu[luRow + k] as number;
      if (uik === 0) continue;
      const kRow = k * nrhs;
      for (let j = 0; j < nrhs; j++) {
        b[iRow + j] = (b[iRow + j] as number) - uik * (b[kRow + j] as number);
      }
    }
    if (reciprocalIsSafe(diag)) {
      const invDiag = 1 / diag;
      for (let j = 0; j < nrhs; j++) {
        b[iRow + j] = (b[iRow + j] as number) * invDiag;
      }
    } else {
      for (let j = 0; j < nrhs; j++) {
        b[iRow + j] = (b[iRow + j] as number) / diag;
      }
    }
  }
}

/**
 * Type-safe array element access for Float64Array.
 * Returns the element at index i, with TypeScript knowing it's always a number.
 * This eliminates the need for `?? 0` fallbacks outside of hot loops.
 *
 * @param arr - The Float64Array to access
 * @param i - Index to access
 * @returns The element at index i
 * @throws {IndexError} If i is out of bounds
 *
 * @internal
 */
export function at(arr: Float64Array, i: number): number {
  const value = arr[i];
  if (value === undefined) {
    throw new IndexError(`Index ${i} is out of bounds for Float64Array length ${arr.length}`, {
      index: i,
      validRange: [0, Math.max(0, arr.length - 1)],
    });
  }
  return value;
}

/**
 * Type-safe array element access for number arrays.
 * Returns the element at index i, with TypeScript knowing it's always a number.
 *
 * @param arr - The number array to access
 * @param i - Index to access
 * @returns The element at index i
 * @throws {IndexError} If i is out of bounds
 *
 * @internal
 */
export function atArr(arr: number[], i: number): number {
  const value = arr[i];
  if (value === undefined) {
    throw new IndexError(`Index ${i} is out of bounds for array length ${arr.length}`, {
      index: i,
      validRange: [0, Math.max(0, arr.length - 1)],
    });
  }
  return value;
}

/**
 * Type-safe element access for Int32Array with explicit bounds checks.
 *
 * @param arr - The Int32Array to access
 * @param i - Index to access
 * @returns The element at index i
 * @throws {IndexError} If i is out of bounds
 *
 * @internal
 */
export function atInt(arr: Int32Array, i: number): number {
  const value = arr[i];
  if (value === undefined) {
    throw new IndexError(`Index ${i} is out of bounds for Int32Array length ${arr.length}`, {
      index: i,
      validRange: [0, Math.max(0, arr.length - 1)],
    });
  }
  return value;
}

/**
 * Retrieves a tensor dimension with bounds checking.
 *
 * @param t - Input tensor
 * @param axis - Axis index (must be valid)
 * @param context - Context string for error messages
 * @returns Dimension size
 * @throws {ShapeError} If the axis does not exist
 *
 * @internal
 */
export function getDim(t: Tensor, axis: number, context: string): number {
  const dim = t.shape[axis];
  if (dim === undefined) {
    throw new ShapeError(`${context}: missing dimension for axis ${axis}`);
  }
  return dim;
}

/**
 * Retrieves a tensor stride with bounds checking.
 *
 * @param t - Input tensor
 * @param axis - Axis index (must be valid)
 * @param context - Context string for error messages
 * @returns Stride value
 * @throws {ShapeError} If the axis does not exist
 *
 * @internal
 */
export function getStride(t: Tensor, axis: number, context: string): number {
  const stride = t.strides[axis];
  if (stride === undefined) {
    throw new ShapeError(`${context}: missing stride for axis ${axis}`);
  }
  return stride;
}

/**
 * Validates that a tensor contains only finite numeric values.
 * Iterates using shape/strides, so views are checked correctly.
 *
 * @param t - Input tensor
 * @param context - Context string for error messages
 * @throws {DTypeError} If the tensor has string or complex dtype
 * @throws {DataValidationError} If any element is NaN or infinite
 *
 * @internal
 */
export function assertFiniteTensor(t: Tensor, context: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${context} does not support complex dtype`);
  }

  if (t.size === 0) return;

  const ndim = t.ndim;
  if (ndim === 0) {
    const val = Number(t.data[t.offset]);
    if (!Number.isFinite(val)) {
      throw new DataValidationError(`${context} contains non-finite values`);
    }
    return;
  }

  const shape = t.shape;
  const strides = t.strides;
  const data = t.data;

  // Fast path: contiguous numeric tensor (any offset), direct array scan
  if (!Array.isArray(data) && !(data instanceof BigInt64Array)) {
    // Check if contiguous by comparing strides
    let contiguous = true;
    let expected = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      if (strides[i] !== expected) {
        contiguous = false;
        break;
      }
      expected *= shape[i] ?? 1;
    }
    if (contiguous) {
      const end = t.offset + t.size;
      for (let i = t.offset; i < end; i++) {
        const val = data[i] as number;
        if (!Number.isFinite(val)) {
          throw new DataValidationError(`${context} contains non-finite values`);
        }
      }
      return;
    }
  }

  const idx = new Array<number>(ndim).fill(0);
  let offset = t.offset;

  for (let count = 0; count < t.size; count++) {
    const val = Number(data[offset]);
    if (!Number.isFinite(val)) {
      throw new DataValidationError(`${context} contains non-finite values`);
    }

    for (let d = ndim - 1; d >= 0; d--) {
      const dim = shape[d];
      const stride = strides[d];
      if (dim === undefined || stride === undefined) {
        throw new ShapeError(`${context}: tensor metadata is inconsistent`);
      }
      const idxVal = idx[d] ?? 0;
      const nextIdx = idxVal + 1;
      idx[d] = nextIdx;
      offset += stride;
      if (nextIdx < dim) {
        break;
      }
      offset -= nextIdx * stride;
      idx[d] = 0;
    }
  }
}
