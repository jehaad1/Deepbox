import { DataValidationError, DTypeError, getConfig, IndexError, ShapeError } from "../core";
import { Tensor } from "../ndarray";

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
 * @throws {DTypeError} If tensor has string dtype
 *
 * @internal
 */
export function asMatrix2D(t: Tensor): Matrix2D {
  // Reject string tensors first
  if (t.dtype === "string") {
    throw new DTypeError("String tensors are not supported");
  }

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

  // Check if data is contiguous in row-major (C-order) layout
  // Row-major: elements in same row are adjacent (strideCol=1), rows are cols apart
  const isRowMajorContiguous = strideCol === 1 && strideRow === cols;

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
 *
 * **Time Complexity**:
 * - O(M*N) where M=rows, N=cols
 * - Optimized path for contiguous data: O(M*N) with better cache locality
 *
 * @param t - Input 2D tensor
 * @returns Object containing dimensions and dense data array
 *
 * @internal
 */
export function toDenseMatrix2D(t: Tensor): {
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
        throw new DataValidationError("Input contains non-finite values");
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
        throw new DataValidationError("Input contains non-finite values");
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
 * @returns Dense Float64Array copy of the data
 * @throws {ShapeError} If tensor is not 1D
 * @throws {DTypeError} If tensor has string dtype
 * @throws {DataValidationError} If tensor contains non-finite values
 *
 * @internal
 */
export function toDenseVector1D(t: Tensor): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError("String tensors are not supported");
  }
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
      throw new DataValidationError("Input contains non-finite values");
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
 *
 * @internal
 */
export function fromDenseMatrix2D(rows: number, cols: number, data: Float64Array): Tensor {
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
 * @throws {DataValidationError} If matrix is singular or contains non-finite values
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
  // Create working copy of input matrix
  const lu = new Float64Array(a);
  // Initialize permutation vector to identity
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;

  // Track sign of permutation for determinant calculation
  let pivSign = 1;

  // Main elimination loop over columns
  for (let k = 0; k < n; k++) {
    // Find pivot: row with largest absolute value in column k
    let maxRow = k;
    let maxVal = Math.abs(at(lu, k * n + k));
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(at(lu, i * n + k));
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }

    // Check for singularity or numerical issues
    if (!Number.isFinite(maxVal) || maxVal === 0) {
      throw new DataValidationError("Matrix is singular");
    }

    // Perform row swap if needed (partial pivoting)
    if (maxRow !== k) {
      // Swap rows k and maxRow in lu matrix
      for (let j = 0; j < n; j++) {
        const tmp = at(lu, k * n + j);
        lu[k * n + j] = at(lu, maxRow * n + j);
        lu[maxRow * n + j] = tmp;
      }
      // Update permutation vector
      const tp = atInt(piv, k);
      piv[k] = atInt(piv, maxRow);
      piv[maxRow] = tp;
      // Flip sign for determinant
      pivSign = -pivSign;
    }

    // Perform elimination for rows below pivot
    const pivot = at(lu, k * n + k);
    for (let i = k + 1; i < n; i++) {
      // Compute multiplier (stored in L part)
      lu[i * n + k] = at(lu, i * n + k) / pivot;
      const lik = at(lu, i * n + k);
      // Update row i: row_i = row_i - lik * row_k
      for (let j = k + 1; j < n; j++) {
        lu[i * n + j] = at(lu, i * n + j) - lik * at(lu, k * n + j);
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

  // Step 2: Forward substitution - solve L*Y = P*B
  // L has unit diagonal (implicit 1s), so no division needed
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < nrhs; j++) {
      let sum = at(b, i * nrhs + j);
      // Subtract contributions from already-solved variables
      for (let k = 0; k < i; k++) {
        sum -= at(lu, i * n + k) * at(b, k * nrhs + j);
      }
      b[i * nrhs + j] = sum;
    }
  }

  // Step 3: Backward substitution - solve U*X = Y
  // U has explicit diagonal, must divide by it
  for (let i = n - 1; i >= 0; i--) {
    const diag = at(lu, i * n + i);
    // Check for singularity
    if (diag === 0) throw new DataValidationError("Matrix is singular");
    for (let j = 0; j < nrhs; j++) {
      let sum = at(b, i * nrhs + j);
      // Subtract contributions from already-solved variables
      for (let k = i + 1; k < n; k++) {
        sum -= at(lu, i * n + k) * at(b, k * nrhs + j);
      }
      // Divide by diagonal element
      b[i * nrhs + j] = sum / diag;
    }
  }
}

/**
 * Type-safe array element access for Float64Array.
 * Returns the element at index i, with TypeScript knowing it's always a number.
 * This eliminates the need for `?? 0` fallbacks in tight loops.
 *
 * **IMPORTANT**: Caller must ensure index is within bounds.
 * This function does NOT perform bounds checking for performance.
 *
 * @param arr - The Float64Array to access
 * @param i - Index to access (must be valid)
 * @returns The element at index i
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
 * **IMPORTANT**: Caller must ensure index is within bounds.
 *
 * @param arr - The number array to access
 * @param i - Index to access (must be valid)
 * @returns The element at index i
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
 *
 * @internal
 */
export function assertFiniteTensor(t: Tensor, context: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
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

  // Fast path: contiguous zero-offset numeric tensor — direct array scan
  if (!Array.isArray(data) && !(data instanceof BigInt64Array) && t.offset === 0) {
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
      for (let i = 0; i < t.size; i++) {
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
