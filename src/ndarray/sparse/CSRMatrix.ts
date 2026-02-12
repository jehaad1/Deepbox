import {
  getBigIntElement,
  getNumericElement,
  IndexError,
  type Shape,
  ShapeError,
} from "../../core";
import { type Tensor, Tensor as TensorImpl } from "../tensor/Tensor";

export type CSRMatrixInit = {
  readonly data: Float64Array;
  readonly indices: Int32Array;
  readonly indptr: Int32Array;
  readonly shape: Shape;
};

/**
 * Compressed Sparse Row (CSR) matrix representation.
 *
 * CSR format stores a sparse matrix using three arrays:
 * - `data`: Non-zero values in row-major order
 * - `indices`: Column indices of non-zero values
 * - `indptr`: Row pointers (indptr[i] to indptr[i+1] gives the range of data/indices for row i)
 *
 * This format is efficient for:
 * - Row slicing
 * - Matrix-vector products
 * - Arithmetic operations
 *
 * @example
 * ```ts
 * import { CSRMatrix } from 'deepbox/ndarray';
 *
 * // Create a 3x3 sparse matrix with values at (0,0)=1, (1,2)=2, (2,1)=3
 * const sparse = CSRMatrix.fromCOO({
 *   rows: 3, cols: 3,
 *   rowIndices: new Int32Array([0, 1, 2]),
 *   colIndices: new Int32Array([0, 2, 1]),
 *   values: new Float64Array([1, 2, 3])
 * });
 *
 * // Convert to dense for operations
 * const dense = sparse.toDense();
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format) | Wikipedia: CSR}
 */
export class CSRMatrix {
  readonly data: Float64Array;
  readonly indices: Int32Array;
  readonly indptr: Int32Array;
  readonly shape: Shape;

  constructor(init: CSRMatrixInit) {
    const [rows, cols] = init.shape;
    if (rows === undefined || cols === undefined || init.shape.length !== 2) {
      throw new ShapeError(`CSRMatrix shape must be 2D; received [${init.shape}]`);
    }
    if (rows < 0 || cols < 0) {
      throw new ShapeError(`CSRMatrix shape must be non-negative; received [${init.shape}]`);
    }
    if (init.indptr.length !== rows + 1) {
      throw new ShapeError(
        `CSRMatrix indptr length must be rows+1 (${rows + 1}); received ${init.indptr.length}`
      );
    }
    if (init.data.length !== init.indices.length) {
      throw new ShapeError(
        `CSRMatrix data/indices length mismatch: ${init.data.length} vs ${init.indices.length}`
      );
    }
    if ((init.indptr[0] ?? 0) !== 0) {
      throw new ShapeError("CSRMatrix indptr[0] must be 0");
    }
    const nnz = init.data.length;
    const lastPtr = init.indptr[rows] ?? 0;
    if (lastPtr !== nnz) {
      throw new ShapeError(
        `CSRMatrix indptr last value must equal nnz (${nnz}); received ${lastPtr}`
      );
    }
    let prevPtr = init.indptr[0] ?? 0;
    for (let i = 1; i < init.indptr.length; i++) {
      const ptr = init.indptr[i] ?? 0;
      if (ptr < prevPtr) {
        throw new ShapeError("CSRMatrix indptr must be non-decreasing");
      }
      if (ptr < 0 || ptr > nnz) {
        throw new ShapeError(
          `CSRMatrix indptr entries must be between 0 and nnz (${nnz}); received ${ptr}`
        );
      }
      prevPtr = ptr;
    }
    for (let i = 0; i < init.indices.length; i++) {
      const col = init.indices[i] ?? 0;
      if (col < 0 || col >= cols) {
        throw new IndexError(`CSRMatrix column index ${col} is out of bounds`, {
          index: col,
          validRange: [0, cols === 0 ? -1 : cols - 1],
        });
      }
    }

    this.data = init.data;
    this.indices = init.indices;
    this.indptr = init.indptr;
    this.shape = init.shape;
  }

  /** Number of non-zero elements in the matrix */
  get nnz(): number {
    return this.data.length;
  }

  /** Number of rows in the matrix */
  get rows(): number {
    return this.shape[0] ?? 0;
  }

  /** Number of columns in the matrix */
  get cols(): number {
    return this.shape[1] ?? 0;
  }

  /**
   * Convert the sparse matrix to a dense Tensor.
   *
   * @returns Dense 2D Tensor representation
   *
   * @example
   * ```ts
   * const dense = sparse.toDense();
   * console.log(dense.shape);  // [rows, cols]
   * ```
   */
  toDense(): Tensor {
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const out = new Float64Array(rows * cols);

    for (let r = 0; r < rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;
      for (let p = start; p < end; p++) {
        const c = this.indices[p] ?? 0;
        out[r * cols + c] = this.data[p] ?? 0;
      }
    }

    return TensorImpl.fromTypedArray({
      data: out,
      shape: [rows, cols],
      dtype: "float64",
      device: "cpu",
    });
  }

  /**
   * Add two sparse matrices element-wise.
   *
   * Both matrices must have the same shape.
   *
   * @param other - Matrix to add
   * @returns New CSRMatrix containing the sum
   * @throws {ShapeError} If shapes don't match
   *
   * @example
   * ```ts
   * const c = a.add(b);  // c = a + b
   * ```
   */
  add(other: CSRMatrix): CSRMatrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new ShapeError(`Cannot add matrices with shapes [${this.shape}] and [${other.shape}]`);
    }

    // Use a map-based approach to handle overlapping indices
    const resultData: number[] = [];
    const resultIndices: number[] = [];
    const resultIndptr: number[] = [0];

    for (let r = 0; r < this.rows; r++) {
      // Collect values from both matrices for this row
      const rowValues = new Map<number, number>();

      // Add values from this matrix
      const thisStart = this.indptr[r] ?? 0;
      const thisEnd = this.indptr[r + 1] ?? thisStart;
      for (let p = thisStart; p < thisEnd; p++) {
        const c = this.indices[p] ?? 0;
        const v = this.data[p] ?? 0;
        rowValues.set(c, (rowValues.get(c) ?? 0) + v);
      }

      // Add values from other matrix
      const otherStart = other.indptr[r] ?? 0;
      const otherEnd = other.indptr[r + 1] ?? otherStart;
      for (let p = otherStart; p < otherEnd; p++) {
        const c = other.indices[p] ?? 0;
        const v = other.data[p] ?? 0;
        rowValues.set(c, (rowValues.get(c) ?? 0) + v);
      }

      // Sort by column index and add non-zero values
      const sortedCols = Array.from(rowValues.keys()).sort((a, b) => a - b);
      for (const c of sortedCols) {
        const v = rowValues.get(c) ?? 0;
        if (v !== 0) {
          resultIndices.push(c);
          resultData.push(v);
        }
      }
      resultIndptr.push(resultData.length);
    }

    return new CSRMatrix({
      data: new Float64Array(resultData),
      indices: new Int32Array(resultIndices),
      indptr: new Int32Array(resultIndptr),
      shape: this.shape,
    });
  }

  /**
   * Subtract another sparse matrix element-wise.
   *
   * Both matrices must have the same shape.
   *
   * @param other - Matrix to subtract
   * @returns New CSRMatrix containing the difference
   * @throws {ShapeError} If shapes don't match
   *
   * @example
   * ```ts
   * const c = a.sub(b);  // c = a - b
   * ```
   */
  sub(other: CSRMatrix): CSRMatrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new ShapeError(
        `Cannot subtract matrices with shapes [${this.shape}] and [${other.shape}]`
      );
    }

    // Use a map-based approach to handle overlapping indices
    const resultData: number[] = [];
    const resultIndices: number[] = [];
    const resultIndptr: number[] = [0];

    for (let r = 0; r < this.rows; r++) {
      const rowValues = new Map<number, number>();

      // Add values from this matrix
      const thisStart = this.indptr[r] ?? 0;
      const thisEnd = this.indptr[r + 1] ?? thisStart;
      for (let p = thisStart; p < thisEnd; p++) {
        const c = this.indices[p] ?? 0;
        const v = this.data[p] ?? 0;
        rowValues.set(c, (rowValues.get(c) ?? 0) + v);
      }

      // Subtract values from other matrix
      const otherStart = other.indptr[r] ?? 0;
      const otherEnd = other.indptr[r + 1] ?? otherStart;
      for (let p = otherStart; p < otherEnd; p++) {
        const c = other.indices[p] ?? 0;
        const v = other.data[p] ?? 0;
        rowValues.set(c, (rowValues.get(c) ?? 0) - v);
      }

      // Sort by column index and add non-zero values
      const sortedCols = Array.from(rowValues.keys()).sort((a, b) => a - b);
      for (const c of sortedCols) {
        const v = rowValues.get(c) ?? 0;
        if (v !== 0) {
          resultIndices.push(c);
          resultData.push(v);
        }
      }
      resultIndptr.push(resultData.length);
    }

    return new CSRMatrix({
      data: new Float64Array(resultData),
      indices: new Int32Array(resultIndices),
      indptr: new Int32Array(resultIndptr),
      shape: this.shape,
    });
  }

  /**
   * Multiply all elements by a scalar value.
   *
   * @param scalar - Value to multiply by
   * @returns New CSRMatrix with scaled values
   *
   * @example
   * ```ts
   * const scaled = matrix.scale(2.0);  // Double all values
   * ```
   */
  scale(scalar: number): CSRMatrix {
    if (scalar === 0) {
      // Return empty sparse matrix
      return new CSRMatrix({
        data: new Float64Array(0),
        indices: new Int32Array(0),
        indptr: new Int32Array(this.rows + 1),
        shape: this.shape,
      });
    }

    const newData = new Float64Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      newData[i] = (this.data[i] ?? 0) * scalar;
    }

    return new CSRMatrix({
      data: newData,
      indices: this.indices.slice(),
      indptr: this.indptr.slice(),
      shape: this.shape,
    });
  }

  /**
   * Element-wise multiplication (Hadamard product) with another sparse matrix.
   *
   * Both matrices must have the same shape.
   *
   * @param other - Matrix to multiply with
   * @returns New CSRMatrix containing the element-wise product
   * @throws {ShapeError} If shapes don't match
   *
   * @example
   * ```ts
   * const c = a.multiply(b);  // c[i,j] = a[i,j] * b[i,j]
   * ```
   */
  multiply(other: CSRMatrix): CSRMatrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new ShapeError(
        `Cannot multiply matrices with shapes [${this.shape}] and [${other.shape}]`
      );
    }

    const resultData: number[] = [];
    const resultIndices: number[] = [];
    const resultIndptr: number[] = [0];

    for (let r = 0; r < this.rows; r++) {
      // Build a map of column -> value for the other matrix's row
      const otherRow = new Map<number, number>();
      const otherStart = other.indptr[r] ?? 0;
      const otherEnd = other.indptr[r + 1] ?? otherStart;
      for (let p = otherStart; p < otherEnd; p++) {
        otherRow.set(other.indices[p] ?? 0, other.data[p] ?? 0);
      }

      // Multiply matching elements
      const thisStart = this.indptr[r] ?? 0;
      const thisEnd = this.indptr[r + 1] ?? thisStart;
      for (let p = thisStart; p < thisEnd; p++) {
        const c = this.indices[p] ?? 0;
        const otherVal = otherRow.get(c);
        if (otherVal !== undefined) {
          const product = (this.data[p] ?? 0) * otherVal;
          if (product !== 0) {
            resultIndices.push(c);
            resultData.push(product);
          }
        }
      }
      resultIndptr.push(resultData.length);
    }

    return new CSRMatrix({
      data: new Float64Array(resultData),
      indices: new Int32Array(resultIndices),
      indptr: new Int32Array(resultIndptr),
      shape: this.shape,
    });
  }

  /**
   * Matrix multiplication with a dense vector.
   *
   * Computes y = A * x where A is this sparse matrix and x is a dense vector.
   *
   * @param vector - Dense vector (1D Tensor or Float64Array)
   * @returns Dense result vector as Tensor
   * @throws {ShapeError} If vector length doesn't match matrix columns
   *
   * @example
   * ```ts
   * const x = tensor([1, 2, 3]);
   * const y = sparse.matvec(x);  // y = A * x
   * ```
   */
  matvec(vector: Tensor | Float64Array): Tensor {
    const vecData = vector instanceof Float64Array ? vector : this.tensorToFloat64(vector);
    const vecLen = vecData.length;

    if (vecLen !== this.cols) {
      throw new ShapeError(`Vector length ${vecLen} doesn't match matrix columns ${this.cols}`);
    }

    const result = new Float64Array(this.rows);

    for (let r = 0; r < this.rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;
      let sum = 0;
      for (let p = start; p < end; p++) {
        const c = this.indices[p] ?? 0;
        sum += (this.data[p] ?? 0) * (vecData[c] ?? 0);
      }
      result[r] = sum;
    }

    return TensorImpl.fromTypedArray({
      data: result,
      shape: [this.rows],
      dtype: "float64",
      device: "cpu",
    });
  }

  /**
   * Matrix multiplication with a dense matrix.
   *
   * Computes C = A * B where A is this sparse matrix and B is a dense matrix.
   *
   * @param dense - Dense matrix (2D Tensor)
   * @returns Dense result matrix as Tensor
   * @throws {ShapeError} If inner dimensions don't match
   *
   * @example
   * ```ts
   * const B = tensor([[1, 2], [3, 4], [5, 6]]);
   * const C = sparse.matmul(B);  // C = A * B
   * ```
   */
  matmul(dense: Tensor): Tensor {
    if (dense.ndim !== 2) {
      throw new ShapeError(`Expected 2D tensor, got ${dense.ndim}D`);
    }

    const denseRows = dense.shape[0] ?? 0;
    const denseCols = dense.shape[1] ?? 0;

    if (this.cols !== denseRows) {
      throw new ShapeError(
        `Cannot multiply: sparse matrix columns (${this.cols}) != dense matrix rows (${denseRows})`
      );
    }

    const denseData = this.tensorToFloat64(dense);
    const result = new Float64Array(this.rows * denseCols);

    for (let r = 0; r < this.rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;

      for (let dc = 0; dc < denseCols; dc++) {
        let sum = 0;
        for (let p = start; p < end; p++) {
          const c = this.indices[p] ?? 0;
          sum += (this.data[p] ?? 0) * (denseData[c * denseCols + dc] ?? 0);
        }
        result[r * denseCols + dc] = sum;
      }
    }

    return TensorImpl.fromTypedArray({
      data: result,
      shape: [this.rows, denseCols],
      dtype: "float64",
      device: "cpu",
    });
  }

  /**
   * Transpose the sparse matrix.
   *
   * @returns New CSRMatrix representing the transpose
   *
   * @example
   * ```ts
   * const At = A.transpose();  // At[i,j] = A[j,i]
   * ```
   */
  transpose(): CSRMatrix {
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;

    // Count non-zeros per column (which become rows in transpose)
    const colCounts = new Int32Array(cols);
    for (let i = 0; i < this.indices.length; i++) {
      const colIdx = this.indices[i] ?? 0;
      colCounts[colIdx] = (colCounts[colIdx] ?? 0) + 1;
    }

    // Build indptr for transpose
    const newIndptr = new Int32Array(cols + 1);
    for (let c = 0; c < cols; c++) {
      newIndptr[c + 1] = (newIndptr[c] ?? 0) + (colCounts[c] ?? 0);
    }

    // Build data and indices
    const newData = new Float64Array(this.nnz);
    const newIndices = new Int32Array(this.nnz);
    const colNext = newIndptr.slice(0, cols);

    for (let r = 0; r < rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;
      for (let p = start; p < end; p++) {
        const c = this.indices[p] ?? 0;
        const pos = colNext[c] ?? 0;
        newData[pos] = this.data[p] ?? 0;
        newIndices[pos] = r;
        colNext[c] = pos + 1;
      }
    }

    return new CSRMatrix({
      data: newData,
      indices: newIndices,
      indptr: newIndptr,
      shape: [cols, rows],
    });
  }

  /**
   * Get a specific element from the matrix.
   *
   * @param row - Row index
   * @param col - Column index
   * @returns Value at the specified position (0 if not stored)
   * @throws {RangeError} If indices are out of bounds
   *
   * @example
   * ```ts
   * const value = matrix.get(1, 2);
   * ```
   */
  get(row: number, col: number): number {
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
      throw new IndexError(`Index (${row}, ${col}) out of bounds for shape [${this.shape}]`);
    }

    const start = this.indptr[row] ?? 0;
    const end = this.indptr[row + 1] ?? start;

    // Binary search for the column
    let lo = start;
    let hi = end;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const midCol = this.indices[mid] ?? 0;
      if (midCol < col) {
        lo = mid + 1;
      } else if (midCol > col) {
        hi = mid;
      } else {
        return this.data[mid] ?? 0;
      }
    }
    return 0;
  }

  /**
   * Create a copy of this matrix.
   *
   * @returns New CSRMatrix with copied data
   */
  copy(): CSRMatrix {
    return new CSRMatrix({
      data: this.data.slice(),
      indices: this.indices.slice(),
      indptr: this.indptr.slice(),
      shape: this.shape,
    });
  }

  /** Helper to convert Tensor to Float64Array */
  private tensorToFloat64(t: Tensor): Float64Array {
    if (t.dtype === "string") {
      throw new ShapeError("Cannot convert string tensor to numeric array");
    }
    if (
      t.data instanceof Float64Array &&
      t.offset === 0 &&
      t.strides.every((s, i) => {
        let expected = 1;
        for (let j = i + 1; j < t.ndim; j++) expected *= t.shape[j] ?? 1;
        return s === expected;
      })
    ) {
      return t.data;
    }
    // Copy with stride handling
    const out = new Float64Array(t.size);
    const strides = t.strides;
    const shape = t.shape;
    for (let i = 0; i < t.size; i++) {
      let offset = t.offset;
      let rem = i;
      for (let d = t.ndim - 1; d >= 0; d--) {
        const dim = shape[d] ?? 1;
        offset += (rem % dim) * (strides[d] ?? 0);
        rem = Math.floor(rem / dim);
      }
      const tData = t.data;
      if (Array.isArray(tData)) {
        throw new ShapeError("Cannot convert string tensor to numeric array");
      }
      out[i] =
        tData instanceof BigInt64Array
          ? Number(getBigIntElement(tData, offset))
          : getNumericElement(tData, offset);
    }
    return out;
  }

  /**
   * Create a sparse matrix from COO (Coordinate List) format.
   *
   * @param args - COO format specification
   * @param args.rows - Number of rows
   * @param args.cols - Number of columns
   * @param args.rowIndices - Row indices of non-zero values
   * @param args.colIndices - Column indices of non-zero values
   * @param args.values - Non-zero values
   * @param args.sort - Whether to sort entries (default: true)
   * @returns New CSRMatrix
   *
   * @example
   * ```ts
   * const sparse = CSRMatrix.fromCOO({
   *   rows: 3, cols: 3,
   *   rowIndices: new Int32Array([0, 1, 2]),
   *   colIndices: new Int32Array([0, 2, 1]),
   *   values: new Float64Array([1, 2, 3])
   * });
   * ```
   */
  static fromCOO(args: {
    readonly rows: number;
    readonly cols: number;
    readonly rowIndices: Int32Array;
    readonly colIndices: Int32Array;
    readonly values: Float64Array;
    readonly sort?: boolean;
  }): CSRMatrix {
    const { rows, cols, rowIndices, colIndices, values } = args;
    if (rowIndices.length !== colIndices.length || rowIndices.length !== values.length) {
      throw new ShapeError("COO arrays must have the same length");
    }

    const nnz = values.length;
    const order = new Int32Array(nnz);
    for (let i = 0; i < nnz; i++) order[i] = i;

    const shouldSort = args.sort ?? true;
    if (shouldSort) {
      order.sort((a, b) => {
        const ra = rowIndices[a] ?? 0;
        const rb = rowIndices[b] ?? 0;
        if (ra !== rb) return ra - rb;
        return (colIndices[a] ?? 0) - (colIndices[b] ?? 0);
      });
    }

    const indptr = new Int32Array(rows + 1);
    for (let k = 0; k < nnz; k++) {
      const i = order[k] ?? 0;
      const r = rowIndices[i] ?? 0;
      if (r < 0 || r >= rows) {
        throw new IndexError(`row index out of bounds: ${r}`);
      }
      indptr[r + 1] = (indptr[r + 1] ?? 0) + 1;
    }

    for (let r = 0; r < rows; r++) {
      indptr[r + 1] = (indptr[r + 1] ?? 0) + (indptr[r] ?? 0);
    }

    const indices = new Int32Array(nnz);
    const data = new Float64Array(nnz);
    const next = indptr.slice();

    for (let k = 0; k < nnz; k++) {
      const i = order[k] ?? 0;
      const r = rowIndices[i] ?? 0;
      const c = colIndices[i] ?? 0;
      if (c < 0 || c >= cols) {
        throw new IndexError(`col index out of bounds: ${c}`);
      }
      const pos = next[r] ?? 0;
      indices[pos] = c;
      data[pos] = values[i] ?? 0;
      next[r] = pos + 1;
    }

    return new CSRMatrix({ data, indices, indptr, shape: [rows, cols] });
  }
}
