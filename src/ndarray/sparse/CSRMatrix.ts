import {
  DTypeError,
  getBigIntElement,
  IndexError,
  InvalidParameterError,
  type Shape,
  ShapeError,
} from "../../core";
import { type Tensor, Tensor as TensorImpl } from "../tensor/Tensor";

/** Initialization data for constructing a {@link CSRMatrix}. */
export type CSRMatrixInit = {
  readonly data: Float64Array;
  readonly indices: Int32Array;
  readonly indptr: Int32Array;
  readonly shape: Shape;
};

/**
 * Convert the contents of a numeric tensor to a row-major Float64Array.
 *
 * Contiguous float64 tensors are returned without copying (the result may be a view, so callers
 * must treat it as read-only). Strided views, other dtypes and int64 tensors are copied.
 */
function tensorToFloat64(t: Tensor): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError("Cannot convert string tensor to numeric array");
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`Cannot convert ${t.dtype} tensor to a real numeric array`);
  }
  const src = t.data;
  if (Array.isArray(src)) {
    throw new DTypeError("Cannot convert string tensor to numeric array");
  }
  const size = t.size;
  const shape = t.shape;
  const strides = t.strides;
  const ndim = t.ndim;

  let contiguous = strides.length === ndim;
  if (contiguous) {
    let expected = 1;
    for (let d = ndim - 1; d >= 0; d--) {
      const dim = shape[d] ?? 1;
      // Dimensions of length 1 never contribute to an offset, whatever their stride.
      if (dim !== 1 && strides[d] !== expected) {
        contiguous = false;
        break;
      }
      expected *= dim;
    }
  }

  if (contiguous) {
    const base = t.offset;
    if (src instanceof Float64Array) {
      if (base === 0 && src.length === size) return src;
      return src.subarray(base, base + size);
    }
    const out = new Float64Array(size);
    if (src instanceof BigInt64Array) {
      for (let i = 0; i < size; i++) out[i] = Number(getBigIntElement(src, base + i));
    } else {
      for (let i = 0; i < size; i++) out[i] = src[base + i] ?? 0;
    }
    return out;
  }

  // Strided view: walk the logical index space with an odometer.
  const out = new Float64Array(size);
  const coords = new Array<number>(ndim).fill(0);
  let offset = t.offset;
  for (let i = 0; i < size; i++) {
    out[i] =
      src instanceof BigInt64Array ? Number(getBigIntElement(src, offset)) : (src[offset] ?? 0);
    for (let d = ndim - 1; d >= 0; d--) {
      const c = (coords[d] ?? 0) + 1;
      const stride = strides[d] ?? 0;
      if (c < (shape[d] ?? 1)) {
        coords[d] = c;
        offset += stride;
        break;
      }
      offset -= (c - 1) * stride;
      coords[d] = 0;
    }
  }
  return out;
}

/** Throw unless `value` is a non-negative integer (used for shapes and sizes). */
function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new InvalidParameterError(
      `${name} must be a non-negative integer; received ${value}`,
      name,
      value
    );
  }
}

/** Throw unless `value` is an integer (used for row/column positions). */
function assertInteger(value: number, name: string): void {
  if (!Number.isInteger(value)) {
    throw new InvalidParameterError(`${name} must be an integer; received ${value}`, name, value);
  }
}

/**
 * Sparse accumulator for one output row: a dense value array plus a list of touched columns,
 * so that building a row costs O(touched) rather than O(cols). Allocated once per operation and
 * reused for every row.
 */
class RowAccumulator {
  private readonly values: Float64Array;
  private readonly mark: Int32Array;
  private readonly touched: Int32Array;
  private count = 0;
  private stamp = 0;

  constructor(cols: number) {
    this.values = new Float64Array(cols);
    this.mark = new Int32Array(cols);
    this.touched = new Int32Array(cols);
  }

  /** Start a new row. */
  begin(): void {
    this.stamp++;
    this.count = 0;
  }

  add(col: number, value: number): void {
    if (this.mark[col] === this.stamp) {
      this.values[col] = (this.values[col] ?? 0) + value;
    } else {
      this.mark[col] = this.stamp;
      this.values[col] = value;
      this.touched[this.count++] = col;
    }
  }

  has(col: number): boolean {
    return this.mark[col] === this.stamp;
  }

  get(col: number): number {
    return this.values[col] ?? 0;
  }

  /** Touched columns in ascending order. */
  sortedColumns(): Int32Array {
    const cols = this.touched.subarray(0, this.count);
    cols.sort();
    return cols;
  }
}

/** Growable CSR output buffer. */
class CSRBuilder {
  private data: Float64Array;
  private indices: Int32Array;
  private length = 0;
  private readonly indptr: Int32Array;

  constructor(rows: number, capacity: number) {
    const cap = Math.max(capacity, 1);
    this.data = new Float64Array(cap);
    this.indices = new Int32Array(cap);
    this.indptr = new Int32Array(rows + 1);
  }

  push(col: number, value: number): void {
    if (this.length === this.data.length) {
      const cap = this.data.length * 2;
      const data = new Float64Array(cap);
      data.set(this.data);
      const indices = new Int32Array(cap);
      indices.set(this.indices);
      this.data = data;
      this.indices = indices;
    }
    this.data[this.length] = value;
    this.indices[this.length] = col;
    this.length++;
  }

  /** Close row `row`: everything pushed so far belongs to rows up to and including it. */
  endRow(row: number): void {
    this.indptr[row + 1] = this.length;
  }

  /** Emit the accumulator's columns in ascending order, optionally dropping exact zeros. */
  flush(acc: RowAccumulator, row: number, dropZeros: boolean): void {
    const cols = acc.sortedColumns();
    for (let i = 0; i < cols.length; i++) {
      const c = cols[i] ?? 0;
      const v = acc.get(c);
      if (!dropZeros || v !== 0) this.push(c, v);
    }
    this.endRow(row);
  }

  finish(shape: Shape): CSRMatrixInit {
    return {
      data: this.data.slice(0, this.length),
      indices: this.indices.slice(0, this.length),
      indptr: this.indptr,
      shape,
    };
  }
}

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
 * The constructor stores the arrays it is given without copying. Column indices within a row
 * should be strictly increasing (canonical form, see {@link CSRMatrix.hasCanonicalFormat}); all
 * methods also give the right result for unsorted rows or repeated column indices, treating
 * repeated entries as a sum as SciPy does, but `get` and `getCol` are faster on canonical input.
 * Results produced by the arithmetic methods are always canonical.
 *
 * Only entries that are stored take part in `matvec`, `matmul`, `multiply` and `spmm`, so a
 * NaN or Infinity in the dense operand does not propagate through a structural zero.
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
 * @see {@link https://deepbox.dev/docs/ndarray-sparse | Deepbox Sparse Matrices}
 */
export class CSRMatrix {
  readonly data: Float64Array;
  readonly indices: Int32Array;
  readonly indptr: Int32Array;
  readonly shape: Shape;
  private readonly canonical: boolean;

  constructor(init: CSRMatrixInit) {
    const [rows, cols] = init.shape;
    if (rows === undefined || cols === undefined || init.shape.length !== 2) {
      throw new ShapeError(`CSRMatrix shape must be 2D; received [${init.shape}]`);
    }
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
      throw new ShapeError(
        `CSRMatrix shape must be non-negative integers; received [${init.shape}]`
      );
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
    let prevPtr = 0;
    for (let i = 1; i < init.indptr.length; i++) {
      const ptr = init.indptr[i] ?? 0;
      if (ptr < prevPtr) {
        throw new ShapeError("CSRMatrix indptr must be non-decreasing");
      }
      prevPtr = ptr;
    }
    for (let i = 0; i < nnz; i++) {
      const col = init.indices[i] ?? 0;
      if (col < 0 || col >= cols) {
        throw new IndexError(`CSRMatrix column index ${col} is out of bounds`, {
          index: col,
          validRange: [0, cols === 0 ? -1 : cols - 1],
        });
      }
    }

    let canonical = true;
    for (let r = 0; r < rows && canonical; r++) {
      const end = init.indptr[r + 1] ?? 0;
      for (let p = (init.indptr[r] ?? 0) + 1; p < end; p++) {
        if ((init.indices[p] ?? 0) <= (init.indices[p - 1] ?? 0)) {
          canonical = false;
          break;
        }
      }
    }

    this.data = init.data;
    this.indices = init.indices;
    this.indptr = init.indptr;
    this.shape = [rows, cols];
    this.canonical = canonical;
  }

  /** Number of stored entries (explicitly stored zeros included). */
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
   * Whether column indices are strictly increasing within every row (sorted, no repeated
   * entries). Matrices built by `fromCOO`, `eye`, `diag`, `fromDense` and by the arithmetic
   * methods are always canonical.
   */
  get hasCanonicalFormat(): boolean {
    return this.canonical;
  }

  /**
   * Convert the sparse matrix to a dense Tensor.
   *
   * @returns Dense 2D float64 Tensor representation
   *
   * @example
   * ```ts
   * const dense = sparse.toDense();
   * console.log(dense.shape);  // [rows, cols]
   * ```
   */
  toDense(): Tensor {
    const rows = this.rows;
    const cols = this.cols;
    const out = new Float64Array(rows * cols);

    for (let r = 0; r < rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;
      const rowBase = r * cols;
      for (let p = start; p < end; p++) {
        const i = rowBase + (this.indices[p] ?? 0);
        out[i] = (out[i] ?? 0) + (this.data[p] ?? 0);
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
   * Create a CSR matrix from a dense 2D tensor, storing every entry that is not exactly zero.
   * NaN and Infinity are stored, since they are not zero.
   *
   * @param dense - 2D numeric tensor
   * @returns New canonical CSRMatrix
   * @throws {ShapeError} If the tensor is not 2D
   * @throws {DTypeError} If the tensor holds strings or complex numbers
   *
   * @example
   * ```ts
   * const A = CSRMatrix.fromDense(tensor([[1, 0], [0, 2]]));
   * A.nnz; // 2
   * ```
   */
  static fromDense(dense: Tensor): CSRMatrix {
    if (dense.ndim !== 2) {
      throw new ShapeError(`Expected 2D tensor, got ${dense.ndim}D`);
    }
    const rows = dense.shape[0] ?? 0;
    const cols = dense.shape[1] ?? 0;
    const values = tensorToFloat64(dense);

    let nnz = 0;
    for (let i = 0; i < values.length; i++) {
      if (values[i] !== 0) nnz++;
    }
    const data = new Float64Array(nnz);
    const indices = new Int32Array(nnz);
    const indptr = new Int32Array(rows + 1);
    let k = 0;
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      for (let c = 0; c < cols; c++) {
        const v = values[base + c] ?? 0;
        if (v !== 0) {
          data[k] = v;
          indices[k] = c;
          k++;
        }
      }
      indptr[r + 1] = k;
    }
    return new CSRMatrix({ data, indices, indptr, shape: [rows, cols] });
  }

  /**
   * Return an equivalent matrix in canonical form: columns sorted within each row and repeated
   * entries summed. Explicitly stored zeros are kept. A canonical matrix is returned as a copy.
   *
   * @returns New canonical CSRMatrix
   */
  canonicalize(): CSRMatrix {
    if (this.canonical) return this.copy();
    const acc = new RowAccumulator(this.cols);
    const builder = new CSRBuilder(this.rows, this.nnz);
    for (let r = 0; r < this.rows; r++) {
      acc.begin();
      const end = this.indptr[r + 1] ?? 0;
      for (let p = this.indptr[r] ?? 0; p < end; p++) {
        acc.add(this.indices[p] ?? 0, this.data[p] ?? 0);
      }
      builder.flush(acc, r, false);
    }
    return new CSRMatrix(builder.finish(this.shape));
  }

  /**
   * Combine two same-shape matrices as `this + sign * other`, dropping exact zeros.
   */
  private combine(other: CSRMatrix, sign: 1 | -1): CSRMatrix {
    const acc = new RowAccumulator(this.cols);
    const builder = new CSRBuilder(this.rows, this.nnz + other.nnz);
    for (let r = 0; r < this.rows; r++) {
      acc.begin();
      const thisEnd = this.indptr[r + 1] ?? 0;
      for (let p = this.indptr[r] ?? 0; p < thisEnd; p++) {
        acc.add(this.indices[p] ?? 0, this.data[p] ?? 0);
      }
      const otherEnd = other.indptr[r + 1] ?? 0;
      for (let p = other.indptr[r] ?? 0; p < otherEnd; p++) {
        acc.add(other.indices[p] ?? 0, sign * (other.data[p] ?? 0));
      }
      builder.flush(acc, r, true);
    }
    return new CSRMatrix(builder.finish(this.shape));
  }

  /**
   * Add two sparse matrices element-wise.
   *
   * Both matrices must have the same shape. Entries that cancel to exactly zero are not stored
   * in the result.
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
    return this.combine(other, 1);
  }

  /**
   * Subtract another sparse matrix element-wise.
   *
   * Both matrices must have the same shape. Entries that cancel to exactly zero are not stored
   * in the result.
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
    return this.combine(other, -1);
  }

  /**
   * Multiply all elements by a scalar value.
   *
   * Multiplying by zero returns a matrix with no stored entries, except that NaN and Infinity
   * entries stay stored (as NaN), because `0 * Infinity` is NaN.
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
      const builder = new CSRBuilder(this.rows, 0);
      for (let r = 0; r < this.rows; r++) {
        const end = this.indptr[r + 1] ?? 0;
        for (let p = this.indptr[r] ?? 0; p < end; p++) {
          const v = (this.data[p] ?? 0) * scalar;
          if (v !== 0) builder.push(this.indices[p] ?? 0, v);
        }
        builder.endRow(r);
      }
      return new CSRMatrix(builder.finish(this.shape));
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
   * Both matrices must have the same shape. Only positions stored in both matrices contribute;
   * products that are exactly zero are not stored.
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

    const otherRow = new RowAccumulator(this.cols);
    const acc = new RowAccumulator(this.cols);
    const builder = new CSRBuilder(this.rows, Math.min(this.nnz, other.nnz));

    for (let r = 0; r < this.rows; r++) {
      otherRow.begin();
      const otherEnd = other.indptr[r + 1] ?? 0;
      for (let p = other.indptr[r] ?? 0; p < otherEnd; p++) {
        otherRow.add(other.indices[p] ?? 0, other.data[p] ?? 0);
      }

      acc.begin();
      const thisEnd = this.indptr[r + 1] ?? 0;
      for (let p = this.indptr[r] ?? 0; p < thisEnd; p++) {
        const c = this.indices[p] ?? 0;
        if (otherRow.has(c)) {
          acc.add(c, (this.data[p] ?? 0) * otherRow.get(c));
        }
      }
      builder.flush(acc, r, true);
    }

    return new CSRMatrix(builder.finish(this.shape));
  }

  /**
   * Matrix multiplication with a dense vector.
   *
   * Computes y = A * x where A is this sparse matrix and x is a dense vector.
   *
   * @param vector - Dense vector (1D Tensor, Float64Array, or a tensor whose only non-singleton
   *   dimension is the vector, such as [n, 1])
   * @returns 1D dense result vector as Tensor
   * @throws {ShapeError} If the tensor has more than one non-singleton dimension or its length
   *   doesn't match the matrix columns
   *
   * @example
   * ```ts
   * const x = tensor([1, 2, 3]);
   * const y = sparse.matvec(x);  // y = A * x
   * ```
   */
  matvec(vector: Tensor | Float64Array): Tensor {
    let vecData: Float64Array;
    if (vector instanceof Float64Array) {
      vecData = vector;
    } else {
      // A vector may carry singleton dimensions (for example [n, 1] or [1, n]); anything with
      // more than one non-singleton dimension is a matrix and would be silently flattened.
      let nonSingleton = 0;
      for (const dim of vector.shape) {
        if (dim !== 1) nonSingleton++;
      }
      if (nonSingleton > 1) {
        throw new ShapeError(
          `matvec expects a vector, got shape [${vector.shape}]; use matmul for matrix operands`
        );
      }
      vecData = tensorToFloat64(vector);
    }
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
        sum += (this.data[p] ?? 0) * (vecData[this.indices[p] ?? 0] ?? 0);
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

    const denseData = tensorToFloat64(dense);
    const result = new Float64Array(this.rows * denseCols);

    for (let r = 0; r < this.rows; r++) {
      const start = this.indptr[r] ?? 0;
      const end = this.indptr[r + 1] ?? start;
      const outBase = r * denseCols;

      // Row-major friendly order: each stored entry updates one whole output row slice.
      for (let p = start; p < end; p++) {
        const v = this.data[p] ?? 0;
        const inBase = (this.indices[p] ?? 0) * denseCols;
        for (let dc = 0; dc < denseCols; dc++) {
          result[outBase + dc] = (result[outBase + dc] ?? 0) + v * (denseData[inBase + dc] ?? 0);
        }
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
    const rows = this.rows;
    const cols = this.cols;

    // Count non-zeros per column (which become rows in transpose)
    const newIndptr = new Int32Array(cols + 1);
    for (let i = 0; i < this.indices.length; i++) {
      const colIdx = this.indices[i] ?? 0;
      newIndptr[colIdx + 1] = (newIndptr[colIdx + 1] ?? 0) + 1;
    }
    for (let c = 0; c < cols; c++) {
      newIndptr[c + 1] = (newIndptr[c + 1] ?? 0) + (newIndptr[c] ?? 0);
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
   * @throws {InvalidParameterError} If an index is not an integer
   * @throws {IndexError} If an index is out of bounds
   *
   * @example
   * ```ts
   * const value = matrix.get(1, 2);
   * ```
   */
  get(row: number, col: number): number {
    assertInteger(row, "row");
    assertInteger(col, "col");
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
      throw new IndexError(`Index (${row}, ${col}) out of bounds for shape [${this.shape}]`);
    }
    return this.lookup(row, col);
  }

  /** Value at (row, col) with already-validated indices. */
  private lookup(row: number, col: number): number {
    const start = this.indptr[row] ?? 0;
    const end = this.indptr[row + 1] ?? start;

    if (!this.canonical) {
      // Unsorted or repeated columns: sum every stored entry at this position.
      let sum = 0;
      for (let p = start; p < end; p++) {
        if ((this.indices[p] ?? 0) === col) sum += this.data[p] ?? 0;
      }
      return sum;
    }

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

  /**
   * Sparse-sparse matrix multiplication.
   *
   * Computes C = A * B where both A and B are CSR matrices. Products that cancel to exactly
   * zero are not stored in the result.
   *
   * @param other - Sparse matrix to multiply with
   * @returns New CSRMatrix containing the product
   * @throws {ShapeError} If inner dimensions don't match
   *
   * @example
   * ```ts
   * const C = A.spmm(B);  // C = A * B (sparse × sparse)
   * ```
   */
  spmm(other: CSRMatrix): CSRMatrix {
    if (this.cols !== other.rows) {
      throw new ShapeError(
        `Cannot multiply: left columns (${this.cols}) != right rows (${other.rows})`
      );
    }

    const acc = new RowAccumulator(other.cols);
    const builder = new CSRBuilder(this.rows, this.nnz + other.nnz);

    for (let r = 0; r < this.rows; r++) {
      acc.begin();
      const aEnd = this.indptr[r + 1] ?? 0;

      for (let pa = this.indptr[r] ?? 0; pa < aEnd; pa++) {
        const k = this.indices[pa] ?? 0;
        const aVal = this.data[pa] ?? 0;

        // Multiply with row k of other
        const bEnd = other.indptr[k + 1] ?? 0;
        for (let pb = other.indptr[k] ?? 0; pb < bEnd; pb++) {
          acc.add(other.indices[pb] ?? 0, aVal * (other.data[pb] ?? 0));
        }
      }

      builder.flush(acc, r, true);
    }

    return new CSRMatrix(builder.finish([this.rows, other.cols]));
  }

  /**
   * Extract a contiguous range of rows as a new CSRMatrix.
   *
   * Out-of-range bounds are clamped to `[0, rows]`, and an empty or reversed range gives a
   * matrix with zero rows. Infinity is accepted as a bound.
   *
   * @param start - First row index (inclusive)
   * @param end - Last row index (exclusive)
   * @returns New CSRMatrix containing only the selected rows
   * @throws {InvalidParameterError} If a bound is NaN or not an integer
   *
   * @example
   * ```ts
   * const sub = matrix.sliceRows(1, 3);  // rows 1 and 2
   * ```
   */
  sliceRows(start: number, end: number): CSRMatrix {
    for (const [name, value] of [
      ["start", start],
      ["end", end],
    ] as const) {
      if (!Number.isInteger(value) && value !== Infinity && value !== -Infinity) {
        throw new InvalidParameterError(
          `${name} must be an integer; received ${value}`,
          name,
          value
        );
      }
    }
    if (start < 0) start = 0;
    if (end > this.rows) end = this.rows;
    if (start >= end) {
      // start > end (e.g. sliceRows(3, 1)) yields an empty matrix; the
      // indptr length must not go negative.
      return new CSRMatrix({
        data: new Float64Array(0),
        indices: new Int32Array(0),
        indptr: new Int32Array(1),
        shape: [0, this.cols],
      });
    }

    const pStart = this.indptr[start] ?? 0;
    const pEnd = this.indptr[end] ?? pStart;
    const newData = this.data.slice(pStart, pEnd);
    const newIndices = this.indices.slice(pStart, pEnd);
    const newRows = end - start;
    const newIndptr = new Int32Array(newRows + 1);
    for (let i = 0; i <= newRows; i++) {
      newIndptr[i] = (this.indptr[start + i] ?? 0) - pStart;
    }

    return new CSRMatrix({
      data: newData,
      indices: newIndices,
      indptr: newIndptr,
      shape: [newRows, this.cols],
    });
  }

  /**
   * Extract a single row as a 1D dense Float64Array.
   *
   * @param row - Row index
   * @returns Dense array of the row values
   * @throws {InvalidParameterError} If `row` is not an integer
   * @throws {IndexError} If `row` is out of bounds
   *
   * @example
   * ```ts
   * const row = matrix.getRow(0);  // [1, 0, 2, 0, ...]
   * ```
   */
  getRow(row: number): Float64Array {
    assertInteger(row, "row");
    if (row < 0 || row >= this.rows) {
      throw new IndexError(`Row index ${row} out of bounds for ${this.rows} rows`);
    }
    const out = new Float64Array(this.cols);
    const start = this.indptr[row] ?? 0;
    const end = this.indptr[row + 1] ?? start;
    for (let p = start; p < end; p++) {
      const c = this.indices[p] ?? 0;
      out[c] = (out[c] ?? 0) + (this.data[p] ?? 0);
    }
    return out;
  }

  /**
   * Extract a single column as a 1D dense Float64Array.
   *
   * @param col - Column index
   * @returns Dense array of the column values
   * @throws {InvalidParameterError} If `col` is not an integer
   * @throws {IndexError} If `col` is out of bounds
   *
   * @example
   * ```ts
   * const col = matrix.getCol(2);  // [0, 2, 0, ...]
   * ```
   */
  getCol(col: number): Float64Array {
    assertInteger(col, "col");
    if (col < 0 || col >= this.cols) {
      throw new IndexError(`Column index ${col} out of bounds for ${this.cols} columns`);
    }
    const out = new Float64Array(this.rows);
    for (let r = 0; r < this.rows; r++) {
      out[r] = this.lookup(r, col);
    }
    return out;
  }

  /**
   * Create a sparse identity matrix.
   *
   * @param n - Size of the identity matrix (n × n)
   * @returns CSRMatrix identity
   * @throws {InvalidParameterError} If `n` is not a non-negative integer
   *
   * @example
   * ```ts
   * const I = CSRMatrix.eye(4);  // 4×4 identity
   * ```
   */
  static eye(n: number): CSRMatrix {
    assertNonNegativeInteger(n, "n");
    const data = new Float64Array(n).fill(1);
    const indices = new Int32Array(n);
    const indptr = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      indices[i] = i;
      indptr[i + 1] = i + 1;
    }
    return new CSRMatrix({ data, indices, indptr, shape: [n, n] });
  }

  /**
   * Create a sparse diagonal matrix from values. Zero values are not stored.
   *
   * @param values - Diagonal values
   * @returns CSRMatrix with values on the diagonal
   *
   * @example
   * ```ts
   * const D = CSRMatrix.diag(new Float64Array([1, 2, 3]));
   * ```
   */
  static diag(values: Float64Array): CSRMatrix {
    const n = values.length;
    const data = new Float64Array(n);
    const indices = new Int32Array(n);
    const indptr = new Int32Array(n + 1);
    let nnz = 0;
    for (let i = 0; i < n; i++) {
      const v = values[i] ?? 0;
      if (v !== 0) {
        data[nnz] = v;
        indices[nnz] = i;
        nnz++;
      }
      indptr[i + 1] = nnz;
    }
    return new CSRMatrix({
      data: data.slice(0, nnz),
      indices: indices.slice(0, nnz),
      indptr,
      shape: [n, n],
    });
  }

  /**
   * Create a sparse matrix from COO (Coordinate List) format.
   *
   * Entries may be given in any order. The result is always canonical: columns are sorted within
   * each row and duplicate (row, col) entries are summed (SciPy semantics). Summed entries that
   * add up to exactly zero stay stored as explicit zeros.
   *
   * @param args - COO format specification
   * @param args.rows - Number of rows
   * @param args.cols - Number of columns
   * @param args.rowIndices - Row indices of non-zero values
   * @param args.colIndices - Column indices of non-zero values
   * @param args.values - Non-zero values
   * @param args.sort - Deprecated and ignored. Entries are always sorted so that the matrix is
   *   valid whatever the input order.
   * @returns New CSRMatrix
   * @throws {ShapeError} If `rows`/`cols` are not non-negative integers or the arrays differ in
   *   length
   * @throws {IndexError} If a row or column index is out of bounds
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
    /** @deprecated Ignored: entries are always sorted. */
    readonly sort?: boolean;
  }): CSRMatrix {
    const { rows, cols, rowIndices, colIndices, values } = args;
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
      throw new ShapeError(
        `CSRMatrix shape must be non-negative integers; received [${rows}, ${cols}]`
      );
    }
    if (rowIndices.length !== colIndices.length || rowIndices.length !== values.length) {
      throw new ShapeError("COO arrays must have the same length");
    }

    const nnz = values.length;

    // Counting sort by row (stable), validating bounds on the way.
    const indptr = new Int32Array(rows + 1);
    for (let i = 0; i < nnz; i++) {
      const r = rowIndices[i] ?? 0;
      if (r < 0 || r >= rows) {
        throw new IndexError(`row index out of bounds: ${r}`);
      }
      const c = colIndices[i] ?? 0;
      if (c < 0 || c >= cols) {
        throw new IndexError(`col index out of bounds: ${c}`);
      }
      indptr[r + 1] = (indptr[r + 1] ?? 0) + 1;
    }
    for (let r = 0; r < rows; r++) {
      indptr[r + 1] = (indptr[r + 1] ?? 0) + (indptr[r] ?? 0);
    }

    const indices = new Int32Array(nnz);
    const data = new Float64Array(nnz);
    const next = indptr.slice();
    for (let i = 0; i < nnz; i++) {
      const r = rowIndices[i] ?? 0;
      const pos = next[r] ?? 0;
      indices[pos] = colIndices[i] ?? 0;
      data[pos] = values[i] ?? 0;
      next[r] = pos + 1;
    }

    // Sort columns inside each row (only rows that are not already sorted), then sum
    // duplicates, which are adjacent after sorting.
    const dedupIndptr = new Int32Array(rows + 1);
    let outNnz = 0;
    for (let r = 0; r < rows; r++) {
      const rowStart = indptr[r] ?? 0;
      const rowEnd = indptr[r + 1] ?? 0;

      let sorted = true;
      for (let p = rowStart + 1; p < rowEnd; p++) {
        if ((indices[p] ?? 0) < (indices[p - 1] ?? 0)) {
          sorted = false;
          break;
        }
      }
      if (!sorted) {
        const len = rowEnd - rowStart;
        const perm = new Int32Array(len);
        for (let k = 0; k < len; k++) perm[k] = k;
        perm.sort((a, b) => (indices[rowStart + a] ?? 0) - (indices[rowStart + b] ?? 0));
        const sortedCols = new Int32Array(len);
        const sortedVals = new Float64Array(len);
        for (let k = 0; k < len; k++) {
          const src = rowStart + (perm[k] ?? 0);
          sortedCols[k] = indices[src] ?? 0;
          sortedVals[k] = data[src] ?? 0;
        }
        indices.set(sortedCols, rowStart);
        data.set(sortedVals, rowStart);
      }

      const outRowStart = outNnz;
      for (let p = rowStart; p < rowEnd; p++) {
        const c = indices[p] ?? 0;
        const v = data[p] ?? 0;
        if (outNnz > outRowStart && indices[outNnz - 1] === c) {
          data[outNnz - 1] = (data[outNnz - 1] ?? 0) + v;
        } else {
          indices[outNnz] = c;
          data[outNnz] = v;
          outNnz++;
        }
      }
      dedupIndptr[r + 1] = outNnz;
    }

    if (outNnz === nnz) {
      return new CSRMatrix({ data, indices, indptr, shape: [rows, cols] });
    }
    return new CSRMatrix({
      data: data.slice(0, outNnz),
      indices: indices.slice(0, outNnz),
      indptr: dedupIndptr,
      shape: [rows, cols],
    });
  }
}
