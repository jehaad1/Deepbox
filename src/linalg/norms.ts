import {
  type Axis,
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  normalizeAxes,
  normalizeAxis,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import {
  assertFiniteTensor,
  fromDenseMatrix2D,
  getDim,
  getStride,
  toDenseVector1D,
} from "./_internal";
import { svdvals } from "./decomposition/svd";
import { inv } from "./inverse";

type NumericData = ArrayLike<number | bigint>;
type MatrixOrd = number | "fro" | "nuc";

/**
 * Sums of squares outside [LOW, MAX_VALUE] may have lost precision to
 * underflow or overflowed, so they are recomputed with scaling.
 */
const SUMSQ_LOW = 1e-280;

function elementsOf(x: Tensor): NumericData {
  return x.data as NumericData;
}

/** sqrt(sum v^2) of a strided run, scaled by the largest magnitude. */
function scaledNorm2(data: NumericData, base: number, stride: number, len: number): number {
  let max = 0;
  for (let k = 0; k < len; k++) {
    const a = Math.abs(Number(data[base + k * stride]));
    if (a > max) max = a;
  }
  if (max === 0) return 0;
  let sum = 0;
  for (let k = 0; k < len; k++) {
    const r = Number(data[base + k * stride]) / max;
    sum += r * r;
  }
  return max * Math.sqrt(sum);
}

/**
 * Lp "norm" of `len` elements starting at `base` with step `stride`.
 * Supports 0, 1, 2, +-Infinity and any other finite order. The 2-norm and the
 * general p-norm fall back to a scaled evaluation when the plain sum
 * overflows or underflows.
 */
function vectorNormStrided(
  data: NumericData,
  base: number,
  stride: number,
  len: number,
  order: number
): number {
  if (len === 0) return 0;

  if (order === 0) {
    let count = 0;
    for (let k = 0; k < len; k++) if (Number(data[base + k * stride]) !== 0) count++;
    return count;
  }
  if (order === 1) {
    let sum = 0;
    for (let k = 0; k < len; k++) sum += Math.abs(Number(data[base + k * stride]));
    return sum;
  }
  if (order === 2) {
    let sum = 0;
    for (let k = 0; k < len; k++) {
      const v = Number(data[base + k * stride]);
      sum += v * v;
    }
    if (sum >= SUMSQ_LOW && sum <= Number.MAX_VALUE) return Math.sqrt(sum);
    return scaledNorm2(data, base, stride, len);
  }
  if (order === Number.POSITIVE_INFINITY) {
    let max = 0;
    for (let k = 0; k < len; k++) max = Math.max(max, Math.abs(Number(data[base + k * stride])));
    return max;
  }
  if (order === Number.NEGATIVE_INFINITY) {
    let min = Number.POSITIVE_INFINITY;
    for (let k = 0; k < len; k++) min = Math.min(min, Math.abs(Number(data[base + k * stride])));
    return min;
  }

  let sum = 0;
  for (let k = 0; k < len; k++) sum += Math.abs(Number(data[base + k * stride])) ** order;
  if (sum >= SUMSQ_LOW && sum <= Number.MAX_VALUE) return sum ** (1 / order);

  // Scale by the dominant element (largest for p > 0, smallest for p < 0) so
  // that every term |v / ref|^p lies in [0, 1].
  let ref = order > 0 ? 0 : Number.POSITIVE_INFINITY;
  for (let k = 0; k < len; k++) {
    const a = Math.abs(Number(data[base + k * stride]));
    ref = order > 0 ? Math.max(ref, a) : Math.min(ref, a);
  }
  if (ref === 0) return 0;
  let scaled = 0;
  for (let k = 0; k < len; k++) {
    scaled += (Math.abs(Number(data[base + k * stride])) / ref) ** order;
  }
  return ref * scaled ** (1 / order);
}

function vectorOrderOf(order: number | "fro"): number {
  const value = order === "fro" ? 2 : order;
  if (Number.isNaN(value)) {
    throw new InvalidParameterError("ord must be a valid number", "ord", value);
  }
  // Finite negative orders are valid vector "norms" (NumPy accepts ord=-1, -2, ...);
  // the generic (sum |v|^p)^(1/p) branch handles them.
  return value;
}

function assertMatrixOrder(ord: MatrixOrd, allowNuc: boolean): void {
  if (
    ord === "fro" ||
    (allowNuc && ord === "nuc") ||
    ord === 1 ||
    ord === -1 ||
    ord === 2 ||
    ord === -2 ||
    ord === Number.POSITIVE_INFINITY ||
    ord === Number.NEGATIVE_INFINITY
  ) {
    return;
  }
  throw new InvalidParameterError(
    `Invalid norm order '${String(ord)}' for matrix norm. ` +
      `Valid orders are: 1, -1, 2, -2, Infinity, -Infinity, 'fro'${allowNuc ? ", 'nuc'" : ""}.`,
    "ord",
    ord
  );
}

function singularValuesOf(
  data: NumericData,
  base: number,
  rows: number,
  cols: number,
  sRow: number,
  sCol: number
): Float64Array {
  const dense = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      dense[i * cols + j] = Number(data[base + i * sRow + j * sCol]);
    }
  }
  return toDenseVector1D(svdvals(fromDenseMatrix2D(rows, cols, dense)));
}

/**
 * Matrix norm of the (rows x cols) slice whose element (i, j) is at
 * `base + i * sRow + j * sCol`. `ord` must already be validated.
 */
function matrixNormSlice(
  data: NumericData,
  base: number,
  rows: number,
  cols: number,
  sRow: number,
  sCol: number,
  ord: MatrixOrd
): number {
  if (rows === 0 || cols === 0) return 0;

  if (ord === "fro") {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = Number(data[base + i * sRow + j * sCol]);
        sum += v * v;
      }
    }
    if (sum >= SUMSQ_LOW && sum <= Number.MAX_VALUE) return Math.sqrt(sum);
    // Scaled recomputation (underflow or overflow of the plain sum).
    let max = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        max = Math.max(max, Math.abs(Number(data[base + i * sRow + j * sCol])));
      }
    }
    if (max === 0) return 0;
    let scaled = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const r = Number(data[base + i * sRow + j * sCol]) / max;
        scaled += r * r;
      }
    }
    return max * Math.sqrt(scaled);
  }

  if (ord === 1 || ord === -1) {
    // Max / min absolute column sum.
    let best = ord === 1 ? 0 : Number.POSITIVE_INFINITY;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        colSum += Math.abs(Number(data[base + i * sRow + j * sCol]));
      }
      best = ord === 1 ? Math.max(best, colSum) : Math.min(best, colSum);
    }
    return best;
  }

  if (ord === Number.POSITIVE_INFINITY || ord === Number.NEGATIVE_INFINITY) {
    // Max / min absolute row sum.
    let best = ord === Number.POSITIVE_INFINITY ? 0 : Number.POSITIVE_INFINITY;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        rowSum += Math.abs(Number(data[base + i * sRow + j * sCol]));
      }
      best = ord === Number.POSITIVE_INFINITY ? Math.max(best, rowSum) : Math.min(best, rowSum);
    }
    return best;
  }

  // 2, -2 and nuc go through the singular values (sorted in descending order).
  const s = singularValuesOf(data, base, rows, cols, sRow, sCol);
  if (s.length === 0) return 0;
  if (ord === 2) return s[0] as number;
  if (ord === -2) return Math.abs(s[s.length - 1] as number);
  let sum = 0;
  for (let i = 0; i < s.length; i++) sum += Math.abs(s[i] as number);
  return sum;
}

/** Visit every index of `shape` in row-major order, passing the matching strided offset. */
function forEachOuter(
  shape: readonly number[],
  strides: readonly number[],
  baseOffset: number,
  visit: (offset: number, flat: number) => void
): void {
  let count = 1;
  for (const d of shape) count *= d;
  const idx = new Array<number>(shape.length).fill(0);
  let offset = baseOffset;
  for (let flat = 0; flat < count; flat++) {
    visit(offset, flat);
    for (let d = shape.length - 1; d >= 0; d--) {
      const stride = strides[d] as number;
      const next = (idx[d] as number) + 1;
      idx[d] = next;
      offset += stride;
      if (next < (shape[d] as number)) break;
      offset -= next * stride;
      idx[d] = 0;
    }
  }
}

function onesShapeResult(value: number, ndim: number): Tensor | number {
  if (ndim === 0) return value;
  return tensor(new Float64Array([value])).view(new Array<number>(ndim).fill(1));
}

/**
 * Matrix or vector norm.
 *
 * Computes various matrix and vector norms. Sums of squares are rescaled when
 * the plain sum would overflow or underflow, so `norm([3e200, 4e200])` is
 * `5e200` rather than `Infinity`.
 *
 * **Parameters**:
 * @param x - Input array (real numeric dtype)
 * @param ord - Order of the norm:
 *   For vectors:
 *   - undefined or 2: L2 norm (Euclidean)
 *   - 1: L1 norm (Manhattan)
 *   - Infinity: Max norm
 *   - -Infinity: Min norm
 *   - 0: L0 "norm" (number of non-zero elements)
 *   - p: Lp norm for any other finite p, including negative p
 *     (`(sum |v|^p)^(1/p)`, as in NumPy)
 *   For matrices:
 *   - 'fro': Frobenius norm
 *   - 'nuc': Nuclear norm (sum of singular values)
 *   - 1: Max column sum
 *   - -1: Min column sum
 *   - 2: Largest singular value
 *   - -2: Smallest singular value
 *   - Infinity: Max row sum
 *   - -Infinity: Min row sum
 *
 *   With `ord` omitted and no `axis`, a 1-D input gives the 2-norm, a 2-D input the
 *   Frobenius norm, and an input with more than two dimensions the 2-norm of
 *   all its elements (like `numpy.linalg.norm`).
 * @param axis - Axis (vector norm) or two axes (matrix norm) along which to compute the norm
 * @param keepdims - Keep reduced dimensions with size 1
 *
 * **Returns**: Norm value (a number when every axis is reduced and `keepdims` is false,
 * otherwise a float64 tensor)
 *
 * @example
 * ```ts
 * import { norm } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * norm(tensor([3, 4]));                          // 5
 * norm(tensor([[1, 2], [3, 4]]), 'fro');         // 5.477...
 * norm(tensor([[1, 2], [3, 4]]), 1);             // 6 (max column sum)
 * norm(tensor([[1, 2], [3, 4]]), 2, 1);          // tensor([2.236, 5]) (row-wise 2-norms)
 * ```
 *
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If norm order or axis values are invalid
 * @throws {ShapeError} If axis configuration is incompatible with input
 *
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */
export function norm(x: Tensor, ord?: number | "fro" | "nuc"): number;
export function norm(
  x: Tensor,
  ord: number | "fro" | "nuc" | undefined,
  axis: Axis | Axis[],
  keepdims?: boolean
): Tensor | number;
export function norm(
  x: Tensor,
  ord?: number | "fro" | "nuc",
  axis?: Axis | Axis[],
  keepdims?: boolean
): Tensor | number;
export function norm(
  x: Tensor,
  ord?: number | "fro" | "nuc",
  axis?: Axis | Axis[],
  keepdims = false
): Tensor | number {
  if (x.dtype === "complex64" || x.dtype === "complex128") {
    throw new DTypeError("norm() does not support complex dtype");
  }
  assertFiniteTensor(x, "norm()");

  const data = elementsOf(x);
  const ndim = x.ndim;
  const axes = axis !== undefined ? normalizeAxes(axis, ndim) : undefined;

  /** Copy every element of x (any layout) into a dense row-major array. */
  const collectValues = (): Float64Array => {
    const values = new Float64Array(x.size);
    if (x.size === 0) return values;

    if (ndim === 0) {
      values[0] = Number(data[x.offset]);
      return values;
    }

    // Fast path: contiguous zero-offset numeric tensor
    if (
      !Array.isArray(x.data) &&
      !(x.data instanceof BigInt64Array) &&
      x.offset === 0 &&
      isContiguous(x.shape, x.strides)
    ) {
      for (let i = 0; i < x.size; i++) values[i] = Number(data[i]);
      return values;
    }

    let count = 0;
    forEachOuter(x.shape, x.strides, x.offset, (offset) => {
      values[count++] = Number(data[offset]);
    });
    return values;
  };

  const flatVectorNorm = (order: number): number => {
    const values = collectValues();
    return vectorNormStrided(values, 0, 1, values.length, order);
  };

  const vectorNormAxis = (axisValue: number, order: number, keep: boolean): Tensor | number => {
    const ax = normalizeAxis(axisValue, ndim);
    const axisDim = getDim(x, ax, "norm()");
    const axisStride = getStride(x, ax, "norm()");

    const outerShape: number[] = [];
    const outerStrides: number[] = [];
    const outShape: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i === ax) {
        if (keep) outShape.push(1);
      } else {
        const dim = getDim(x, i, "norm()");
        outerShape.push(dim);
        outerStrides.push(getStride(x, i, "norm()"));
        outShape.push(dim);
      }
    }

    let outSize = 1;
    for (const d of outerShape) outSize *= d;
    const out = new Float64Array(outSize);
    forEachOuter(outerShape, outerStrides, x.offset, (offset, flat) => {
      out[flat] = vectorNormStrided(data, offset, axisStride, axisDim, order);
    });

    if (outShape.length === 0) return out[0] as number;
    return tensor(out).view(outShape);
  };

  // ---- Explicit axis -------------------------------------------------------
  if (axes !== undefined) {
    if (axes.length === 0) {
      // Every element is reduced.
      if (ord === "nuc") {
        throw new InvalidParameterError(
          "axis is only supported for vector norms, not nuclear norm, unless two axes are given",
          "axis",
          axis
        );
      }
      const value = flatVectorNorm(vectorOrderOf(ord === undefined ? 2 : ord));
      return keepdims ? onesShapeResult(value, ndim) : value;
    }

    if (axes.length === 1) {
      if (ord === "nuc") {
        throw new InvalidParameterError(
          "axis is only supported for vector norms, not nuclear norm, unless two axes are given",
          "axis",
          axis
        );
      }
      const ax = axes[0] as number;
      return vectorNormAxis(ax, vectorOrderOf(ord === undefined ? 2 : ord), keepdims);
    }

    if (axes.length === 2) {
      const ax0 = axes[0] as number;
      const ax1 = axes[1] as number;
      const matOrd: MatrixOrd = ord === undefined ? "fro" : ord;
      assertMatrixOrder(matOrd, true);

      const dimRow = getDim(x, ax0, "norm()");
      const dimCol = getDim(x, ax1, "norm()");
      const strideRow = getStride(x, ax0, "norm()");
      const strideCol = getStride(x, ax1, "norm()");

      const outerShape: number[] = [];
      const outerStrides: number[] = [];
      const kdShape: number[] = [];
      for (let i = 0; i < ndim; i++) {
        if (i === ax0 || i === ax1) {
          kdShape.push(1);
        } else {
          const dim = getDim(x, i, "norm()");
          outerShape.push(dim);
          outerStrides.push(getStride(x, i, "norm()"));
          kdShape.push(dim);
        }
      }

      let outerSize = 1;
      for (const d of outerShape) outerSize *= d;
      const results = new Float64Array(outerSize);
      forEachOuter(outerShape, outerStrides, x.offset, (offset, flat) => {
        results[flat] = matrixNormSlice(data, offset, dimRow, dimCol, strideRow, strideCol, matOrd);
      });

      if (keepdims) return tensor(results).view(kdShape);
      if (outerShape.length === 0) return results[0] as number;
      return tensor(results).view(outerShape);
    }

    throw new ShapeError("axis has invalid length for input");
  }

  // ---- No axis: reduce over every element ---------------------------------
  if (ndim > 2) {
    if (ord !== undefined) {
      throw new ShapeError("norm requires 1D or 2D input when axis is omitted and ord is given");
    }
    const value = flatVectorNorm(2);
    return keepdims ? onesShapeResult(value, ndim) : value;
  }

  if (ndim < 2) {
    if (ord === "nuc") {
      throw new InvalidParameterError("Invalid norm order 'nuc' for vector norm.", "ord", ord);
    }
    const value = flatVectorNorm(vectorOrderOf(ord === undefined ? 2 : ord));
    return keepdims ? onesShapeResult(value, ndim) : value;
  }

  // 2-D matrix
  const matOrd: MatrixOrd = ord === undefined ? "fro" : ord;
  assertMatrixOrder(matOrd, true);
  const s0 = getStride(x, 0, "norm()");
  const s1 = getStride(x, 1, "norm()");
  const rows = getDim(x, 0, "norm()");
  const cols = getDim(x, 1, "norm()");

  let value: number;
  if (
    matOrd === "fro" &&
    !Array.isArray(x.data) &&
    !(x.data instanceof BigInt64Array) &&
    x.offset === 0 &&
    s1 === 1 &&
    s0 === cols
  ) {
    // Fast path: contiguous numeric data reads monomorphically over a
    // zero-based view (the generic `Number(x.data[...])` path pays a
    // megamorphic union access + Number() coercion per element, ~10x).
    const flat = x.data as Float32Array | Float64Array | Int32Array | Uint8Array;
    const n = rows * cols;
    let acc0 = 0;
    let acc1 = 0;
    let i = 0;
    for (; i + 2 <= n; i += 2) {
      const a = flat[i] as number;
      const b = flat[i + 1] as number;
      acc0 += a * a;
      acc1 += b * b;
    }
    if (i < n) {
      const a = flat[i] as number;
      acc0 += a * a;
    }
    const sum = acc0 + acc1;
    value =
      sum >= SUMSQ_LOW && sum <= Number.MAX_VALUE
        ? Math.sqrt(sum)
        : matrixNormSlice(data, x.offset, rows, cols, s0, s1, "fro");
  } else {
    value = matrixNormSlice(data, x.offset, rows, cols, s0, s1, matOrd);
  }
  return keepdims ? onesShapeResult(value, ndim) : value;
}

/**
 * Condition number of a matrix.
 *
 * Measures how sensitive the solution of A*x=b is to changes in b.
 * Large condition number indicates ill-conditioned matrix.
 *
 * **Formula**: cond(A) = ||A|| * ||A^(-1)||
 *
 * **Parameters**:
 * @param a - Input matrix of shape (M, N)
 * @param p - Norm order, as in `numpy.linalg.cond`:
 *   - undefined or 2: ratio of the largest to the smallest singular value (any M, N)
 *   - -2: ratio of the smallest to the largest singular value (any M, N)
 *   - 'fro' or 'nuc': Frobenius or nuclear norm of A times that of its (pseudo-)inverse,
 *     computed from the singular values (any M, N)
 *   - 1, -1, Infinity, -Infinity: matrix norm of A times that of A^(-1) (square A only)
 *
 * **Returns**: Condition number (>= 1 for p != -2, Infinity for singular matrices).
 * Singular values at or below `max(M, N) * eps * largest singular value` count as zero,
 * and an empty matrix gives Infinity.
 *
 * @example
 * ```ts
 * import { cond } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * cond(tensor([[1, 2], [3, 4]]));      // 14.93... (2-norm)
 * cond(tensor([[1, 2], [3, 4]]), 1);   // 21
 * ```
 *
 * @throws {ShapeError} If input is not a 2D matrix, or is not square for p in
 *   {1, -1, Infinity, -Infinity}
 * @throws {DTypeError} If input has string or complex dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If p is unsupported
 */
export function cond(a: Tensor, p?: number | "fro" | "nuc"): number {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  if (a.dtype === "complex64" || a.dtype === "complex128") {
    throw new DTypeError("cond() does not support complex dtype");
  }
  assertFiniteTensor(a, "cond()");

  if (p !== undefined) {
    try {
      assertMatrixOrder(p, true);
    } catch {
      throw new InvalidParameterError(
        `Unsupported norm order '${String(p)}' for condition number. ` +
          "Valid orders are: 2, -2, 1, -1, Infinity, -Infinity, 'fro', 'nuc'.",
        "p",
        p
      );
    }
  }

  const m = getDim(a, 0, "cond()");
  const n = getDim(a, 1, "cond()");
  const k = Math.min(m, n);
  if (k === 0) return Infinity;

  if (p === 1 || p === -1 || p === Number.POSITIVE_INFINITY || p === Number.NEGATIVE_INFINITY) {
    if (m !== n) {
      throw new ShapeError(`cond() with p=${String(p)} requires a square matrix; got (${m}, ${n})`);
    }
    let aInv: Tensor;
    try {
      aInv = inv(a);
    } catch (err) {
      if (err instanceof DataValidationError && /singular/i.test(err.message)) {
        return Infinity;
      }
      throw err;
    }
    const value = norm(a, p) * norm(aInv, p);
    return Number.isNaN(value) ? Infinity : value;
  }

  // Only singular values are needed, so skip U/V accumulation.
  const sDense = toDenseVector1D(svdvals(a));
  if (sDense.length === 0) return Infinity;

  // A rank-deficient matrix has a numerically-zero smallest singular value.
  // Different SVD backends bottom it out at slightly different tiny values,
  // so treat any singular value below a relative tolerance as exactly zero
  // (→ infinite condition number) rather than testing `=== 0`.
  const sMax = sDense[0] as number;
  const sMin = sDense[k - 1] as number;
  const singularTol = sMax * Number.EPSILON * Math.max(m, n);

  if (p === "fro" || p === "nuc") {
    let sum = 0;
    let sumInv = 0;
    for (let i = 0; i < sDense.length; i++) {
      const si = sDense[i] as number;
      if (si <= singularTol) return Infinity;
      if (p === "fro") {
        // Scale by sMax so squares of tiny or huge singular values do not underflow.
        const r = si / sMax;
        sum += r * r;
        sumInv += (sMax / si) ** 2;
      } else {
        sum += si;
        sumInv += 1 / si;
      }
    }
    return p === "fro" ? Math.sqrt(sum) * Math.sqrt(sumInv) : sum * sumInv;
  }

  if (sMin <= singularTol) return p === -2 ? 0 : Infinity;
  return p === -2 ? sMin / sMax : sMax / sMin;
}
