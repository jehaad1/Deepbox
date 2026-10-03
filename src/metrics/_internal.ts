/**
 * Shared internal helpers for the metrics module.
 *
 * These functions are used across classification, regression, and clustering metrics
 * to handle strided tensor access and input validation.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox documentation}
 */

import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import { Tensor } from "../ndarray";

/**
 * Converts a flat (logical) index to a physical buffer offset.
 *
 * @internal
 */
export type FlatOffsetter = (flatIndex: number) => number;

/**
 * Labels read into a dense, row-major array: a `Float64Array` for numeric
 * (and bool) tensors, a `string[]` for string tensors and a `bigint[]` for
 * int64 tensors.
 *
 * @internal
 */
export type DenseLabels = Float64Array | string[] | bigint[];

/**
 * Whether a view with the given shape and strides walks its buffer in plain
 * row-major order with no gaps (dimensions of size 1 may carry any stride).
 *
 * @internal
 */
export function isRowMajorContiguous(
  shape: readonly number[],
  strides: readonly number[]
): boolean {
  let expected = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    const dim = shape[d] ?? 0;
    if (dim === 1) continue;
    if ((strides[d] ?? 0) !== expected) return false;
    expected *= dim;
  }
  return true;
}

/**
 * Copy the logical elements of any host-backed tensor into a dense row-major
 * array, honouring strides and offset. `get(i)` is called once per element;
 * the odometer walk is shared by the numeric and the string/int64 readers.
 */
function gatherStrided<T>(t: Tensor, out: { [i: number]: T }, get: (physical: number) => T): void {
  const shape = t.shape;
  const ndim = shape.length;
  const strides = t.strides;
  const size = t.size;
  if (size === 0) return;

  if (ndim <= 1) {
    const s0 = strides[0] ?? 1;
    const offset = t.offset;
    for (let i = 0; i < size; i++) out[i] = get(offset + i * s0);
    return;
  }

  const inner = shape[ndim - 1] ?? 1;
  const innerStride = strides[ndim - 1] ?? 0;
  const outer = size / inner;
  const coords = new Array<number>(ndim - 1).fill(0);
  let base = t.offset;
  let pos = 0;
  for (let b = 0; b < outer; b++) {
    let idx = base;
    for (let j = 0; j < inner; j++) {
      out[pos++] = get(idx);
      idx += innerStride;
    }
    for (let d = ndim - 2; d >= 0; d--) {
      const next = (coords[d] ?? 0) + 1;
      base += strides[d] ?? 0;
      if (next < (shape[d] ?? 0)) {
        coords[d] = next;
        break;
      }
      base -= (strides[d] ?? 0) * (shape[d] ?? 0);
      coords[d] = 0;
    }
  }
}

/**
 * Dense-read a tensor's logical elements into a Float64Array when it is a
 * plain numeric typed array (not string/int64), else return null. No
 * finiteness check, so callers decide how to treat NaN/Infinity.
 *
 * @internal
 */
export function tryDenseNumeric(t: Tensor): Float64Array | null {
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;
  const size = t.size;
  const out = new Float64Array(size);
  if (size === 0) return out;

  if (isRowMajorContiguous(t.shape, t.strides)) {
    const offset = t.offset;
    out.set(offset === 0 && data.length === size ? data : data.subarray(offset, offset + size));
    return out;
  }
  gatherStrided(t, out, (physical) => data[physical] as number);
  return out;
}

/**
 * Read a numeric tensor's logical elements into a dense Float64Array in
 * row-major order, honouring strides/offset. Optionally validates that every
 * value is finite in the same pass. Metric hot loops then index `[i]`
 * directly instead of paying a per-element offsetter closure + bounds-checked
 * `getNumericElement` + finite-check call.
 *
 * Throws for string/int64 tensors (callers guard dtype upstream).
 *
 * @internal
 */
export function denseFloat64(t: Tensor, name: string, checkFinite = true): Float64Array {
  const out = tryDenseNumeric(t);
  if (out === null) {
    throw new DataValidationError(`${name} must be a numeric (non-int64) tensor`);
  }
  if (checkFinite) {
    for (let i = 0; i < out.length; i++) {
      const v = out[i] as number;
      if (!Number.isFinite(v)) assertFiniteNumber(v, name, `index ${i}`);
    }
  }
  return out;
}

/**
 * Read a label tensor into a dense array whose element type depends on the
 * tensor dtype (see {@link DenseLabels}). Numeric labels must be finite.
 * Works on strided views.
 *
 * @internal
 */
export function denseLabels(t: Tensor, name: string): DenseLabels {
  const data = t.data;
  if (Array.isArray(data)) {
    const out = new Array<string>(t.size);
    if (isRowMajorContiguous(t.shape, t.strides)) {
      const offset = t.offset;
      for (let i = 0; i < out.length; i++) out[i] = data[offset + i] as string;
    } else {
      gatherStrided(t, out, (physical) => data[physical] as string);
    }
    return out;
  }
  if (data instanceof BigInt64Array) {
    const out = new Array<bigint>(t.size);
    gatherStrided(t, out, (physical) => data[physical] as bigint);
    return out;
  }
  return denseFloat64(t, name, true);
}

/**
 * Compute row-major logical strides from a tensor shape.
 *
 * @internal
 */
export function computeLogicalStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i];
    if (dim === undefined) {
      throw new ShapeError("Tensor shape must be fully defined");
    }
    strides[i] = stride;
    stride *= dim;
  }
  return strides;
}

/**
 * Build a function that maps flat (logical) indices to physical buffer offsets.
 *
 * Handles arbitrary strides (views, slices, transposes).
 *
 * @internal
 */
export function createFlatOffsetter(t: Tensor): FlatOffsetter {
  const base = t.offset;

  if (t.ndim <= 1) {
    const stride0 = t.strides[0] ?? 1;
    return (flatIndex: number) => base + flatIndex * stride0;
  }

  const logicalStrides = computeLogicalStrides(t.shape);
  const strides = t.strides;

  return (flatIndex: number) => {
    let rem = flatIndex;
    let offset = base;

    for (let axis = 0; axis < logicalStrides.length; axis++) {
      const axisLogicalStride = logicalStrides[axis] ?? 1;
      const coord = Math.floor(rem / axisLogicalStride);
      rem -= coord * axisLogicalStride;
      offset += coord * (strides[axis] ?? 0);
    }

    return offset;
  };
}

/**
 * Assert that a numeric value is finite.
 *
 * @param value - The value to check
 * @param name - Name of the tensor for error messages
 * @param detail - Additional detail for the error message (e.g., "index 5")
 *
 * @internal
 */
export function assertFiniteNumber(value: number, name: string, detail: string): void {
  if (!Number.isFinite(value)) {
    throw new DataValidationError(
      `${name} must contain only finite numbers; found ${String(value)} at ${detail}`
    );
  }
}

/**
 * Assert that a tensor is 1D or a column vector.
 *
 * @internal
 */
export function assertVectorLike(t: Tensor, name: string): void {
  if (t.ndim <= 1) return;
  if (t.ndim === 2 && (t.shape[1] ?? 0) === 1) return;
  throw new ShapeError(`${name} must be 1D or a column vector; got shape [${t.shape.join(", ")}]`);
}

/**
 * Assert that two tensors have the same size and are vector-like.
 *
 * @internal
 */
export function assertSameSizeVectors(a: Tensor, b: Tensor, nameA: string, nameB: string): void {
  assertSameSize(a, b, nameA, nameB);
  assertVectorLike(a, nameA);
  assertVectorLike(b, nameB);
}

/**
 * Assert that two tensors have the same size (without vector-like check).
 *
 * @internal
 */
export function assertSameSize(a: Tensor, b: Tensor, nameA: string, nameB: string): void {
  if (a.size !== b.size) {
    throw new ShapeError(
      `${nameA} (size ${a.size}) and ${nameB} (size ${b.size}) must have same size`
    );
  }
}

/**
 * Read a numeric tensor into a dense Float64Array in row-major order.
 *
 * Handles strided views and int64 (converted to numbers), and rejects string
 * tensors and non-finite values. Shared by the metric implementations in this module.
 *
 * @internal
 */
export function readFiniteFloat64(t: Tensor, name: string): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} must be numeric (string tensors not supported)`);
  }

  let out: Float64Array;
  const data = t.data;
  if (data instanceof BigInt64Array) {
    out = new Float64Array(t.size);
    const offsetter = createFlatOffsetter(t);
    for (let i = 0; i < out.length; i++) out[i] = Number(data[offsetter(i)]);
  } else {
    const dense = tryDenseNumeric(t);
    if (dense === null) throw new DTypeError(`${name} must be a numeric tensor`);
    out = dense;
  }

  for (let i = 0; i < out.length; i++) {
    const v = out[i] as number;
    if (!Number.isFinite(v)) assertFiniteNumber(v, name, `index ${i}`);
  }
  return out;
}

/**
 * Neumaier (compensated) sum. Keeps the error independent of the number of terms,
 * which a plain running sum does not for long inputs. When the sum overflows, the
 * plain running sum (an infinity) is returned rather than the NaN the correction
 * term would produce.
 *
 * @internal
 */
export function compensatedSum(values: ArrayLike<number>): number {
  let sum = 0;
  let comp = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    const t = sum + v;
    comp += Math.abs(sum) >= Math.abs(v) ? sum - t + v : v - t + sum;
    sum = t;
  }
  const total = sum + comp;
  return Number.isNaN(total) && !Number.isNaN(sum) ? sum : total;
}

/**
 * Euclidean distance between two rows stored in flat row-major buffers. The result
 * stays finite and non-zero for very large or very small coordinates, where a plain
 * sum of squares would overflow or underflow.
 *
 * @internal
 */
export function euclideanDistance(
  a: Float64Array,
  ia: number,
  b: Float64Array,
  ib: number,
  d: number
): number {
  let sq = 0;
  for (let f = 0; f < d; f++) {
    const diff = (a[ia + f] as number) - (b[ib + f] as number);
    sq += diff * diff;
  }
  if (sq > 1e-290 && sq < 1e290) return Math.sqrt(sq);

  let scale = 0;
  for (let f = 0; f < d; f++) {
    const diff = Math.abs((a[ia + f] as number) - (b[ib + f] as number));
    if (diff > scale) scale = diff;
  }
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  let scaled = 0;
  for (let f = 0; f < d; f++) {
    const r = ((a[ia + f] as number) - (b[ib + f] as number)) / scale;
    scaled += r * r;
  }
  return scale * Math.sqrt(scaled);
}

/**
 * Per-sample weights accepted by the metrics: a numeric tensor (1D or a column
 * vector) or any array-like of numbers, with one value per sample.
 *
 * @example
 * ```ts
 * import { accuracy } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1]);
 * const yPred = tensor([0, 1, 0]);
 * accuracy(yTrue, yPred, { sampleWeight: [1, 1, 2] }); // as an array
 * accuracy(yTrue, yPred, { sampleWeight: tensor([1, 1, 2]) }); // as a tensor
 * ```
 */
export type SampleWeightInput = Tensor | ArrayLike<number>;

/**
 * Options shared by the metrics that accept per-sample weights.
 *
 * @example
 * ```ts
 * import { mse } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * mse(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] }); // 0.475
 * ```
 */
export type WeightedMetricOptions = {
  /**
   * One finite weight per sample. Each sample counts `sampleWeight[i]` times, as in
   * scikit-learn's `sample_weight`. Negative weights are allowed; a weight vector
   * that sums to zero makes an averaged metric undefined and throws.
   */
  readonly sampleWeight?: SampleWeightInput;
};

/**
 * Validate optional sample weights and read them into a private Float64Array.
 *
 * @internal
 */
export function readSampleWeight(
  weights: SampleWeightInput | undefined,
  n: number,
  name = "sampleWeight"
): Float64Array | undefined {
  if (weights === undefined) return undefined;
  let out: Float64Array;
  if (weights instanceof Tensor) {
    assertVectorLike(weights, name);
    out = readFiniteFloat64(weights, name);
  } else {
    out = new Float64Array(weights.length);
    for (let i = 0; i < out.length; i++) {
      const v = Number(weights[i]);
      if (!Number.isFinite(v)) assertFiniteNumber(v, name, `index ${i}`);
      out[i] = v;
    }
  }
  if (out.length !== n) {
    throw new ShapeError(`${name} must have one value per sample; got ${out.length} for ${n}`);
  }
  return out;
}

/**
 * Sum of the weights; throws when it is zero because a weighted mean is undefined.
 *
 * @internal
 */
export function nonZeroWeightSum(weights: Float64Array, name = "sampleWeight"): number {
  const total = compensatedSum(weights);
  if (total === 0) {
    throw new InvalidParameterError(
      `${name} must not sum to zero (the weighted average is undefined)`,
      name,
      total
    );
  }
  return total;
}
