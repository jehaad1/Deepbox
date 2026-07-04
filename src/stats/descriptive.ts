import { InvalidParameterError } from "../core";
import { Tensor, tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import {
  type AxisLike,
  computeStrides,
  copyContiguousToF64,
  forEachIndexOffset,
  getNumberAt,
  normalizeAxes,
  quickMedianF64,
  quickSelectF64,
  reducedShape,
  reduceMean,
  reduceVariance,
} from "./_internal";

/**
 * Narrows the second positional argument of a reduction to an options object.
 *
 * An {@link AxisLike} is always a `number`, `string` alias, or an array of
 * those, so any non-null, non-array object at the axis position is an
 * options bag (the recommended, self-documenting call form) rather than an
 * axis specifier. Used to route the options-object overloads below without
 * disturbing the historical positional signatures.
 */
function isReductionOptions(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

/**
 * Options bag for {@link mean}. Recommended over trailing positional flags,
 * which are easy to confuse with the differently-typed flags on other
 * reductions (e.g. `skewness(t, axis, bias)`).
 */
export interface MeanOptions {
  /** Axis or axes along which to reduce (undefined = all axes). */
  axis?: AxisLike;
  /** If true, reduced axes are retained with size 1 (default: false). */
  keepdims?: boolean;
}

/**
 * Options bag for {@link variance} and {@link std}. Recommended over trailing
 * positional flags: the `keepdims`/`ddof` pair is easy to transpose.
 */
export interface VarianceOptions {
  /** Axis or axes along which to reduce (undefined = all axes). */
  axis?: AxisLike;
  /** If true, reduced axes are retained with size 1 (default: false). */
  keepdims?: boolean;
  /** Delta degrees of freedom (0 = population, 1 = sample, default: 0). */
  ddof?: number;
}

/**
 * Options bag for {@link skewness}. Recommended so the sample-correction flag
 * is named rather than a bare trailing boolean.
 */
export interface SkewnessOptions {
  /** Axis or axes along which to reduce (undefined = all axes). */
  axis?: AxisLike;
  /** If false, applies the unbiased Fisher-Pearson correction (default: true). */
  bias?: boolean;
}

/**
 * Options bag for {@link kurtosis}. Recommended so the two independent flags
 * (`fisher`, `bias`) are named rather than positional booleans.
 */
export interface KurtosisOptions {
  /** Axis or axes along which to reduce (undefined = all axes). */
  axis?: AxisLike;
  /** If true, returns excess kurtosis (subtract 3, default: true). */
  fisher?: boolean;
  /** If false, applies the bias correction (requires at least 4 samples, default: true). */
  bias?: boolean;
}

/**
 * Computes the arithmetic mean along specified axes.
 *
 * The mean is the sum of all values divided by the count.
 * Supports axis-wise reduction with optional dimension preservation.
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. The options object avoids the trailing-boolean footgun where the same
 * argument position means different things across the stats reductions.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the mean (undefined = all axes)
 * @param keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @returns Tensor containing mean values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2, 3], [4, 5, 6]]);
 * mean(t);                         // Returns tensor([3.5]) - mean of all elements
 * mean(t, 0);                      // Returns tensor([2.5, 3.5, 4.5]) - column means
 * mean(t, 1);                      // Returns tensor([2, 5]) - row means
 * mean(t, 1, true);                // Returns tensor([[2], [5]]) - keepdims (positional)
 * mean(t, { axis: 1, keepdims: true }); // Recommended options-object form
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Infinity is handled according to standard arithmetic rules
 * - Mixed Infinity values (Infinity + -Infinity) result in NaN
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function mean(t: Tensor, options: MeanOptions): Tensor;
export function mean(t: Tensor, axis?: AxisLike, keepdims?: boolean): Tensor;
export function mean(t: Tensor, axisOrOptions?: AxisLike | MeanOptions, keepdims = false): Tensor {
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as MeanOptions;
    return reduceMean(t, o.axis, o.keepdims ?? false);
  }
  return reduceMean(t, axisOrOptions as AxisLike | undefined, keepdims);
}

/**
 * Computes the median (50th percentile) along specified axes.
 *
 * The median is the middle value when data is sorted. For even-sized arrays,
 * it's the average of the two middle values. More robust to outliers than mean.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the median (undefined = all axes)
 * @param keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @returns Tensor containing median values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * median(t);  // Returns tensor([3])
 *
 * const t2 = tensor([1, 2, 3, 4]);
 * median(t2); // Returns tensor([2.5]) - average of 2 and 3
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs result in NaN output (NaN sorts to end)
 * - Infinity values are sorted naturally (±Infinity at extremes)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function median(t: Tensor, axis?: AxisLike, keepdims = false): Tensor {
  const axes = normalizeAxes(axis, t.ndim);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("median() requires at least one element", "size", t.size);
    }
    let values: Float64Array;
    const raw = t.data;
    if (
      !Array.isArray(raw) &&
      !(raw instanceof BigInt64Array) &&
      isContiguous(t.shape, t.strides)
    ) {
      // Contiguous numeric fast path: monomorphic bulk copy.
      values = copyContiguousToF64(raw, t.offset, t.size);
    } else {
      values = new Float64Array(t.size);
      let w = 0;
      forEachIndexOffset(t, (off) => {
        values[w++] = getNumberAt(t, off);
      });
    }
    const outShape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
    // O(n) quickselect instead of an O(n log n) sort; quickMedianF64 propagates
    // NaN (NumPy semantics) and mutates `values`, which is our private copy.
    const result = quickMedianF64(values);
    const out = new Float64Array(1);
    out[0] = result;
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, keepdims);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const buckets: (number[] | undefined)[] = new Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);

  const reduce = new Set(axes);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    if (keepdims) {
      for (let i = 0; i < t.ndim; i++) {
        const s = outStrides[i] ?? 0;
        const v = reduce.has(i) ? 0 : (idx[i] ?? 0);
        outFlat += v * s;
      }
    } else {
      let oi = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (reduce.has(i)) continue;
        outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
        oi++;
      }
    }

    const val = getNumberAt(t, off);
    if (Number.isNaN(val)) {
      nanFlags[outFlat] = true;
      return;
    }
    const arr = buckets[outFlat] ?? [];
    arr.push(val);
    buckets[outFlat] = arr;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    if (nanFlags[i]) {
      out[i] = Number.NaN;
      continue;
    }
    const arr = buckets[i] ?? [];
    if (arr.length === 0) {
      throw new InvalidParameterError(
        "median() reduction over empty axis is undefined",
        "axis",
        arr.length
      );
    }
    arr.sort((a, b) => a - b);
    const mid = Math.floor(arr.length / 2);
    out[i] = arr.length % 2 === 0 ? ((arr[mid - 1] ?? 0) + (arr[mid] ?? 0)) / 2 : (arr[mid] ?? 0);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the mode (most frequent value) along specified axis.
 *
 * The mode is the value that appears most frequently in the dataset.
 * If multiple values have the same maximum frequency, returns the smallest value.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the mode (undefined = all axes)
 * @returns Tensor containing mode values
 * @throws {InvalidParameterError} If tensor is empty
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 2, 3, 3, 3]);
 * mode(t);  // Returns tensor([3]) - most frequent value
 *
 * const t2 = tensor([[1, 2, 2], [3, 3, 4]]);
 * mode(t2, 1);  // Returns tensor([2, 3]) - mode of each row
 * ```
 *
 * @remarks
 * NaN inputs propagate to NaN output.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function mode(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("mode() requires at least one element", "size", t.size);
    }
    const freq = new Map<number, number>();
    let maxFreq = 0;
    let modeVal = Number.POSITIVE_INFINITY;
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const val = getNumberAt(t, off);
      if (Number.isNaN(val)) {
        hasNaN = true;
        return;
      }
      const count = (freq.get(val) ?? 0) + 1;
      freq.set(val, count);
      if (count > maxFreq || (count === maxFreq && val < modeVal)) {
        maxFreq = count;
        modeVal = val;
      }
    });
    if (hasNaN) {
      return tensor([Number.NaN]);
    }
    return tensor([modeVal]);
  }

  // Axis-wise mode implemented via per-output frequency maps.
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const maps: Array<Map<number, number> | undefined> = new Array(outSize);
  const bestCounts = new Int32Array(outSize);
  const bestValues = new Float64Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);
  const reduce = new Set(axes);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }

    const val = getNumberAt(t, off);
    if (Number.isNaN(val)) {
      nanFlags[outFlat] = true;
      return;
    }
    const m = maps[outFlat] ?? new Map<number, number>();
    const next = (m.get(val) ?? 0) + 1;
    m.set(val, next);
    maps[outFlat] = m;
    const currentBestCount = bestCounts[outFlat] ?? 0;
    const currentBestValue = bestValues[outFlat] ?? Number.POSITIVE_INFINITY;
    if (next > currentBestCount || (next === currentBestCount && val < currentBestValue)) {
      bestCounts[outFlat] = next;
      bestValues[outFlat] = val;
    }
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    if (nanFlags[i]) {
      out[i] = Number.NaN;
      continue;
    }
    if ((bestCounts[i] ?? 0) === 0) {
      throw new InvalidParameterError(
        "mode() reduction over empty axis is undefined",
        "axis",
        bestCounts[i] ?? 0
      );
    }
    out[i] = bestValues[i] ?? Number.NaN;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the standard deviation along specified axes.
 *
 * Standard deviation is the square root of variance, measuring spread of data.
 * Uses Welford's algorithm for numerical stability via the variance function.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute std (undefined = all axes)
 * @param keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @param ddof - Delta degrees of freedom (0 = population, 1 = sample, default: 0)
 * @returns Tensor containing standard deviation values
 * @throws {InvalidParameterError} If tensor is empty, ddof < 0, ddof >= sample size, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `std(t, { ddof: 1 })` over `std(t, 0, false, 1)`, whose trailing
 * `keepdims`/`ddof` booleans/numbers are easy to transpose.
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * std(t);                     // Population std (ddof=0)
 * std(t, 0, false, 1);        // Sample std (ddof=1, positional)
 * std(t, { ddof: 1 });        // Recommended options-object form
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Infinity inputs result in NaN (infinite standard deviation)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function std(t: Tensor, options: VarianceOptions): Tensor;
export function std(t: Tensor, axis?: AxisLike, keepdims?: boolean, ddof?: number): Tensor;
export function std(
  t: Tensor,
  axisOrOptions?: AxisLike | VarianceOptions,
  keepdims = false,
  ddof = 0
): Tensor {
  let axis: AxisLike | undefined;
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as VarianceOptions;
    axis = o.axis;
    keepdims = o.keepdims ?? false;
    ddof = o.ddof ?? 0;
  } else {
    axis = axisOrOptions as AxisLike | undefined;
  }
  const v = reduceVariance(t, axis, keepdims, ddof);
  const out = new Float64Array(v.size);
  for (let i = 0; i < v.size; i++) {
    out[i] = Math.sqrt(getNumberAt(v, v.offset + i));
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: v.shape,
    dtype: "float64",
    device: v.device,
  });
}

/**
 * Computes the variance along specified axes.
 *
 * Variance measures the average squared deviation from the mean.
 * Uses Welford's online algorithm for numerical stability.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute variance (undefined = all axes)
 * @param keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @param ddof - Delta degrees of freedom (0 = population, 1 = sample, default: 0)
 * @returns Tensor containing variance values
 * @throws {InvalidParameterError} If tensor is empty, ddof < 0, ddof >= sample size, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `variance(t, { ddof: 1 })` over `variance(t, 0, false, 1)`.
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * variance(t);                  // Population variance: 2.0
 * variance(t, 0, false, 1);     // Sample variance: 2.5 (positional)
 * variance(t, { ddof: 1 });     // Recommended options-object form
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Infinity inputs result in NaN (infinite variance)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function variance(t: Tensor, options: VarianceOptions): Tensor;
export function variance(t: Tensor, axis?: AxisLike, keepdims?: boolean, ddof?: number): Tensor;
export function variance(
  t: Tensor,
  axisOrOptions?: AxisLike | VarianceOptions,
  keepdims = false,
  ddof = 0
): Tensor {
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as VarianceOptions;
    return reduceVariance(t, o.axis, o.keepdims ?? false, o.ddof ?? 0);
  }
  return reduceVariance(t, axisOrOptions as AxisLike | undefined, keepdims, ddof);
}

/**
 * Central moments (m2, m3, m4 about the mean) of a whole numeric tensor in two
 * passes — gather+mean, then one narrowed loop accumulating the squared, cubed
 * and quartic deviations together. Used by the full-reduction fast paths of
 * `skewness`/`kurtosis`, which otherwise re-fetch the scalar mean/variance and
 * recompute a sqrt per element (several extra passes). Returns null for
 * non-numeric or empty input so callers fall back to the generic path.
 */
function fullTensorCentralMoments(
  t: Tensor
): { n: number; m2: number; m3: number; m4: number } | null {
  if (t.dtype === "string") return null;
  const n = t.size;
  if (n === 0) return null;
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;

  // Materialize a contiguous Float64Array once (a narrowed memcpy for contiguous
  // float64 input) so both accumulation passes run as monomorphic typed-array
  // loops — no `forEachIndexOffset` closure or `getNumberAt` indirection, which
  // otherwise dominate at a few thousand elements.
  const src = isContiguous(t.shape, t.strides) ? copyContiguousToF64(data, t.offset, n) : null;
  if (src === null) return null;

  let sum = 0;
  for (let i = 0; i < n; i++) sum += src[i] as number;
  const m = sum / n;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  for (let i = 0; i < n; i++) {
    const d = (src[i] as number) - m;
    const d2 = d * d;
    s2 += d2;
    s3 += d2 * d;
    s4 += d2 * d2;
  }
  return { n, m2: s2 / n, m3: s3 / n, m4: s4 / n };
}

/**
 * Computes the skewness (third standardized moment) along specified axis.
 *
 * Skewness measures the asymmetry of the probability distribution.
 * - Negative skew: left tail is longer (mean < median)
 * - Zero skew: symmetric distribution (normal distribution)
 * - Positive skew: right tail is longer (mean > median)
 *
 * Uses Fisher's moment coefficient: E[(X - μ)³] / σ³
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `skewness(t, { bias: false })`: the bare trailing boolean is
 * visually identical to `keepdims` on `mean`/`std` but means something else.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute skewness (undefined = all axes)
 * @param bias - If false, applies the unbiased Fisher-Pearson correction (default: true)
 * @returns Tensor containing skewness values
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * skewness(t);                    // Returns ~0 (symmetric)
 * skewness(t, undefined, false);  // Unbiased correction (positional)
 * skewness(t, { bias: false });   // Recommended options-object form
 *
 * const t2 = tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]);
 * skewness(t2); // Positive skew (right-tailed)
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Returns NaN for constant input (zero variance)
 * - Unbiased correction requires at least 3 samples; otherwise returns NaN
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function skewness(t: Tensor, options: SkewnessOptions): Tensor;
export function skewness(t: Tensor, axis?: AxisLike, bias?: boolean): Tensor;
export function skewness(
  t: Tensor,
  axisOrOptions?: AxisLike | SkewnessOptions,
  bias = true
): Tensor {
  let axis: AxisLike | undefined;
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as SkewnessOptions;
    axis = o.axis;
    bias = o.bias ?? true;
  } else {
    axis = axisOrOptions as AxisLike | undefined;
  }
  const axes = normalizeAxes(axis, t.ndim);

  // Fast path: reduction over the entire tensor → a single scalar. Two passes
  // instead of the four-pass generic path, with no per-element scalar refetch.
  // (Empty axes is this codebase's "reduce all" shorthand for `axis=undefined`.)
  if (axes.length === 0 || axes.length === t.ndim) {
    const mom = fullTensorCentralMoments(t);
    if (mom) {
      const { n, m2, m3 } = mom;
      let g1: number;
      if (!Number.isFinite(m2) || m2 === 0) {
        g1 = Number.NaN;
      } else {
        g1 = m3 / m2 ** 1.5;
        if (!bias) {
          g1 = n < 3 ? Number.NaN : g1 * (Math.sqrt(n * (n - 1)) / (n - 2));
        }
      }
      return Tensor.fromTypedArray({
        data: new Float64Array([g1]),
        shape: [],
        dtype: "float64",
        device: t.device,
      });
    }
  }

  const mu = reduceMean(t, axis, false);
  const sigma2 = reduceVariance(t, axis, false, 0);

  const reduce = new Set(axes);
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sumCube = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }

    const m = getNumberAt(mu, mu.offset + outFlat);
    const v = Math.sqrt(getNumberAt(sigma2, sigma2.offset + outFlat));
    const x = getNumberAt(t, off);
    if (!Number.isFinite(v) || v === 0) {
      sumCube[outFlat] = NaN;
    } else {
      const z = (x - m) / v;
      sumCube[outFlat] = (sumCube[outFlat] ?? 0) + z * z * z;
    }
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const n = counts[i] ?? 0;
    if (n === 0) {
      out[i] = NaN;
      continue;
    }
    let g1 = (sumCube[i] ?? NaN) / n;
    if (!bias) {
      // Fisher-Pearson unbiased correction for sample skewness
      if (n < 3) {
        g1 = NaN;
      } else {
        g1 *= Math.sqrt(n * (n - 1)) / (n - 2);
      }
    }
    out[i] = g1;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the kurtosis (fourth standardized moment) along specified axis.
 *
 * Kurtosis measures the "tailedness" of the probability distribution.
 * - Negative excess kurtosis: lighter tails than normal (platykurtic)
 * - Zero excess kurtosis: same tails as normal distribution (mesokurtic)
 * - Positive excess kurtosis: heavier tails than normal (leptokurtic)
 *
 * Uses Fisher's definition: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `kurtosis(t, { fisher: false })`: it names the two independent
 * flags instead of relying on their order.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute kurtosis (undefined = all axes)
 * @param fisher - If true, returns excess kurtosis (subtract 3, default: true)
 * @param bias - If false, applies bias correction (requires at least 4 samples, default: true)
 * @returns Tensor containing kurtosis values
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * kurtosis(t, undefined, true);   // Excess kurtosis (Fisher, positional)
 * kurtosis(t, undefined, false);  // Raw kurtosis (Pearson, positional)
 * kurtosis(t, { fisher: false }); // Recommended options-object form
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Returns NaN for constant input (zero variance)
 * - Unbiased correction requires at least 4 samples; otherwise returns NaN
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function kurtosis(t: Tensor, options: KurtosisOptions): Tensor;
export function kurtosis(t: Tensor, axis?: AxisLike, fisher?: boolean, bias?: boolean): Tensor;
export function kurtosis(
  t: Tensor,
  axisOrOptions?: AxisLike | KurtosisOptions,
  fisher = true,
  bias = true
): Tensor {
  let axis: AxisLike | undefined;
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as KurtosisOptions;
    axis = o.axis;
    fisher = o.fisher ?? true;
    bias = o.bias ?? true;
  } else {
    axis = axisOrOptions as AxisLike | undefined;
  }
  const axes = normalizeAxes(axis, t.ndim);

  // Fast path: reduction over the entire tensor → a single scalar (two passes,
  // no per-element scalar refetch/sqrt). Empty axes = "reduce all".
  if (axes.length === 0 || axes.length === t.ndim) {
    const mom = fullTensorCentralMoments(t);
    if (mom) {
      const { n, m2, m4 } = mom;
      let out: number;
      if (!Number.isFinite(m2) || m2 === 0) {
        out = Number.NaN;
      } else {
        let g2 = m4 / (m2 * m2);
        if (!bias) {
          if (n < 4) {
            g2 = Number.NaN;
          } else {
            const excess = g2 - 3;
            const adj = ((n + 1) * excess + 6) * ((n - 1) / ((n - 2) * (n - 3)));
            g2 = adj + 3;
          }
        }
        out = fisher ? g2 - 3 : g2;
      }
      return Tensor.fromTypedArray({
        data: new Float64Array([out]),
        shape: [],
        dtype: "float64",
        device: t.device,
      });
    }
  }

  const mu = reduceMean(t, axis, false);
  const sigma2 = reduceVariance(t, axis, false, 0);

  const reduce = new Set(axes);
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sumQuad = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }

    const m = getNumberAt(mu, mu.offset + outFlat);
    const v = Math.sqrt(getNumberAt(sigma2, sigma2.offset + outFlat));
    const x = getNumberAt(t, off);
    if (!Number.isFinite(v) || v === 0) {
      sumQuad[outFlat] = NaN;
    } else {
      const z = (x - m) / v;
      sumQuad[outFlat] = (sumQuad[outFlat] ?? 0) + z ** 4;
    }
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const n = counts[i] ?? 0;
    if (n === 0) {
      out[i] = NaN;
      continue;
    }
    let g2 = (sumQuad[i] ?? NaN) / n; // raw kurtosis
    if (!bias) {
      // Unbiased k-statistics correction for excess kurtosis
      if (n < 4) {
        g2 = NaN;
      } else {
        const excess = g2 - 3;
        const adj = ((n + 1) * excess + 6) * ((n - 1) / ((n - 2) * (n - 3)));
        g2 = adj + 3;
      }
    }
    out[i] = fisher ? g2 - 3 : g2;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes quantiles along specified axes.
 *
 * Quantiles are cut points dividing the range of a probability distribution.
 * Uses linear interpolation between data points.
 *
 * @param t - Input tensor
 * @param q - Quantile(s) to compute, in range [0, 1] (0.5 = median)
 * @param axis - Axis or axes along which to compute quantiles (undefined = all axes)
 * @returns Tensor containing quantile values
 * @throws {InvalidParameterError} If q is not in [0, 1], tensor is empty, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * quantile(t, 0.5);        // Returns tensor([3]) - median
 * quantile(t, [0.25, 0.75]); // Returns tensor([2, 4]) - quartiles
 * quantile(t, 0.95);       // Returns tensor([4.8]) - 95th percentile
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function quantile(t: Tensor, q: number | number[], axis?: AxisLike): Tensor {
  const qVals = Array.isArray(q) ? q : [q];
  for (const v of qVals) {
    if (!Number.isFinite(v) || v < 0 || v > 1) {
      throw new InvalidParameterError("q must be in [0, 1]", "q", v);
    }
  }

  const axes = normalizeAxes(axis, t.ndim);
  const reduce = new Set(axes);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("quantile() requires at least one element", "size", t.size);
    }
    const arr = new Float64Array(t.size);
    let hasNaN = false;
    const raw = t.data;
    if (
      !Array.isArray(raw) &&
      !(raw instanceof BigInt64Array) &&
      isContiguous(t.shape, t.strides)
    ) {
      // Contiguous numeric fast path: bulk copy + comparator-free typed sort
      // (the closure/number[] push + comparator sort cost ~6x on 1K inputs).
      const off = t.offset;
      for (let i = 0; i < t.size; i++) {
        const v = raw[off + i] as number;
        if (Number.isNaN(v)) hasNaN = true;
        arr[i] = v;
      }
    } else {
      let w = 0;
      forEachIndexOffset(t, (o) => {
        const v = getNumberAt(t, o);
        if (Number.isNaN(v)) hasNaN = true;
        arr[w++] = v;
      });
    }
    if (hasNaN) {
      return tensor(
        qVals.map(() => Number.NaN),
        { dtype: "float64" }
      );
    }
    const n = arr.length;
    const results: number[] = [];
    // For a handful of quantiles, O(n) quickselect per rank beats an
    // O(n log n) sort. quickSelectF64 mutates `arr` (our private copy); a
    // re-partition on the already-permuted array stays correct. Fall back to a
    // full sort when there are many quantiles (k ≳ log2 n) or tiny inputs.
    if (qVals.length <= 8 && n >= 64) {
      for (const qVal of qVals) {
        const idx = qVal * (n - 1);
        const lower = Math.floor(idx);
        const weight = idx - lower;
        const loVal = quickSelectF64(arr, lower);
        if (weight === 0) {
          results.push(loVal);
        } else {
          // The (lower+1)-th smallest is the min of the right partition, since
          // quickselect leaves arr[lower+1..] all >= arr[lower].
          let hiVal = arr[lower + 1] as number;
          for (let i = lower + 2; i < n; i++) {
            const v = arr[i] as number;
            if (v < hiVal) hiVal = v;
          }
          results.push(loVal * (1 - weight) + hiVal * weight);
        }
      }
    } else {
      arr.sort();
      for (const qVal of qVals) {
        const idx = qVal * (n - 1);
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        const weight = idx - lower;
        results.push((arr[lower] ?? 0) * (1 - weight) + (arr[upper] ?? 0) * weight);
      }
    }
    return tensor(results, { dtype: "float64" });
  }

  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const buckets: (number[] | undefined)[] = new Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const val = getNumberAt(t, off);
    if (Number.isNaN(val)) {
      nanFlags[outFlat] = true;
      return;
    }
    const arr = buckets[outFlat] ?? [];
    arr.push(val);
    buckets[outFlat] = arr;
  });

  // Output shape: (qVals.length, ...reducedShape)
  const finalShape = [qVals.length, ...outShape];
  const finalSize = qVals.length * outSize;
  const out = new Float64Array(finalSize);

  for (let g = 0; g < outSize; g++) {
    if (nanFlags[g]) {
      for (let qi = 0; qi < qVals.length; qi++) {
        out[qi * outSize + g] = Number.NaN;
      }
      continue;
    }
    const arr = buckets[g] ?? [];
    if (arr.length === 0) {
      throw new InvalidParameterError(
        "quantile() reduction over empty axis is undefined",
        "axis",
        arr.length
      );
    }
    arr.sort((a, b) => a - b);
    for (let qi = 0; qi < qVals.length; qi++) {
      const qVal = qVals[qi] ?? 0;
      const idx = qVal * (arr.length - 1);
      const lower = Math.floor(idx);
      const upper = Math.ceil(idx);
      const weight = idx - lower;
      out[qi * outSize + g] = (arr[lower] ?? 0) * (1 - weight) + (arr[upper] ?? 0) * weight;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: finalShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes percentiles along specified axes.
 *
 * Percentiles are quantiles expressed as percentages (0-100 instead of 0-1).
 * This is a convenience wrapper around quantile().
 *
 * @param t - Input tensor
 * @param q - Percentile(s) to compute, in range [0, 100] (50 = median)
 * @param axis - Axis or axes along which to compute percentiles (undefined = all axes)
 * @returns Tensor containing percentile values
 * @throws {InvalidParameterError} If q is not in [0, 100], tensor is empty, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * percentile(t, 50);       // Returns tensor([3]) - median
 * percentile(t, [25, 75]); // Returns tensor([2, 4]) - quartiles
 * percentile(t, 95);       // Returns tensor([4.8]) - 95th percentile
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function percentile(t: Tensor, q: number | number[], axis?: AxisLike): Tensor {
  const qArr = Array.isArray(q) ? q : [q];
  for (const v of qArr) {
    if (!Number.isFinite(v) || v < 0 || v > 100) {
      throw new InvalidParameterError("q must be in [0, 100]", "q", v);
    }
  }
  // Percentile is just quantile with q/100 (convert percentage to fraction)
  const qVals = Array.isArray(q) ? q.map((v) => v / 100) : q / 100;
  return quantile(t, qVals, axis);
}

/**
 * Computes the n-th central moment about the mean.
 *
 * The n-th moment is defined as: E[(X - μ)ⁿ]
 * - n=1: Always 0 (by definition of mean)
 * - n=2: Variance
 * - n=3: Related to skewness
 * - n=4: Related to kurtosis
 *
 * @param t - Input tensor
 * @param n - Order of the moment (must be non-negative integer)
 * @param axis - Axis or axes along which to compute moment (undefined = all axes)
 * @returns Tensor containing moment values
 * @throws {InvalidParameterError} If n is not a non-negative integer
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * moment(t, 1);  // Returns ~0 (first moment about mean)
 * moment(t, 2);  // Returns variance
 * moment(t, 3);  // Returns third moment (related to skewness)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function moment(t: Tensor, n: number, axis?: AxisLike): Tensor {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("n must be a non-negative integer", "n", n);
  }

  const mu = reduceMean(t, axis, false);
  const axes = normalizeAxes(axis, t.ndim);
  const reduce = new Set(axes);
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const m = getNumberAt(mu, mu.offset + outFlat);
    const x = getNumberAt(t, off);
    sums[outFlat] = (sums[outFlat] ?? 0) + (x - m) ** n;
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const c = counts[i] ?? 0;
    out[i] = c === 0 ? NaN : (sums[i] ?? 0) / c;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the geometric mean along specified axis.
 *
 * The geometric mean is the n-th root of the product of n values.
 * Computed as: exp(mean(log(x))) for numerical stability.
 * Useful for averaging ratios, growth rates, and multiplicative processes.
 *
 * @param t - Input tensor (all values must be > 0)
 * @param axis - Axis or axes along which to compute geometric mean (undefined = all axes)
 * @returns Tensor containing geometric mean values
 * @throws {InvalidParameterError} If any value is <= 0
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 4, 8]);
 * geometricMean(t);  // Returns ~2.83 (⁴√(1*2*4*8))
 *
 * // Growth rates: 10% and 20% growth
 * const growth = tensor([1.1, 1.2]);
 * geometricMean(growth);  // Returns ~1.149 (average growth rate)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
/**
 * Contiguous numeric tensor → a `Float64Array` copy for whole-tensor reductions,
 * or null when the input isn't a simple contiguous numeric buffer (caller falls
 * back to the generic strided path).
 */
function fullReduceDenseF64(t: Tensor): Float64Array | null {
  if (t.dtype === "string") return null;
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;
  if (!isContiguous(t.shape, t.strides)) return null;
  return copyContiguousToF64(data, t.offset, t.size);
}

export function geometricMean(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError(
      "geometricMean() requires at least one element",
      "size",
      t.size
    );
  }

  // Fast path: whole-tensor reduction over a contiguous numeric buffer — one
  // narrowed loop instead of the forEachIndexOffset closure + getNumberAt.
  if (axes.length === 0 || axes.length === t.ndim) {
    const src = fullReduceDenseF64(t);
    if (src && src.length > 0) {
      const n = src.length;
      let s = 0;
      for (let i = 0; i < n; i++) {
        const x = src[i] as number;
        if (x <= 0) {
          throw new InvalidParameterError(
            "geometricMean() requires all values to be > 0",
            "value",
            x
          );
        }
        s += Math.log(x);
      }
      return Tensor.fromTypedArray({
        data: new Float64Array([Math.exp(s / n)]),
        shape: [],
        dtype: "float64",
        device: t.device,
      });
    }
  }

  const reduce = new Set(axes);
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const x = getNumberAt(t, off);
    if (x <= 0)
      throw new InvalidParameterError("geometricMean() requires all values to be > 0", "value", x);
    sums[outFlat] = (sums[outFlat] ?? 0) + Math.log(x);
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const c = counts[i] ?? 0;
    if (c === 0) {
      throw new InvalidParameterError(
        "geometricMean() reduction over empty axis is undefined",
        "axis",
        c
      );
    }
    out[i] = Math.exp((sums[i] ?? 0) / c);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the harmonic mean along specified axis.
 *
 * The harmonic mean is the reciprocal of the arithmetic mean of reciprocals.
 * Computed as: n / sum(1/x)
 * Useful for averaging rates and ratios (e.g., speeds, densities).
 *
 * @param t - Input tensor (all values must be > 0)
 * @param axis - Axis or axes along which to compute harmonic mean (undefined = all axes)
 * @returns Tensor containing harmonic mean values
 * @throws {InvalidParameterError} If any value is <= 0
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 4]);
 * harmonicMean(t);  // Returns ~1.71 (3 / (1/1 + 1/2 + 1/4))
 *
 * // Average speed: 60 mph for half distance, 40 mph for other half
 * const speeds = tensor([60, 40]);
 * harmonicMean(speeds);  // Returns 48 mph (correct average)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function harmonicMean(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError("harmonicMean() requires at least one element", "size", t.size);
  }

  // Fast path: whole-tensor reduction over a contiguous numeric buffer.
  if (axes.length === 0 || axes.length === t.ndim) {
    const src = fullReduceDenseF64(t);
    if (src && src.length > 0) {
      const n = src.length;
      let s = 0;
      for (let i = 0; i < n; i++) {
        const x = src[i] as number;
        if (x <= 0) {
          throw new InvalidParameterError(
            "harmonicMean() requires all values to be > 0",
            "value",
            x
          );
        }
        s += 1 / x;
      }
      return Tensor.fromTypedArray({
        data: new Float64Array([n / s]),
        shape: [],
        dtype: "float64",
        device: t.device,
      });
    }
  }

  const reduce = new Set(axes);
  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const x = getNumberAt(t, off);
    if (x <= 0) {
      throw new InvalidParameterError("harmonicMean() requires all values to be > 0", "value", x);
    }
    sums[outFlat] = (sums[outFlat] ?? 0) + 1 / x;
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const c = counts[i] ?? 0;
    if (c === 0) {
      throw new InvalidParameterError(
        "harmonicMean() reduction over empty axis is undefined",
        "axis",
        c
      );
    }
    out[i] = c / (sums[i] ?? NaN);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the trimmed mean (mean after removing outliers from both tails).
 *
 * Removes a specified proportion of extreme values from both ends before computing mean.
 * More robust to outliers than regular mean, less extreme than median.
 *
 * @param t - Input tensor
 * @param proportiontocut - Fraction to cut from each tail, in range [0, 0.5)
 * @param axis - Axis or axes along which to compute trimmed mean (undefined = all axes)
 * @returns Tensor containing trimmed mean values
 * @throws {InvalidParameterError} If proportiontocut is not in [0, 0.5), tensor is empty, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 100]); // 100 is outlier
 * mean(t);                    // Returns ~19.17 (affected by outlier)
 * trimMean(t, 0.2);          // Returns 3.5 (removes 1 and 100)
 * trimMean(t, 0.1);          // Returns ~22.8 (removes only 100)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function trimMean(t: Tensor, proportiontocut: number, axis?: AxisLike): Tensor {
  if (!Number.isFinite(proportiontocut) || proportiontocut < 0 || proportiontocut >= 0.5) {
    throw new InvalidParameterError(
      "proportiontocut must be a finite number in range [0, 0.5)",
      "proportiontocut",
      proportiontocut
    );
  }

  const axes = normalizeAxes(axis, t.ndim);
  const reduce = new Set(axes);

  if (axes.length === 0) {
    const n = t.size;
    if (n === 0)
      throw new InvalidParameterError("trimMean() requires at least one element", "size", n);
    // Gather into a flat Float64Array (no boxed number[]) while accumulating the
    // total and a NaN flag in the same pass.
    const arr = new Float64Array(n);
    let w = 0;
    let total = 0;
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const v = getNumberAt(t, off);
      arr[w++] = v;
      total += v;
      if (Number.isNaN(v)) hasNaN = true;
    });
    if (hasNaN) {
      return tensor([Number.NaN]);
    }
    const nTrim = Math.floor(n * proportiontocut);
    if (nTrim === 0) {
      return tensor([total / n]);
    }
    // Trimmed mean = (total − smallest nTrim − largest nTrim) / (n − 2·nTrim).
    // Two O(n) quickselect partitions beat a full O(n log n) sort: after
    // quickselect(k), arr[0..k) holds the k smallest and arr[k..n) the rest.
    quickSelectF64(arr, nTrim);
    let bottom = 0;
    for (let i = 0; i < nTrim; i++) bottom += arr[i] as number;
    quickSelectF64(arr, n - nTrim);
    let topSum = 0;
    for (let i = n - nTrim; i < n; i++) topSum += arr[i] as number;
    return tensor([(total - bottom - topSum) / (n - 2 * nTrim)]);
  }

  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const buckets: (number[] | undefined)[] = new Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const val = getNumberAt(t, off);
    if (Number.isNaN(val)) {
      nanFlags[outFlat] = true;
      return;
    }
    const arr = buckets[outFlat] ?? [];
    arr.push(val);
    buckets[outFlat] = arr;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    if (nanFlags[i]) {
      out[i] = Number.NaN;
      continue;
    }
    const arr = buckets[i] ?? [];
    if (arr.length === 0) {
      throw new InvalidParameterError(
        "trimMean() reduction over empty axis is undefined",
        "axis",
        arr.length
      );
    }
    arr.sort((a, b) => a - b);
    const nTrim = Math.floor(arr.length * proportiontocut);
    const trimmed = arr.slice(nTrim, arr.length - nTrim);
    const sum = trimmed.reduce((a, b) => a + b, 0);
    out[i] = sum / trimmed.length;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Compute the z-score of each element relative to the sample mean and std.
 *
 * z = (x - mean) / std
 *
 * @param t - Input tensor
 * @param ddof - Delta degrees of freedom for std (default 0)
 * @returns Tensor of z-scores (flattened)
 */
export function zscore(t: Tensor, ddof = 0): Tensor {
  if (t.size === 0) {
    throw new InvalidParameterError("zscore() requires non-empty tensor", "t");
  }

  // Fast path: contiguous numeric input. Standardize in three narrowed passes
  // over one Float64Array (mean, variance, write) — no boxed number[] gather.
  // Output is row-major, matching the contiguous input positions.
  const dense = fullReduceDenseF64(t);
  if (dense) {
    const n = dense.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += dense[i] as number;
    const m = sum / n;
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = (dense[i] as number) - m;
      ss += d * d;
    }
    const s = Math.sqrt(ss / (n - ddof));
    const out = new Float64Array(n);
    if (s !== 0) {
      const inv = 1 / s;
      for (let i = 0; i < n; i++) out[i] = ((dense[i] as number) - m) * inv;
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: [...t.shape],
      dtype: "float64",
      device: t.device,
    });
  }

  // Collect all values (non-contiguous / non-numeric fallback)
  const vals: number[] = [];
  forEachIndexOffset(t, (off) => {
    vals.push(getNumberAt(t, off));
  });

  // Compute global mean
  const m = vals.reduce((a, b) => a + b, 0) / vals.length;

  // Compute global std
  let ss = 0;
  for (const v of vals) {
    ss += (v - m) ** 2;
  }
  const s = Math.sqrt(ss / (vals.length - ddof));

  const out = new Float64Array(vals.length);
  for (let i = 0; i < vals.length; i++) {
    out[i] = s === 0 ? 0 : ((vals[i] ?? 0) - m) / s;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: [...t.shape],
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Compute Cohen's d effect size between two samples.
 *
 * d = (mean1 - mean2) / pooled_std
 *
 * @param a - First sample (1-D array of numbers)
 * @param b - Second sample (1-D array of numbers)
 * @returns Cohen's d value
 */
export function cohenD(a: number[], b: number[]): number {
  if (a.length < 2 || b.length < 2) {
    throw new InvalidParameterError("cohenD requires at least 2 observations per group", "a");
  }
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / (a.length - 1);
  const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / (b.length - 1);
  const pooled = Math.sqrt(((a.length - 1) * va + (b.length - 1) * vb) / (a.length + b.length - 2));
  return pooled === 0 ? 0 : (ma - mb) / pooled;
}

/**
 * Non-parametric bootstrap: resample a statistic.
 *
 * @param data - 1-D array of observations
 * @param statFn - Function that computes a statistic from a sample
 * @param options - nResamples (default 1000), seed (optional), confidenceLevel (default 0.95)
 * @returns Object with estimate, ci (confidence interval), and samples array
 */
export function bootstrap(
  data: number[],
  statFn: (sample: number[]) => number,
  options: {
    nResamples?: number;
    seed?: number;
    confidenceLevel?: number;
  } = {}
): { estimate: number; ci: [number, number]; samples: number[] } {
  const nResamples = options.nResamples ?? 1000;
  const confidenceLevel = options.confidenceLevel ?? 0.95;
  const n = data.length;
  if (n === 0) {
    throw new InvalidParameterError("bootstrap requires non-empty data", "data");
  }

  // Simple seeded RNG (LCG)
  let rngState = options.seed ?? Date.now() ^ 0xdeadbeef;
  const nextRng = () => {
    rngState = (rngState * 1664525 + 1013904223) & 0x7fffffff;
    return rngState / 0x7fffffff;
  };

  const samples: number[] = [];
  for (let r = 0; r < nResamples; r++) {
    const resample: number[] = [];
    for (let i = 0; i < n; i++) {
      resample.push(data[Math.floor(nextRng() * n)]!);
    }
    samples.push(statFn(resample));
  }

  samples.sort((a, b) => a - b);
  const estimate = statFn(data);
  const alpha = 1 - confidenceLevel;
  const lo = samples[Math.floor((alpha / 2) * nResamples)]!;
  const hi = samples[Math.floor((1 - alpha / 2) * nResamples)]!;

  return { estimate, ci: [lo, hi], samples };
}

/**
 * Computes the standard error of the mean (SEM).
 *
 * SEM = std(x, ddof=1) / sqrt(n)
 *
 * The SEM quantifies how precisely the sample mean estimates the population mean.
 * Smaller SEM indicates a more precise estimate.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute SEM (undefined = all axes)
 * @param ddof - Delta degrees of freedom for std computation (default: 1)
 * @returns Tensor containing SEM values
 * @throws {InvalidParameterError} If tensor is empty or has fewer observations than ddof+1
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * sem(t);  // Returns std(t, ddof=1) / sqrt(5)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function sem(t: Tensor, axis?: AxisLike, ddof = 1): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  const reduce = new Set(axes);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("sem() requires at least one element", "size", t.size);
    }
    const vals: number[] = [];
    forEachIndexOffset(t, (off) => {
      vals.push(getNumberAt(t, off));
    });
    const n = vals.length;
    if (n <= ddof) {
      throw new InvalidParameterError(
        `sem() requires more than ddof=${ddof} observations`,
        "ddof",
        ddof
      );
    }
    const m = vals.reduce((a, b) => a + b, 0) / n;
    let ss = 0;
    for (const v of vals) {
      ss += (v - m) ** 2;
    }
    const s = Math.sqrt(ss / (n - ddof));
    const out = new Float64Array(1);
    out[0] = s / Math.sqrt(n);
    return Tensor.fromTypedArray({
      data: out,
      shape: [],
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, false);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);
  const sumsSq = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const val = getNumberAt(t, off);
    sums[outFlat] = (sums[outFlat] ?? 0) + val;
    counts[outFlat] = (counts[outFlat] ?? 0) + 1;
  });

  // Second pass: compute sum of squared deviations
  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
      oi++;
    }
    const val = getNumberAt(t, off);
    const n = counts[outFlat] ?? 1;
    const m = (sums[outFlat] ?? 0) / n;
    sumsSq[outFlat] = (sumsSq[outFlat] ?? 0) + (val - m) ** 2;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const n = counts[i] ?? 0;
    if (n <= ddof) {
      out[i] = NaN;
    } else {
      const s = Math.sqrt((sumsSq[i] ?? 0) / (n - ddof));
      out[i] = s / Math.sqrt(n);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes the interquartile range (IQR).
 *
 * IQR = Q3 - Q1 = percentile(75) - percentile(25)
 *
 * The IQR is a robust measure of spread that is less sensitive to outliers
 * than the standard deviation.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute IQR (undefined = all axes)
 * @returns Tensor containing IQR values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
 * iqr(t);  // Returns tensor([4.5]) - Q3(7.75) - Q1(3.25)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function iqr(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  const reduce = new Set(axes);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("iqr() requires at least one element", "size", t.size);
    }
    const arr: number[] = [];
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const v = getNumberAt(t, off);
      if (Number.isNaN(v)) hasNaN = true;
      arr.push(v);
    });
    if (hasNaN) return tensor([NaN]);
    arr.sort((a, b) => a - b);
    const q1Idx = 0.25 * (arr.length - 1);
    const q3Idx = 0.75 * (arr.length - 1);
    const interp = (idx: number): number => {
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      const w = idx - lo;
      return (arr[lo] ?? 0) * (1 - w) + (arr[hi] ?? 0) * w;
    };
    const out = new Float64Array(1);
    out[0] = interp(q3Idx) - interp(q1Idx);
    return Tensor.fromTypedArray({
      data: out,
      shape: [],
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, false);
  const outStridesArr = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const buckets: (number[] | undefined)[] = new Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    let oi = 0;
    for (let i = 0; i < t.ndim; i++) {
      if (reduce.has(i)) continue;
      outFlat += (idx[i] ?? 0) * (outStridesArr[oi] ?? 0);
      oi++;
    }
    const val = getNumberAt(t, off);
    if (Number.isNaN(val)) {
      nanFlags[outFlat] = true;
      return;
    }
    const bucket = buckets[outFlat] ?? [];
    bucket.push(val);
    buckets[outFlat] = bucket;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    if (nanFlags[i]) {
      out[i] = NaN;
      continue;
    }
    const arr = buckets[i] ?? [];
    if (arr.length === 0) {
      throw new InvalidParameterError(
        "iqr() reduction over empty axis is undefined",
        "axis",
        arr.length
      );
    }
    arr.sort((a, b) => a - b);
    const q1Idx = 0.25 * (arr.length - 1);
    const q3Idx = 0.75 * (arr.length - 1);
    const lo1 = Math.floor(q1Idx);
    const hi1 = Math.ceil(q1Idx);
    const w1 = q1Idx - lo1;
    const q1 = (arr[lo1] ?? 0) * (1 - w1) + (arr[hi1] ?? 0) * w1;
    const lo3 = Math.floor(q3Idx);
    const hi3 = Math.ceil(q3Idx);
    const w3 = q3Idx - lo3;
    const q3 = (arr[lo3] ?? 0) * (1 - w3) + (arr[hi3] ?? 0) * w3;
    out[i] = q3 - q1;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}
