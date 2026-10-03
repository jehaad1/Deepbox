import { InvalidParameterError } from "../core";
import { Tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import { __random } from "../random/random";
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
 * One-element `float64` tensor of shape `[1]`.
 *
 * `tensor([x])` would use the configured default dtype (float32 unless changed),
 * which silently rounds the value; reductions here always return float64.
 */
function scalarF64(value: number, device: Tensor["device"]): Tensor {
  return Tensor.fromTypedArray({
    data: new Float64Array([value]),
    shape: [1],
    dtype: "float64",
    device,
  });
}

/** NumPy-style linear interpolation between two order statistics (`0 < w < 1`). */
function lerp(lo: number, hi: number, w: number): number {
  if (lo === hi) return lo;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return lo * (1 - w) + hi * w;
  const diff = hi - lo;
  return w >= 0.5 ? hi - diff * (1 - w) : lo + diff * w;
}

/**
 * The `q`-quantile (0 <= q <= 1) of an ascending-sorted, NaN-free array using
 * linear interpolation between the two nearest order statistics (NumPy's
 * default `linear` method).
 */
function quantileOfSorted(sorted: ArrayLike<number>, q: number): number {
  const pos = q * (sorted.length - 1);
  const lower = Math.floor(pos);
  const weight = pos - lower;
  const lo = sorted[lower] as number;
  if (weight === 0) return lo;
  return lerp(lo, sorted[lower + 1] as number, weight);
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
 * mean(t);                         // Returns scalar tensor 3.5 - mean of all elements
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
 * it's the average of the two middle values. Less sensitive to outliers than the mean.
 *
 * Accepts either an options object or the positional `(axis, keepdims)` form,
 * like {@link mean}.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the median (undefined = all axes)
 * @param keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @returns `float64` tensor containing median values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * median(t);  // Returns scalar tensor 3
 *
 * const t2 = tensor([1, 2, 3, 4]);
 * median(t2); // Returns scalar tensor 2.5 - average of 2 and 3
 *
 * median(tensor([[1, 2], [3, 8]]), { axis: 0 }); // [2, 5]
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs result in NaN output (NaN sorts to end)
 * - Infinity values are sorted naturally (±Infinity at extremes)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function median(t: Tensor, options: MeanOptions): Tensor;
export function median(t: Tensor, axis?: AxisLike, keepdims?: boolean): Tensor;
export function median(
  t: Tensor,
  axisOrOptions?: AxisLike | MeanOptions,
  keepdimsArg = false
): Tensor {
  let axis: AxisLike | undefined;
  let keepdims = keepdimsArg;
  if (isReductionOptions(axisOrOptions)) {
    const o = axisOrOptions as MeanOptions;
    axis = o.axis;
    keepdims = o.keepdims ?? false;
  } else {
    axis = axisOrOptions as AxisLike | undefined;
  }
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
 * @returns `float64` tensor containing mode values. A full reduction (no `axis`) has shape `[1]`;
 *   otherwise the reduced axes are removed.
 * @throws {InvalidParameterError} If tensor is empty
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
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
    return scalarF64(hasNaN ? Number.NaN : modeVal, t.device);
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
 * Computed from {@link variance}, which avoids the cancellation of the naive
 * `E[x²] - E[x]²` formula.
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `std(t, { ddof: 1 })` over `std(t, 0, false, 1)`, whose trailing
 * `keepdims`/`ddof` booleans/numbers are easy to transpose.
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
 * Uses a two-pass algorithm (mean, then squared deviations) or Welford's online update,
 * both of which are numerically stable.
 *
 * Accepts either an options object (recommended) or the historical positional
 * form. Prefer `variance(t, { ddof: 1 })` over `variance(t, 0, false, 1)`.
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

/** Block length for the two-level summation used by the moment fast path. */
const MOMENT_BLOCK = 256;

/**
 * SciPy's test for a sample whose variance is rounding noise around a constant:
 * `m2 <= (eps * mean)^2`. A constant column such as `[0.1, 0.1, 0.1]` has a
 * mean that is off by one ulp, so its computed variance is tiny but not zero;
 * without this test the standardized moments of such data come out as
 * meaningless finite numbers (for example a skewness of -1). Non-finite `m2`
 * (NaN or infinite input) is also treated as degenerate.
 */
function isDegenerateVariance(m2: number, mean: number): boolean {
  if (!Number.isFinite(m2)) return true;
  const tol = Number.EPSILON * mean;
  return m2 <= tol * tol;
}

/**
 * True when a finite variance is only rounding noise around a constant sample
 * (same tolerance as {@link isDegenerateVariance}). Unlike that helper, NaN and
 * infinite variances are not reported as constant, so non-finite data keeps
 * propagating as NaN.
 */
function isConstantVariance(v: number, mean: number): boolean {
  return Number.isFinite(v) && isDegenerateVariance(v, mean);
}

/**
 * Central moments (m2, m3, m4 about the mean) of a whole numeric tensor in two
 * passes: gather and mean, then one narrowed loop accumulating the squared, cubed
 * and quartic deviations together. Used by the full-reduction fast paths of
 * `skewness`/`kurtosis`, which otherwise re-fetch the scalar mean/variance and
 * recompute a sqrt per element (several extra passes). Sums are accumulated in
 * blocks so rounding error does not grow linearly with the sample size.
 * Returns null for non-numeric, empty or non-contiguous input so callers fall
 * back to the generic path.
 */
function fullTensorCentralMoments(
  t: Tensor
): { n: number; mean: number; m2: number; m3: number; m4: number } | null {
  if (t.dtype === "string") return null;
  const n = t.size;
  if (n === 0) return null;
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;

  // Materialize a contiguous Float64Array once (a narrowed memcpy for contiguous
  // float64 input) so both accumulation passes run as monomorphic typed-array
  // loops, with no `forEachIndexOffset` closure or `getNumberAt` indirection, which
  // otherwise dominate at a few thousand elements.
  const src = isContiguous(t.shape, t.strides) ? copyContiguousToF64(data, t.offset, n) : null;
  if (src === null) return null;

  let sum = 0;
  for (let start = 0; start < n; start += MOMENT_BLOCK) {
    const end = Math.min(n, start + MOMENT_BLOCK);
    let s = 0;
    for (let i = start; i < end; i++) s += src[i] as number;
    sum += s;
  }
  const m = sum / n;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  for (let start = 0; start < n; start += MOMENT_BLOCK) {
    const end = Math.min(n, start + MOMENT_BLOCK);
    let b2 = 0;
    let b3 = 0;
    let b4 = 0;
    for (let i = start; i < end; i++) {
      const d = (src[i] as number) - m;
      const d2 = d * d;
      b2 += d2;
      b3 += d2 * d;
      b4 += d2 * d2;
    }
    s2 += b2;
    s3 += b3;
    s4 += b4;
  }
  return { n, mean: m, m2: s2 / n, m3: s3 / n, m4: s4 / n };
}

/**
 * Per-output mean and population standard deviation (ddof = 0) for the
 * axis-wise standardized-moment loops. The standard deviation is NaN where the
 * variance is degenerate (see {@link isDegenerateVariance}), which makes every
 * standardized value of that output NaN.
 */
function standardizers(
  t: Tensor,
  axis: AxisLike | undefined
): { means: Float64Array; sds: Float64Array } {
  const mu = reduceMean(t, axis, false);
  const sigma2 = reduceVariance(t, axis, false, 0);
  const means = new Float64Array(mu.size);
  const sds = new Float64Array(mu.size);
  for (let i = 0; i < mu.size; i++) {
    const m = getNumberAt(mu, mu.offset + i);
    const v = getNumberAt(sigma2, sigma2.offset + i);
    means[i] = m;
    sds[i] = isDegenerateVariance(v, m) ? Number.NaN : Math.sqrt(v);
  }
  return { means, sds };
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
 * @throws {InvalidParameterError} If the tensor is empty or the reduction is over an empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * skewness(t);                    // Returns ~0 (symmetric)
 * skewness(t, undefined, false);  // Unbiased correction (positional)
 * skewness(t, { bias: false });   // Recommended options-object form
 *
 * const t2 = tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]);
 * skewness(t2); // Negative skew (the longer tail is on the left)
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Returns NaN for constant input (zero variance, judged like SciPy: the
 *   variance is at most `(eps * mean)²`)
 * - Unbiased correction (`bias: false`) requires at least 3 samples; with fewer it returns NaN,
 *   as pandas does (SciPy silently returns the uncorrected value instead)
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
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError("skewness() requires at least one element", "size", t.size);
  }

  // Fast path: reduction over the entire tensor to a single scalar. Two passes
  // instead of the four-pass generic path, with no per-element scalar refetch.
  // (Empty axes is this codebase's "reduce all" shorthand for `axis=undefined`.)
  if (axes.length === 0 || axes.length === t.ndim) {
    const mom = fullTensorCentralMoments(t);
    if (mom) {
      const { n, mean: mu, m2, m3 } = mom;
      let g1: number;
      if (isDegenerateVariance(m2, mu)) {
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

  const { means, sds } = standardizers(t, axis);

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

    const z = (getNumberAt(t, off) - (means[outFlat] as number)) / (sds[outFlat] as number);
    sumCube[outFlat] = (sumCube[outFlat] as number) + z * z * z;
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
 * @throws {InvalidParameterError} If the tensor is empty or the reduction is over an empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
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
 * - Returns NaN for constant input (zero variance, judged like SciPy: the
 *   variance is at most `(eps * mean)²`)
 * - Unbiased correction (`bias: false`) requires at least 4 samples; with fewer it returns NaN,
 *   as pandas does (SciPy silently returns the uncorrected value instead)
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
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError("kurtosis() requires at least one element", "size", t.size);
  }

  // Fast path: reduction over the entire tensor to a single scalar (two passes,
  // no per-element scalar refetch/sqrt). Empty axes = "reduce all".
  if (axes.length === 0 || axes.length === t.ndim) {
    const mom = fullTensorCentralMoments(t);
    if (mom) {
      const { n, mean: mu, m2, m4 } = mom;
      let out: number;
      if (isDegenerateVariance(m2, mu)) {
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

  const { means, sds } = standardizers(t, axis);

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

    const z = (getNumberAt(t, off) - (means[outFlat] as number)) / (sds[outFlat] as number);
    const z2 = z * z;
    sumQuad[outFlat] = (sumQuad[outFlat] as number) + z2 * z2;
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
 * Uses linear interpolation between data points (NumPy's default `linear` method).
 *
 * @param t - Input tensor
 * @param q - Quantile(s) to compute, in range [0, 1] (0.5 = median)
 * @param axis - Axis or axes along which to compute quantiles (undefined = all axes)
 * @returns `float64` tensor with one leading entry per quantile: shape `[q.length]` for a full
 *   reduction, `[q.length, ...reducedShape]` when `axis` is given (`q.length` is 1 for a scalar `q`)
 * @throws {InvalidParameterError} If q is not in [0, 1], tensor is empty, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * quantile(t, 0.5);        // Returns tensor([3]) - median
 * quantile(t, [0.25, 0.75]); // Returns tensor([2, 4]) - quartiles
 * quantile(t, 0.95);       // Returns tensor([4.8]) - 95th percentile
 * ```
 *
 * @remarks
 * A slice containing NaN gives NaN. Infinite values interpolate to the infinite value
 * rather than NaN (`quantile([1, Infinity], 1)` is `Infinity`).
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
      return Tensor.fromTypedArray({
        data: new Float64Array(qVals.length).fill(Number.NaN),
        shape: [qVals.length],
        dtype: "float64",
        device: t.device,
      });
    }
    const n = arr.length;
    const results = new Float64Array(qVals.length);
    let ri = 0;
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
          results[ri++] = loVal;
        } else {
          // The (lower+1)-th smallest is the min of the right partition, since
          // quickselect leaves arr[lower+1..] all >= arr[lower].
          let hiVal = arr[lower + 1] as number;
          for (let i = lower + 2; i < n; i++) {
            const v = arr[i] as number;
            if (v < hiVal) hiVal = v;
          }
          results[ri++] = lerp(loVal, hiVal, weight);
        }
      }
    } else {
      arr.sort();
      for (const qVal of qVals) results[ri++] = quantileOfSorted(arr, qVal);
    }
    return Tensor.fromTypedArray({
      data: results,
      shape: [qVals.length],
      dtype: "float64",
      device: t.device,
    });
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
      out[qi * outSize + g] = quantileOfSorted(arr, qVals[qi] ?? 0);
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
 * - n=0: Always 1 (NaN if the data contain NaN or Infinity)
 * - n=1: Always 0 (by definition of mean; NaN if the data contain NaN or Infinity)
 * - n=2: Population variance
 * - n=3: Related to skewness
 * - n=4: Related to kurtosis
 *
 * @param t - Input tensor
 * @param n - Order of the moment (must be non-negative integer)
 * @param axis - Axis or axes along which to compute moment (undefined = all axes)
 * @returns Tensor containing moment values
 * @throws {InvalidParameterError} If n is not a non-negative integer, the tensor is empty, or the
 *   reduction is over an empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * moment(t, 1);  // Returns 0 (first moment about mean)
 * moment(t, 2);  // Returns 2 (population variance)
 * moment(t, 3);  // Returns 0 (third moment of a symmetric sample)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function moment(t: Tensor, n: number, axis?: AxisLike): Tensor {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("n must be a non-negative integer", "n", n);
  }

  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError("moment() requires at least one element", "size", t.size);
  }
  const mu = reduceMean(t, axis, false);
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
    // The zeroth moment is exactly one and the first exactly zero; summing (x - m)
    // would leave a rounding residue of order eps * |m|. `(x - m) * 0` is 0 for
    // finite data and keeps NaN for NaN/Infinity (note that NaN ** 0 is 1 in JS).
    const term = n === 0 ? 1 + (x - m) * 0 : n === 1 ? (x - m) * 0 : (x - m) ** n;
    sums[outFlat] = (sums[outFlat] ?? 0) + term;
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
 * Returns a `Float64Array` copy of a contiguous numeric tensor for whole-tensor reductions,
 * or null when the input is not a simple contiguous numeric buffer (the caller falls
 * back to the generic strided path).
 */
function fullReduceDenseF64(t: Tensor): Float64Array | null {
  if (t.dtype === "string") return null;
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;
  if (!isContiguous(t.shape, t.strides)) return null;
  return copyContiguousToF64(data, t.offset, t.size);
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
export function geometricMean(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError(
      "geometricMean() requires at least one element",
      "size",
      t.size
    );
  }

  // Fast path: whole-tensor reduction over a contiguous numeric buffer, using one
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
 * Sum of `arr[from .. to)` accumulated in blocks of 256 so rounding error does
 * not grow linearly with the length.
 */
function blockSum(arr: Float64Array, from: number, to: number): number {
  let total = 0;
  for (let start = from; start < to; start += 256) {
    const end = Math.min(to, start + 256);
    let s = 0;
    for (let i = start; i < end; i++) s += arr[i] as number;
    total += s;
  }
  return total;
}

/**
 * Computes the trimmed mean (mean after removing outliers from both tails).
 *
 * Removes a specified proportion of extreme values from both ends before computing mean.
 * Less sensitive to outliers than the regular mean, less extreme than the median. The number of
 * values removed from each tail is `floor(n * proportiontocut)`, as in
 * `scipy.stats.trim_mean`.
 *
 * @param t - Input tensor
 * @param proportiontocut - Fraction to cut from each tail, in range [0, 0.5)
 * @param axis - Axis or axes along which to compute trimmed mean (undefined = all axes)
 * @returns `float64` tensor of trimmed means. For a full reduction (no `axis`) the result has
 *   shape `[1]`; otherwise the reduced axes are removed.
 * @throws {InvalidParameterError} If proportiontocut is not in [0, 0.5), tensor is empty, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 100]); // 100 is outlier
 * mean(t);                    // Returns ~19.17 (affected by outlier)
 * trimMean(t, 0.2);          // Returns 3.5 (removes 1 and 100)
 * trimMean(t, 0.1);          // Returns ~19.17 (floor(6 * 0.1) = 0 values removed)
 * ```
 *
 * @remarks
 * NaN in a slice makes that slice's result NaN. Infinite values are dropped like any other
 * extreme value when they fall in a trimmed tail.
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
    // Gather into a flat Float64Array (no boxed number[]) and flag NaN.
    let arr = fullReduceDenseF64(t);
    let hasNaN = false;
    if (arr === null) {
      arr = new Float64Array(n);
      let w = 0;
      const gathered = arr;
      forEachIndexOffset(t, (off) => {
        gathered[w++] = getNumberAt(t, off);
      });
    }
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(arr[i] as number)) {
        hasNaN = true;
        break;
      }
    }
    if (hasNaN) {
      return scalarF64(Number.NaN, t.device);
    }
    const nTrim = Math.floor(n * proportiontocut);
    if (nTrim === 0) {
      return scalarF64(blockSum(arr, 0, n) / n, t.device);
    }
    // Two O(n) quickselect partitions beat a full O(n log n) sort. The first
    // moves the nTrim smallest values to arr[0..nTrim); the second, on the
    // remaining n - nTrim values, moves the (n - 2·nTrim) smallest of them to
    // the front. The kept middle is then summed directly: subtracting the
    // trimmed tails from a grand total instead would lose the kept values
    // entirely next to a huge outlier and give NaN for infinite tails.
    quickSelectF64(arr, nTrim);
    const rest = arr.subarray(nTrim);
    const keep = n - 2 * nTrim;
    quickSelectF64(rest, keep);
    return scalarF64(blockSum(rest, 0, keep) / keep, t.device);
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
    let sum = 0;
    for (let j = nTrim; j < arr.length - nTrim; j++) sum += arr[j] as number;
    out[i] = sum / (arr.length - 2 * nTrim);
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
 * Without `axis` the mean and standard deviation are taken over every element
 * (note that `scipy.stats.zscore` defaults to `axis=0`; pass `axis` to get that
 * behavior). The output always has the same shape as the input. A slice with
 * zero standard deviation maps to zeros instead of NaN.
 *
 * @param t - Input tensor
 * @param ddof - Delta degrees of freedom for std (default 0)
 * @param axis - Axis or axes along which to standardize (undefined = all elements)
 * @returns `float64` tensor of z-scores with the same shape as `t`
 * @throws {InvalidParameterError} If the tensor is empty, `ddof` is negative or not finite, or
 *   `ddof` is not smaller than the number of elements being standardized together
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * zscore(tensor([1, 2, 3, 4, 5]));                  // (x - 3) / sqrt(2)
 * zscore(tensor([[1, 2], [3, 6]]), 0, 0);           // standardize each column
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function zscore(t: Tensor, ddof = 0, axis?: AxisLike): Tensor {
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be a non-negative finite number", "ddof", ddof);
  }
  if (t.size === 0) {
    throw new InvalidParameterError("zscore() requires non-empty tensor", "t");
  }
  const axes = normalizeAxes(axis, t.ndim);

  if (axes.length > 0 && axes.length < t.ndim) {
    return zscoreAlongAxes(t, axes, ddof);
  }

  if (t.size <= ddof) {
    throw new InvalidParameterError(
      `ddof=${ddof} >= size=${t.size}, standard deviation undefined`,
      "ddof",
      ddof
    );
  }

  // Fast path: contiguous numeric input. Standardize in three narrowed passes
  // over one Float64Array (mean, variance, write), with no boxed number[] gather.
  // Output is row-major, matching the contiguous input positions.
  let dense = fullReduceDenseF64(t);
  if (dense === null) {
    // Non-contiguous / non-numeric fallback: gather in row-major order.
    const gathered = new Float64Array(t.size);
    let w = 0;
    forEachIndexOffset(t, (off) => {
      gathered[w++] = getNumberAt(t, off);
    });
    dense = gathered;
  }
  const n = dense.length;
  const m = blockSum(dense, 0, n) / n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = (dense[i] as number) - m;
    ss += d * d;
  }
  const v = ss / (n - ddof);
  const s = Math.sqrt(v);
  const out = new Float64Array(n);
  // A constant sample (including one whose computed variance is only rounding
  // noise, such as [0.1, 0.1, 0.1]) maps to zeros.
  if (s !== 0 && !isConstantVariance(v, m)) {
    for (let i = 0; i < n; i++) out[i] = ((dense[i] as number) - m) / s;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: [...t.shape],
    dtype: "float64",
    device: t.device,
  });
}

/** Axis-wise {@link zscore} for a non-empty strict subset of axes. */
function zscoreAlongAxes(t: Tensor, axes: readonly number[], ddof: number): Tensor {
  const mu = reduceMean(t, axes, true);
  const variance = reduceVariance(t, axes, true, ddof);
  const keptShape = reducedShape(t.shape, axes, true);
  const keptStrides = computeStrides(keptShape);
  const reduce = new Set(axes);
  const contrib = new Array<number>(t.ndim).fill(0);
  for (let i = 0; i < t.ndim; i++) {
    if (!reduce.has(i)) contrib[i] = keptStrides[i] ?? 0;
  }
  const means = new Float64Array(mu.size);
  const sds = new Float64Array(mu.size);
  for (let i = 0; i < mu.size; i++) {
    const m = getNumberAt(mu, mu.offset + i);
    const v = getNumberAt(variance, variance.offset + i);
    means[i] = m;
    // Zero marks a constant slice, which standardizes to zeros.
    sds[i] = isConstantVariance(v, m) ? 0 : Math.sqrt(v);
  }
  const out = new Float64Array(t.size);
  let w = 0;
  forEachIndexOffset(t, (off, idx) => {
    let k = 0;
    for (let i = 0; i < t.ndim; i++) k += (idx[i] ?? 0) * (contrib[i] ?? 0);
    const s = sds[k] as number;
    out[w++] = s === 0 ? 0 : (getNumberAt(t, off) - (means[k] as number)) / s;
  });
  return Tensor.fromTypedArray({
    data: out,
    shape: [...t.shape],
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Mean and sample variance (ddof = 1) of a sample with at least two values. A sample of
 * one repeated finite value is reported exactly (its mean is that value and its variance
 * is 0): the computed mean of `[0.1, 0.1, 0.1]` is off by one ulp, which would otherwise
 * leave a variance of about 1e-34 instead of 0.
 */
function meanAndSampleVariance(x: readonly number[]): [number, number] {
  const first = x[0] as number;
  if (Number.isFinite(first) && x.every((v) => v === first)) return [first, 0];
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  const v = x.reduce((s, e) => s + (e - m) ** 2, 0) / (x.length - 1);
  return [m, v];
}

/**
 * Compute Cohen's d effect size between two samples.
 *
 * d = (mean1 - mean2) / pooled_std
 *
 * The pooled standard deviation uses the sample variance (ddof = 1) of each group.
 * When both groups are constant the pooled standard deviation is 0: the result is 0
 * if the two means are equal and `Infinity` or `-Infinity` (the sign of the mean
 * difference) otherwise.
 *
 * @param a - First sample (1-D array of numbers)
 * @param b - Second sample (1-D array of numbers)
 * @returns Cohen's d value
 * @throws {InvalidParameterError} If either sample has fewer than 2 observations
 *
 * @example
 * ```ts
 * cohenD([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]); // -2 / sqrt(2.5) ≈ -1.265
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function cohenD(a: readonly number[], b: readonly number[]): number {
  if (a.length < 2 || b.length < 2) {
    const shortGroup = a.length < 2 ? "a" : "b";
    throw new InvalidParameterError(
      "cohenD requires at least 2 observations per group",
      shortGroup,
      shortGroup === "a" ? a.length : b.length
    );
  }
  const [ma, va] = meanAndSampleVariance(a);
  const [mb, vb] = meanAndSampleVariance(b);
  const pooled = Math.sqrt(((a.length - 1) * va + (b.length - 1) * vb) / (a.length + b.length - 2));
  if (pooled === 0) {
    const diff = ma - mb;
    return diff === 0 ? 0 : diff > 0 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  }
  return (ma - mb) / pooled;
}

/** Options for {@link bootstrap}. */
export interface BootstrapOptions {
  /** Number of resamples to draw (positive integer, default: 1000). */
  nResamples?: number | undefined;
  /**
   * Seed for a private, reproducible generator. When omitted the shared random
   * generator is used, so `setSeed` from `deepbox/random` makes the result
   * reproducible too.
   */
  seed?: number | undefined;
  /** Confidence level of the interval, in (0, 1) (default: 0.95). */
  confidenceLevel?: number | undefined;
}

/** Result of {@link bootstrap}. */
export interface BootstrapResult {
  /** The statistic evaluated on the original data. */
  estimate: number;
  /** Percentile confidence interval `[lower, upper]`. */
  ci: [number, number];
  /** The statistic on every resample, in ascending order. */
  samples: number[];
}

/**
 * Non-parametric bootstrap: resample a statistic.
 *
 * Draws `nResamples` samples of the same size as `data` with replacement,
 * evaluates `statFn` on each, and reports the percentile interval of those values
 * (linear interpolation between order statistics, like `numpy.percentile`). If
 * `statFn` returns NaN on any resample the interval is `[NaN, NaN]`.
 *
 * @param data - 1-D array of observations
 * @param statFn - Function that computes a statistic from a sample. It receives a fresh array
 *   on every call, so it may modify it.
 * @param options - nResamples (default 1000), seed (optional), confidenceLevel (default 0.95)
 * @returns Object with estimate, ci (confidence interval), and samples array
 * @throws {InvalidParameterError} If `data` is empty, `nResamples` is not a positive integer,
 *   `confidenceLevel` is not in (0, 1), or `seed` is not finite
 *
 * @example
 * ```ts
 * const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
 * const { estimate, ci } = bootstrap([2, 4, 4, 5, 7, 9], mean, { seed: 42 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function bootstrap(
  data: readonly number[],
  statFn: (sample: number[]) => number,
  options: BootstrapOptions = {}
): BootstrapResult {
  const nResamples = options.nResamples ?? 1000;
  const confidenceLevel = options.confidenceLevel ?? 0.95;
  const n = data.length;
  if (n === 0) {
    throw new InvalidParameterError("bootstrap requires non-empty data", "data");
  }
  if (!Number.isInteger(nResamples) || nResamples < 1) {
    throw new InvalidParameterError(
      "nResamples must be a positive integer",
      "nResamples",
      nResamples
    );
  }
  if (!(confidenceLevel > 0 && confidenceLevel < 1)) {
    throw new InvalidParameterError(
      "confidenceLevel must be in the open interval (0, 1)",
      "confidenceLevel",
      confidenceLevel
    );
  }
  if (options.seed !== undefined && !Number.isFinite(options.seed)) {
    throw new InvalidParameterError("seed must be a finite number", "seed", options.seed);
  }

  // Seeded runs use a small private LCG so results are reproducible per seed;
  // unseeded runs draw from the shared generator (which honors `setSeed`).
  let nextRng: () => number;
  if (options.seed === undefined) {
    nextRng = __random;
  } else {
    let rngState = Math.trunc(options.seed) >>> 0;
    nextRng = () => {
      rngState = (rngState * 1664525 + 1013904223) & 0x7fffffff;
      // Divide by 2^31 (not 2^31 - 1) so the result stays strictly below 1.
      return rngState / 0x80000000;
    };
  }

  const samples = new Float64Array(nResamples);
  for (let r = 0; r < nResamples; r++) {
    const resample: number[] = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      resample[i] = data[Math.floor(nextRng() * n)] as number;
    }
    samples[r] = statFn(resample);
  }

  // Typed-array sort is numeric and places NaN last.
  samples.sort();
  const estimate = statFn(data.slice());
  // Percentile interval with linear interpolation (NumPy's default), symmetric in
  // both tails. A NaN statistic on any resample makes the interval undefined.
  const alpha = 1 - confidenceLevel;
  const hasNaN = Number.isNaN(samples[nResamples - 1] as number);
  const lo = hasNaN ? Number.NaN : quantileOfSorted(samples, alpha / 2);
  const hi = hasNaN ? Number.NaN : quantileOfSorted(samples, 1 - alpha / 2);

  return { estimate, ci: [lo, hi], samples: Array.from(samples) };
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
 * @throws {InvalidParameterError} If tensor is empty, the reduction is over an empty axis, or it
 *   has no more than `ddof` observations
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * sem(t);  // Returns std(t, ddof=1) / sqrt(5) ≈ 0.7071
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function sem(t: Tensor, axis?: AxisLike, ddof = 1): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  const count = axes.length === 0 ? t.size : axes.reduce((acc, ax) => acc * (t.shape[ax] ?? 0), 1);
  if (count === 0) {
    throw new InvalidParameterError(
      axes.length === 0
        ? "sem() requires at least one element"
        : "sem() reduction over empty axis is undefined",
      axes.length === 0 ? "size" : "axis",
      count
    );
  }
  if (count <= ddof) {
    throw new InvalidParameterError(
      `sem() requires more than ddof=${ddof} observations`,
      "ddof",
      ddof
    );
  }
  const v = reduceVariance(t, axis, false, ddof);
  const root = Math.sqrt(count);
  const out = new Float64Array(v.size);
  for (let i = 0; i < v.size; i++) {
    out[i] = Math.sqrt(getNumberAt(v, v.offset + i)) / root;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: v.shape,
    dtype: "float64",
    device: v.device,
  });
}

/**
 * Computes the interquartile range (IQR).
 *
 * IQR = Q3 - Q1 = percentile(75) - percentile(25), with linear interpolation
 * between order statistics (NumPy's default).
 *
 * The IQR is a measure of spread that is less sensitive to outliers
 * than the standard deviation.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute IQR (undefined = all axes)
 * @returns Tensor containing IQR values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
 * iqr(t);  // Returns 4.5 (a scalar tensor): Q3 (7.75) - Q1 (3.25)
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
    let arr = fullReduceDenseF64(t);
    if (arr === null) {
      arr = new Float64Array(t.size);
      let w = 0;
      const gathered = arr;
      forEachIndexOffset(t, (off) => {
        gathered[w++] = getNumberAt(t, off);
      });
    }
    let value: number;
    if (arr.some((v) => Number.isNaN(v))) {
      value = Number.NaN;
    } else {
      arr.sort();
      value = quantileOfSorted(arr, 0.75) - quantileOfSorted(arr, 0.25);
    }
    return Tensor.fromTypedArray({
      data: new Float64Array([value]),
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
    out[i] = quantileOfSorted(arr, 0.75) - quantileOfSorted(arr, 0.25);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}
