import { InvalidParameterError } from "../core";
import { Tensor, tensor } from "../ndarray";
import {
  type AxisLike,
  computeStrides,
  forEachIndexOffset,
  getNumberAt,
  normalizeAxes,
  reducedShape,
  reduceMean,
  reduceVariance,
} from "./_internal";

/**
 * Computes the arithmetic mean along specified axes.
 *
 * The mean is the sum of all values divided by the count.
 * Supports axis-wise reduction with optional dimension preservation.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the mean (undefined = all axes)
 * @param _keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @returns Tensor containing mean values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2, 3], [4, 5, 6]]);
 * mean(t);           // Returns tensor([3.5]) - mean of all elements
 * mean(t, 0);        // Returns tensor([2.5, 3.5, 4.5]) - column means
 * mean(t, 1);        // Returns tensor([2, 5]) - row means
 * mean(t, 1, true);  // Returns tensor([[2], [5]]) - keepdims
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
export function mean(t: Tensor, axis?: AxisLike, _keepdims = false): Tensor {
  return reduceMean(t, axis, _keepdims);
}

/**
 * Computes the median (50th percentile) along specified axes.
 *
 * The median is the middle value when data is sorted. For even-sized arrays,
 * it's the average of the two middle values. More robust to outliers than mean.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the median (undefined = all axes)
 * @param _keepdims - If true, reduced axes are retained with size 1 (default: false)
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
export function median(t: Tensor, axis?: AxisLike, _keepdims = false): Tensor {
  const axes = normalizeAxes(axis, t.ndim);

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("median() requires at least one element", "size", t.size);
    }
    const values: number[] = [];
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const v = getNumberAt(t, off);
      if (Number.isNaN(v)) hasNaN = true;
      values.push(v);
    });
    const outShape = _keepdims ? new Array<number>(t.ndim).fill(1) : [];
    if (hasNaN) {
      const out = new Float64Array(1);
      out[0] = Number.NaN;
      return Tensor.fromTypedArray({
        data: out,
        shape: outShape,
        dtype: "float64",
        device: t.device,
      });
    }
    values.sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    const result =
      values.length % 2 === 0
        ? ((values[mid - 1] ?? 0) + (values[mid] ?? 0)) / 2
        : (values[mid] ?? 0);
    const out = new Float64Array(1);
    out[0] = result;
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, _keepdims);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const buckets: (number[] | undefined)[] = new Array(outSize);
  const nanFlags = new Array<boolean>(outSize).fill(false);

  const reduce = new Set(axes);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    if (_keepdims) {
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
 * @param _keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @param ddof - Delta degrees of freedom (0 = population, 1 = sample, default: 0)
 * @returns Tensor containing standard deviation values
 * @throws {InvalidParameterError} If tensor is empty, ddof < 0, ddof >= sample size, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * std(t);        // Population std (ddof=0)
 * std(t, 0, false, 1);  // Sample std (ddof=1)
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Infinity inputs result in NaN (infinite standard deviation)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function std(t: Tensor, axis?: AxisLike, _keepdims = false, ddof = 0): Tensor {
  const v = reduceVariance(t, axis, _keepdims, ddof);
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
 * @param _keepdims - If true, reduced axes are retained with size 1 (default: false)
 * @param ddof - Delta degrees of freedom (0 = population, 1 = sample, default: 0)
 * @returns Tensor containing variance values
 * @throws {InvalidParameterError} If tensor is empty, ddof < 0, ddof >= sample size, or reduction over empty axis
 * @throws {IndexError} If axis is out of bounds
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * variance(t);              // Population variance: 2.0
 * variance(t, 0, false, 1); // Sample variance: 2.5
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN output
 * - Infinity inputs result in NaN (infinite variance)
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Descriptive Statistics}
 */
export function variance(t: Tensor, axis?: AxisLike, _keepdims = false, ddof = 0): Tensor {
  return reduceVariance(t, axis, _keepdims, ddof);
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
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute skewness (undefined = all axes)
 * @param bias - If false, applies the unbiased Fisher-Pearson correction (default: true)
 * @returns Tensor containing skewness values
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * skewness(t);  // Returns ~0 (symmetric)
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
export function skewness(t: Tensor, axis?: AxisLike, bias = true): Tensor {
  const mu = reduceMean(t, axis, false);
  const sigma2 = reduceVariance(t, axis, false, 0);

  const axes = normalizeAxes(axis, t.ndim);
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
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute kurtosis (undefined = all axes)
 * @param fisher - If true, returns excess kurtosis (subtract 3, default: true)
 * @param bias - If false, applies bias correction (requires at least 4 samples, default: true)
 * @returns Tensor containing kurtosis values
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * kurtosis(t, undefined, true);  // Excess kurtosis (Fisher)
 * kurtosis(t, undefined, false); // Raw kurtosis (Pearson)
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
export function kurtosis(t: Tensor, axis?: AxisLike, fisher = true, bias = true): Tensor {
  const mu = reduceMean(t, axis, false);
  const sigma2 = reduceVariance(t, axis, false, 0);

  const axes = normalizeAxes(axis, t.ndim);
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
    const arr: number[] = [];
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const v = getNumberAt(t, off);
      if (Number.isNaN(v)) hasNaN = true;
      arr.push(v);
    });
    if (hasNaN) {
      return tensor(qVals.map(() => Number.NaN));
    }
    arr.sort((a, b) => a - b);
    const results: number[] = [];
    for (const qVal of qVals) {
      const idx = qVal * (arr.length - 1);
      const lower = Math.floor(idx);
      const upper = Math.ceil(idx);
      const weight = idx - lower;
      results.push((arr[lower] ?? 0) * (1 - weight) + (arr[upper] ?? 0) * weight);
    }
    return tensor(results);
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
export function geometricMean(t: Tensor, axis?: AxisLike): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0 && t.size === 0) {
    throw new InvalidParameterError(
      "geometricMean() requires at least one element",
      "size",
      t.size
    );
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
    const arr: number[] = [];
    let hasNaN = false;
    forEachIndexOffset(t, (off) => {
      const v = getNumberAt(t, off);
      if (Number.isNaN(v)) hasNaN = true;
      arr.push(v);
    });
    if (arr.length === 0)
      throw new InvalidParameterError(
        "trimMean() requires at least one element",
        "size",
        arr.length
      );
    if (hasNaN) {
      return tensor([Number.NaN]);
    }
    arr.sort((a, b) => a - b);
    const nTrim = Math.floor(arr.length * proportiontocut);
    const trimmed = arr.slice(nTrim, arr.length - nTrim);
    const sum = trimmed.reduce((a, b) => a + b, 0);
    return tensor([sum / trimmed.length]);
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
