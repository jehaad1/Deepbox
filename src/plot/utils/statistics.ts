/**
 * Statistical utilities for plot calculations.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

/**
 * Linear interpolation written the way NumPy's `lerp` is, so results agree with
 * `numpy.percentile` to the last bit: start from the nearer end point.
 */
function lerp(a: number, b: number, t: number): number {
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

/**
 * Calculates quartiles using linear interpolation between the closest ranks
 * (NumPy's default and matplotlib's boxplot default), so box extents and the
 * 1.5·IQR whiskers match matplotlib. (This is NOT Excel's/Tukey's hinge
 * method, which the previous median-of-medians implementation used.)
 *
 * `sortedData` must already be sorted ascending and free of NaN. An empty input returns zeros.
 * @internal
 */
export function calculateQuartiles(sortedData: ArrayLike<number>): {
  readonly q1: number;
  readonly median: number;
  readonly q3: number;
} {
  const n = sortedData.length;
  if (n === 0) {
    return { q1: 0, median: 0, q3: 0 };
  }
  if (n === 1) {
    const val = sortedData[0] ?? 0;
    return { q1: val, median: val, q3: val };
  }

  // Linear-interpolation percentile (numpy 'linear' / matplotlib default).
  const percentile = (p: number): number => {
    const idx = (p / 100) * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(n - 1, lo + 1);
    return lerp(sortedData[lo] ?? 0, sortedData[hi] ?? 0, idx - lo);
  };

  return { q1: percentile(25), median: percentile(50), q3: percentile(75) };
}

/**
 * Calculates whiskers for boxplot using 1.5 * IQR rule.
 * Whiskers extend to the most extreme data points within 1.5 * IQR of Q1/Q3.
 * Points beyond whiskers are classified as outliers. NaN values are ignored.
 *
 * With very uneven data the most extreme in-fence value can lie inside the box (for example
 * `[1, 2, 3, 100]` has q3 = 27.25 but its largest in-fence value is 3). A whisker never ends
 * inside the box, so the lower whisker is at most `q1` and the upper whisker at least `q3`, as
 * in matplotlib. When no value is inside the fences both whiskers equal the quartiles.
 * @internal
 */
export function calculateWhiskers(
  sortedData: ArrayLike<number>,
  q1: number,
  q3: number
): {
  readonly lowerWhisker: number;
  readonly upperWhisker: number;
  readonly outliers: readonly number[];
} {
  if (sortedData.length === 0) {
    return { lowerWhisker: 0, upperWhisker: 0, outliers: [] };
  }

  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  const outliers: number[] = [];
  let lowerWhisker: number | null = null;
  let upperWhisker: number | null = null;

  // Find whisker endpoints (most extreme values within bounds)
  // Since data is sorted, first non-outlier is lower whisker, last non-outlier is upper whisker
  for (let i = 0; i < sortedData.length; i++) {
    const value = sortedData[i] ?? Number.NaN;
    if (Number.isNaN(value)) continue;
    if (value < lowerBound || value > upperBound) {
      outliers.push(value);
    } else {
      // Within bounds - track as potential whisker
      if (lowerWhisker === null) {
        lowerWhisker = value;
      }
      upperWhisker = value;
    }
  }

  // If no values fall within the fences (all outliers), whiskers collapse to the quartiles.
  // Otherwise they never end inside the box.
  lowerWhisker = lowerWhisker === null ? q1 : Math.min(lowerWhisker, q1);
  upperWhisker = upperWhisker === null ? q3 : Math.max(upperWhisker, q3);

  return { lowerWhisker, upperWhisker, outliers };
}

/**
 * Rule-of-thumb bandwidth `factor(n) * s`, where `s` is the sample standard deviation
 * (n - 1 denominator) of the finite values. Falls back to a tenth of the data range when the
 * result is zero or undefined (a single value or constant data), and to 1 when the range is
 * zero too. Non-finite values are ignored. Always returns a finite, positive number.
 */
function ruleOfThumbBandwidth(data: ArrayLike<number>, factor: (n: number) => number): number {
  let n = 0;
  let mean = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? Number.NaN;
    if (!Number.isFinite(v)) continue;
    n++;
    mean += (v - mean) / n; // running mean avoids overflow for large values
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (n === 0) return 1;

  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? Number.NaN;
    if (!Number.isFinite(v)) continue;
    const d = v - mean;
    sumSq += d * d;
  }
  const stdDev = n > 1 ? Math.sqrt(sumSq / (n - 1)) : 0;
  const bandwidth = stdDev * factor(n);
  if (Number.isFinite(bandwidth) && bandwidth > 0) return bandwidth;

  const range = max - min;
  return Number.isFinite(range) && range > 0 ? range * 0.1 : 1;
}

/**
 * Silverman's rule-of-thumb bandwidth for a Gaussian kernel,
 * `(3n/4)^(-1/5) * s`, where `s` is the sample standard deviation (n - 1 denominator).
 * This is the factor SciPy's `gaussian_kde(bw_method="silverman")` uses in one dimension.
 *
 * Falls back to a tenth of the data range when the deviation is zero or undefined (a single
 * value or constant data), and to 1 when the range is zero too. Non-finite values are ignored.
 * Always returns a finite, positive number.
 * @internal
 */
export function silvermanBandwidth(data: ArrayLike<number>): number {
  return ruleOfThumbBandwidth(data, (n) => (0.75 * n) ** -0.2);
}

/**
 * Scott's rule-of-thumb bandwidth for a Gaussian kernel, `n^(-1/5) * s`, where `s` is the
 * sample standard deviation (n - 1 denominator). This is the factor SciPy's
 * `gaussian_kde(bw_method="scott")` and matplotlib's `violinplot` use in one dimension.
 * The fallbacks are those of {@link silvermanBandwidth}.
 * @internal
 */
export function scottBandwidth(data: ArrayLike<number>): number {
  return ruleOfThumbBandwidth(data, (n) => n ** -0.2);
}

/**
 * Gaussian kernel density estimate of `data`, evaluated at `points`.
 *
 * The result integrates to 1 over the real line. If `bandwidth` is not a positive finite
 * number, {@link silvermanBandwidth} is used. Non-finite data values are ignored; with no
 * usable data the density is 0 everywhere.
 * @param data - Sample values
 * @param points - Locations at which to evaluate the density
 * @param bandwidth - Kernel standard deviation in data units; pass 0 for the automatic choice
 * @internal
 */
export function kernelDensityEstimation(
  data: ArrayLike<number>,
  points: ArrayLike<number>,
  bandwidth: number
): readonly number[] {
  const m = points.length;
  const result = new Float64Array(m);

  const finite: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? Number.NaN;
    if (Number.isFinite(v)) finite.push(v);
  }
  const n = finite.length;
  if (n === 0) return Array.from(result);

  const h = Number.isFinite(bandwidth) && bandwidth > 0 ? bandwidth : silvermanBandwidth(finite);
  const norm = 1 / (h * Math.sqrt(2 * Math.PI) * n);

  for (let i = 0; i < m; i++) {
    const x = points[i] ?? Number.NaN;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const u = (x - (finite[j] ?? 0)) / h;
      sum += Math.exp(-0.5 * u * u);
    }
    result[i] = sum * norm;
  }

  return Array.from(result);
}
