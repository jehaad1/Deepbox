/**
 * Statistical utilities for plot calculations.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

/**
 * Calculates quartiles using linear interpolation between the closest ranks
 * (NumPy's default and matplotlib's boxplot default), so box extents and the
 * 1.5·IQR whiskers match matplotlib. (This is NOT Excel's/Tukey's hinge
 * method, which the previous median-of-medians implementation used.)
 * @internal
 */
export function calculateQuartiles(sortedData: readonly number[]): {
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
    const hi = Math.ceil(idx);
    const w = idx - lo;
    return (sortedData[lo] ?? 0) * (1 - w) + (sortedData[hi] ?? 0) * w;
  };

  return { q1: percentile(25), median: percentile(50), q3: percentile(75) };
}

/**
 * Calculates whiskers for boxplot using 1.5 * IQR rule.
 * Whiskers extend to the most extreme data points within 1.5 * IQR of Q1/Q3.
 * Points beyond whiskers are classified as outliers.
 * @internal
 */
export function calculateWhiskers(
  sortedData: readonly number[],
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
  for (const value of sortedData) {
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

  // If no values fall within the fences (all outliers), whiskers should collapse to the quartiles
  if (lowerWhisker === null) {
    lowerWhisker = q1;
  }
  if (upperWhisker === null) {
    upperWhisker = q3;
  }

  return { lowerWhisker, upperWhisker, outliers };
}

/**
 * Kernel density estimation for violin plots using Gaussian kernel.
 * @internal
 */
export function kernelDensityEstimation(
  data: readonly number[],
  points: readonly number[],
  bandwidth: number
): readonly number[] {
  const n = data.length;
  const m = points.length;
  const result = new Float64Array(m);

  // Return zeros for empty data
  if (n === 0) {
    return Array.from(result); // All zeros
  }

  // Silverman's rule of thumb for bandwidth if not provided
  if (!Number.isFinite(bandwidth) || bandwidth <= 0) {
    let sum = 0;
    let sumSq = 0;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      const v = data[i] ?? 0;
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const mean = sum / n;
    const variance = Math.max(0, sumSq / n - mean * mean);
    const stdDev = Math.sqrt(variance);
    const silverman = 1.06 * stdDev * n ** -0.2;
    if (Number.isFinite(silverman) && silverman > 0) {
      bandwidth = silverman;
    } else {
      const range = max - min;
      bandwidth = Number.isFinite(range) && range > 0 ? range * 0.1 : 1;
    }
  }
  if (!Number.isFinite(bandwidth) || bandwidth <= 0) {
    bandwidth = 1;
  }

  const invSqrt2PiBandwidth = 1 / (bandwidth * Math.sqrt(2 * Math.PI));

  for (let i = 0; i < m; i++) {
    const x = points[i] ?? 0;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const u = (x - (data[j] ?? 0)) / bandwidth;
      sum += Math.exp(-0.5 * u * u);
    }
    result[i] = (sum * invSqrt2PiBandwidth) / n;
  }

  return Array.from(result);
}
