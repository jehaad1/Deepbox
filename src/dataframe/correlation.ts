/**
 * Pairwise correlation coefficients used by DataFrame.corr.
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import type { CorrelationMethod } from "./types";
import { compensatedSum } from "./utils";

/** Rounding can push a coefficient a hair past +-1; keep it inside the valid range. */
const clamp = (r: number): number => (r > 1 ? 1 : r < -1 ? -1 : r);

/**
 * Pearson correlation of two equally long samples. NaN for fewer than two
 * observations or when either sample is constant.
 */
export const pearson = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
  const n = x.length;
  if (n < 2) return Number.NaN;
  const mx = compensatedSum(x) / n;
  const my = compensatedSum(y) / n;
  let num = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    const dx = (x[i] as number) - mx;
    const dy = (y[i] as number) - my;
    num += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  if (sx === 0 || sy === 0) return Number.NaN;
  return clamp(num / (Math.sqrt(sx) * Math.sqrt(sy)));
};

/** Index of the first element of the ascending `sorted` that is not less than `v`. */
const lowerBound = (sorted: Float64Array, v: number): number => {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((sorted[mid] as number) < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

/**
 * Average ranks (1-based, ties share the mean of their positions) of a sample. The values are
 * sorted once with the native numeric sort, and each rank is found by binary search.
 */
export const averageRanks = (values: ArrayLike<number>): Float64Array => {
  const n = values.length;
  const sorted = Float64Array.from(values).sort();
  // Average rank of the run of equal values that starts at, or contains, each sorted position.
  const rankAt = new Float64Array(n);
  let start = 0;
  while (start < n) {
    let end = start;
    const v = sorted[start] as number;
    while (end + 1 < n && (sorted[end + 1] as number) === v) end++;
    const rank = (start + end) / 2 + 1;
    for (let k = start; k <= end; k++) rankAt[k] = rank;
    start = end + 1;
  }
  const ranks = new Float64Array(n);
  for (let i = 0; i < n; i++) ranks[i] = rankAt[lowerBound(sorted, values[i] as number)] as number;
  return ranks;
};

/** Dense ranks (0-based; equal values share a rank, no gaps) of a sample. */
const denseRanks = (values: ArrayLike<number>): Float64Array => {
  const n = values.length;
  const sorted = Float64Array.from(values).sort();
  let unique = 0;
  for (let i = 0; i < n; i++) {
    if (i === 0 || (sorted[i] as number) !== (sorted[i - 1] as number)) {
      sorted[unique++] = sorted[i] as number;
    }
  }
  const levels = sorted.subarray(0, unique);
  const ranks = new Float64Array(n);
  for (let i = 0; i < n; i++) ranks[i] = lowerBound(levels, values[i] as number);
  return ranks;
};

/** Spearman rank correlation: Pearson correlation of the average ranks. */
export const spearman = (x: ArrayLike<number>, y: ArrayLike<number>): number =>
  pearson(averageRanks(x), averageRanks(y));

/** Sum of t(t-1)/2 over the runs of equal values in an ascending sample. */
const tiePairs = (sorted: ArrayLike<number>): number => {
  let total = 0;
  let run = 1;
  for (let i = 1; i <= sorted.length; i++) {
    if (i < sorted.length && sorted[i] === sorted[i - 1]) {
      run++;
    } else {
      total += (run * (run - 1)) / 2;
      run = 1;
    }
  }
  return total;
};

/** Counts the pairs i < j with seq[i] > seq[j] by merge sort; sorts `seq` in place. */
const countInversions = (seq: Float64Array): number => {
  const n = seq.length;
  let src: Float64Array = seq;
  let dst: Float64Array = new Float64Array(n);
  let inversions = 0;
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n);
      const hi = Math.min(lo + 2 * width, n);
      let i = lo;
      let j = mid;
      let k = lo;
      while (i < mid && j < hi) {
        if ((src[i] as number) <= (src[j] as number)) {
          dst[k++] = src[i++] as number;
        } else {
          inversions += mid - i;
          dst[k++] = src[j++] as number;
        }
      }
      while (i < mid) dst[k++] = src[i++] as number;
      while (j < hi) dst[k++] = src[j++] as number;
    }
    const swap = src;
    src = dst;
    dst = swap;
  }
  return inversions;
};

/**
 * Kendall's tau-b of two equally long samples in O(n log n) (Knight's algorithm).
 * NaN for fewer than two observations or when either sample is constant.
 *
 * Both samples are replaced by their dense ranks, which keeps every tie. The pairs are then
 * ordered by sorting the packed keys `rankX * n + rankY` with the native numeric sort.
 */
export const kendall = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
  const n = x.length;
  if (n < 2) return Number.NaN;
  const rx = denseRanks(x);
  const ry = denseRanks(y);
  const keys = new Float64Array(n);
  for (let i = 0; i < n; i++) keys[i] = (rx[i] as number) * n + (ry[i] as number);
  keys.sort();
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const key = keys[i] as number;
    const high = Math.floor(key / n);
    xs[i] = high;
    ys[i] = key - high * n;
  }
  // Pairs tied in both x and y: runs of equal keys.
  let joint = 0;
  let run = 1;
  for (let i = 1; i <= n; i++) {
    if (i < n && keys[i] === keys[i - 1]) {
      run++;
    } else {
      joint += (run * (run - 1)) / 2;
      run = 1;
    }
  }
  const tiesX = tiePairs(xs);
  const discordant = countInversions(Float64Array.from(ys));
  const tiesY = tiePairs(Float64Array.from(ys).sort());
  const total = (n * (n - 1)) / 2;
  const denom = Math.sqrt((total - tiesX) * (total - tiesY));
  if (denom === 0) return Number.NaN;
  return clamp((total - tiesX - tiesY + joint - 2 * discordant) / denom);
};

/** Correlation of two equally long, NaN-free samples by the named method. */
export const correlate = (
  method: CorrelationMethod,
  x: ArrayLike<number>,
  y: ArrayLike<number>
): number => {
  if (method === "spearman") return spearman(x, y);
  if (method === "kendall") return kendall(x, y);
  return pearson(x, y);
};
