/**
 * Correlation and covariance measures.
 *
 * @module stats/correlation
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */

import { InvalidParameterError, ShapeError } from "../core";
import { Tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import {
  assertSameSize,
  copyContiguousToF64,
  correlationPValue,
  forEachIndexOffset,
  getNumberAt,
  normalSf,
  rankData,
} from "./_internal";

/**
 * Converts a tensor to a dense flat Float64Array (row-major).
 * Helper function for correlation computations.
 *
 * @param t - Input tensor
 * @returns Flattened Float64Array containing all tensor values
 */
function toDenseFlatArray(t: Tensor): Float64Array {
  const raw = t.data;
  if (!Array.isArray(raw) && !(raw instanceof BigInt64Array) && isContiguous(t.shape, t.strides)) {
    return copyContiguousToF64(raw, t.offset, t.size);
  }
  const out = new Float64Array(t.size);
  let i = 0;
  forEachIndexOffset(t, (off) => {
    out[i] = getNumberAt(t, off);
    i++;
  });
  return out;
}

/** True if any element is NaN. */
function containsNaN(a: Float64Array): boolean {
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i] as number)) return true;
  }
  return false;
}

/**
 * Magnitudes beyond these bounds are rescaled before squaring, so sums of
 * squared deviations neither overflow nor underflow.
 */
const SCALE_HIGH = 1e50;
const SCALE_LOW = 1e-50;

/** Alternative hypothesis of a correlation test, as in `scipy.stats`. */
export type CorrelationAlternative = "two-sided" | "less" | "greater";

/** Options shared by the correlation tests that return a p-value. */
export interface CorrelationTestOptions {
  /**
   * Alternative hypothesis. `"two-sided"` (default) tests for a nonzero
   * correlation, `"greater"` for a positive one and `"less"` for a negative one.
   */
  readonly alternative?: CorrelationAlternative;
}

/** Options for {@link kendalltau}. */
export interface KendallTauOptions extends CorrelationTestOptions {
  /**
   * `"b"` (default) divides by the geometric mean of the untied pair counts and
   * so accounts for ties in both samples. `"c"` is Stuart's tau-c, which is
   * meant for rank tables with a different number of categories per variable.
   */
  readonly variant?: "b" | "c";
  /**
   * How the p-value is computed. `"auto"` (default) uses the exact permutation
   * distribution for small untied samples and the normal approximation
   * otherwise. `"exact"` requires untied input. `"asymptotic"` always uses the
   * normal approximation.
   */
  readonly method?: "auto" | "exact" | "asymptotic";
}

function resolveAlternative(fn: string, options: CorrelationTestOptions): CorrelationAlternative {
  const alternative: unknown = options.alternative ?? "two-sided";
  if (alternative !== "two-sided" && alternative !== "less" && alternative !== "greater") {
    throw new InvalidParameterError(
      `${fn}() alternative must be "two-sided", "less" or "greater"`,
      "alternative",
      alternative
    );
  }
  return alternative;
}

/**
 * p-value of correlation `r` under t(df) for the given alternative, derived from
 * the two-sided value (the null distribution is symmetric about 0). `r = 0`
 * gives 0.5 for the one-sided alternatives.
 */
function correlationPValueFor(r: number, df: number, alternative: CorrelationAlternative): number {
  const twoSided = correlationPValue(r, df);
  if (alternative === "two-sided" || Number.isNaN(twoSided)) return twoSided;
  if (r === 0) return 0.5;
  const inDirection = alternative === "greater" ? r > 0 : r < 0;
  return inDirection ? twoSided / 2 : 1 - twoSided / 2;
}

/**
 * Computes Pearson correlation coefficient from two dense arrays.
 *
 * Uses the two-pass formula r = Σ(dx·dy) / sqrt(Σdx² · Σdy²). Inputs with
 * extreme magnitudes (above 1e50 or below 1e-50) are rescaled first, since
 * r is scale invariant. The result is clamped to [-1, 1].
 *
 * @param x - First array
 * @param y - Second array (must have same length as x)
 * @param fnName - Public function name used in error messages
 * @returns Pearson correlation coefficient in [-1, 1]; NaN if any value is NaN or infinite
 * @throws {InvalidParameterError} If either array has constant values (zero variance)
 */
function pearsonFromDense(x: Float64Array, y: Float64Array, fnName: string): number {
  const n = x.length;
  const x0 = x[0] as number;
  const y0 = y[0] as number;
  let constX = true;
  let constY = true;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] as number;
    const yi = y[i] as number;
    if (!Number.isFinite(xi) || !Number.isFinite(yi)) return Number.NaN;
    if (xi !== x0) constX = false;
    if (yi !== y0) constY = false;
    const ax = Math.abs(xi);
    const ay = Math.abs(yi);
    if (ax > maxX) maxX = ax;
    if (ay > maxY) maxY = ay;
  }
  // An exact equality test: a mean-based test would flag e.g. [0.1, 0.1, 0.1]
  // as varying because its computed mean is off by one ulp.
  if (constX || constY) {
    throw new InvalidParameterError(
      `${fnName}() is undefined for constant input`,
      "input",
      "constant"
    );
  }
  const sx = maxX > SCALE_HIGH || maxX < SCALE_LOW ? 1 / maxX : 1;
  const sy = maxY > SCALE_HIGH || maxY < SCALE_LOW ? 1 / maxY : 1;

  // First pass: compute means
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += (x[i] as number) * sx;
    sumY += (y[i] as number) * sy;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  // Second pass: compute covariance and variances
  let num = 0; // Numerator: covariance
  let denX = 0; // Denominator: variance of X
  let denY = 0; // Denominator: variance of Y
  for (let i = 0; i < n; i++) {
    const dx = (x[i] as number) * sx - meanX;
    const dy = (y[i] as number) * sy - meanY;
    num += dx * dy; // Sum of products of deviations
    denX += dx * dx; // Sum of squared deviations for X
    denY += dy * dy; // Sum of squared deviations for Y
  }

  // After the optional rescale denX * denY can neither overflow nor underflow.
  const den = Math.sqrt(denX * denY);
  if (den === 0) {
    throw new InvalidParameterError(
      `${fnName}() is undefined for constant input`,
      "input",
      "constant"
    );
  }
  const r = num / den;
  return r > 1 ? 1 : r < -1 ? -1 : r;
}

/**
 * Pearson correlation coefficient.
 *
 * Measures linear correlation between two variables.
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @param options - `alternative` hypothesis of the test (default `"two-sided"`)
 * @returns Tuple of [correlation coefficient in [-1, 1], p-value]
 * @throws {InvalidParameterError} If tensors have different sizes, < 2 samples, constant input,
 *   or an invalid `alternative`
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 6, 8, 10]);
 * const [r, p] = pearsonr(x, y);  // r = 1.0 (perfect linear)
 *
 * const [r2, pGreater] = pearsonr(x, y, { alternative: "greater" });
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN correlation and NaN p-value
 * - Infinity inputs result in NaN correlation and NaN p-value
 *
 * The p-value is the tail of the t-distribution with n - 2 degrees of freedom
 * (the same value `scipy.stats.pearsonr` returns). For n = 2 the coefficient is
 * exactly -1 or 1 and the p-value is 1, as in SciPy. Tensors are flattened, so
 * both inputs are treated as 1D samples in row-major order.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function pearsonr(
  x: Tensor,
  y: Tensor,
  options: CorrelationTestOptions = {}
): [number, number] {
  const alternative = resolveAlternative("pearsonr", options);
  assertSameSize(x, y, "pearsonr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("pearsonr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);
  const r = pearsonFromDense(xd, yd, "pearsonr");

  // Under H0: rho = 0, the test statistic t = r*sqrt((n-2)/(1-r^2)) follows t(n-2)
  if (n === 2) {
    // Two points always lie on a line: no degrees of freedom are left.
    return Number.isNaN(r) ? [r, Number.NaN] : [Math.round(r), 1];
  }
  return [r, correlationPValueFor(r, n - 2, alternative)];
}

/**
 * Computes Spearman's rank correlation coefficient.
 *
 * Non-parametric measure of monotonic relationship between two variables.
 * Computed as Pearson correlation of rank values.
 * - rho = 1: Perfect monotonic increasing relationship
 * - rho = 0: No monotonic relationship
 * - rho = -1: Perfect monotonic decreasing relationship
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @param options - `alternative` hypothesis of the test (default `"two-sided"`)
 * @returns Tuple of [correlation coefficient, p-value]
 * @throws {InvalidParameterError} If tensors have different sizes, < 2 samples, constant input,
 *   or an invalid `alternative`
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 6, 8, 10]);
 * const [rho, p] = spearmanr(x, y);  // rho = 1.0 (perfect monotonic)
 * ```
 *
 * @remarks
 * Ties are assigned average ranks. If either input contains NaN, both the
 * coefficient and the p-value are NaN (as in `scipy.stats.spearmanr`).
 * Positive and negative infinity are ordinary extreme values. The p-value uses
 * the t-distribution with n - 2 degrees of freedom and is NaN when n = 2.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function spearmanr(
  x: Tensor,
  y: Tensor,
  options: CorrelationTestOptions = {}
): [number, number] {
  const alternative = resolveAlternative("spearmanr", options);
  assertSameSize(x, y, "spearmanr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("spearmanr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);
  if (containsNaN(xd) || containsNaN(yd)) return [Number.NaN, Number.NaN];

  // Convert values to average ranks (1-indexed) with tie correction.
  const rankX = rankData(xd).ranks;
  const rankY = rankData(yd).ranks;

  // Compute Pearson correlation of ranks
  const rho = pearsonFromDense(rankX, rankY, "spearmanr");
  // Test statistic follows t-distribution under H0: rho = 0
  const df = n - 2;
  return [rho, df > 0 ? correlationPValueFor(rho, df, alternative) : Number.NaN];
}

/**
 * Kendall tie summary for one sample: number of tied pairs (sum of t(t-1)/2),
 * the sums over tie groups of t(t-1), t(t-1)(2t+5) and t(t-1)(t-2), and the
 * number of distinct values.
 */
function tieSummary(vals: Float64Array): {
  pairs: number;
  sumT: number;
  sumT2: number;
  sumT3: number;
  distinct: number;
} {
  const sorted = Float64Array.from(vals).sort();
  let sumT = 0;
  let sumT2 = 0;
  let sumT3 = 0;
  let distinct = 0;
  for (let i = 0; i < sorted.length; ) {
    let j = i + 1;
    while (j < sorted.length && sorted[j] === sorted[i]) j++;
    const t = j - i;
    distinct++;
    if (t > 1) {
      sumT += t * (t - 1);
      sumT2 += t * (t - 1) * (2 * t + 5);
      sumT3 += t * (t - 1) * (t - 2);
    }
    i = j;
  }
  return { pairs: sumT / 2, sumT, sumT2, sumT3, distinct };
}

/**
 * Number of discordant pairs (strict order reversals between x and y) among
 * pairs untied in x, plus the number of pairs tied in both x and y, computed in
 * O(n log n) via Knight's algorithm (merge-sort inversion count). Combined with
 * the per-variable tied-pair counts this gives Kendall's
 * S = (concordant - discordant) = tot - xtie - ytie + ntie - 2*dis.
 *
 * @internal
 */
function kendallDiscordant(
  xd: Float64Array,
  yd: Float64Array,
  n: number
): { dis: number; ntie: number } {
  // Order by (x asc, then y asc). Inputs are NaN-free here.
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => {
    const xa = xd[a] as number;
    const xb = xd[b] as number;
    if (xa < xb) return -1;
    if (xa > xb) return 1;
    const ya = yd[a] as number;
    const yb = yd[b] as number;
    return ya < yb ? -1 : ya > yb ? 1 : 0;
  });

  // Pairs tied in both x and y: runs of identical (x, y) in the sorted order.
  let ntie = 0;
  for (let i = 0; i < n; ) {
    const a = order[i] as number;
    let j = i + 1;
    while (j < n && xd[order[j] as number] === xd[a] && yd[order[j] as number] === yd[a]) {
      j++;
    }
    const t = j - i;
    if (t > 1) ntie += (t * (t - 1)) / 2;
    i = j;
  }

  // Discordant pairs = strict inversions in the y-sequence taken in (x,y) order.
  const yByX = new Float64Array(n);
  for (let i = 0; i < n; i++) yByX[i] = yd[order[i] as number] as number;
  return { dis: countInversions(yByX, n), ntie };
}

/** Count strict inversions (a[i] > a[j], i < j) via bottom-up merge sort. */
function countInversions(a: Float64Array, n: number): number {
  if (n < 2) return 0;
  let buf = new Float64Array(a);
  let tmp = new Float64Array(n);
  let inv = 0;
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n);
      const hi = Math.min(lo + 2 * width, n);
      let i = lo;
      let j = mid;
      let k = lo;
      while (i < mid && j < hi) {
        if ((buf[i] as number) <= (buf[j] as number)) {
          tmp[k++] = buf[i++] as number;
        } else {
          // buf[i] > buf[j]: buf[j] jumps ahead of the (mid - i) remaining
          // left-run elements, each of which is > buf[j], so that many inversions.
          inv += mid - i;
          tmp[k++] = buf[j++] as number;
        }
      }
      while (i < mid) tmp[k++] = buf[i++] as number;
      while (j < hi) tmp[k++] = buf[j++] as number;
    }
    const swap = buf;
    buf = tmp;
    tmp = swap;
  }
  return inv;
}

/**
 * P(D <= k) where D is the number of inversions of a uniformly random
 * permutation of `n` items (the null distribution of Kendall's discordant-pair
 * count when there are no ties; Kendall, "Rank Correlation Methods", 4th ed.,
 * 1970). The distribution is symmetric about n(n-1)/4, so only the half up to
 * the mean is built, by dynamic programming over probabilities (not counts, so
 * nothing overflows for large `n`). Cost is O(n * min(k, n(n-1)/2 - k)).
 */
function inversionCdf(n: number, k: number): number {
  const tot = (n * (n - 1)) / 2;
  if (k < 0) return 0;
  if (k >= tot) return 1;
  if (2 * k > tot) return 1 - inversionCdf(n, tot - k - 1);

  // probs[j] = P(D = j) for the current number of items.
  let probs = new Float64Array(k + 1);
  let next = new Float64Array(k + 1);
  probs[0] = 1;
  for (let size = 2; size <= n; size++) {
    // Adding an item adds 0..size-1 new inversions with equal probability.
    let window = 0;
    for (let j = 0; j <= k; j++) {
      window += probs[j] as number;
      if (j >= size) window -= probs[j - size] as number;
      next[j] = window / size;
    }
    const swap = probs;
    probs = next;
    next = swap;
  }
  let sum = 0;
  for (let j = 0; j <= k; j++) sum += probs[j] as number;
  return Math.min(1, sum);
}

/**
 * Exact p-value of Kendall's tau for `n` observations with no ties, given the
 * number of discordant pairs. Same definition as
 * `scipy.stats.kendalltau(method="exact")`: for the two-sided test, 2·P(D <= c)
 * with c = min(discordant, n(n-1)/2 - discordant), capped at 1.
 */
function kendallExactPValue(
  n: number,
  discordant: number,
  alternative: CorrelationAlternative
): number {
  const tot = (n * (n - 1)) / 2;
  if (alternative === "greater") return inversionCdf(n, discordant);
  if (alternative === "less") return inversionCdf(n, tot - discordant);
  return Math.min(1, 2 * inversionCdf(n, Math.min(discordant, tot - discordant)));
}

/** Largest dynamic-programming workload (items times inversion count) allowed for `method: "exact"`. */
const KENDALL_EXACT_MAX_WORK = 5e8;

/**
 * Computes Kendall's tau correlation coefficient.
 *
 * Non-parametric measure of ordinal association based on concordant/discordant pairs.
 * Less sensitive to outliers than Pearson, and with a simple probabilistic interpretation.
 * - tau = 1: All pairs concordant (perfect agreement)
 * - tau = 0: Equal concordant and discordant pairs
 * - tau = -1: All pairs discordant (perfect disagreement)
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @param options - `alternative` (default `"two-sided"`), `variant` (`"b"`, default, or
 *   `"c"`) and `method` (`"auto"`, default, `"exact"` or `"asymptotic"`)
 * @returns Tuple of [tau coefficient, p-value]
 * @throws {InvalidParameterError} If tensors have different sizes or < 2 samples, an option
 *   is invalid, `method` is `"exact"` but there are ties, or the exact computation is too large
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([1, 3, 2, 4, 5]);
 * const [tau, p] = kendalltau(x, y);  // tau = 0.8, p = 0.0833 (exact)
 * ```
 *
 * @remarks
 * The default is the tau-b variant with tie correction. Ties are excluded from
 * concordant/discordant counts and reduce the denominator.
 *
 * With `method: "auto"` the p-value follows `scipy.stats.kendalltau`: when there are
 * no ties and either n <= 33 or at most one pair is discordant (or concordant), it
 * is the exact p-value of the permutation distribution; otherwise it
 * is a normal approximation with tie-corrected variance. The p-value is the same
 * for both variants. If either input contains NaN, tau and the p-value are NaN.
 * If one input is constant, tau and the p-value are NaN as well.
 *
 * @complexity O(n log n) via Knight's merge-sort algorithm for the
 * concordant-minus-discordant score (the exact p-value adds O(n · c) work for
 * n <= 33).
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function kendalltau(
  x: Tensor,
  y: Tensor,
  options: KendallTauOptions = {}
): [number, number] {
  const alternative = resolveAlternative("kendalltau", options);
  const variant: unknown = options.variant ?? "b";
  if (variant !== "b" && variant !== "c") {
    throw new InvalidParameterError(
      `kendalltau() variant must be "b" or "c"; received ${String(variant)}`,
      "variant",
      variant
    );
  }
  const method: unknown = options.method ?? "auto";
  if (method !== "auto" && method !== "exact" && method !== "asymptotic") {
    throw new InvalidParameterError(
      `kendalltau() method must be "auto", "exact" or "asymptotic"; received ${String(method)}`,
      "method",
      method
    );
  }
  assertSameSize(x, y, "kendalltau");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("kendalltau() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);
  if (containsNaN(xd) || containsNaN(yd)) return [Number.NaN, Number.NaN];

  const tieX = tieSummary(xd);
  const tieY = tieSummary(yd);
  const { dis, ntie } = kendallDiscordant(xd, yd, n);

  // s = concordant - discordant over pairs untied in both x and y, by
  // inclusion-exclusion: s = tot - xtie - ytie + ntie - 2*dis, where tot =
  // n(n-1)/2 and xtie/ytie/ntie are the tied-pair counts in x, y and jointly.
  const n0 = (n * (n - 1)) / 2;
  const s = n0 - tieX.pairs - tieY.pairs + ntie - 2 * dis;

  const denomB = Math.sqrt((n0 - tieX.pairs) * (n0 - tieY.pairs));
  if (denomB === 0) return [Number.NaN, Number.NaN];
  let tau: number;
  if (variant === "b") {
    tau = s / denomB;
  } else {
    const m = Math.min(tieX.distinct, tieY.distinct);
    tau = (2 * s) / ((n * n * (m - 1)) / m);
  }
  tau = Math.max(-1, Math.min(1, tau));

  const untied = tieX.pairs === 0 && tieY.pairs === 0;
  if (method === "exact" && !untied) {
    throw new InvalidParameterError(
      'kendalltau() method "exact" cannot be used when there are ties; use "auto" or "asymptotic"',
      "method",
      method
    );
  }
  const useExact =
    untied &&
    (method === "exact" || (method === "auto" && (n <= 33 || Math.min(dis, n0 - dis) <= 1)));
  if (useExact) {
    if (method === "exact" && n * Math.min(dis, n0 - dis) > KENDALL_EXACT_MAX_WORK) {
      throw new InvalidParameterError(
        'kendalltau() method "exact" is too expensive for this sample size; use "asymptotic"',
        "n",
        n
      );
    }
    return [tau, kendallExactPValue(n, dis, alternative)];
  }

  // Normal approximation variance with tie correction (standard).
  let varS =
    (n * (n - 1) * (2 * n + 5) - tieX.sumT2 - tieY.sumT2) / 18 +
    (tieX.sumT * tieY.sumT) / (2 * n * (n - 1));
  if (n > 2) {
    varS += (tieX.sumT3 * tieY.sumT3) / (9 * n * (n - 1) * (n - 2));
  }
  if (!(varS > 0)) return [tau, Number.NaN];

  const z = s / Math.sqrt(varS);
  const pValue =
    alternative === "two-sided"
      ? 2 * normalSf(Math.abs(z))
      : alternative === "greater"
        ? normalSf(z)
        : normalSf(-z);
  return [tau, pValue];
}

/**
 * Gathers the columns of a tensor into dense `Float64Array`s.
 *
 * @param x - 2D tensor of shape (nObs, nVar)
 * @returns One array of length nObs per column
 */
function columnsOf(x: Tensor): Float64Array[] {
  const nObs = x.shape[0] ?? 0;
  const nVar = x.shape[1] ?? 0;
  const s0 = x.strides[0] ?? 0;
  const s1 = x.strides[1] ?? 0;
  const base = x.offset;
  const data = x.data;
  const cols: Float64Array[] = [];
  const direct = !Array.isArray(data) && !(data instanceof BigInt64Array);
  for (let j = 0; j < nVar; j++) {
    const col = new Float64Array(nObs);
    if (direct) {
      for (let i = 0; i < nObs; i++) col[i] = data[base + i * s0 + j * s1] as number;
    } else {
      for (let i = 0; i < nObs; i++) col[i] = getNumberAt(x, base + i * s0 + j * s1);
    }
    cols.push(col);
  }
  return cols;
}

/** Subtracts each column's mean in place and returns the means. */
function centerColumns(cols: Float64Array[]): Float64Array {
  const means = new Float64Array(cols.length);
  for (let j = 0; j < cols.length; j++) {
    const col = cols[j] as Float64Array;
    const nObs = col.length;
    let s = 0;
    for (let i = 0; i < nObs; i++) s += col[i] as number;
    const m = s / nObs;
    means[j] = m;
    for (let i = 0; i < nObs; i++) col[i] = (col[i] as number) - m;
  }
  return means;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

/**
 * Covariance matrix of the given columns (each of length nObs), symmetric,
 * row-major nVar x nVar. Consumes `cols` (they are centered in place).
 */
function covarianceOfColumns(cols: Float64Array[], ddof: number): Float64Array {
  const nVar = cols.length;
  const nObs = nVar > 0 ? (cols[0] as Float64Array).length : 0;
  centerColumns(cols);
  const out = new Float64Array(nVar * nVar);
  for (let a = 0; a < nVar; a++) {
    for (let b = a; b < nVar; b++) {
      const v = dot(cols[a] as Float64Array, cols[b] as Float64Array) / (nObs - ddof);
      out[a * nVar + b] = v;
      out[b * nVar + a] = v;
    }
  }
  return out;
}

/**
 * Pearson correlation matrix of the given columns (each of length nObs).
 * A column that is constant or contains a non-finite value gets NaN in its
 * whole row and column, including the diagonal. Consumes `cols`.
 */
function correlationOfColumns(cols: Float64Array[]): Float64Array {
  const nVar = cols.length;
  const bad = new Array<boolean>(nVar).fill(false);
  const norms = new Float64Array(nVar);
  for (let j = 0; j < nVar; j++) {
    const col = cols[j] as Float64Array;
    const first = col[0] as number;
    let constant = true;
    let finite = true;
    let maxAbs = 0;
    for (let i = 0; i < col.length; i++) {
      const v = col[i] as number;
      if (!Number.isFinite(v)) finite = false;
      if (v !== first) constant = false;
      const av = Math.abs(v);
      if (av > maxAbs) maxAbs = av;
    }
    if (!finite || constant) {
      bad[j] = true;
      continue;
    }
    // Center, then rescale to max |deviation| = 1 so squares cannot overflow
    // or underflow; correlation is invariant to the per-column scale.
    centerColumns([col]);
    if (maxAbs > SCALE_HIGH || maxAbs < SCALE_LOW) {
      let maxDev = 0;
      for (let i = 0; i < col.length; i++) {
        const av = Math.abs(col[i] as number);
        if (av > maxDev) maxDev = av;
      }
      const inv = 1 / maxDev;
      for (let i = 0; i < col.length; i++) col[i] = (col[i] as number) * inv;
    }
    norms[j] = Math.sqrt(dot(col, col));
  }

  const out = new Float64Array(nVar * nVar);
  for (let a = 0; a < nVar; a++) {
    for (let b = a; b < nVar; b++) {
      let v: number;
      if (bad[a] || bad[b]) {
        v = Number.NaN;
      } else if (a === b) {
        v = 1;
      } else {
        const den = (norms[a] as number) * (norms[b] as number);
        const r =
          den === 0 ? Number.NaN : dot(cols[a] as Float64Array, cols[b] as Float64Array) / den;
        v = r > 1 ? 1 : r < -1 ? -1 : r;
      }
      out[a * nVar + b] = v;
      out[b * nVar + a] = v;
    }
  }
  return out;
}

/**
 * Computes the Pearson correlation coefficient matrix.
 *
 * For two variables, returns 2x2 correlation matrix.
 * For a 2D tensor, treats each column as a variable and computes pairwise correlations.
 *
 * @param x - Input tensor (1D or 2D)
 * @param y - Optional second tensor (if provided, computes correlation between x and y)
 * @returns Correlation matrix (symmetric with 1s on diagonal)
 * @throws {InvalidParameterError} If < 2 observations or size mismatch
 * @throws {ShapeError} If tensor is not 1D or 2D
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 5, 4, 5]);
 * corrcoef(x, y);  // Returns [[1.0, 0.7746], [0.7746, 1.0]]
 *
 * const data = tensor([[1, 2], [3, 4], [5, 6]]);
 * corrcoef(data);  // Returns 2x2 correlation matrix for 2 variables
 * ```
 *
 * @remarks
 * A variable with zero variance (or with a NaN/infinite value) has NaN
 * correlations with everything, including itself, as in `numpy.corrcoef`.
 * Unlike {@link pearsonr}, this does not throw for constant input. When `y` is
 * given, both tensors are flattened and treated as two variables. Matrix input
 * uses observations as rows and variables as columns (NumPy's `rowvar=False`).
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function corrcoef(x: Tensor, y?: Tensor): Tensor {
  if (y) {
    assertSameSize(x, y, "corrcoef");
    if (x.size < 2) {
      throw new InvalidParameterError(
        "corrcoef() requires at least 2 observations",
        "nObs",
        x.size
      );
    }
    const corr = correlationOfColumns([toDenseFlatArray(x), toDenseFlatArray(y)]);
    return Tensor.fromTypedArray({
      data: corr,
      shape: [2, 2],
      dtype: "float64",
      device: x.device,
    });
  }

  if (x.ndim === 1) {
    if (x.size < 2) {
      throw new InvalidParameterError(
        "corrcoef() requires at least 2 observations",
        "nObs",
        x.size
      );
    }
    const corr = correlationOfColumns([toDenseFlatArray(x)]);
    return Tensor.fromTypedArray({
      data: corr,
      shape: [1, 1],
      dtype: "float64",
      device: x.device,
    });
  }

  if (x.ndim !== 2) {
    throw new ShapeError("corrcoef() expects a 1D or 2D tensor");
  }

  // Treat columns as variables (rowvar=false style): shape (nObs, nVar)
  const nObs = x.shape[0] ?? 0; // Number of observations (rows)
  const nVar = x.shape[1] ?? 0; // Number of variables (columns)
  if (nObs < 2) {
    throw new InvalidParameterError("corrcoef() requires at least 2 observations", "nObs", nObs);
  }

  const corr = correlationOfColumns(columnsOf(x));
  return Tensor.fromTypedArray({
    data: corr,
    shape: [nVar, nVar],
    dtype: "float64",
    device: x.device,
  });
}

/**
 * Computes the covariance matrix.
 *
 * Covariance measures how two variables change together.
 * For two variables, returns 2x2 covariance matrix.
 * For a 2D tensor, treats each column as a variable.
 *
 * @param x - Input tensor (1D or 2D)
 * @param y - Optional second tensor (if provided, computes covariance between x and y)
 * @param ddof - Delta degrees of freedom (0 = population, 1 = sample, default: 1)
 * @returns Covariance matrix (symmetric)
 * @throws {InvalidParameterError} If tensor is empty, ddof is negative or not finite,
 *   ddof >= sample size, or size mismatch
 * @throws {ShapeError} If tensor is not 1D or 2D
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 5, 4, 5]);
 * cov(x, y);  // Returns 2x2 covariance matrix
 *
 * const data = tensor([[1, 2], [3, 4], [5, 6]]);
 * cov(data);  // Returns 2x2 covariance matrix for 2 variables
 * ```
 *
 * @remarks
 * A 1D `x` (without `y`) gives a 1x1 matrix. Matrix input uses observations as
 * rows and variables as columns (NumPy's `rowvar=False`). When `y` is given,
 * both tensors are flattened and treated as two variables.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function cov(x: Tensor, y?: Tensor, ddof = 1): Tensor {
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be a non-negative finite number", "ddof", ddof);
  }

  if (y) {
    assertSameSize(x, y, "cov");
    const n = x.size;
    if (n === 0) throw new InvalidParameterError("cov() requires at least one element", "n", n);
    if (n <= ddof)
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${n}, covariance undefined`,
        "ddof",
        ddof
      );
    const out = covarianceOfColumns([toDenseFlatArray(x), toDenseFlatArray(y)], ddof);
    return Tensor.fromTypedArray({ data: out, shape: [2, 2], dtype: "float64", device: x.device });
  }

  if (x.ndim === 1) {
    const n = x.size;
    if (n === 0) throw new InvalidParameterError("cov() requires at least one element", "n", n);
    if (n <= ddof)
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${n}, covariance undefined`,
        "ddof",
        ddof
      );
    const out = covarianceOfColumns([toDenseFlatArray(x)], ddof);
    return Tensor.fromTypedArray({ data: out, shape: [1, 1], dtype: "float64", device: x.device });
  }

  if (x.ndim !== 2) {
    throw new ShapeError("cov() expects a 1D or 2D tensor");
  }

  const nObs = x.shape[0] ?? 0;
  const nVar = x.shape[1] ?? 0;
  if (nObs === 0)
    throw new InvalidParameterError("cov() requires at least one observation", "nObs", nObs);
  if (nObs <= ddof)
    throw new InvalidParameterError(
      `ddof=${ddof} >= nObs=${nObs}, covariance undefined`,
      "ddof",
      ddof
    );

  const out = covarianceOfColumns(columnsOf(x), ddof);
  return Tensor.fromTypedArray({
    data: out,
    shape: [nVar, nVar],
    dtype: "float64",
    device: x.device,
  });
}

/**
 * Computes the point-biserial correlation coefficient.
 *
 * Measures the correlation between a binary variable and a continuous variable.
 * Equivalent to Pearson r where one variable is dichotomous (0/1).
 *
 * @param x - Binary tensor (values must be 0 or 1)
 * @param y - Continuous tensor (must have same size as x)
 * @param options - `alternative` hypothesis of the test (default `"two-sided"`)
 * @returns Tuple of [correlation coefficient, p-value]
 * @throws {InvalidParameterError} If x contains non-binary values, sizes differ, < 2 samples,
 *   either input is constant (for x: all zeros or all ones), or `alternative` is invalid
 *
 * @example
 * ```ts
 * const gender = tensor([0, 1, 1, 0, 1, 0]);
 * const score = tensor([72, 85, 91, 68, 88, 75]);
 * const [r, p] = pointbiserialr(gender, score);
 * ```
 *
 * @remarks
 * The coefficient and p-value are those of {@link pearsonr}: the p-value is 1 when
 * n = 2, and both are NaN when `y` contains NaN or infinite values.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function pointbiserialr(
  x: Tensor,
  y: Tensor,
  options: CorrelationTestOptions = {}
): [number, number] {
  const alternative = resolveAlternative("pointbiserialr", options);
  assertSameSize(x, y, "pointbiserialr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("pointbiserialr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);

  // Validate binary
  for (let i = 0; i < n; i++) {
    const v = xd[i] as number;
    if (v !== 0 && v !== 1) {
      throw new InvalidParameterError("pointbiserialr() requires binary (0/1) values in x", "x", v);
    }
  }

  // Point-biserial is equivalent to Pearson r for binary x
  const r = pearsonFromDense(xd, yd, "pointbiserialr");
  if (n === 2) return Number.isNaN(r) ? [r, Number.NaN] : [Math.round(r), 1];
  return [r, correlationPValueFor(r, n - 2, alternative)];
}

/**
 * Computes partial correlation between two variables controlling for confounders.
 *
 * Partial correlation measures the linear relationship between x and y
 * after removing the effect of one or more confounding variables (z).
 *
 * Both x and y are regressed on the confounders (with an intercept) by
 * orthogonal projection, and the Pearson correlation of the two residual
 * vectors is returned.
 *
 * @param x - First variable tensor (1D)
 * @param y - Second variable tensor (1D, same size as x)
 * @param z - Confounding variable(s): a single 1D tensor or array of 1D tensors
 * @returns Tuple of [partial correlation coefficient, two-tailed p-value]
 * @throws {InvalidParameterError} If sizes don't match, there are fewer than
 *   `confounders + 3` samples, x or y is constant, or x or y is (numerically) an exact
 *   linear combination of the confounders
 *
 * @example
 * ```ts
 * const age = tensor([25, 30, 35, 40, 45]);
 * const income = tensor([30, 40, 55, 60, 75]);
 * const education = tensor([12, 14, 16, 18, 20]);
 * const [r, p] = partialcorr(income, age, education);
 * ```
 *
 * @remarks
 * The p-value uses the t-distribution with n - 2 - k degrees of freedom (k =
 * number of confounders). Constant or perfectly collinear confounders carry no
 * extra information and are ignored; the degrees of freedom still count them.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function partialcorr(x: Tensor, y: Tensor, z: Tensor | Tensor[]): [number, number] {
  assertSameSize(x, y, "partialcorr");
  const n = x.size;

  const confounders = Array.isArray(z) ? z : [z];
  for (const zi of confounders) {
    assertSameSize(x, zi, "partialcorr");
  }

  if (n < confounders.length + 3) {
    throw new InvalidParameterError(
      `partialcorr() requires at least ${confounders.length + 3} samples for ${confounders.length} confounder(s)`,
      "n",
      n
    );
  }

  const xRaw = toDenseFlatArray(x);
  const yRaw = toDenseFlatArray(y);
  // Reports constant input / non-finite values with partialcorr()'s own name.
  pearsonFromDense(xRaw, yRaw, "partialcorr");

  // Orthonormal basis of the centered confounders (modified Gram-Schmidt with
  // re-orthogonalization). A confounder that is constant or lies in the span
  // of the earlier ones adds nothing and is dropped.
  const basis: Float64Array[] = [];
  for (const zi of confounders) {
    const v = toDenseFlatArray(zi);
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(v[i] as number)) return [Number.NaN, Number.NaN];
    }
    centerColumns([v]);
    const before = Math.sqrt(dot(v, v));
    if (before === 0) continue;
    project(v, basis);
    const after = Math.sqrt(dot(v, v));
    if (after <= 1e-10 * before) continue;
    for (let i = 0; i < n; i++) v[i] = (v[i] as number) / after;
    basis.push(v);
  }

  const residualize = (raw: Float64Array, label: string): Float64Array => {
    const v = Float64Array.from(raw);
    centerColumns([v]);
    const before = Math.sqrt(dot(v, v));
    project(v, basis);
    const after = Math.sqrt(dot(v, v));
    if (after <= 1e-10 * before) {
      throw new InvalidParameterError(
        `partialcorr() is undefined when ${label} is a linear combination of the confounders`,
        label,
        "collinear"
      );
    }
    return v;
  };

  const xResid = residualize(xRaw, "x");
  const yResid = residualize(yRaw, "y");

  const r = pearsonFromDense(xResid, yResid, "partialcorr");
  // n >= confounders + 3 was checked above, so df >= 1.
  return [r, correlationPValue(r, n - 2 - confounders.length)];
}

/** Removes from `v` (in place) its components along the orthonormal `basis`, twice for stability. */
function project(v: Float64Array, basis: readonly Float64Array[]): void {
  for (let pass = 0; pass < 2; pass++) {
    for (const q of basis) {
      const c = dot(v, q);
      for (let i = 0; i < v.length; i++) v[i] = (v[i] as number) - c * (q[i] as number);
    }
  }
}
