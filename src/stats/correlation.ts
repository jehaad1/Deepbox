import { InvalidParameterError, ShapeError } from "../core";
import { Tensor, tensor } from "../ndarray";
import {
  assertSameSize,
  forEachIndexOffset,
  getNumberAt,
  normalCdf,
  rankData,
  studentTCdf,
} from "./_internal";

/**
 * Converts a tensor to a dense flat Float64Array.
 * Helper function for correlation computations.
 *
 * @param t - Input tensor
 * @returns Flattened Float64Array containing all tensor values
 */
function toDenseFlatArray(t: Tensor): Float64Array {
  const out = new Float64Array(t.size);
  let i = 0;
  forEachIndexOffset(t, (off) => {
    out[i] = getNumberAt(t, off);
    i++;
  });
  return out;
}

/**
 * Computes Pearson correlation coefficient from two dense arrays.
 *
 * Uses the standard formula: r = cov(X,Y) / (std(X) * std(Y))
 * Computed efficiently in a single pass.
 *
 * @param x - First array
 * @param y - Second array (must have same length as x)
 * @returns Pearson correlation coefficient in [-1, 1]
 * @throws {InvalidParameterError} If arrays have constant values (zero variance)
 */
function pearsonFromDense(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  // First pass: compute means
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i] ?? 0;
    sumY += y[i] ?? 0;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  // Second pass: compute covariance and variances
  let num = 0; // Numerator: covariance
  let denX = 0; // Denominator: variance of X
  let denY = 0; // Denominator: variance of Y
  for (let i = 0; i < n; i++) {
    const dx = (x[i] ?? 0) - meanX;
    const dy = (y[i] ?? 0) - meanY;
    num += dx * dy; // Sum of products of deviations
    denX += dx * dx; // Sum of squared deviations for X
    denY += dy * dy; // Sum of squared deviations for Y
  }

  // Compute correlation: r = cov(X,Y) / (std(X) * std(Y))
  const den = Math.sqrt(denX * denY);
  if (den === 0) {
    throw new InvalidParameterError(
      "pearsonr() is undefined for constant input",
      "input",
      "constant"
    );
  }
  return num / den;
}

/**
 * Pearson correlation coefficient.
 *
 * Measures linear correlation between two variables.
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @returns Tuple of [correlation coefficient in [-1, 1], two-tailed p-value]
 * @throws {InvalidParameterError} If tensors have different sizes, < 2 samples, or constant input
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 6, 8, 10]);
 * const [r, p] = pearsonr(x, y);  // r = 1.0 (perfect linear)
 * ```
 *
 * @remarks
 * This function follows IEEE 754 semantics for special values:
 * - NaN inputs propagate to NaN correlation
 * - Infinity inputs result in NaN correlation
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function pearsonr(x: Tensor, y: Tensor): [number, number] {
  assertSameSize(x, y, "pearsonr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("pearsonr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);
  const r = pearsonFromDense(xd, yd);

  // Compute p-value using t-distribution
  // Under H0: ρ=0, the test statistic t = r*sqrt((n-2)/(1-r²)) follows t(n-2)
  const df = n - 2;
  if (df <= 0) {
    return [r, NaN]; // Cannot compute p-value with < 2 degrees of freedom
  }
  const tStat = r * Math.sqrt(df / (1 - r * r));
  const pValue = 2 * (1 - studentTCdf(Math.abs(tStat), df)); // Two-tailed test
  return [r, pValue];
}

/**
 * Computes Spearman's rank correlation coefficient.
 *
 * Non-parametric measure of monotonic relationship between two variables.
 * Computed as Pearson correlation of rank values.
 * - ρ = 1: Perfect monotonic increasing relationship
 * - ρ = 0: No monotonic relationship
 * - ρ = -1: Perfect monotonic decreasing relationship
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @returns Tuple of [correlation coefficient, p-value]
 * @throws {InvalidParameterError} If tensors have different sizes, < 2 samples, or constant input
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([2, 4, 6, 8, 10]);
 * const [rho, p] = spearmanr(x, y);  // rho = 1.0 (perfect monotonic)
 * ```
 *
 * @remarks
 * Ties are assigned average ranks.
 * NaN values are ranked according to JavaScript sort behavior.
 * Infinity values are sorted naturally (±Infinity at extremes).
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function spearmanr(x: Tensor, y: Tensor): [number, number] {
  assertSameSize(x, y, "spearmanr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("spearmanr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);

  // Convert values to average ranks (1-indexed) with tie correction.
  const rankX = rankData(xd).ranks;
  const rankY = rankData(yd).ranks;

  // Compute Pearson correlation of ranks
  const rho = pearsonFromDense(rankX, rankY);
  // Test statistic follows t-distribution under H0: ρ=0
  const df = n - 2;
  const tStat = rho * Math.sqrt(df / (1 - rho * rho));
  const pValue = df > 0 ? 2 * (1 - studentTCdf(Math.abs(tStat), df)) : NaN;
  return [rho, pValue];
}

/**
 * Computes Kendall's tau correlation coefficient.
 *
 * Non-parametric measure of ordinal association based on concordant/discordant pairs.
 * More robust to outliers than Spearman, but computationally more expensive.
 * - τ = 1: All pairs concordant (perfect agreement)
 * - τ = 0: Equal concordant and discordant pairs
 * - τ = -1: All pairs discordant (perfect disagreement)
 *
 * @param x - First tensor
 * @param y - Second tensor (must have same size as x)
 * @returns Tuple of [tau coefficient, p-value]
 * @throws {InvalidParameterError} If tensors have different sizes or < 2 samples
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([1, 3, 2, 4, 5]);
 * const [tau, p] = kendalltau(x, y);  // Mostly concordant
 * ```
 *
 * @remarks
 * This implementation uses the tau-b variant with tie correction.
 * Ties are excluded from concordant/discordant counts and reduce the denominator.
 * The p-value uses a normal approximation with tie-corrected variance.
 *
 * @complexity O(n log n) via Knight's merge-sort algorithm for the
 * concordant-minus-discordant score.
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
/**
 * Kendall's S = (concordant - discordant), counting only pairs untied in both
 * x and y, computed in O(n log n) via Knight's algorithm (merge-sort inversion
 * count). Result is identical to the naive O(n²) double loop.
 *
 * @internal
 */
function kendallScore(xd: Float64Array, yd: Float64Array, n: number): number {
  // Order by (x asc, then y asc).
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => {
    const dx = (xd[a] as number) - (xd[b] as number);
    if (dx !== 0) return dx;
    return (yd[a] as number) - (yd[b] as number);
  });

  // Tied-pair counts: t(t-1)/2 summed over equal-value runs.
  const tiePairs = (equal: (a: number, b: number) => boolean): number => {
    let sum = 0;
    for (let i = 0; i < n; ) {
      let j = i + 1;
      while (j < n && equal(order[j - 1] as number, order[j] as number)) j++;
      const t = j - i;
      if (t > 1) sum += (t * (t - 1)) / 2;
      i = j;
    }
    return sum;
  };

  const xtie = tiePairs((a, b) => (xd[a] as number) === (xd[b] as number));
  const ntie = tiePairs(
    (a, b) => (xd[a] as number) === (xd[b] as number) && (yd[a] as number) === (yd[b] as number)
  );

  // ytie needs y sorted on its own.
  const yorder = new Int32Array(n);
  for (let i = 0; i < n; i++) yorder[i] = i;
  yorder.sort((a, b) => (yd[a] as number) - (yd[b] as number));
  let ytie = 0;
  for (let i = 0; i < n; ) {
    let j = i + 1;
    while (j < n && (yd[yorder[j] as number] as number) === (yd[yorder[i] as number] as number))
      j++;
    const t = j - i;
    if (t > 1) ytie += (t * (t - 1)) / 2;
    i = j;
  }

  // Discordant pairs = strict inversions in the y-sequence taken in (x,y) order.
  const yByX = new Float64Array(n);
  for (let i = 0; i < n; i++) yByX[i] = yd[order[i] as number] as number;
  const dis = countInversions(yByX, n);

  const tot = (n * (n - 1)) / 2;
  return tot - xtie - ytie + ntie - 2 * dis;
}

/** Count strict inversions (a[i] > a[j], i < j) via bottom-up merge sort. */
function countInversions(a: Float64Array, n: number): number {
  if (n < 2) return 0;
  const buf = new Float64Array(a);
  const tmp = new Float64Array(n);
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
          // left-run elements, each of which is > buf[j] → that many inversions.
          inv += mid - i;
          tmp[k++] = buf[j++] as number;
        }
      }
      while (i < mid) tmp[k++] = buf[i++] as number;
      while (j < hi) tmp[k++] = buf[j++] as number;
    }
    buf.set(tmp);
  }
  return inv;
}

export function kendalltau(x: Tensor, y: Tensor): [number, number] {
  assertSameSize(x, y, "kendalltau");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("kendalltau() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);

  // Knight's O(n log n) algorithm for s = (concordant - discordant), counting
  // only pairs untied in both x and y. Sort by (x, then y); the number of
  // strict inversions in the resulting y-sequence (via merge sort) is exactly
  // the discordant count for pairs with distinct x. By inclusion-exclusion,
  //   s = tot - xtie - ytie + ntie - 2*dis
  // where tot = n(n-1)/2, xtie/ytie/ntie are the tied-pair counts in x, y and
  // the joint (x, y). This is identical to the O(n²) double-loop result but
  // scales to large n (the previous loop was O(n²)).
  const n0 = (n * (n - 1)) / 2;
  const s = kendallScore(xd, yd, n);
  // Tie summaries for tau-b denominator and variance corrections.
  const tieSums = (
    vals: Float64Array
  ): {
    nTies: number;
    sumT: number;
    sumT2: number;
    sumT3: number;
  } => {
    const sorted = Array.from(vals).sort((a, b) => a - b);
    let sumT = 0;
    let sumT2 = 0;
    let sumT3 = 0;
    for (let i = 0; i < sorted.length; ) {
      let j = i + 1;
      while (j < sorted.length && sorted[j] === sorted[i]) j++;
      const t = j - i;
      if (t > 1) {
        sumT += t * (t - 1);
        sumT2 += t * (t - 1) * (2 * t + 5);
        sumT3 += t * (t - 1) * (t - 2);
      }
      i = j;
    }
    return { nTies: sumT / 2, sumT, sumT2, sumT3 };
  };

  const tieX = tieSums(xd);
  const tieY = tieSums(yd);
  const denom = Math.sqrt((n0 - tieX.nTies) * (n0 - tieY.nTies));
  const tau = denom === 0 ? NaN : s / denom;

  // Normal approximation for p-value with tie correction
  // Normal approximation variance with tie correction (standard).
  let varS =
    (n * (n - 1) * (2 * n + 5) - tieX.sumT2 - tieY.sumT2) / 18 +
    (tieX.sumT * tieY.sumT) / (2 * n * (n - 1));
  if (n > 2) {
    varS += (tieX.sumT3 * tieY.sumT3) / (9 * n * (n - 1) * (n - 2));
  }

  const pValue = varS <= 0 ? NaN : 2 * (1 - normalCdf(Math.abs(s / Math.sqrt(varS))));
  return [tau, pValue];
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
 * corrcoef(x, y);  // Returns [[1.0, 0.8], [0.8, 1.0]]
 *
 * const data = tensor([[1, 2], [3, 4], [5, 6]]);
 * corrcoef(data);  // Returns 2x2 correlation matrix for 2 variables
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function corrcoef(x: Tensor, y?: Tensor): Tensor {
  if (y) {
    const [r] = pearsonr(x, y);
    return tensor([
      [1.0, r],
      [r, 1.0],
    ]);
  }

  if (x.ndim === 1) {
    if (x.size < 2) {
      throw new InvalidParameterError(
        "corrcoef() requires at least 2 observations",
        "nObs",
        x.size
      );
    }
    return tensor([[1.0]]);
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

  const s0 = x.strides[0] ?? 0;
  const s1 = x.strides[1] ?? 0;
  const xOff = x.offset;
  const xData = x.data;

  // Fast path: direct numeric array access (non-BigInt, non-string)
  const directAccess = !Array.isArray(xData) && !(xData instanceof BigInt64Array);

  const means = new Float64Array(nVar);
  if (directAccess) {
    for (let j = 0; j < nVar; j++) {
      let s = 0;
      for (let i = 0; i < nObs; i++) {
        s += xData[xOff + i * s0 + j * s1] as number;
      }
      means[j] = s / nObs;
    }
  } else {
    for (let j = 0; j < nVar; j++) {
      let s = 0;
      for (let i = 0; i < nObs; i++) {
        s += getNumberAt(x, xOff + i * s0 + j * s1);
      }
      means[j] = s / nObs;
    }
  }

  const cov = new Float64Array(nVar * nVar);
  const ddof = 1;
  if (directAccess) {
    for (let a = 0; a < nVar; a++) {
      const ma = means[a] as number;
      for (let b = a; b < nVar; b++) {
        const mb = means[b] as number;
        let s = 0;
        for (let i = 0; i < nObs; i++) {
          const base = xOff + i * s0;
          s += ((xData[base + a * s1] as number) - ma) * ((xData[base + b * s1] as number) - mb);
        }
        const v = s / (nObs - ddof);
        cov[a * nVar + b] = v;
        cov[b * nVar + a] = v;
      }
    }
  } else {
    for (let a = 0; a < nVar; a++) {
      for (let b = a; b < nVar; b++) {
        let s = 0;
        for (let i = 0; i < nObs; i++) {
          const offA = xOff + i * s0 + a * s1;
          const offB = xOff + i * s0 + b * s1;
          s += (getNumberAt(x, offA) - (means[a] ?? 0)) * (getNumberAt(x, offB) - (means[b] ?? 0));
        }
        const v = s / (nObs - ddof);
        cov[a * nVar + b] = v;
        cov[b * nVar + a] = v;
      }
    }
  }

  const corr = new Float64Array(nVar * nVar);
  for (let i = 0; i < nVar; i++) {
    for (let j = 0; j < nVar; j++) {
      const v = cov[i * nVar + j] ?? 0;
      const vi = cov[i * nVar + i] ?? 0;
      const vj = cov[j * nVar + j] ?? 0;
      const den = Math.sqrt(vi * vj);
      corr[i * nVar + j] = den === 0 ? NaN : v / den;
    }
  }

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
 * @throws {InvalidParameterError} If tensor is empty, ddof < 0, ddof >= sample size, or size mismatch
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
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
/**
 * Computes the point-biserial correlation coefficient.
 *
 * Measures the correlation between a binary variable and a continuous variable.
 * Equivalent to Pearson r where one variable is dichotomous (0/1).
 *
 * @param x - Binary tensor (values must be 0 or 1)
 * @param y - Continuous tensor (must have same size as x)
 * @returns Tuple of [correlation coefficient, two-tailed p-value]
 * @throws {InvalidParameterError} If x contains non-binary values, sizes differ, or < 2 samples
 *
 * @example
 * ```ts
 * const gender = tensor([0, 1, 1, 0, 1, 0]);
 * const score = tensor([72, 85, 91, 68, 88, 75]);
 * const [r, p] = pointbiserialr(gender, score);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox Correlations}
 */
export function pointbiserialr(x: Tensor, y: Tensor): [number, number] {
  assertSameSize(x, y, "pointbiserialr");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("pointbiserialr() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);

  // Validate binary
  for (let i = 0; i < n; i++) {
    const v = xd[i] ?? 0;
    if (v !== 0 && v !== 1) {
      throw new InvalidParameterError("pointbiserialr() requires binary (0/1) values in x", "x", v);
    }
  }

  // Point-biserial is equivalent to Pearson r for binary x
  const r = pearsonFromDense(xd, yd);
  const df = n - 2;
  if (df <= 0) {
    return [r, NaN];
  }
  const tStat = r * Math.sqrt(df / (1 - r * r));
  const pValue = 2 * (1 - studentTCdf(Math.abs(tStat), df));
  return [r, pValue];
}

/**
 * Computes partial correlation between two variables controlling for confounders.
 *
 * Partial correlation measures the linear relationship between x and y
 * after removing the effect of one or more confounding variables (z).
 *
 * Uses the recursive formula for single confounder and matrix inversion
 * approach for multiple confounders.
 *
 * @param x - First variable tensor (1D)
 * @param y - Second variable tensor (1D, same size as x)
 * @param z - Confounding variable(s): a single 1D tensor or array of 1D tensors
 * @returns Tuple of [partial correlation coefficient, two-tailed p-value]
 * @throws {InvalidParameterError} If sizes don't match or < 3 samples
 *
 * @example
 * ```ts
 * const age = tensor([25, 30, 35, 40, 45]);
 * const income = tensor([30, 40, 55, 60, 75]);
 * const education = tensor([12, 14, 16, 18, 20]);
 * const [r, p] = partialcorr(income, age, education);
 * ```
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

  // Residualize: for each variable, regress out the confounders via OLS
  const residualize = (v: Float64Array): Float64Array => {
    // Build design matrix Z (n x k) and compute residuals: v - Z * (Z'Z)^{-1} Z'v
    const k = confounders.length;
    const zArrays: Float64Array[] = confounders.map(toDenseFlatArray);

    // Compute Z'Z (k x k) and Z'v (k x 1)
    const ZtZ = new Float64Array(k * k);
    const Ztv = new Float64Array(k);
    for (let a = 0; a < k; a++) {
      const za = zArrays[a];
      if (!za) continue;
      for (let b = a; b < k; b++) {
        const zb = zArrays[b];
        if (!zb) continue;
        let s = 0;
        for (let i = 0; i < n; i++) {
          s += (za[i] ?? 0) * (zb[i] ?? 0);
        }
        ZtZ[a * k + b] = s;
        ZtZ[b * k + a] = s;
      }
      let sv = 0;
      for (let i = 0; i < n; i++) {
        sv += (za[i] ?? 0) * (v[i] ?? 0);
      }
      Ztv[a] = sv;
    }

    // Solve ZtZ * beta = Ztv via Gauss elimination
    const aug = new Float64Array(k * (k + 1));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        aug[i * (k + 1) + j] = ZtZ[i * k + j] ?? 0;
      }
      aug[i * (k + 1) + k] = Ztv[i] ?? 0;
    }

    for (let col = 0; col < k; col++) {
      // Partial pivoting
      let maxRow = col;
      let maxVal = Math.abs(aug[col * (k + 1) + col] ?? 0);
      for (let row = col + 1; row < k; row++) {
        const absVal = Math.abs(aug[row * (k + 1) + col] ?? 0);
        if (absVal > maxVal) {
          maxVal = absVal;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j <= k; j++) {
          const tmp = aug[col * (k + 1) + j] ?? 0;
          aug[col * (k + 1) + j] = aug[maxRow * (k + 1) + j] ?? 0;
          aug[maxRow * (k + 1) + j] = tmp;
        }
      }

      const pivot = aug[col * (k + 1) + col] ?? 0;
      if (Math.abs(pivot) < 1e-15) continue;

      for (let row = col + 1; row < k; row++) {
        const factor = (aug[row * (k + 1) + col] ?? 0) / pivot;
        for (let j = col; j <= k; j++) {
          aug[row * (k + 1) + j] =
            (aug[row * (k + 1) + j] ?? 0) - factor * (aug[col * (k + 1) + j] ?? 0);
        }
      }
    }

    // Back-substitution
    const beta = new Float64Array(k);
    for (let i = k - 1; i >= 0; i--) {
      let s = aug[i * (k + 1) + k] ?? 0;
      for (let j = i + 1; j < k; j++) {
        s -= (aug[i * (k + 1) + j] ?? 0) * (beta[j] ?? 0);
      }
      const diag = aug[i * (k + 1) + i] ?? 0;
      beta[i] = Math.abs(diag) < 1e-15 ? 0 : s / diag;
    }

    // Compute residuals: v - Z * beta
    const resid = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < k; j++) {
        const zj = zArrays[j];
        pred += (zj ? (zj[i] ?? 0) : 0) * (beta[j] ?? 0);
      }
      resid[i] = (v[i] ?? 0) - pred;
    }
    return resid;
  };

  const xResid = residualize(toDenseFlatArray(x));
  const yResid = residualize(toDenseFlatArray(y));

  const r = pearsonFromDense(xResid, yResid);
  const df = n - 2 - confounders.length;
  if (df <= 0) {
    return [r, NaN];
  }
  const rClamped = Math.min(Math.max(r, -1 + 1e-15), 1 - 1e-15);
  const tStat = rClamped * Math.sqrt(df / (1 - rClamped * rClamped));
  const pValue = 2 * (1 - studentTCdf(Math.abs(tStat), df));
  return [r, pValue];
}

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

    const xd = toDenseFlatArray(x);
    const yd = toDenseFlatArray(y);

    let meanX = 0;
    let meanY = 0;
    for (let i = 0; i < n; i++) {
      meanX += xd[i] ?? 0;
      meanY += yd[i] ?? 0;
    }
    meanX /= n;
    meanY /= n;

    let varX = 0;
    let varY = 0;
    let covXY = 0;
    for (let i = 0; i < n; i++) {
      const dx = (xd[i] ?? 0) - meanX;
      const dy = (yd[i] ?? 0) - meanY;
      varX += dx * dx;
      varY += dy * dy;
      covXY += dx * dy;
    }
    varX /= n - ddof;
    varY /= n - ddof;
    covXY /= n - ddof;

    return tensor([
      [varX, covXY],
      [covXY, varY],
    ]);
  }

  if (x.ndim === 1) {
    const n = x.size;
    if (n === 0) throw new InvalidParameterError("cov() requires at least one element", "n", n);
    if (ddof < 0) {
      throw new InvalidParameterError("ddof must be non-negative", "ddof", ddof);
    }
    if (n <= ddof)
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${n}, covariance undefined`,
        "ddof",
        ddof
      );

    const xd = toDenseFlatArray(x);
    let meanX = 0;
    for (let i = 0; i < n; i++) meanX += xd[i] ?? 0;
    meanX /= n;
    let varX = 0;
    for (let i = 0; i < n; i++) {
      const dx = (xd[i] ?? 0) - meanX;
      varX += dx * dx;
    }
    varX /= n - ddof;
    return tensor([[varX]]);
  }

  if (x.ndim !== 2) {
    throw new ShapeError("cov() expects a 1D or 2D tensor");
  }

  const nObs = x.shape[0] ?? 0;
  const nVar = x.shape[1] ?? 0;
  if (nObs === 0)
    throw new InvalidParameterError("cov() requires at least one observation", "nObs", nObs);
  if (ddof < 0) {
    throw new InvalidParameterError("ddof must be non-negative", "ddof", ddof);
  }
  if (nObs <= ddof)
    throw new InvalidParameterError(
      `ddof=${ddof} >= nObs=${nObs}, covariance undefined`,
      "ddof",
      ddof
    );

  const means = new Float64Array(nVar);
  for (let j = 0; j < nVar; j++) {
    let s = 0;
    for (let i = 0; i < nObs; i++) {
      const off = x.offset + i * (x.strides[0] ?? 0) + j * (x.strides[1] ?? 0);
      s += getNumberAt(x, off);
    }
    means[j] = s / nObs;
  }

  const out = new Float64Array(nVar * nVar);
  for (let a = 0; a < nVar; a++) {
    for (let b = a; b < nVar; b++) {
      let s = 0;
      for (let i = 0; i < nObs; i++) {
        const offA = x.offset + i * (x.strides[0] ?? 0) + a * (x.strides[1] ?? 0);
        const offB = x.offset + i * (x.strides[0] ?? 0) + b * (x.strides[1] ?? 0);
        s += (getNumberAt(x, offA) - (means[a] ?? 0)) * (getNumberAt(x, offB) - (means[b] ?? 0));
      }
      const v = s / (nObs - ddof);
      out[a * nVar + b] = v;
      out[b * nVar + a] = v;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [nVar, nVar],
    dtype: "float64",
    device: x.device,
  });
}
