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
 * @see {@link https://deepbox.dev/docs/stats-correlation | Deepbox Correlations}
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
 * @see {@link https://deepbox.dev/docs/stats-correlation | Deepbox Correlations}
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
 * @complexity O(n²) - suitable for moderate sample sizes (n < 10,000)
 *
 * @see {@link https://deepbox.dev/docs/stats-correlation | Deepbox Correlations}
 */
export function kendalltau(x: Tensor, y: Tensor): [number, number] {
  assertSameSize(x, y, "kendalltau");
  const n = x.size;
  if (n < 2) {
    throw new InvalidParameterError("kendalltau() requires at least 2 paired samples", "n", n);
  }

  const xd = toDenseFlatArray(x);
  const yd = toDenseFlatArray(y);

  // Count concordant and discordant pairs (ties excluded)
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n - 1; i++) {
    const xi = xd[i] ?? 0;
    const yi = yd[i] ?? 0;
    for (let j = i + 1; j < n; j++) {
      const signX = Math.sign((xd[j] ?? 0) - xi);
      const signY = Math.sign((yd[j] ?? 0) - yi);
      if (signX === 0 || signY === 0) continue;
      if (signX === signY) concordant++;
      else discordant++;
    }
  }

  const n0 = (n * (n - 1)) / 2;
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
  const s = concordant - discordant;
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
 * @see {@link https://deepbox.dev/docs/stats-correlation | Deepbox Correlations}
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
 * @see {@link https://deepbox.dev/docs/stats-correlation | Deepbox Correlations}
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
