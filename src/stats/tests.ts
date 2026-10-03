import { InvalidParameterError } from "../core";
import { Tensor } from "../ndarray";
import {
  chiSquareSf,
  forEachIndexOffset,
  fSf,
  getNumberAt,
  logGamma,
  normalCdf,
  normalPpf,
  normalSf,
  rankData,
  studentTCdf,
  studentTSf,
} from "./_internal";

/**
 * Result of a statistical hypothesis test.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export type TestResult = {
  statistic: number;
  pvalue: number;
};

/**
 * Alternative hypothesis for tests that support one-sided p-values.
 *
 * - `"two-sided"`: the parameter differs from the null value (default).
 * - `"greater"`: the parameter (or first sample) is greater than the null value.
 * - `"less"`: the parameter (or first sample) is less than the null value.
 */
export type TestAlternative = "two-sided" | "less" | "greater";

function assertAlternative(
  fn: string,
  alternative: unknown
): asserts alternative is TestAlternative {
  if (alternative !== "two-sided" && alternative !== "less" && alternative !== "greater") {
    throw new InvalidParameterError(
      `${fn}() alternative must be "two-sided", "less" or "greater"`,
      "alternative",
      alternative
    );
  }
}

/** p-value of a t statistic with `df` degrees of freedom for the given alternative. */
function tPvalue(t: number, df: number, alternative: TestAlternative): number {
  if (Number.isNaN(t)) return Number.NaN;
  if (alternative === "greater") return studentTSf(t, df);
  if (alternative === "less") return studentTCdf(t, df);
  return Math.min(1, 2 * studentTSf(Math.abs(t), df));
}

/** p-value of a z statistic for the given alternative. */
function zPvalue(z: number, alternative: TestAlternative): number {
  if (Number.isNaN(z)) return Number.NaN;
  if (alternative === "greater") return normalSf(z);
  if (alternative === "less") return normalSf(-z);
  return Math.min(1, 2 * normalSf(Math.abs(z)));
}

function toDenseSortedArray1D(x: Tensor): Float64Array {
  if (x.size < 1) {
    return new Float64Array(0);
  }
  const out = new Float64Array(x.size);
  let i = 0;
  forEachIndexOffset(x, (off) => {
    out[i] = getNumberAt(x, off);
    i++;
  });
  // The default TypedArray sort is numeric and orders NaN last; it is also
  // much faster than a comparator callback.
  out.sort();
  return out;
}

function toDenseArray1D(x: Tensor): Float64Array {
  const out = new Float64Array(x.size);
  let i = 0;
  forEachIndexOffset(x, (off) => {
    out[i] = getNumberAt(x, off);
    i++;
  });
  return out;
}

function hasNaN(x: ArrayLike<number>): boolean {
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) return true;
  }
  return false;
}

function meanAndM2(x: Float64Array): { mean: number; m2: number } {
  if (x.length === 0) {
    throw new InvalidParameterError("expected at least one element", "length", x.length);
  }
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] ?? 0;
    const n = i + 1;
    const delta = v - mean;
    mean += delta / n;
    const delta2 = v - mean;
    m2 += delta * delta2;
  }
  return { mean, m2 };
}

function shapiroWilk(x: Float64Array): TestResult {
  // Ported from Algorithm AS R94 (Royston, 1995), following implementations
  // derived from the original FORTRAN and common ports.
  const n = x.length;
  if (n < 3 || n > 5000) {
    throw new InvalidParameterError("shapiro() sample size must be between 3 and 5000", "n", n);
  }

  const range = (x[n - 1] ?? 0) - (x[0] ?? 0);
  if (range === 0) {
    throw new InvalidParameterError("shapiro() all x values are identical", "range", range);
  }

  const small = 1e-19;
  if (range < small) {
    throw new InvalidParameterError("shapiro() range is too small", "range", range);
  }

  const nn2 = Math.floor(n / 2);
  const a = new Float64Array(nn2 + 1); // 1-based

  const g: readonly number[] = [-2.273, 0.459];
  const c1: readonly number[] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
  const c2: readonly number[] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
  const c3: readonly number[] = [0.544, -0.39978, 0.025054, -6.714e-4];
  const c4: readonly number[] = [1.3822, -0.77857, 0.062767, -0.0020322];
  const c5: readonly number[] = [-1.5861, -0.31082, -0.083751, 0.0038915];
  const c6: readonly number[] = [-0.4803, -0.082676, 0.0030302];

  const poly = (cc: readonly number[], x0: number): number => {
    let p = cc[cc.length - 1] ?? 0;
    for (let j = cc.length - 2; j >= 0; j--) {
      p = p * x0 + (cc[j] ?? 0);
    }
    return p;
  };

  const sign = (v: number): number => (v === 0 ? 0 : v > 0 ? 1 : -1);

  const an = n;
  if (n === 3) {
    a[1] = Math.SQRT1_2;
  } else {
    const an25 = an + 0.25;
    let summ2 = 0;
    for (let i = 1; i <= nn2; i++) {
      // Expected values of normal order statistics (Blom's approximation):
      // m_i = Φ⁻¹((i − 0.375) / (n + 0.25)).
      const p = (i - 0.375) / an25;
      const z = normalPpf(p);
      a[i] = z;
      summ2 += z * z;
    }
    summ2 *= 2;
    const ssumm2 = Math.sqrt(summ2);
    const rsn = 1 / Math.sqrt(an);

    const a1 = poly(c1, rsn) - (a[1] ?? 0) / ssumm2;
    let i1: number;
    let fac: number;
    if (n > 5) {
      i1 = 3;
      const a2 = -((a[2] ?? 0) / ssumm2) + poly(c2, rsn);
      fac = Math.sqrt(
        (summ2 - 2 * (a[1] ?? 0) * (a[1] ?? 0) - 2 * (a[2] ?? 0) * (a[2] ?? 0)) /
          (1 - 2 * a1 * a1 - 2 * a2 * a2)
      );
      a[2] = a2;
    } else {
      i1 = 2;
      fac = Math.sqrt((summ2 - 2 * (a[1] ?? 0) * (a[1] ?? 0)) / (1 - 2 * a1 * a1));
    }
    a[1] = a1;
    for (let i = i1; i <= nn2; i++) {
      a[i] = -((a[i] ?? 0) / fac);
    }
  }

  // Check sort order and compute scaled sums
  let xx = (x[0] ?? 0) / range;
  let sx = xx;
  let sa = -(a[1] ?? 0);
  for (let i = 1, j = n - 1; i < n; j--) {
    const xi = (x[i] ?? 0) / range;
    if (xx - xi > small) {
      throw new InvalidParameterError("shapiro() data is not sorted", "data", "unsorted");
    }
    sx += xi;
    i++;
    if (i !== j) {
      sa += sign(i - j) * (a[Math.min(i, j)] ?? 0);
    }
    xx = xi;
  }

  sa /= n;
  sx /= n;

  let ssa = 0;
  let ssx = 0;
  let sax = 0;
  for (let i = 0, j = n - 1; i < n; i++, j--) {
    const asa = i !== j ? sign(i - j) * (a[1 + Math.min(i, j)] ?? 0) - sa : -sa;
    const xsx = (x[i] ?? 0) / range - sx;
    ssa += asa * asa;
    ssx += xsx * xsx;
    sax += asa * xsx;
  }

  const ssassx = Math.sqrt(ssa * ssx);
  const w1 = ((ssassx - sax) * (ssassx + sax)) / (ssa * ssx);
  const w = 1 - w1;

  if (n === 3) {
    const pi6 = 1.90985931710274;
    const stqr = 1.0471975511966;
    let pw = pi6 * (Math.asin(Math.sqrt(w)) - stqr);
    if (pw < 0) pw = 0;
    if (pw > 1) pw = 1;
    return { statistic: w, pvalue: pw };
  }

  const y = Math.log(w1);
  const lnN = Math.log(an);
  let m: number;
  let s: number;
  if (n <= 11) {
    const gamma = poly(g, an);
    if (y >= gamma) {
      return { statistic: w, pvalue: 0 };
    }
    const yy = -Math.log(gamma - y);
    m = poly(c3, an);
    s = Math.exp(poly(c4, an));
    const z = (yy - m) / s;
    // Shapiro-Wilk p-value is the upper tail of the transformed statistic.
    return { statistic: w, pvalue: normalSf(z) };
  }
  m = poly(c5, lnN);
  s = Math.exp(poly(c6, lnN));
  const z = (y - m) / s;
  // Shapiro-Wilk p-value is the upper tail of the transformed statistic.
  return { statistic: w, pvalue: normalSf(z) };
}

/**
 * One-sample t-test.
 *
 * Tests whether the mean of a sample differs from a hypothesized population mean.
 *
 * @param a - Sample values (any shape; the tensor is flattened)
 * @param popmean - Hypothesized population mean
 * @param alternative - `"two-sided"` (default), `"less"` or `"greater"`
 * @returns The t statistic (`df = n - 1`) and the p-value
 * @throws {InvalidParameterError} If fewer than 2 values are given, the input is constant,
 *   or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { ttest1samp } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = ttest1samp(tensor([2.1, 1.9, 2.4, 2.2, 2.0]), 2.0);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ttest1samp}.
 */
export function ttest_1samp(
  a: Tensor,
  popmean: number,
  alternative: TestAlternative = "two-sided"
): TestResult {
  assertAlternative("ttest_1samp", alternative);
  const x = toDenseArray1D(a);
  const n = x.length;
  if (n < 2) {
    throw new InvalidParameterError("ttest_1samp() requires at least 2 samples", "n", n);
  }

  // mean
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i] ?? 0;
  mean /= n;

  // sample variance
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - mean;
    m2 += d * d;
  }
  const variance = m2 / (n - 1);
  const std = Math.sqrt(variance);
  if (std === 0) {
    throw new InvalidParameterError("ttest_1samp() is undefined for constant input", "std", std);
  }
  const tstat = (mean - popmean) / (std / Math.sqrt(n));

  const df = n - 1;
  return { statistic: tstat, pvalue: tPvalue(tstat, df, alternative) };
}

/** Options of {@link ttest_ind}. */
export interface TTestIndOptions {
  /** Assume equal population variances (default: `true`); `false` gives Welch's t-test. */
  equalVar?: boolean;
  /** Alternative hypothesis (default: `"two-sided"`). */
  alternative?: TestAlternative;
}

/**
 * Independent two-sample t-test.
 *
 * Tests whether means of two independent samples are equal. By default the
 * samples are assumed to have equal variance (Student's t-test); pass
 * `equalVar = false` for Welch's t-test with Welch-Satterthwaite degrees of freedom.
 *
 * @param a - First sample (any shape; the tensor is flattened)
 * @param b - Second sample
 * @param equalVar - Assume equal population variances (default: `true`), or a
 *   {@link TTestIndOptions} object
 * @param alternative - `"two-sided"` (default), `"less"` (mean of `a` is less than
 *   mean of `b`) or `"greater"`
 * @returns The t statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If a sample has fewer than 2 values, the pooled variance is
 *   zero, or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { ttestInd } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([2.1, 1.9, 2.4, 2.2, 2.0]);
 * const b = tensor([2.8, 3.1, 2.6, 3.0]);
 * ttestInd(a, b, { equalVar: false, alternative: "less" });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ttestInd}.
 */
export function ttest_ind(
  a: Tensor,
  b: Tensor,
  equalVar: boolean | TTestIndOptions = true,
  alternative: TestAlternative = "two-sided"
): TestResult {
  if (typeof equalVar === "object" && equalVar !== null) {
    alternative = equalVar.alternative ?? alternative;
    equalVar = equalVar.equalVar ?? true;
  }
  assertAlternative("ttest_ind", alternative);
  const xa = toDenseArray1D(a);
  const xb = toDenseArray1D(b);
  const na = xa.length;
  const nb = xb.length;
  if (na < 2 || nb < 2) {
    throw new InvalidParameterError("ttest_ind() requires at least 2 samples in each group", "n", {
      na,
      nb,
    });
  }

  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < na; i++) meanA += xa[i] ?? 0;
  for (let i = 0; i < nb; i++) meanB += xb[i] ?? 0;
  meanA /= na;
  meanB /= nb;

  let ssa = 0;
  let ssb = 0;
  for (let i = 0; i < na; i++) {
    const d = (xa[i] ?? 0) - meanA;
    ssa += d * d;
  }
  for (let i = 0; i < nb; i++) {
    const d = (xb[i] ?? 0) - meanB;
    ssb += d * d;
  }
  const varA = ssa / (na - 1);
  const varB = ssb / (nb - 1);

  let tstat: number;
  let df: number;

  if (equalVar) {
    const pooledVar = ((na - 1) * varA + (nb - 1) * varB) / (na + nb - 2);
    const denom = Math.sqrt(pooledVar * (1 / na + 1 / nb));
    if (denom === 0)
      throw new InvalidParameterError(
        "ttest_ind() is undefined for constant input",
        "denom",
        denom
      );
    tstat = (meanA - meanB) / denom;
    df = na + nb - 2;
  } else {
    const denom = Math.sqrt(varA / na + varB / nb);
    if (denom === 0)
      throw new InvalidParameterError(
        "ttest_ind() is undefined for constant input",
        "denom",
        denom
      );
    tstat = (meanA - meanB) / denom;
    df = (varA / na + varB / nb) ** 2 / ((varA / na) ** 2 / (na - 1) + (varB / nb) ** 2 / (nb - 1));
  }

  return { statistic: tstat, pvalue: tPvalue(tstat, df, alternative) };
}

/**
 * Paired-sample t-test.
 *
 * Tests whether means of two related samples are equal. Tensors are flattened
 * in row-major order and paired element by element.
 *
 * @param a - First sample
 * @param b - Second sample (same number of elements as `a`)
 * @param alternative - `"two-sided"` (default), `"less"` (mean of `a - b` is less
 *   than zero) or `"greater"`
 * @returns The t statistic (`df = n - 1`) and the p-value
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ttestRel}.
 */
export function ttest_rel(
  a: Tensor,
  b: Tensor,
  alternative: TestAlternative = "two-sided"
): TestResult {
  assertAlternative("ttest_rel", alternative);
  if (a.size !== b.size) {
    throw new InvalidParameterError("ttest_rel() requires paired samples of equal length", "size", {
      a: a.size,
      b: b.size,
    });
  }
  const n = a.size;
  if (n < 2) {
    throw new InvalidParameterError("ttest_rel() requires at least 2 paired samples", "n", n);
  }

  // Differences, paired in flattened row-major order.
  const diffs = toDenseArray1D(a);
  const bd = toDenseArray1D(b);
  for (let k = 0; k < n; k++) {
    diffs[k] = (diffs[k] ?? 0) - (bd[k] ?? 0);
  }

  let mean = 0;
  for (let k = 0; k < n; k++) mean += diffs[k] ?? 0;
  mean /= n;

  let ss = 0;
  for (let k = 0; k < n; k++) {
    const d = (diffs[k] ?? 0) - mean;
    ss += d * d;
  }
  const varDiff = ss / (n - 1);
  const stdDiff = Math.sqrt(varDiff);
  if (stdDiff === 0) {
    throw new InvalidParameterError(
      "ttest_rel() is undefined for constant differences",
      "stdDiff",
      stdDiff
    );
  }
  const tstat = mean / (stdDiff / Math.sqrt(n));
  const df = n - 1;
  return { statistic: tstat, pvalue: tPvalue(tstat, df, alternative) };
}

/**
 * Chi-square goodness of fit test.
 *
 * Observed and expected frequencies must be non-negative and sum to the same total
 * (within floating-point tolerance). Without `f_exp` the categories are assumed to be
 * equally likely.
 *
 * @param f_obs - Observed frequencies (flattened)
 * @param f_exp - Expected frequencies (same number of elements as `f_obs`); uniform if omitted
 * @param ddof - "Delta degrees of freedom": the p-value uses `k - 1 - ddof` degrees of
 *   freedom, where `k` is the number of categories (default: 0)
 * @returns The chi-square statistic and p-value
 *
 * @example
 * ```ts
 * import { chisquare } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = chisquare(tensor([16, 18, 16, 14, 12, 12]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function chisquare(f_obs: Tensor, f_exp?: Tensor, ddof = 0): TestResult {
  const obs = toDenseArray1D(f_obs);
  const n = obs.length;
  if (n < 1) {
    throw new InvalidParameterError("chisquare() requires at least one observed value", "n", n);
  }
  if (!Number.isInteger(ddof) || ddof < 0) {
    throw new InvalidParameterError(
      "chisquare() ddof must be a non-negative integer",
      "ddof",
      ddof
    );
  }
  let chiSq = 0;
  let sumObs = 0;

  for (let i = 0; i < n; i++) {
    const v = obs[i] ?? 0;
    if (!Number.isFinite(v) || v < 0) {
      throw new InvalidParameterError(
        "chisquare() observed frequencies must be finite and >= 0",
        "f_obs",
        v
      );
    }
    sumObs += v;
  }

  if (f_exp && f_obs.size !== f_exp.size) {
    throw new InvalidParameterError(
      "Observed and expected frequency arrays must have the same length",
      "size",
      { f_obs: f_obs.size, f_exp: f_exp.size }
    );
  }

  if (!f_exp) {
    // Uniform expected frequencies
    const expected = sumObs / n;
    if (!Number.isFinite(expected) || expected <= 0) {
      throw new InvalidParameterError(
        "chisquare() expected frequencies must be finite and > 0",
        "expected",
        expected
      );
    }
    for (let i = 0; i < n; i++) {
      const v = obs[i] ?? 0;
      chiSq += (v - expected) ** 2 / expected;
    }
  } else {
    const exp = toDenseArray1D(f_exp);
    let sumExp = 0;
    for (let i = 0; i < n; i++) {
      const v = exp[i] ?? 0;
      if (!Number.isFinite(v) || v <= 0) {
        throw new InvalidParameterError(
          "chisquare() expected frequencies must be finite and > 0",
          "f_exp",
          v
        );
      }
      sumExp += v;
    }

    const rtol = Math.sqrt(Number.EPSILON);
    const denom = Math.max(Math.abs(sumObs), Math.abs(sumExp));
    if (Math.abs(sumObs - sumExp) > rtol * denom) {
      throw new InvalidParameterError(
        "chisquare() expected and observed frequencies must sum to the same value",
        "sum",
        { f_obs: sumObs, f_exp: sumExp }
      );
    }

    for (let i = 0; i < n; i++) {
      const vObs = obs[i] ?? 0;
      const vExp = exp[i] ?? 0;
      chiSq += (vObs - vExp) ** 2 / vExp;
    }
  }

  const df = n - 1 - ddof;
  if (df < 1) {
    throw new InvalidParameterError(
      ddof === 0
        ? "chisquare() requires at least 2 categories (df must be >= 1)"
        : "chisquare() requires k - 1 - ddof >= 1 degrees of freedom",
      "df",
      df
    );
  }
  const pvalue = chiSquareSf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

// Kolmogorov-Smirnov distribution helpers

const PI_SQUARED = Math.PI * Math.PI;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Round half to even, as `numpy.round` does. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Survival function of the Kolmogorov limit distribution, P(K > x). */
function kolmogorovSf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  if (x <= 0) return 1;
  if (x < 1) {
    // Jacobi theta transform: converges quickly for small x, where the
    // alternating series below needs many terms and cancels badly.
    const c = Math.sqrt(2 * Math.PI) / x;
    const x2 = x * x;
    let sum = 0;
    for (let k = 1; k < 200; k += 2) {
      const term = Math.exp(-((k * Math.PI) ** 2) / (8 * x2));
      sum += term;
      if (term < 1e-18 * sum) break;
    }
    return Math.max(0, Math.min(1, 1 - c * sum));
  }
  let sum = 0;
  for (let k = 1; k < 200; k++) {
    const term = Math.exp(-2 * k * k * x * x);
    sum += k % 2 === 1 ? term : -term;
    if (term < 1e-17) break;
  }
  return Math.max(0, Math.min(1, 2 * sum));
}

/**
 * Exact one-sided Smirnov survival function P(D+_n >= d) (Birnbaum and Tingey, 1951):
 * d * sum_{j=0..floor(n (1 - d))} C(n, j) (d + j / n)^(j - 1) (1 - d - j / n)^(n - j).
 * All terms are positive, so the sum is evaluated in log space without cancellation.
 */
function smirnovSf(n: number, d: number): number {
  if (Number.isNaN(d)) return Number.NaN;
  if (d <= 0) return 1;
  if (d >= 1) return 0;
  const jmax = Math.floor(n * (1 - d));
  let logChoose = 0;
  let sum = 0;
  let prevTerm = 0;
  for (let j = 0; j <= jmax; j++) {
    if (j > 0) logChoose += Math.log((n - j + 1) / j);
    const b = 1 - d - j / n;
    if (b <= 0) continue; // the factor b^(n - j) is zero
    const term = Math.exp(logChoose + (j - 1) * Math.log(d + j / n) + (n - j) * Math.log(b));
    sum += term;
    // The terms rise to a single peak and then decay quickly.
    if (term < prevTerm && term < 1e-18 * sum) break;
    prevTerm = term;
  }
  return Math.min(1, d * sum);
}

function matMul(a: Float64Array, b: Float64Array, m: number): Float64Array {
  const c = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < m; k++) {
      const aik = a[i * m + k] as number;
      if (aik === 0) continue;
      const rowB = k * m;
      const rowC = i * m;
      for (let j = 0; j < m; j++) {
        c[rowC + j] = (c[rowC + j] as number) + aik * (b[rowB + j] as number);
      }
    }
  }
  return c;
}

/** Matrix power with a decimal exponent to avoid overflow (Marsaglia, Tsang & Wang, 2003). */
function matPow(a: Float64Array, ea: number, m: number, n: number): { v: Float64Array; e: number } {
  if (n === 1) return { v: a.slice(), e: ea };
  const half = matPow(a, ea, m, Math.floor(n / 2));
  const sq = matMul(half.v, half.v, m);
  let v: Float64Array;
  let e: number;
  if (n % 2 === 0) {
    v = sq;
    e = 2 * half.e;
  } else {
    v = matMul(a, sq, m);
    e = ea + 2 * half.e;
  }
  const mid = Math.floor(m / 2) * m + Math.floor(m / 2);
  if ((v[mid] as number) > 1e140) {
    for (let i = 0; i < v.length; i++) v[i] = (v[i] as number) * 1e-140;
    e += 140;
  }
  return { v, e };
}

/**
 * Exact P(D_n < d) for the one-sample two-sided Kolmogorov-Smirnov statistic
 * (Marsaglia, Tsang & Wang, 2003). Cost grows with (n * d)^3 log n.
 */
function kolmogorovExactCdf(n: number, d: number): number {
  if (d <= 1 / (2 * n)) return 0;
  if (d >= 1) return 1;
  const nd = n * d;
  const k = Math.floor(nd) + 1;
  const m = 2 * k - 1;
  const h = k - nd;
  const H = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      H[i * m + j] = i - j + 1 < 0 ? 0 : 1;
    }
  }
  for (let i = 0; i < m; i++) {
    H[i * m] = (H[i * m] as number) - h ** (i + 1);
    H[(m - 1) * m + i] = (H[(m - 1) * m + i] as number) - h ** (m - i);
  }
  H[(m - 1) * m] = (H[(m - 1) * m] as number) + (2 * h - 1 > 0 ? (2 * h - 1) ** m : 0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i - j + 1 > 0) {
        for (let g = 1; g <= i - j + 1; g++) {
          H[i * m + j] = (H[i * m + j] as number) / g;
        }
      }
    }
  }
  const { v: Q, e: eQ0 } = matPow(H, 0, m, n);
  let eQ = eQ0;
  let s = Q[(k - 1) * m + (k - 1)] as number;
  for (let i = 1; i <= n; i++) {
    s = (s * i) / n;
    if (s < 1e-140) {
      s *= 1e140;
      eQ -= 140;
    }
  }
  return s * 10 ** eQ;
}

/**
 * Pelz-Good (1976) approximation of P(D_n < d) for large n, written with Jacobi theta
 * functions so that it is accurate for small `z = d * sqrt(n)`.
 */
function kolmogorovPelzGoodCdf(n: number, d: number): number {
  if (d <= 0) return 0;
  if (d >= 1) return 1;
  const z = Math.sqrt(n) * d;
  const z2 = z * z;
  const z3 = z2 * z;
  const z4 = z2 * z2;
  const z6 = z4 * z2;
  const z8 = z4 * z4;
  const qlog = -PI_SQUARED / 8 / z2;
  if (qlog < -708) return 0;
  const q = Math.exp(qlog);
  const pi4 = PI_SQUARED * PI_SQUARED;
  const pi6 = pi4 * PI_SQUARED;

  const k1a = -z2;
  const k1b = PI_SQUARED / 4;
  const k2a = 6 * z6 + 2 * z4;
  const k2b = ((2 * z4 - 5 * z2) * PI_SQUARED) / 4;
  const k2c = (pi4 * (1 - 2 * z2)) / 16;
  const k3d = (pi6 * (5 - 30 * z2)) / 64;
  const k3c = (pi4 * (-60 * z2 + 212 * z4)) / 16;
  const k3b = (PI_SQUARED * (135 * z4 - 96 * z6)) / 4;
  const k3a = -30 * z6 - 90 * z8;

  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  const maxk = Math.ceil((16 * z) / Math.PI);
  // Horner scheme for sum c_k q^((2k - 1)^2).
  for (let k = maxk; k >= 1; k--) {
    const m = 2 * k - 1;
    const m2 = m * m;
    const m4 = m2 * m2;
    const m6 = m4 * m2;
    const qpower = q ** (8 * k);
    s0 = s0 * qpower + 1;
    s1 = s1 * qpower + (k1a + k1b * m2);
    s2 = s2 * qpower + (k2a + k2b * m2 + k2c * m4);
    s3 = s3 * qpower + (k3a + k3b * m2 + k3c * m4 + k3d * m6);
  }
  const scale = q * SQRT_2PI;
  s0 = (s0 * scale) / z;
  s1 = (s1 * scale) / (6 * z4);
  s2 = (s2 * scale) / (72 * z ** 7);
  s3 = (s3 * scale) / (6480 * z ** 10);

  const q2 = Math.exp(-PI_SQUARED / 2 / z2);
  const sqrt3z = Math.sqrt(3) * z;
  let k2extra = 0;
  let k3extra = 0;
  for (let k = 1; k <= maxk; k++) {
    const k2 = k * k;
    const qp = q2 ** k2;
    k2extra += k2 * qp;
    k3extra += (sqrt3z + Math.PI * k) * (sqrt3z - Math.PI * k) * k2 * qp;
  }
  s2 += (k2extra * PI_SQUARED * SQRT_2PI) / (-36 * z3);
  s3 += (k3extra * PI_SQUARED * SQRT_2PI) / (216 * z6);

  const sn = Math.sqrt(n);
  return s0 + s1 / sn + s2 / n + s3 / (n * sn);
}

/**
 * P(D_n >= d) for the one-sample two-sided Kolmogorov-Smirnov statistic. This follows
 * the case analysis of Simard and L'Ecuyer (2011) used by SciPy's `kstwo.sf`.
 */
function kolmogorovTwoSidedSf(n: number, d: number): number {
  if (Number.isNaN(d)) return Number.NaN;
  if (d >= 1) return 0;
  if (d <= 0) return 1;
  const t = n * d;
  if (t <= 1) {
    if (t <= 0.5) return 1;
    // Ruben and Gambino: 1/(2n) <= d <= 1/n.
    return 1 - Math.exp(logGamma(n + 1) - n * Math.log(n) + n * Math.log(2 * t - 1));
  }
  if (d >= 0.5 || t >= n - 1) return Math.min(1, 2 * smirnovSf(n, d));
  const nd2 = t * d;
  if (n <= 140) {
    if (nd2 <= 4) return Math.max(0, Math.min(1, 1 - kolmogorovExactCdf(n, d)));
    return Math.min(1, 2 * smirnovSf(n, d));
  }
  if (nd2 >= 370) return 0;
  if (nd2 >= 2.2) return Math.min(1, 2 * smirnovSf(n, d));
  const cdf =
    n <= 100000 && n * d ** 1.5 <= 1.4 ? kolmogorovExactCdf(n, d) : kolmogorovPelzGoodCdf(n, d);
  return Math.max(0, Math.min(1, 1 - cdf));
}

/** p-value method of the Kolmogorov-Smirnov tests. */
type KsMethod = "auto" | "exact" | "asymptotic";

/** Options of {@link kstest} and {@link ks_2samp}. */
export interface KsTestOptions {
  /**
   * Alternative hypothesis (default: `"two-sided"`). For {@link kstest}, `"greater"` means the
   * empirical distribution function lies above the hypothesized one (statistic `D+`) and `"less"`
   * that it lies below (statistic `D-`). For {@link ks_2samp}, `"greater"` means the distribution
   * function of the first sample lies above that of the second.
   */
  alternative?: TestAlternative;
  /**
   * p-value method. `"auto"` (default) uses the exact null distribution. For {@link ks_2samp}
   * it switches to the asymptotic distribution when a sample has more than 10000 values (the
   * same rule as SciPy). One-sided {@link kstest} p-values are always exact.
   */
  method?: KsMethod;
}

function assertKsOptions(
  fn: string,
  options: KsTestOptions
): { alternative: TestAlternative; method: KsMethod } {
  const alternative = options.alternative ?? "two-sided";
  const method = options.method ?? "auto";
  assertAlternative(fn, alternative);
  if (method !== "auto" && method !== "exact" && method !== "asymptotic") {
    throw new InvalidParameterError(
      `${fn}() method must be "auto", "exact" or "asymptotic"`,
      "method",
      method
    );
  }
  return { alternative, method };
}

/**
 * P(D >= d) for the one-sample Kolmogorov-Smirnov statistic. As in SciPy, one-sided p-values
 * always use the exact Smirnov distribution, and the two-sided p-value uses the exact
 * distribution unless `method` is `"asymptotic"`.
 */
function ksOneSamplePvalue(
  d: number,
  n: number,
  alternative: TestAlternative,
  method: KsMethod
): number {
  if (Number.isNaN(d)) return Number.NaN;
  if (alternative !== "two-sided") return smirnovSf(n, d);
  return method === "asymptotic" ? kolmogorovSf(d * Math.sqrt(n)) : kolmogorovTwoSidedSf(n, d);
}

function gcd(a: number, b: number): number {
  let x = a;
  let y = b;
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

/**
 * Exact P(D >= h / lcm(n1, n2)) for the two-sample Kolmogorov-Smirnov statistic under the
 * null hypothesis (no ties). Returns `undefined` when the lattice-path computation would be
 * too expensive, in which case the caller falls back to the asymptotic distribution.
 */
function ksTwoSampleExactPvalue(
  n1: number,
  n2: number,
  d: number,
  oneSided: boolean
): number | undefined {
  const g = gcd(n1, n2);
  const lcm = (n1 / g) * n2;
  const h = Math.round(d * lcm);
  if (h <= 0) return 1;
  if (h > lcm) return 0;

  if (n1 === n2) {
    const n = n1;
    if (oneSided) {
      // P(D+ >= h / n) = C(2n, n - h) / C(2n, n)
      let p = 1;
      for (let j = 0; j < h; j++) p *= (n - j) / (n + j + 1);
      return Math.max(0, Math.min(1, p));
    }
    // Closed form (Hodges, 1958) evaluated with a Horner-like recursion.
    let P = 0;
    for (let k = Math.floor(n / h); k >= 0; k--) {
      let p1 = 1;
      for (let j = 0; j < h; j++) {
        p1 = ((n - k * h - j) * p1) / (n + k * h + j + 1);
      }
      P = p1 * (1 - P);
    }
    return Math.max(0, Math.min(1, 2 * P));
  }

  // Count lattice paths from (0, 0) to (n1, n2) that stay strictly inside the band
  // i * (n2 / g) - j * (n1 / g) < h (and > -h when two-sided), as a probability
  // (path count / C(i + j, i)).
  const A = n2 / g;
  const B = n1 / g;
  const width = oneSided ? n2 + 1 : Math.min(n2 + 1, Math.ceil((2 * h) / B) + 2);
  if ((n1 + 1) * width > 2e8) return undefined;
  let prev = new Float64Array(n2 + 1);
  let cur = new Float64Array(n2 + 1);
  let prevLo = 1;
  let prevHi = 0;
  for (let i = 0; i <= n1; i++) {
    // Only nodes inside the band are written; stale entries outside it are never read.
    const lo = Math.max(0, Math.floor((i * A - h) / B) + 1);
    const hi = oneSided ? n2 : Math.min(n2, Math.ceil((i * A + h) / B) - 1);
    for (let j = lo; j <= hi; j++) {
      if (i === 0 && j === 0) {
        cur[0] = 1;
        continue;
      }
      const up = i > 0 && j >= prevLo && j <= prevHi ? (prev[j] as number) : 0;
      const left = j > lo ? (cur[j - 1] as number) : 0;
      cur[j] = (i * up + j * left) / (i + j);
    }
    const tmp = prev;
    prev = cur;
    cur = tmp;
    prevLo = lo;
    prevHi = hi;
  }
  const inside = prevLo <= n2 && n2 <= prevHi ? (prev[n2] as number) : 0;
  return Math.max(0, Math.min(1, 1 - inside));
}

/**
 * Kolmogorov-Smirnov test for goodness of fit.
 *
 * Compares the empirical distribution of `data` with a fully specified
 * continuous distribution. The p-value comes from the exact distribution of the statistic
 * (Marsaglia-Tsang-Wang, Smirnov and Pelz-Good approximations, as in SciPy's `kstwo`), or
 * from the Kolmogorov limit distribution with `method: "asymptotic"`.
 *
 * @param data - Sample values (flattened)
 * @param cdf - `"norm"` for the standard normal distribution, or a function returning the
 *   cumulative distribution function of the hypothesized distribution at a point
 * @param options - `alternative` (`"two-sided"` by default) and `method`
 *   (see {@link KsTestOptions})
 * @returns The KS statistic (`D`, `D+` or `D-` depending on `alternative`) and the p-value.
 *   Both are `NaN` if the data or the distribution function produce `NaN`.
 * @throws {InvalidParameterError} If `data` is empty, `cdf` is an unknown name, or an option
 *   is invalid
 *
 * @example
 * ```ts
 * import { kstest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * kstest(tensor([0.1, -0.3, 1.2, 0.8, -1.5, 0.4, 2.2]), "norm");
 * // { statistic: 0.2541..., pvalue: 0.6692... }
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function kstest(
  data: Tensor,
  cdf: string | ((x: number) => number),
  options: KsTestOptions = {}
): TestResult {
  const { alternative, method } = assertKsOptions("kstest", options);
  const x = toDenseSortedArray1D(data);
  const n = x.length;
  if (n === 0) {
    throw new InvalidParameterError("kstest() requires at least one element", "n", n);
  }

  if (typeof cdf === "string" && cdf !== "norm") {
    throw new InvalidParameterError(
      `Unsupported distribution: '${cdf}'. Supported distributions: 'norm'`,
      "cdf",
      cdf
    );
  }

  const F = typeof cdf === "string" ? (v: number) => normalCdf(v) : cdf;

  let dPlus = 0;
  let dMinus = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? 0;
    const fi = F(xi);
    if (Number.isNaN(fi) || Number.isNaN(xi)) {
      return { statistic: Number.NaN, pvalue: Number.NaN };
    }
    dPlus = Math.max(dPlus, (i + 1) / n - fi);
    dMinus = Math.max(dMinus, fi - i / n);
  }

  const d =
    alternative === "greater" ? dPlus : alternative === "less" ? dMinus : Math.max(dPlus, dMinus);
  return { statistic: d, pvalue: ksOneSamplePvalue(d, n, alternative, method) };
}

/**
 * D'Agostino-Pearson omnibus test for normality.
 *
 * Combines the skewness test and the Anscombe-Glynn kurtosis test into
 * `K2 = Z1^2 + Z2^2`, which follows a chi-square distribution with 2 degrees of
 * freedom under normality. Matches `scipy.stats.normaltest`.
 *
 * @param a - Sample values (flattened, at least 8 values, not all equal)
 * @returns The `K2` statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than 8 values are given or the input is constant
 *
 * @example
 * ```ts
 * import { normaltest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = normaltest(tensor([2.1, 1.9, 2.4, 2.2, 2.0, 2.6, 1.7, 2.3, 2.5]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function normaltest(a: Tensor): TestResult {
  const x = toDenseArray1D(a);
  const n = x.length;
  if (n < 8) {
    throw new InvalidParameterError("normaltest() requires at least 8 samples", "n", n);
  }
  if (hasNaN(x)) return { statistic: Number.NaN, pvalue: Number.NaN };
  const { mean, m2 } = meanAndM2(x);
  const m2n = m2 / n;
  const std = Math.sqrt(m2n);
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError(
      "normaltest() is undefined for constant input (or when a value is infinite)",
      "std",
      std
    );
  }

  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - mean;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  const skew = m3 / n / m2n ** 1.5;
  const kurt = m4 / n / (m2n * m2n);

  // Skewness component (D'Agostino)
  const y = skew * Math.sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)));
  const beta2 =
    (3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9));
  const w2 = -1 + Math.sqrt(2 * (beta2 - 1));
  const delta = 1 / Math.sqrt(0.5 * Math.log(w2));
  const alpha = Math.sqrt(2 / (w2 - 1));
  const yScaled = y / alpha;
  const z1 = delta * Math.log(yScaled + Math.sqrt(yScaled * yScaled + 1));

  // Kurtosis component (Anscombe-Glynn)
  const e = (3 * (n - 1)) / (n + 1);
  const varb2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  const xval = (kurt - e) / Math.sqrt(varb2);
  const sqrtbeta1 =
    ((6 * (n * n - 5 * n + 2)) / ((n + 7) * (n + 9))) *
    Math.sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)));
  const aTerm = 6 + (8 / sqrtbeta1) * (2 / sqrtbeta1 + Math.sqrt(1 + 4 / (sqrtbeta1 * sqrtbeta1)));
  const term1 = 1 - 2 / (9 * aTerm);
  const denom = 1 + xval * Math.sqrt(2 / (aTerm - 4));
  const term2 =
    denom === 0 ? Number.NaN : Math.sign(denom) * ((1 - 2 / aTerm) / Math.abs(denom)) ** (1 / 3);
  const z2 = (term1 - term2) / Math.sqrt(2 / (9 * aTerm));

  const k2 = z1 * z1 + z2 * z2;
  const pvalue = chiSquareSf(k2, 2);
  return { statistic: k2, pvalue };
}

/**
 * Shapiro-Wilk test for normality.
 *
 * Uses Royston's algorithm AS R94, the same as `scipy.stats.shapiro`. Small values of the
 * `W` statistic indicate departure from normality.
 *
 * @param x - Sample values (flattened, between 3 and 5000 values, not all equal)
 * @returns The `W` statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If the number of values is outside 3 to 5000, or all values
 *   are identical
 *
 * @example
 * ```ts
 * import { shapiro } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = shapiro(tensor([2.1, 1.9, 2.4, 2.2, 2.0, 2.6, 1.7]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function shapiro(x: Tensor): TestResult {
  const sorted = toDenseSortedArray1D(x);
  if (sorted.length >= 3 && sorted.length <= 5000 && hasNaN(sorted)) {
    return { statistic: Number.NaN, pvalue: Number.NaN };
  }
  return shapiroWilk(sorted);
}

/**
 * Result of {@link anderson}.
 *
 * `criticalValues[i]` is the critical value of the statistic at significance level
 * `significanceLevel[i]` (given as a fraction, e.g. `0.05`; SciPy reports percentages).
 * Reject normality at that level when `statistic` exceeds the critical value.
 */
export type AndersonResult = {
  statistic: number;
  criticalValues: number[];
  significanceLevel: number[];
  /**
   * Same as `criticalValues`.
   *
   * @deprecated Prefer `criticalValues`.
   */
  critical_values: number[];
  /**
   * Same as `significanceLevel`.
   *
   * @deprecated Prefer `significanceLevel`.
   */
  significance_level: number[];
  /** Approximate p-value (D'Agostino and Stephens, 1986). */
  pvalue: number;
};

/** log Φ(z), accurate in both tails. */
function logNormalCdf(z: number): number {
  if (z > 0) return Math.log1p(-normalSf(z));
  const tail = normalSf(-z);
  if (tail > 0) return Math.log(tail);
  // Φ(z) underflows below about z = -38; use the leading asymptotic term.
  return -0.5 * z * z - Math.log(-z * Math.sqrt(2 * Math.PI));
}

/**
 * Anderson-Darling test for normality.
 *
 * The mean and standard deviation (with `n - 1` in the denominator) are estimated from
 * the data. Critical values are the asymptotic points for the case of unknown mean and
 * variance (0.561, 0.631, 0.752, 0.873, 1.035), divided by the small-sample factor
 * `1 + 0.75 / n + 2.25 / n^2` (D'Agostino and Stephens, 1986), the same as SciPy's
 * `anderson(x, "norm")`.
 *
 * @param x - Sample values (flattened, at least 2 values, not all equal)
 * @returns The statistic A², critical values at the significance levels
 *   15%, 10%, 5%, 2.5% and 1% (as fractions) and an approximate p-value
 *
 * @example
 * ```ts
 * import { anderson } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const res = anderson(tensor([1, 2, 3, 4, 5.5]));
 * // res.statistic is about 0.1436; res.criticalValues[2] (5% level) is 0.606
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function anderson(x: Tensor): AndersonResult {
  const sorted = toDenseSortedArray1D(x);
  const n = sorted.length;
  if (n < 1) {
    throw new InvalidParameterError("anderson() requires at least one element", "n", n);
  }
  if (n < 2) {
    throw new InvalidParameterError(
      "anderson() requires at least 2 elements (n = 1 given)",
      "n",
      n
    );
  }
  const correction = 1 + 0.75 / n + 2.25 / (n * n);
  const baseCritical = [0.561, 0.631, 0.752, 0.873, 1.035];
  const critical_values = baseCritical.map((v) => Math.round((v / correction) * 1000) / 1000);
  const significance_level = [0.15, 0.1, 0.05, 0.025, 0.01];
  if (hasNaN(sorted)) {
    return {
      statistic: Number.NaN,
      criticalValues: critical_values.slice(),
      significanceLevel: significance_level.slice(),
      critical_values,
      significance_level,
      pvalue: Number.NaN,
    };
  }
  const { mean, m2 } = meanAndM2(sorted);
  const std = Math.sqrt(m2 / (n - 1));
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError(
      "anderson() is undefined for constant input (or when a value is infinite)",
      "std",
      std
    );
  }

  // A^2 = -n - (1/n) sum_{i=1..n} (2i-1) [ln Phi(z_i) + ln(1 - Phi(z_{n+1-i}))]
  let A2 = 0;
  for (let i = 0; i < n; i++) {
    const zi = ((sorted[i] ?? 0) - mean) / std;
    const zj = ((sorted[n - 1 - i] ?? 0) - mean) / std;
    A2 += (2 * (i + 1) - 1) * (logNormalCdf(zi) + logNormalCdf(-zj));
  }
  A2 = -n - A2 / n;

  // p-value from the modified statistic A*^2 (D'Agostino and Stephens, 1986, table 4.9).
  const aa = A2 * correction;
  let pvalue: number;
  if (Number.isNaN(aa)) pvalue = Number.NaN;
  else if (aa < 0.2) pvalue = 1 - Math.exp(-13.436 + 101.14 * aa - 223.73 * aa * aa);
  else if (aa < 0.34) pvalue = 1 - Math.exp(-8.318 + 42.796 * aa - 59.938 * aa * aa);
  else if (aa < 0.6) pvalue = Math.exp(0.9177 - 4.279 * aa - 1.38 * aa * aa);
  else if (aa < 10) pvalue = Math.exp(1.2937 - 5.709 * aa + 0.0186 * aa * aa);
  else pvalue = 3.7e-24;

  return {
    statistic: A2,
    criticalValues: critical_values.slice(),
    significanceLevel: significance_level.slice(),
    critical_values,
    significance_level,
    pvalue,
  };
}

/** Options shared by {@link mannwhitneyu} and {@link wilcoxon}. */
export interface RankTestOptions {
  /** Alternative hypothesis (default: `"two-sided"`). */
  alternative?: TestAlternative;
  /**
   * p-value method. `"auto"` (default) uses the exact null distribution for small samples
   * without ties and the normal approximation otherwise; `"exact"` and `"asymptotic"`
   * force one of them.
   */
  method?: "auto" | "exact" | "asymptotic";
  /**
   * Apply a continuity correction of 0.5 in the normal approximation. Default: `true` for
   * {@link mannwhitneyu} and `false` for {@link wilcoxon} (the same defaults as SciPy).
   */
  correction?: boolean;
}

function assertMethod(
  fn: string,
  method: unknown
): asserts method is "auto" | "exact" | "asymptotic" {
  if (method !== "auto" && method !== "exact" && method !== "asymptotic") {
    throw new InvalidParameterError(
      `${fn}() method must be "auto", "exact" or "asymptotic"`,
      "method",
      method
    );
  }
}

/**
 * Null probability mass function of the Mann-Whitney U statistic for sample sizes
 * `n1` and `n2` without ties: pmf[u] = P(U = u), u = 0..n1*n2.
 */
function mannWhitneyNullPmf(n1: number, n2: number): Float64Array {
  const m = Math.min(n1, n2);
  const n = Math.max(n1, n2);
  const size = m * n;
  // Coefficients of the Gaussian binomial [m + n choose m]_q = prod (1 - q^(n+i)) / (1 - q^i).
  const poly = new Float64Array(size + 1);
  poly[0] = 1;
  for (let i = 1; i <= m; i++) {
    for (let k = i; k <= size; k++) {
      poly[k] = (poly[k] as number) + (poly[k - i] as number);
    }
    const shift = n + i;
    for (let k = size; k >= shift; k--) {
      poly[k] = (poly[k] as number) - (poly[k - shift] as number);
    }
  }
  let total = 0;
  for (let k = 0; k <= size; k++) total += poly[k] as number;
  for (let k = 0; k <= size; k++) poly[k] = (poly[k] as number) / total;
  return poly;
}

/** P(U >= k) from a null pmf that is symmetric about its midpoint. */
function pmfUpperTail(pmf: Float64Array, k: number): number {
  const size = pmf.length - 1;
  if (k <= 0) return 1;
  if (k > size) return 0;
  let s = 0;
  if (k >= size / 2) {
    for (let j = size; j >= k; j--) s += pmf[j] as number;
    return Math.min(1, s);
  }
  for (let j = 0; j < k; j++) s += pmf[j] as number;
  return Math.max(0, 1 - s);
}

/** P(U <= k) from a null pmf that is symmetric about its midpoint. */
function pmfLowerTail(pmf: Float64Array, k: number): number {
  const size = pmf.length - 1;
  if (k < 0) return 0;
  if (k >= size) return 1;
  let s = 0;
  if (k <= size / 2) {
    for (let j = 0; j <= k; j++) s += pmf[j] as number;
    return Math.min(1, s);
  }
  for (let j = size; j > k; j--) s += pmf[j] as number;
  return Math.max(0, 1 - s);
}

/**
 * Mann-Whitney U test (non-parametric).
 *
 * Tests whether two independent samples come from the same distribution.
 *
 * The p-value is exact when one of the samples has at most 8 values and there are no
 * ties (`method: "auto"`, as in SciPy). Otherwise the normal approximation with tie
 * correction and a continuity correction of 0.5 is used.
 *
 * Like SciPy, the returned statistic is always `U1`, the statistic of `x`
 * (`U2 = nx * ny - U1` is the statistic of `y`). The p-value of the two-sided test is
 * computed from the larger of `U1` and `U2`.
 *
 * @param x - First sample
 * @param y - Second sample
 * @param options - `alternative`, `method` and `correction` (see {@link RankTestOptions});
 *   `"greater"` tests whether `x` is stochastically greater than `y`
 * @returns `U1` and the p-value (`NaN` when all values are tied, or when a value is `NaN`)
 * @throws {InvalidParameterError} If a sample is empty, an option is invalid, or
 *   `method: "exact"` is requested for data with ties or for very large samples
 *
 * @example
 * ```ts
 * import { mannwhitneyu } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * mannwhitneyu(tensor([19, 22, 16, 29, 24]), tensor([20, 11, 17, 12]));
 * // { statistic: 17, pvalue: 0.1111... }
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function mannwhitneyu(x: Tensor, y: Tensor, options: RankTestOptions = {}): TestResult {
  const alternative = options.alternative ?? "two-sided";
  const method = options.method ?? "auto";
  assertAlternative("mannwhitneyu", alternative);
  assertMethod("mannwhitneyu", method);
  const nx = x.size;
  const ny = y.size;

  if (nx < 1 || ny < 1) {
    throw new InvalidParameterError("Both samples must be non-empty", "size", {
      x: nx,
      y: ny,
    });
  }

  const xVals = toDenseArray1D(x);
  const yVals = toDenseArray1D(y);
  const n = nx + ny;
  const combined = new Float64Array(n);
  combined.set(xVals, 0);
  combined.set(yVals, nx);

  if (hasNaN(combined)) return { statistic: Number.NaN, pvalue: Number.NaN };

  const { ranks, tieSum } = rankData(combined);

  // Rank sum for group 1 (first sample)
  let R1 = 0;
  for (let i = 0; i < nx; i++) {
    R1 += ranks[i] ?? 0;
  }

  const U1 = R1 - (nx * (nx + 1)) / 2;
  const U2 = nx * ny - U1;
  const statistic = U1;

  const hasTies = tieSum > 0;
  const small = Math.min(nx, ny) <= 8 && nx * ny <= 4e6;
  if (method === "exact" && hasTies) {
    throw new InvalidParameterError(
      'mannwhitneyu() method "exact" is not available when the data contain ties',
      "method",
      method
    );
  }
  if (method === "exact" && Math.min(nx, ny) ** 2 * Math.max(nx, ny) > 4e9) {
    throw new InvalidParameterError(
      'mannwhitneyu() method "exact" is too expensive for these sample sizes',
      "method",
      { nx, ny }
    );
  }
  const useExact = method === "exact" || (method === "auto" && small && !hasTies);

  // Statistic whose upper tail gives the p-value: U1 for "greater", U2 for "less"
  // and the larger of the two for the two-sided test.
  const uTail = alternative === "greater" ? U1 : alternative === "less" ? U2 : Math.max(U1, U2);
  const factor = alternative === "two-sided" ? 2 : 1;

  if (useExact) {
    const pmf = mannWhitneyNullPmf(nx, ny);
    return { statistic, pvalue: Math.min(1, factor * pmfUpperTail(pmf, Math.round(uTail))) };
  }

  const meanU = (nx * ny) / 2;
  // Tie correction uses sum(t^3 - t) over tied groups.
  const tieAdj = n > 1 ? tieSum / (n * (n - 1)) : 0;
  const varU = (nx * ny * (n + 1 - tieAdj)) / 12;
  if (varU <= 0) {
    return { statistic, pvalue: Number.NaN };
  }
  const continuity = (options.correction ?? true) ? 0.5 : 0;
  const z = (uTail - meanU - continuity) / Math.sqrt(varU);
  return { statistic, pvalue: Math.max(0, Math.min(1, factor * normalSf(z))) };
}

/**
 * Null probability mass function of the sum of the weights selected by independent
 * fair coin flips: pmf[s] = P(sum of selected weights = s). Weights are positive integers.
 */
function signFlipNullPmf(weights: readonly number[]): Float64Array {
  let total = 0;
  for (const w of weights) total += w;
  const pmf = new Float64Array(total + 1);
  pmf[0] = 1;
  let reach = 0;
  for (const w of weights) {
    reach += w;
    for (let s = reach; s >= 0; s--) {
      const stay = pmf[s] as number;
      const take = s >= w ? (pmf[s - w] as number) : 0;
      pmf[s] = 0.5 * (stay + take);
    }
  }
  return pmf;
}

/** Options of {@link wilcoxon}. */
export interface WilcoxonOptions extends RankTestOptions {
  /**
   * Treatment of zero differences (default: `"wilcox"`).
   *
   * - `"wilcox"`: discard zero differences.
   * - `"pratt"`: rank the zero differences together with the others, then drop their ranks
   *   from the statistic (Pratt, 1959).
   * - `"zsplit"`: rank the zero differences and split their ranks equally between `W+` and `W-`.
   */
  zeroMethod?: "wilcox" | "pratt" | "zsplit";
}

/**
 * Wilcoxon signed-rank test (non-parametric paired test).
 *
 * With `method: "auto"` (the SciPy default) the p-value is exact for at most 50 differences
 * without ties or zeros, comes from the exact sign-flip distribution of the tied ranks for at
 * most 13 differences with ties or zeros, and uses the normal approximation with tie
 * correction otherwise. Zero differences are discarded unless `zeroMethod` says otherwise.
 *
 * Like SciPy, the returned statistic is `min(W+, W-)` for the two-sided test and `W+`, the
 * sum of the ranks of the positive differences, for one-sided tests.
 *
 * @param x - First sample, or the differences when `y` is omitted
 * @param y - Second sample (same number of elements as `x`)
 * @param options - `alternative`, `method`, `correction` and `zeroMethod`
 *   (see {@link WilcoxonOptions}); `"greater"` tests whether `x - y` is stochastically greater
 *   than zero
 * @returns The statistic and the p-value (`NaN` if the normal approximation has zero variance
 *   or a value is `NaN`)
 * @throws {InvalidParameterError} If the lengths differ, all differences are zero, or an option
 *   is invalid
 *
 * @example
 * ```ts
 * import { wilcoxon } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * wilcoxon(tensor([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]));
 * // { statistic: 1, pvalue: 0.03125 }
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function wilcoxon(x: Tensor, y?: Tensor, options: WilcoxonOptions = {}): TestResult {
  const alternative = options.alternative ?? "two-sided";
  const method = options.method ?? "auto";
  const zeroMethod = options.zeroMethod ?? "wilcox";
  assertAlternative("wilcoxon", alternative);
  assertMethod("wilcoxon", method);
  if (zeroMethod !== "wilcox" && zeroMethod !== "pratt" && zeroMethod !== "zsplit") {
    throw new InvalidParameterError(
      'wilcoxon() zeroMethod must be "wilcox", "pratt" or "zsplit"',
      "zeroMethod",
      zeroMethod
    );
  }
  const n = x.size;
  let d: Float64Array;
  if (y) {
    if (x.size !== y.size) {
      throw new InvalidParameterError("Paired samples must have equal length", "size", {
        x: x.size,
        y: y.size,
      });
    }
    d = toDenseArray1D(x);
    const yd = toDenseArray1D(y);
    for (let i = 0; i < n; i++) d[i] = (d[i] ?? 0) - (yd[i] ?? 0);
  } else {
    d = toDenseArray1D(x);
  }

  let nZero = 0;
  for (let i = 0; i < n; i++) if (d[i] === 0) nZero++;
  if (n === 0) {
    throw new InvalidParameterError("wilcoxon() requires at least one observation", "n", n);
  }
  if (nZero === n) {
    throw new InvalidParameterError(
      "wilcoxon() is undefined when all differences are zero",
      "diffs",
      0
    );
  }
  if (hasNaN(d)) return { statistic: Number.NaN, pvalue: Number.NaN };

  // Zero differences take part in the ranking only for "pratt" and "zsplit".
  const keepZeros = zeroMethod !== "wilcox";
  const used = keepZeros ? n : n - nZero;
  const absDiffs = new Float64Array(used);
  const signs = new Int8Array(used);
  for (let i = 0, k = 0; i < n; i++) {
    const v = d[i] ?? 0;
    if (v === 0 && !keepZeros) continue;
    absDiffs[k] = Math.abs(v);
    signs[k] = v > 0 ? 1 : v < 0 ? -1 : 0;
    k++;
  }

  const { ranks, tieSum: rawTieSum } = rankData(absDiffs);
  let rPlus = 0;
  let rMinus = 0;
  let rZero = 0;
  for (let i = 0; i < used; i++) {
    const r = ranks[i] ?? 0;
    if (signs[i] === 1) rPlus += r;
    else if (signs[i] === -1) rMinus += r;
    else rZero += r;
  }
  if (zeroMethod === "zsplit") {
    rPlus += rZero / 2;
    rMinus += rZero / 2;
  }
  // Ties among the zeros are excluded from the tie correction for Pratt's method. Zeros have
  // the smallest absolute value, so they form the first group of tied ranks.
  const zeroTies = zeroMethod === "pratt" && nZero > 1 ? nZero ** 3 - nZero : 0;
  const tieSum = rawTieSum - zeroTies;
  const hasTiesOrZeros = rawTieSum > 0 || nZero > 0;
  const statistic = alternative === "two-sided" ? Math.min(rPlus, rMinus) : rPlus;

  let useMethod: "exact" | "permutation" | "asymptotic";
  if (method === "exact") useMethod = "exact";
  else if (method === "asymptotic") useMethod = "asymptotic";
  else if (n > 50) useMethod = "asymptotic";
  else if (!hasTiesOrZeros) useMethod = "exact";
  else if (n <= 13) useMethod = "permutation";
  else useMethod = "asymptotic";

  const tails = (pmf: Float64Array, upper: number, lower: number): number => {
    const sf = pmfUpperTail(pmf, upper);
    const cdf = pmfLowerTail(pmf, lower);
    const p = alternative === "greater" ? sf : alternative === "less" ? cdf : 2 * Math.min(sf, cdf);
    return Math.max(0, Math.min(1, p));
  };

  if (useMethod === "exact") {
    if (used > 5000) {
      throw new InvalidParameterError(
        'wilcoxon() method "exact" is too expensive for more than 5000 differences',
        "method",
        used
      );
    }
    const pmf = signFlipNullPmf(Array.from({ length: used }, (_, i) => i + 1));
    // The exact distribution is defined for integers: round the statistic conservatively
    // when ties or zeros make it fractional.
    return { statistic, pvalue: tails(pmf, Math.floor(rPlus), Math.ceil(rPlus)) };
  }

  if (useMethod === "permutation") {
    // Ranks are multiples of 0.5, so doubled ranks are integer weights. Zero differences keep
    // a zero sign under sign flips, so they only shift the distribution by a constant.
    const weights: number[] = [];
    for (let i = 0; i < used; i++) {
      if (signs[i] !== 0) weights.push(Math.round(2 * (ranks[i] ?? 0)));
    }
    const pmf = signFlipNullPmf(weights);
    const shift = zeroMethod === "zsplit" ? Math.round(rZero) : 0;
    const shifted = new Float64Array(pmf.length + shift);
    shifted.set(pmf, shift);
    const obs = Math.round(2 * rPlus);
    return { statistic, pvalue: tails(shifted, obs, obs) };
  }

  // Normal approximation.
  let meanW = (used * (used + 1)) / 4;
  let seSq = used * (used + 1) * (2 * used + 1);
  if (zeroMethod === "pratt") {
    // Adjustment for the discarded zero ranks (Cureton, 1967).
    meanW -= (nZero * (nZero + 1)) / 4;
    seSq -= nZero * (nZero + 1) * (2 * nZero + 1);
  }
  const varW = (seSq - tieSum / 2) / 24;
  if (!(varW > 0)) {
    return { statistic, pvalue: Number.NaN };
  }
  const stdW = Math.sqrt(varW);
  let z = (rPlus - meanW) / stdW;
  if (options.correction === true) {
    const sign = alternative === "greater" ? 1 : alternative === "less" ? -1 : Math.sign(z);
    z -= (sign * 0.5) / stdW;
  }
  return { statistic, pvalue: zPvalue(z, alternative) };
}

/**
 * Kruskal-Wallis H-test (non-parametric version of ANOVA).
 *
 * Tests whether two or more independent samples come from the same distribution. The
 * p-value uses the chi-square approximation with `k - 1` degrees of freedom and the tie
 * correction, as `scipy.stats.kruskal` does.
 *
 * @param samples - Two or more samples (flattened)
 * @returns The H statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than 2 samples are given, a sample is empty, or all
 *   values are identical
 *
 * @example
 * ```ts
 * import { kruskal } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = kruskal(tensor([2.9, 3.0, 2.5, 2.6]), tensor([3.8, 2.7, 4.0]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function kruskal(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("kruskal() requires at least 2 groups", "k", k);
  }

  let N = 0;
  const sizes = new Array<number | undefined>(k);
  const validatedSamples = new Array<Tensor | undefined>(k);
  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size < 1) {
      throw new InvalidParameterError("kruskal() requires non-empty samples", "size", {
        group: g,
        size: sample?.size ?? 0,
      });
    }
    validatedSamples[g] = sample;
    sizes[g] = sample.size;
    N += sample.size;
  }

  const combined = new Float64Array(N);
  const groupIndex = new Int32Array(N);
  let idx = 0;
  for (let g = 0; g < k; g++) {
    const sample = validatedSamples[g];
    if (!sample) {
      throw new InvalidParameterError("kruskal() requires non-empty samples", "sample", g);
    }
    const vals = toDenseArray1D(sample);
    for (let i = 0; i < vals.length; i++) {
      combined[idx] = vals[i] ?? 0;
      groupIndex[idx] = g;
      idx++;
    }
  }

  if (hasNaN(combined)) return { statistic: Number.NaN, pvalue: Number.NaN };

  const { ranks, tieSum } = rankData(combined);

  const rankSums = new Float64Array(k);
  for (let i = 0; i < N; i++) {
    const group = groupIndex[i] ?? 0;
    const rank = ranks[i] ?? 0;
    rankSums[group] = (rankSums[group] ?? 0) + rank;
  }

  let H = 0;
  for (let g = 0; g < k; g++) {
    const rs = rankSums[g] ?? 0;
    const sz = sizes[g] ?? 1;
    H += (rs * rs) / sz;
  }
  H = (12 / (N * (N + 1))) * H - 3 * (N + 1);

  // Tie correction factor for H statistic.
  const tieCorrection = N > 1 ? 1 - tieSum / (N * N * N - N) : 1;
  if (tieCorrection <= 0) {
    throw new InvalidParameterError(
      "kruskal() is undefined when all numbers are identical",
      "tieCorrection",
      tieCorrection
    );
  }
  H /= tieCorrection;

  const df = k - 1;
  const pvalue = chiSquareSf(H, df);

  return { statistic: H, pvalue };
}

/**
 * Friedman test (non-parametric repeated measures ANOVA).
 *
 * Tests whether `k` related samples (measurements of the same `n` subjects) have the same
 * distribution. The p-value uses the chi-square approximation with `k - 1` degrees of
 * freedom and the tie correction, as `scipy.stats.friedmanchisquare` does.
 *
 * @param samples - Three or more samples of equal length, paired by position
 * @returns The chi-square statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than 3 samples are given, the lengths differ, a
 *   sample is empty, or all values are identical within every block
 *
 * @example
 * ```ts
 * import { friedmanchisquare } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = friedmanchisquare(
 *   tensor([7.0, 9.9, 8.5, 5.1, 10.3]),
 *   tensor([5.3, 5.7, 4.7, 3.5, 7.7]),
 *   tensor([4.9, 7.6, 5.5, 2.8, 8.4])
 * );
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function friedmanchisquare(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 3) {
    throw new InvalidParameterError(
      "friedmanchisquare() requires at least 3 related samples",
      "k",
      k
    );
  }
  const n = samples[0]?.size ?? 0;
  if (n < 1) {
    throw new InvalidParameterError(
      "friedmanchisquare() requires all samples to be non-empty",
      "n",
      n
    );
  }

  for (let i = 1; i < k; i++) {
    const sample = samples[i];
    if (sample && sample.size !== n) {
      throw new InvalidParameterError(
        "All samples must have the same length for Friedman test",
        "size",
        { expected: n, got: sample.size, sampleIndex: i }
      );
    }
  }

  // Rank within each block (row)
  const denseSamples = samples.map((sample) =>
    sample ? toDenseArray1D(sample) : new Float64Array(0)
  );
  if (denseSamples.some(hasNaN)) return { statistic: Number.NaN, pvalue: Number.NaN };
  const rankSums = new Float64Array(k);
  let tieSum = 0;

  for (let i = 0; i < n; i++) {
    const block = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const arr = denseSamples[j];
      block[j] = arr?.[i] ?? 0;
    }
    const ranked = rankData(block);
    tieSum += ranked.tieSum;
    for (let j = 0; j < k; j++) {
      const rank = ranked.ranks[j] ?? 0;
      rankSums[j] = (rankSums[j] ?? 0) + rank;
    }
  }

  let chiSq = 0;
  for (let j = 0; j < k; j++) {
    const rs = rankSums[j] ?? 0;
    chiSq += rs * rs;
  }
  chiSq = (12 / (n * k * (k + 1))) * chiSq - 3 * n * (k + 1);

  // Tie correction factor for Friedman chi-square.
  const tieCorrection = n > 0 ? 1 - tieSum / (n * k * (k * k - 1)) : 1;
  if (tieCorrection <= 0) {
    throw new InvalidParameterError(
      "friedmanchisquare() is undefined when all numbers are identical within blocks",
      "tieCorrection",
      tieCorrection
    );
  }
  chiSq /= tieCorrection;

  const df = k - 1;
  const pvalue = chiSquareSf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/** Location estimate used to center the groups in Levene's and Fligner-Killeen's tests. */
export type VarianceTestCenter = "mean" | "median" | "trimmed";

/** Center of a group: its mean, median, or mean after trimming `trim` values from each end. */
function groupCenter(values: Float64Array, center: VarianceTestCenter, trim: number): number {
  const n = values.length;
  if (center === "mean") {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[i] ?? 0;
    return sum / n;
  }
  const sorted = Float64Array.from(values).sort();
  if (center === "median") {
    const mid = Math.floor(n / 2);
    return n % 2 === 0 ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2 : (sorted[mid] ?? 0);
  }
  let sum = 0;
  for (let i = trim; i < n - trim; i++) sum += sorted[i] ?? 0;
  return sum / (n - 2 * trim);
}

function assertCenter(fn: string, center: unknown, proportiontocut: unknown): void {
  if (center !== "mean" && center !== "median" && center !== "trimmed") {
    throw new InvalidParameterError(
      `${fn}() center must be "mean", "median" or "trimmed"`,
      "center",
      center
    );
  }
  if (
    typeof proportiontocut !== "number" ||
    !Number.isFinite(proportiontocut) ||
    proportiontocut < 0 ||
    proportiontocut >= 0.5
  ) {
    throw new InvalidParameterError(
      `${fn}() proportiontocut must be in [0, 0.5)`,
      "proportiontocut",
      proportiontocut
    );
  }
}

/** Options of {@link levene} and {@link fligner}. */
export interface VarianceTestOptions {
  /** How each group is centered: `"median"` (default), `"mean"` or `"trimmed"`. */
  center?: VarianceTestCenter;
  /** Fraction trimmed from each end for `center: "trimmed"` (default: 0.05). */
  proportiontocut?: number;
}

/**
 * Levene's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances. It is an analysis of variance of the
 * absolute deviations of each group from its center, and is less sensitive than Bartlett's
 * test to departures from normality.
 *
 * The first argument may be the centering method (`"median"`, the Brown-Forsythe variant;
 * `"mean"`, the original Levene test; or `"trimmed"`), an options object, or the first sample
 * (in which case the median is used, as in SciPy).
 *
 * @param center - Centering method, or {@link VarianceTestOptions}
 * @param samples - Two or more sample tensors to compare (at least 2 values each)
 * @returns The W statistic and its p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than two groups are given, a group has fewer than
 *   2 values, or an option is invalid
 *
 * @example
 * ```ts
 * import { levene } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const group1 = tensor([1, 2, 3, 4, 5]);
 * const group2 = tensor([2, 4, 6, 8, 10]);
 * const result = levene('median', group1, group2);
 * console.log(result.pvalue);  // p-value for equal variances
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function levene(center: VarianceTestCenter, ...samples: Tensor[]): TestResult;
export function levene(options: VarianceTestOptions, ...samples: Tensor[]): TestResult;
export function levene(...samples: Tensor[]): TestResult;
export function levene(...args: (VarianceTestCenter | VarianceTestOptions | Tensor)[]): TestResult {
  const { center, proportiontocut, samples } = parseVarianceArgs("levene", args);
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("levene() requires at least 2 groups", "k", k);
  }

  const groups: Float64Array[] = [];
  let anyNaN = false;
  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError("levene() requires all groups to be non-empty", "groupSize", {
        group: g,
        size: sample?.size ?? 0,
      });
    }
    const arr = toDenseArray1D(sample);
    if (arr.length < 2) {
      throw new InvalidParameterError(
        "levene() requires at least 2 samples per group",
        "groupSize",
        arr.length
      );
    }
    if (hasNaN(arr)) anyNaN = true;
    groups.push(arr);
  }
  if (anyNaN) return { statistic: Number.NaN, pvalue: Number.NaN };

  // Absolute deviations from the group center (Z_ij = |Y_ij - center_i|).
  const Z: Float64Array[] = [];
  const groupMeansZ: number[] = [];
  let N = 0;
  let grandSumZ = 0;

  for (const arr of groups) {
    const c = groupCenter(arr, center, Math.floor(arr.length * proportiontocut));
    const zArr = new Float64Array(arr.length);
    let sumZ = 0;
    for (let i = 0; i < arr.length; i++) {
      const absVal = Math.abs((arr[i] ?? 0) - c);
      zArr[i] = absVal;
      sumZ += absVal;
    }
    Z.push(zArr);
    groupMeansZ.push(sumZ / arr.length);
    N += arr.length;
    grandSumZ += sumZ;
  }

  const grandMeanZ = grandSumZ / N;

  // Levene's W statistic (F-test on the absolute deviations).
  let SSB = 0; // Between-group sum of squares
  let SSW = 0; // Within-group sum of squares
  for (let g = 0; g < Z.length; g++) {
    const zArr = Z[g] as Float64Array;
    const meanZ = groupMeansZ[g] ?? 0;
    SSB += zArr.length * (meanZ - grandMeanZ) ** 2;
    for (let i = 0; i < zArr.length; i++) {
      SSW += ((zArr[i] ?? 0) - meanZ) ** 2;
    }
  }

  const dfB = k - 1;
  const dfW = N - k;
  if (SSW === 0) {
    // All deviations within every group are identical: W is infinite if the group
    // means of the deviations differ, and undefined otherwise.
    return SSB === 0
      ? { statistic: Number.NaN, pvalue: Number.NaN }
      : { statistic: Number.POSITIVE_INFINITY, pvalue: 0 };
  }
  const W = SSB / dfB / (SSW / dfW);

  return { statistic: W, pvalue: fSf(W, dfB, dfW) };
}

/** Splits the leading option argument of {@link levene} and {@link fligner} from the samples. */
function parseVarianceArgs(
  fn: string,
  args: readonly (VarianceTestCenter | VarianceTestOptions | Tensor | readonly Tensor[])[]
): { center: VarianceTestCenter; proportiontocut: number; samples: Tensor[] } {
  let center: VarianceTestCenter = "median";
  let proportiontocut = 0.05;
  let rest = args;
  const first = args[0];
  if (typeof first === "string") {
    center = first;
    rest = args.slice(1);
  } else if (first !== undefined && !(first instanceof Tensor) && !Array.isArray(first)) {
    center = (first as VarianceTestOptions).center ?? "median";
    proportiontocut = (first as VarianceTestOptions).proportiontocut ?? 0.05;
    rest = args.slice(1);
  }
  assertCenter(fn, center, proportiontocut);
  const samples: Tensor[] = [];
  for (const item of rest) {
    if (!(item instanceof Tensor)) {
      throw new InvalidParameterError(`${fn}() samples must be tensors`, "samples", typeof item);
    }
    samples.push(item);
  }
  return { center, proportiontocut, samples };
}

/**
 * Bartlett's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances.
 * Assumes data is normally distributed; use Levene's test for non-normal data.
 *
 * @param samples - Two or more sample tensors to compare (at least 2 values each)
 * @returns The chi-square statistic and its p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than two groups are given, a group has fewer than
 *   2 values, or a group has zero variance
 *
 * @example
 * ```ts
 * import { bartlett } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const group1 = tensor([1, 2, 3, 4, 5]);
 * const group2 = tensor([2, 4, 6, 8, 10]);
 * const result = bartlett(group1, group2);
 * console.log(result.pvalue);  // p-value for equal variances
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function bartlett(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("bartlett() requires at least 2 groups", "k", k);
  }

  const variances: number[] = [];
  const sizes: number[] = [];
  let N = 0;
  let anyNaN = false;

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError(
        "bartlett() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: sample?.size ?? 0 }
      );
    }
    const arr = toDenseArray1D(sample);
    const n = arr.length;
    if (n < 2) {
      throw new InvalidParameterError(
        "bartlett() requires at least 2 samples per group",
        "groupSize",
        n
      );
    }
    if (hasNaN(arr)) anyNaN = true;

    let mean = 0;
    for (let i = 0; i < n; i++) mean += arr[i] ?? 0;
    mean /= n;

    // Sample variance (n - 1 denominator)
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = (arr[i] ?? 0) - mean;
      ss += d * d;
    }
    variances.push(ss / (n - 1));
    sizes.push(n);
    N += n;
  }
  if (anyNaN) return { statistic: Number.NaN, pvalue: Number.NaN };

  for (let g = 0; g < k; g++) {
    if ((variances[g] ?? 0) === 0) {
      throw new InvalidParameterError(
        "bartlett() is undefined when a group has zero variance",
        "variance",
        variances[g]
      );
    }
  }

  // Pooled variance
  let pooledNumerator = 0;
  for (let g = 0; g < k; g++) {
    pooledNumerator += ((sizes[g] ?? 1) - 1) * (variances[g] ?? 1);
  }
  const pooledVariance = pooledNumerator / (N - k);

  // T = (N-k) * ln(s_p^2) - sum((n_i - 1) * ln(s_i^2))
  let sumLogVar = 0;
  for (let g = 0; g < k; g++) {
    sumLogVar += ((sizes[g] ?? 1) - 1) * Math.log(variances[g] ?? 1);
  }
  const T = (N - k) * Math.log(pooledVariance) - sumLogVar;

  // Correction factor C
  let sumInvDf = 0;
  for (let g = 0; g < k; g++) {
    sumInvDf += 1 / ((sizes[g] ?? 1) - 1);
  }
  const C = 1 + (1 / (3 * (k - 1))) * (sumInvDf - 1 / (N - k));

  const chiSq = T / C;
  return { statistic: chiSq, pvalue: chiSquareSf(chiSq, k - 1) };
}

/**
 * One-way analysis of variance.
 *
 * Tests whether two or more groups have the same population mean.
 *
 * @param samples - Two or more groups (flattened)
 * @returns The F statistic and the p-value. `F` is `Infinity` (with p-value 0) when the groups
 *   have no within-group variation but different means, and `NaN` when all values are equal.
 * @throws {InvalidParameterError} If fewer than 2 groups are given, a group is empty, or every
 *   group has a single value
 *
 * @example
 * ```ts
 * import { fOneway } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = fOneway(tensor([6, 8, 4, 5, 3, 4]), tensor([8, 12, 9, 11, 6, 8]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link fOneway}.
 */
export function f_oneway(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("f_oneway() requires at least 2 groups", "groups", k);
  }

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError(
        "f_oneway() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: sample?.size ?? 0 }
      );
    }
  }

  let N = 0;
  const means: number[] = [];
  const sizes: number[] = [];
  const groups: Float64Array[] = [];

  // Compute group means
  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample) {
      throw new InvalidParameterError(
        "f_oneway() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: 0 }
      );
    }
    const arr = toDenseArray1D(sample);
    const n = arr.length;
    groups.push(arr);
    sizes.push(n);
    N += n;

    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += arr[i] ?? 0;
    }
    means.push(sum / n);
  }

  // Compute grand mean
  let grandSum = 0;
  for (let g = 0; g < k; g++) {
    grandSum += (means[g] ?? 0) * (sizes[g] ?? 0);
  }
  const grandMean = grandSum / N;

  // Compute between-group and within-group variance
  let SSB = 0; // Between-group sum of squares
  let SSW = 0; // Within-group sum of squares

  for (let g = 0; g < groups.length; g++) {
    const arr = groups[g];
    if (!arr) continue;
    const n = arr.length;
    SSB += n * ((means[g] ?? 0) - grandMean) ** 2;

    for (let i = 0; i < n; i++) {
      SSW += ((arr[i] ?? 0) - (means[g] ?? 0)) ** 2;
    }
  }

  const dfB = k - 1;
  const dfW = N - k;
  if (dfW <= 0) {
    throw new InvalidParameterError(
      "f_oneway() requires at least one group with more than one sample",
      "dfW",
      dfW
    );
  }
  const MSB = SSB / dfB;
  const MSW = SSW / dfW;
  if (MSW === 0) {
    // All within-group values are identical; F is infinite if groups differ, NaN otherwise
    const F = MSB === 0 ? NaN : Infinity;
    return { statistic: F, pvalue: MSB === 0 ? NaN : 0 };
  }
  const F = MSB / MSW;

  const pvalue = fSf(F, dfB, dfW);

  return { statistic: F, pvalue };
}

/**
 * Result of two-way ANOVA test.
 */
export type TwoWayAnovaResult = {
  factorA: TestResult;
  factorB: TestResult;
  interaction: TestResult;
};

/**
 * Two-way ANOVA for balanced designs.
 *
 * Tests main effects of two factors and their interaction.
 * Expects data organized as a 3D array: data[a][b][replication].
 *
 * @param data - 3D array indexed by [factorA level][factorB level][replications]
 * @returns Object with factorA, factorB, and interaction test results
 * @throws {InvalidParameterError} If the design is unbalanced, a factor has fewer than 2
 *   levels, cells have a single replication, or a value is not finite
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link fTwoway}.
 */
export function f_twoway(data: number[][][]): TwoWayAnovaResult {
  const a = data.length;
  if (a < 2) {
    throw new InvalidParameterError(
      "f_twoway() requires at least 2 levels for factor A",
      "data",
      a
    );
  }

  const firstRow = data[0];
  if (!firstRow) {
    throw new InvalidParameterError("f_twoway() data[0] is undefined", "data");
  }
  const b = firstRow.length;
  if (b < 2) {
    throw new InvalidParameterError(
      "f_twoway() requires at least 2 levels for factor B",
      "data",
      b
    );
  }

  const firstCell = firstRow[0];
  if (!firstCell) {
    throw new InvalidParameterError("f_twoway() data[0][0] is undefined", "data");
  }
  const n = firstCell.length;
  if (n < 1) {
    throw new InvalidParameterError(
      "f_twoway() requires at least 1 replication per cell",
      "data",
      n
    );
  }

  // Validate balanced design and compute cell means
  const cellMeans: number[][] = [];
  let grandSum = 0;
  const N = a * b * n;

  for (let i = 0; i < a; i++) {
    const row = data[i];
    if (!row || row.length !== b) {
      throw new InvalidParameterError(
        `f_twoway() factor B must have ${b} levels in all rows`,
        "data"
      );
    }
    cellMeans.push([]);
    for (let j = 0; j < b; j++) {
      const cell = row[j];
      if (!cell || cell.length !== n) {
        throw new InvalidParameterError(`f_twoway() all cells must have ${n} replications`, "data");
      }
      let cellSum = 0;
      for (let k = 0; k < n; k++) {
        const v = cell[k];
        if (v === undefined || !Number.isFinite(v)) {
          throw new InvalidParameterError("f_twoway() data must be finite numbers", "data");
        }
        cellSum += v;
        grandSum += v;
      }
      cellMeans[i]!.push(cellSum / n);
    }
  }

  const grandMean = grandSum / N;

  // Row means (factor A)
  const rowMeans: number[] = [];
  for (let i = 0; i < a; i++) {
    let sum = 0;
    for (let j = 0; j < b; j++) {
      sum += cellMeans[i]![j]!;
    }
    rowMeans.push(sum / b);
  }

  // Column means (factor B)
  const colMeans: number[] = [];
  for (let j = 0; j < b; j++) {
    let sum = 0;
    for (let i = 0; i < a; i++) {
      sum += cellMeans[i]![j]!;
    }
    colMeans.push(sum / a);
  }

  // Sum of squares
  let SSA = 0;
  for (let i = 0; i < a; i++) {
    SSA += b * n * (rowMeans[i]! - grandMean) ** 2;
  }

  let SSB = 0;
  for (let j = 0; j < b; j++) {
    SSB += a * n * (colMeans[j]! - grandMean) ** 2;
  }

  let SSAB = 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      SSAB += n * (cellMeans[i]![j]! - rowMeans[i]! - colMeans[j]! + grandMean) ** 2;
    }
  }

  let SSE = 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      const cell = data[i]![j]!;
      const cm = cellMeans[i]![j]!;
      for (let k = 0; k < n; k++) {
        SSE += (cell[k]! - cm) ** 2;
      }
    }
  }

  const dfA = a - 1;
  const dfB_val = b - 1;
  const dfAB = dfA * dfB_val;
  const dfE = a * b * (n - 1);

  if (dfE <= 0) {
    throw new InvalidParameterError(
      "f_twoway() requires more than 1 replication per cell for error estimation",
      "data",
      n
    );
  }

  const MSA = SSA / dfA;
  const MSB_val = SSB / dfB_val;
  const MSAB = SSAB / dfAB;
  const MSE = SSE / dfE;

  const computeF = (ms: number, df: number): TestResult => {
    if (MSE === 0) {
      const stat = ms === 0 ? NaN : Infinity;
      return { statistic: stat, pvalue: ms === 0 ? NaN : 0 };
    }
    const stat = ms / MSE;
    const pval = fSf(stat, df, dfE);
    return { statistic: stat, pvalue: pval };
  };

  return {
    factorA: computeF(MSA, dfA),
    factorB: computeF(MSB_val, dfB_val),
    interaction: computeF(MSAB, dfAB),
  };
}

// ─── Contingency table analysis ─────────────────────────────────────────────

/** Result of a contingency table analysis (chi-square test). */
export type ContingencyResult = {
  statistic: number;
  pvalue: number;
  dof: number;
  expected: number[][];
};

/**
 * Chi-squared test of independence for a contingency table.
 *
 * Tests whether two categorical variables are independent
 * given their observed frequency table.
 *
 * For 2×2 tables (one degree of freedom) Yates' continuity correction is applied by
 * default, as in SciPy: each observed count is moved by at most 0.5 towards its expected
 * count. Pass `correction = false` for the plain Pearson statistic.
 *
 * @param observed - 2D array of observed frequencies (rows × cols)
 * @param correction - Apply Yates' continuity correction when `dof = 1` (default: `true`)
 * @returns Object with chi2 statistic, p-value, degrees of freedom, and expected frequencies
 * @throws {InvalidParameterError} If the table is smaller than 2×2, ragged, contains
 *   negative or non-finite counts, or has a row or column that sums to zero
 *
 * @example
 * ```ts
 * import { chi2Contingency } from 'deepbox/stats';
 * const result = chi2Contingency([[10, 20, 30], [6, 9, 17]]);
 * console.log(result.pvalue);
 * ```
 * @deprecated Prefer {@link chi2Contingency}.
 */
export function chi2_contingency(
  observed: readonly (readonly number[])[],
  correction = true
): ContingencyResult {
  const nRows = observed.length;
  if (nRows < 2) {
    throw new InvalidParameterError(
      "Contingency table must have at least 2 rows",
      "observed",
      nRows
    );
  }
  const nCols = observed[0]?.length ?? 0;
  if (nCols < 2) {
    throw new InvalidParameterError(
      "Contingency table must have at least 2 columns",
      "observed",
      nCols
    );
  }

  // Validate all rows have same length
  for (let i = 1; i < nRows; i++) {
    if ((observed[i]?.length ?? 0) !== nCols) {
      throw new InvalidParameterError(
        "All rows must have the same number of columns",
        "observed",
        observed[i]?.length ?? 0
      );
    }
  }

  // Compute row and column totals
  const rowTotals = new Float64Array(nRows);
  const colTotals = new Float64Array(nCols);
  let grandTotal = 0;

  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const val = observed[i]?.[j] ?? 0;
      if (!Number.isFinite(val) || val < 0) {
        throw new InvalidParameterError(
          "Observed frequencies must be finite and non-negative",
          "observed",
          val
        );
      }
      rowTotals[i] = (rowTotals[i] ?? 0) + val;
      colTotals[j] = (colTotals[j] ?? 0) + val;
      grandTotal += val;
    }
  }

  if (grandTotal === 0) {
    throw new InvalidParameterError(
      "Contingency table total must be positive",
      "observed",
      grandTotal
    );
  }
  if (rowTotals.some((t) => t === 0) || colTotals.some((t) => t === 0)) {
    throw new InvalidParameterError(
      "Expected frequencies contain a zero because a row or column of the table sums to zero",
      "observed",
      { rowTotals: Array.from(rowTotals), colTotals: Array.from(colTotals) }
    );
  }

  const dof = (nRows - 1) * (nCols - 1);
  const yates = correction && dof === 1;

  // Compute expected frequencies and chi-squared statistic
  const expected: number[][] = [];
  let chi2Stat = 0;

  for (let i = 0; i < nRows; i++) {
    const row: number[] = [];
    for (let j = 0; j < nCols; j++) {
      const exp = ((rowTotals[i] ?? 0) * (colTotals[j] ?? 0)) / grandTotal;
      row.push(exp);
      let obs = observed[i]?.[j] ?? 0;
      if (yates) {
        // Move each count by at most 0.5 towards its expected value.
        const diff = exp - obs;
        obs += Math.sign(diff) * Math.min(0.5, Math.abs(diff));
      }
      chi2Stat += (obs - exp) ** 2 / exp;
    }
    expected.push(row);
  }

  const pvalue = chiSquareSf(chi2Stat, dof);

  return { statistic: chi2Stat, pvalue, dof, expected };
}

/** Largest total count accepted by {@link fisher_exact} (2^40). */
const FISHER_MAX_TOTAL = 2 ** 40;

/**
 * Fisher's exact test for a 2×2 contingency table.
 *
 * Computes the exact p-value for the association between two binary variables.
 * Uses the hypergeometric distribution. The two-sided p-value sums the probabilities of
 * all tables that are at most as likely as the observed one (with a relative tolerance
 * of 1e-7, as in R's `fisher.test`; SciPy 1.17 uses 1e-14, which only matters for tables
 * whose probabilities agree to more than 7 digits without being equal).
 *
 * @param table - 2×2 array of observed counts [[a, b], [c, d]] (non-negative integers)
 * @param alternative - 'two-sided' (default), 'less', or 'greater'
 * @returns Object with the sample odds ratio `a * d / (b * c)` (`Infinity` or `NaN` when the
 *   denominator is zero) and the p-value
 * @throws {InvalidParameterError} If a count is negative, not an integer or not finite, the
 *   total count exceeds 2^40, or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { fisherExact } from 'deepbox/stats';
 * const result = fisherExact([[1, 9], [11, 3]]);
 * console.log(result.pvalue);
 * ```
 * @deprecated Prefer {@link fisherExact}.
 */
export function fisher_exact(
  table: readonly [readonly [number, number], readonly [number, number]],
  alternative: "two-sided" | "less" | "greater" = "two-sided"
): { oddsRatio: number; pvalue: number } {
  assertAlternative("fisher_exact", alternative);
  if (table.length !== 2 || table[0]?.length !== 2 || table[1]?.length !== 2) {
    throw new InvalidParameterError("fisher_exact() requires a 2x2 table", "table", {
      rows: table.length,
    });
  }
  const a = table[0][0];
  const b = table[0][1];
  const c = table[1][0];
  const d = table[1][1];

  for (const v of [a, b, c, d]) {
    if (!Number.isInteger(v) || v < 0) {
      throw new InvalidParameterError(
        "fisher_exact() table entries must be non-negative integers",
        "table",
        v
      );
    }
  }

  const n = a + b + c + d;
  if (n > FISHER_MAX_TOTAL) {
    // The sum runs over about sqrt(n) tables, so the cost grows with the square root of n.
    throw new InvalidParameterError(
      `fisher_exact() supports at most ${FISHER_MAX_TOTAL} observations in total`,
      "table",
      n
    );
  }
  const r1 = a + b; // row 1 total
  const r2 = c + d; // row 2 total
  const c1 = a + c; // col 1 total

  // Sample odds ratio; division by zero follows IEEE rules (Infinity, or NaN for 0 / 0).
  const oddsRatio = (a * d) / (b * c);

  // P(X = k) = C(r1, k) C(r2, c1 - k) / C(n, c1) for k in [kMin, kMax]. The weights
  // w_k = P(X = k) / P(X = mode) follow from the ratio of consecutive probabilities, so the
  // p-value needs no factorials and stays accurate for very large counts.
  const kMin = Math.max(0, c1 - r2);
  const kMax = Math.min(r1, c1);
  const mode = Math.min(kMax, Math.max(kMin, Math.floor(((c1 + 1) * (r1 + 1)) / (n + 2))));
  const up = (k: number): number => ((r1 - k) * (c1 - k)) / ((k + 1) * (r2 - c1 + k + 1));
  const down = (k: number): number => (k * (r2 - c1 + k)) / ((r1 - k + 1) * (c1 - k + 1));
  const tiny = 1e-300;

  // Weight of the observed table.
  // The weights fall monotonically away from the mode: stop once they underflow to zero.
  let wObs = 1;
  for (let k = mode; k < a; k++) {
    wObs *= up(k);
    if (wObs < tiny) {
      wObs = 0;
      break;
    }
  }
  for (let k = mode; k > a; k--) {
    wObs *= down(k);
    if (wObs < tiny) {
      wObs = 0;
      break;
    }
  }

  const include = (w: number, k: number): boolean =>
    alternative === "less" ? k <= a : alternative === "greater" ? k >= a : w <= wObs * (1 + 1e-7);

  let total = 1;
  let tail = include(1, mode) ? 1 : 0;
  for (let k = mode, w = 1; k < kMax; k++) {
    w *= up(k);
    if (w < tiny) break;
    total += w;
    if (include(w, k + 1)) tail += w;
  }
  for (let k = mode, w = 1; k > kMin; k--) {
    w *= down(k);
    if (w < tiny) break;
    total += w;
    if (include(w, k - 1)) tail += w;
  }
  const pvalue = tail / total;

  return { oddsRatio, pvalue: Math.min(1, pvalue) };
}

// ─── Runs test ──────────────────────────────────────────────────────────────

/**
 * Wald-Wolfowitz runs test for randomness.
 *
 * Tests whether a sequence of binary (above/below median) values
 * is random by counting the number of "runs" (consecutive sequences
 * of the same type). Values equal to the median count as "above".
 *
 * @param x - Input tensor of numeric values, in observation order
 * @returns TestResult with z-statistic and two-sided p-value (`NaN` if the data contain `NaN`)
 * @deprecated Prefer {@link runsTest}.
 *
 * @example
 * ```ts
 * import { runsTest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 * const result = runsTest(tensor([1, 2, 1, 2, 1, 2, 1, 2]));
 * ```
 */
export function runs_test(x: Tensor): TestResult {
  const data = toDenseArray1D(x);
  const n = data.length;
  if (n < 2) {
    throw new InvalidParameterError("runs_test requires at least 2 observations", "x.size", n);
  }
  if (hasNaN(data)) return { statistic: Number.NaN, pvalue: Number.NaN };

  // Compute median
  const sorted = new Float64Array(data).sort();
  const med =
    n % 2 === 0
      ? ((sorted[n / 2 - 1] ?? 0) + (sorted[n / 2] ?? 0)) / 2
      : (sorted[Math.floor(n / 2)] ?? 0);

  // Count runs
  let nPlus = 0;
  let nMinus = 0;
  let runs = 1;
  let prev = (data[0] ?? 0) >= med;
  if (prev) nPlus++;
  else nMinus++;

  for (let i = 1; i < n; i++) {
    const cur = (data[i] ?? 0) >= med;
    if (cur) nPlus++;
    else nMinus++;
    if (cur !== prev) {
      runs++;
      prev = cur;
    }
  }

  if (nPlus === 0 || nMinus === 0) {
    // All values on one side of the median
    return { statistic: 0, pvalue: 1 };
  }

  // Expected runs and variance under H0
  const expectedRuns = 1 + (2 * nPlus * nMinus) / n;
  const varRuns = (2 * nPlus * nMinus * (2 * nPlus * nMinus - n)) / (n * n * (n - 1));

  if (varRuns <= 0) {
    return { statistic: 0, pvalue: 1 };
  }

  const z = (runs - expectedRuns) / Math.sqrt(varRuns);
  const pvalue = zPvalue(z, "two-sided");

  return { statistic: z, pvalue };
}

// ─── Lilliefors test ────────────────────────────────────────────────────────

/**
 * Lilliefors test for normality.
 *
 * A variant of the Kolmogorov-Smirnov test where the mean and standard deviation
 * are estimated from the data (rather than specified). The p-value uses the
 * Dallal-Wilkinson (1986) approximation, refined for large p-values with the
 * polynomial fits of Stephens, as in R's `nortest::lillie.test`.
 *
 * @param x - Input tensor of numeric values (at least 4)
 * @returns TestResult with the KS statistic and the approximate p-value. A constant sample
 *   gives `{ statistic: 0, pvalue: 1 }`.
 *
 * @example
 * ```ts
 * import { lilliefors } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 * const result = lilliefors(tensor([1.2, 2.3, 1.8, 2.1, 1.5]));
 * ```
 */
export function lilliefors(x: Tensor): TestResult {
  const data = toDenseSortedArray1D(x);
  const n = data.length;
  if (n < 4) {
    throw new InvalidParameterError(
      "lilliefors test requires at least 4 observations",
      "x.size",
      n
    );
  }
  if (hasNaN(data)) return { statistic: Number.NaN, pvalue: Number.NaN };
  if ((data[0] ?? 0) === (data[n - 1] ?? 0)) {
    // All values identical: the fitted normal is degenerate, so nothing can be tested.
    return { statistic: 0, pvalue: 1 };
  }

  // Estimate mean and std from data
  const { mean, m2 } = meanAndM2(data);
  const std = Math.sqrt(m2 / (n - 1));

  // Compute KS statistic against N(mean, std)
  let dMax = 0;
  for (let i = 0; i < n; i++) {
    const z = ((data[i] ?? 0) - mean) / std;
    const theoreticalCdf = normalCdf(z);
    dMax = Math.max(dMax, (i + 1) / n - theoreticalCdf, theoreticalCdf - i / n);
  }

  // Dallal-Wilkinson approximation. For n > 100 the statistic is rescaled to n = 100.
  const nd = Math.min(n, 100);
  const kd = n > 100 ? dMax * (n / 100) ** 0.49 : dMax;
  let pvalue = Math.exp(
    -7.01256 * kd * kd * (nd + 2.78019) +
      2.99587 * kd * Math.sqrt(nd + 2.78019) -
      0.122119 +
      0.974598 / Math.sqrt(nd) +
      1.67997 / nd
  );
  if (pvalue > 0.1) {
    // The formula above is inaccurate for large p-values; use Stephens' polynomial fits.
    const kk = (Math.sqrt(n) - 0.01 + 0.85 / Math.sqrt(n)) * dMax;
    if (kk <= 0.302) {
      pvalue = 1;
    } else if (kk <= 0.5) {
      pvalue =
        2.76773 - 19.828315 * kk + 80.709644 * kk ** 2 - 138.55152 * kk ** 3 + 81.218052 * kk ** 4;
    } else if (kk <= 0.9) {
      pvalue =
        -4.901232 +
        40.662806 * kk -
        97.490286 * kk ** 2 +
        94.029866 * kk ** 3 -
        32.355711 * kk ** 4;
    } else if (kk <= 1.31) {
      pvalue =
        6.198765 - 19.558097 * kk + 23.186922 * kk ** 2 - 12.234627 * kk ** 3 + 2.423045 * kk ** 4;
    } else {
      pvalue = 0;
    }
  }

  return { statistic: dMax, pvalue: Math.max(0, Math.min(1, pvalue)) };
}

// ─── Fligner-Killeen test ───────────────────────────────────────────────────

/**
 * Fligner-Killeen test for equality of variances.
 *
 * A non-parametric test that is insensitive to departures from normality.
 * Uses the ranks of absolute deviations from the group centers (the median by default).
 *
 * The groups can be passed as an array followed by optional {@link VarianceTestOptions}, or
 * as separate arguments, optionally preceded by the centering method or an options object.
 *
 * @param groups - Array of tensors, one per group (at least 2 groups, at least 2 values each)
 * @param options - `center` (default `"median"`) and `proportiontocut` (default 0.05)
 * @returns The chi-square statistic and its p-value. Both are `NaN` if a value is `NaN` or all
 *   absolute deviations are equal.
 * @throws {InvalidParameterError} If fewer than 2 groups are given, a group has fewer than
 *   2 values, or an option is invalid
 *
 * @example
 * ```ts
 * import { fligner } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const result = fligner([tensor([1, 2, 3]), tensor([4, 5, 6, 7])]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function fligner(groups: readonly Tensor[], options?: VarianceTestOptions): TestResult;
export function fligner(center: VarianceTestCenter, ...samples: Tensor[]): TestResult;
export function fligner(options: VarianceTestOptions, ...samples: Tensor[]): TestResult;
export function fligner(...samples: Tensor[]): TestResult;
export function fligner(
  ...args: (readonly Tensor[] | VarianceTestCenter | VarianceTestOptions | Tensor | undefined)[]
): TestResult {
  let parsed: { center: VarianceTestCenter; proportiontocut: number; samples: Tensor[] };
  const first = args[0];
  if (Array.isArray(first)) {
    const options = (args[1] ?? {}) as VarianceTestOptions;
    parsed = parseVarianceArgs("fligner", [options, ...(first as readonly Tensor[])]);
  } else {
    parsed = parseVarianceArgs(
      "fligner",
      args.filter((a) => a !== undefined)
    );
  }
  const { center, proportiontocut, samples: groups } = parsed;
  if (groups.length < 2) {
    throw new InvalidParameterError(
      "fligner test requires at least 2 groups",
      "groups.length",
      groups.length
    );
  }

  const groupData: Float64Array[] = [];
  let N = 0;
  let anyNaN = false;
  for (const g of groups) {
    const arr = toDenseArray1D(g);
    if (arr.length < 2) {
      throw new InvalidParameterError(
        "Each group must have at least 2 observations",
        "group.size",
        arr.length
      );
    }
    if (hasNaN(arr)) anyNaN = true;
    groupData.push(arr);
    N += arr.length;
  }
  if (anyNaN) return { statistic: Number.NaN, pvalue: Number.NaN };

  // Absolute deviations from the group centers
  const deviations = new Float64Array(N);
  const groupSizes: number[] = [];
  let pos = 0;
  for (const arr of groupData) {
    const n = arr.length;
    groupSizes.push(n);
    const c = groupCenter(arr, center, Math.floor(n * proportiontocut));
    for (let i = 0; i < n; i++) {
      deviations[pos++] = Math.abs((arr[i] ?? 0) - c);
    }
  }

  // Normal scores of the ranks. Fligner-Killeen uses the half-normal score
  // a_i = Phi^-1(1/2 + rank_i / (2 (N + 1))), since the deviations are folded about zero.
  const { ranks } = rankData(deviations);
  const scores = new Float64Array(N);
  let grandMean = 0;
  for (let idx = 0; idx < N; idx++) {
    const score = normalPpf(0.5 + (ranks[idx] ?? 0) / (2 * (N + 1)));
    scores[idx] = score;
    grandMean += score;
  }
  grandMean /= N;

  let numerator = 0;
  let start = 0;
  for (const ng of groupSizes) {
    let sum = 0;
    for (let i = start; i < start + ng; i++) sum += scores[i] ?? 0;
    numerator += ng * (sum / ng - grandMean) ** 2;
    start += ng;
  }

  let denominator = 0;
  for (let idx = 0; idx < N; idx++) {
    denominator += ((scores[idx] ?? 0) - grandMean) ** 2;
  }
  denominator /= N - 1;

  if (denominator < 1e-15) {
    // Every absolute deviation is equal, so the scores carry no information.
    return { statistic: Number.NaN, pvalue: Number.NaN };
  }

  const stat = numerator / denominator;
  return { statistic: stat, pvalue: chiSquareSf(stat, groupSizes.length - 1) };
}

/**
 * Two-sample Kolmogorov-Smirnov test.
 *
 * Tests whether two samples are drawn from the same continuous distribution.
 * The two-sided statistic is the maximum absolute difference between the
 * empirical distribution functions of the two samples.
 *
 * The p-value comes from the exact null distribution when both samples have at most
 * 10000 values (SciPy's `method="auto"`), and from the asymptotic distribution otherwise.
 * With tied values the exact p-value is conservative.
 *
 * @param x - First sample (any shape; the tensor is flattened)
 * @param y - Second sample
 * @param options - `alternative` (`"two-sided"` by default) and `method`
 *   (see {@link KsTestOptions})
 * @returns The statistic (`D`, or `D+` / `D-` for one-sided alternatives) and the p-value.
 *   Both are `NaN` if a sample contains `NaN`.
 * @throws {InvalidParameterError} If a sample is empty or an option is invalid
 *
 * @example
 * ```ts
 * import { ks2samp } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const result = ks2samp(tensor([1, 2, 3]), tensor([1.5, 2.5, 3.5]));
 * console.log(result.statistic, result.pvalue);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ks2samp}.
 */
export function ks_2samp(x: Tensor, y: Tensor, options: KsTestOptions = {}): TestResult {
  const { alternative, method } = assertKsOptions("ks_2samp", options);
  const xs = toDenseSortedArray1D(x);
  const ys = toDenseSortedArray1D(y);
  const nx = xs.length;
  const ny = ys.length;
  if (nx === 0 || ny === 0) {
    throw new InvalidParameterError("ks_2samp() requires non-empty samples", "size", {
      x: nx,
      y: ny,
    });
  }
  if (hasNaN(xs) || hasNaN(ys)) return { statistic: Number.NaN, pvalue: Number.NaN };

  // Largest positive and negative differences of the empirical distribution functions,
  // found by merging the sorted samples. Both pointers move past every value equal to
  // the current one before the difference is evaluated, so tied values inside a sample
  // (or shared between samples) are handled correctly.
  let dPlus = 0;
  let dMinus = 0;
  let i = 0;
  let j = 0;
  while (i < nx && j < ny) {
    const v = Math.min(xs[i] ?? 0, ys[j] ?? 0);
    while (i < nx && (xs[i] ?? 0) <= v) i++;
    while (j < ny && (ys[j] ?? 0) <= v) j++;
    const diff = i / nx - j / ny;
    if (diff > dPlus) dPlus = diff;
    if (-diff > dMinus) dMinus = -diff;
  }
  const d =
    alternative === "greater" ? dPlus : alternative === "less" ? dMinus : Math.max(dPlus, dMinus);
  const oneSided = alternative !== "two-sided";

  if (method === "exact" || (method === "auto" && Math.max(nx, ny) <= 10000)) {
    const exact = ksTwoSampleExactPvalue(nx, ny, d, oneSided);
    if (exact !== undefined) return { statistic: d, pvalue: exact };
    if (method === "exact") {
      throw new InvalidParameterError(
        'ks_2samp() method "exact" is too expensive for these sample sizes',
        "method",
        method
      );
    }
  }

  // Asymptotic p-value (SciPy: Kolmogorov distribution with n rounded to an integer for the
  // two-sided test and Hodges' approximation for one-sided tests).
  const m = Math.max(nx, ny);
  const k = Math.min(nx, ny);
  const en = (m * k) / (m + k);
  if (!oneSided) {
    return { statistic: d, pvalue: kolmogorovTwoSidedSf(Math.max(1, roundHalfEven(en)), d) };
  }
  const z = Math.sqrt(en) * d;
  const expt = -2 * z * z - (2 * z * (m + 2 * k)) / Math.sqrt(m * k * (m + k)) / 3;
  return { statistic: d, pvalue: Math.min(1, Math.exp(expt)) };
}

/** Options of {@link median_test}. */
export interface MedianTestOptions {
  /** Apply Yates' continuity correction when there are two samples (default: `true`). */
  correction?: boolean;
  /**
   * Where values equal to the grand median are counted: `"below"` (default), `"above"`, or
   * `"ignore"` to drop them.
   */
  ties?: "below" | "above" | "ignore";
}

/**
 * Mood's median test.
 *
 * Tests whether two or more samples have the same median. It builds the 2 by k contingency
 * table of counts above and below the grand median and applies the chi-square test of
 * independence. As in SciPy, Yates' continuity correction is applied by default when there
 * are two samples.
 *
 * @param args - Two or more samples (flattened), optionally followed by
 *   {@link MedianTestOptions}
 * @returns The chi-square statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than 2 samples are given, a sample is empty,
 *   all values fall on one side of the grand median, or `ties: "ignore"` leaves a sample empty
 *
 * @example
 * ```ts
 * import { medianTest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const result = medianTest(tensor([1, 2, 3, 4, 5]), tensor([3, 5, 7, 8, 9]));
 * console.log(result.statistic, result.pvalue);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link medianTest}.
 */
export function median_test(...args: (Tensor | MedianTestOptions)[]): TestResult {
  let options: MedianTestOptions = {};
  const samples: Tensor[] = [];
  for (let i = 0; i < args.length; i++) {
    const item = args[i];
    if (item instanceof Tensor) {
      samples.push(item);
    } else if (i === args.length - 1 && item !== undefined && typeof item === "object") {
      options = item;
    } else {
      throw new InvalidParameterError("median_test() samples must be tensors", "samples", i);
    }
  }
  const correction = options.correction ?? true;
  const ties = options.ties ?? "below";
  if (ties !== "below" && ties !== "above" && ties !== "ignore") {
    throw new InvalidParameterError(
      'median_test() ties must be "below", "above" or "ignore"',
      "ties",
      ties
    );
  }
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("median_test() requires at least 2 groups", "groups", k);
  }

  const groups: Float64Array[] = [];
  let total = 0;
  for (const s of samples) {
    const vals = toDenseArray1D(s);
    if (vals.length === 0) {
      throw new InvalidParameterError("median_test() requires non-empty samples", "size", 0);
    }
    groups.push(vals);
    total += vals.length;
  }
  const all = new Float64Array(total);
  let pos = 0;
  for (const vals of groups) {
    all.set(vals, pos);
    pos += vals.length;
  }
  if (hasNaN(all)) return { statistic: Number.NaN, pvalue: Number.NaN };

  all.sort();
  const grandMedian =
    total % 2 === 1
      ? (all[Math.floor(total / 2)] ?? 0)
      : ((all[total / 2 - 1] ?? 0) + (all[total / 2] ?? 0)) / 2;

  // 2 x k contingency table: counts above and below the grand median for each sample
  const above: number[] = new Array(k).fill(0);
  const below: number[] = new Array(k).fill(0);
  for (let g = 0; g < k; g++) {
    for (const v of groups[g] as Float64Array) {
      if (v > grandMedian) above[g] = (above[g] ?? 0) + 1;
      else if (v < grandMedian) below[g] = (below[g] ?? 0) + 1;
      else if (ties === "below") below[g] = (below[g] ?? 0) + 1;
      else if (ties === "above") above[g] = (above[g] ?? 0) + 1;
    }
  }
  const totalAbove = above.reduce((a, b) => a + b, 0);
  const totalBelow = below.reduce((a, b) => a + b, 0);
  if (totalAbove === 0) {
    throw new InvalidParameterError(
      "median_test() found that all values are at or below the grand median",
      "samples",
      grandMedian
    );
  }
  if (totalBelow === 0) {
    throw new InvalidParameterError(
      "median_test() found that all values are at or above the grand median",
      "samples",
      grandMedian
    );
  }
  for (let g = 0; g < k; g++) {
    if ((above[g] ?? 0) + (below[g] ?? 0) === 0) {
      throw new InvalidParameterError(
        "median_test() found a sample whose values all equal the grand median and are ignored",
        "ties",
        g
      );
    }
  }

  const { statistic, pvalue } = chi2_contingency([above, below], correction);
  return { statistic, pvalue };
}

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror SciPy and remain exported (marked
// `@deprecated`) for backward compatibility. These camelCase aliases are the
// recommended names on Deepbox's public surface and refer to the same function.
// ---------------------------------------------------------------------------

/**
 * One-sample t-test.
 *
 * Tests whether the mean of a sample differs from a hypothesized population mean.
 *
 * @param a - Sample values (any shape; the tensor is flattened)
 * @param popmean - Hypothesized population mean
 * @param alternative - `"two-sided"` (default), `"less"` or `"greater"`
 * @returns The t statistic (`df = n - 1`) and the p-value
 * @throws {InvalidParameterError} If fewer than 2 values are given, the input is constant,
 *   or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { ttest1samp } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = ttest1samp(tensor([2.1, 1.9, 2.4, 2.2, 2.0]), 2.0);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const ttest1samp = ttest_1samp;
/**
 * Independent two-sample t-test.
 *
 * Tests whether means of two independent samples are equal. By default the
 * samples are assumed to have equal variance (Student's t-test); pass
 * `equalVar = false` for Welch's t-test with Welch-Satterthwaite degrees of freedom.
 *
 * @param a - First sample (any shape; the tensor is flattened)
 * @param b - Second sample
 * @param equalVar - Assume equal population variances (default: `true`), or a
 *   {@link TTestIndOptions} object
 * @param alternative - `"two-sided"` (default), `"less"` (mean of `a` is less than
 *   mean of `b`) or `"greater"`
 * @returns The t statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If a sample has fewer than 2 values, the pooled variance is
 *   zero, or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { ttestInd } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([2.1, 1.9, 2.4, 2.2, 2.0]);
 * const b = tensor([2.8, 3.1, 2.6, 3.0]);
 * ttestInd(a, b, { equalVar: false, alternative: "less" });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const ttestInd = ttest_ind;
/**
 * Paired-sample t-test.
 *
 * Tests whether means of two related samples are equal. Tensors are flattened
 * in row-major order and paired element by element.
 *
 * @param a - First sample
 * @param b - Second sample (same number of elements as `a`)
 * @param alternative - `"two-sided"` (default), `"less"` (mean of `a - b` is less
 *   than zero) or `"greater"`
 * @returns The t statistic (`df = n - 1`) and the p-value
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const ttestRel = ttest_rel;
/**
 * One-way analysis of variance.
 *
 * Tests whether two or more groups have the same population mean.
 *
 * @param samples - Two or more groups (flattened)
 * @returns The F statistic and the p-value. `F` is `Infinity` (with p-value 0) when the groups
 *   have no within-group variation but different means, and `NaN` when all values are equal.
 * @throws {InvalidParameterError} If fewer than 2 groups are given, a group is empty, or every
 *   group has a single value
 *
 * @example
 * ```ts
 * import { fOneway } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { statistic, pvalue } = fOneway(tensor([6, 8, 4, 5, 3, 4]), tensor([8, 12, 9, 11, 6, 8]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const fOneway = f_oneway;
/**
 * Chi-squared test of independence for a contingency table.
 *
 * Tests whether two categorical variables are independent
 * given their observed frequency table.
 *
 * For 2×2 tables (one degree of freedom) Yates' continuity correction is applied by
 * default, as in SciPy: each observed count is moved by at most 0.5 towards its expected
 * count. Pass `correction = false` for the plain Pearson statistic.
 *
 * @param observed - 2D array of observed frequencies (rows × cols)
 * @param correction - Apply Yates' continuity correction when `dof = 1` (default: `true`)
 * @returns Object with chi2 statistic, p-value, degrees of freedom, and expected frequencies
 * @throws {InvalidParameterError} If the table is smaller than 2×2, ragged, contains
 *   negative or non-finite counts, or has a row or column that sums to zero
 *
 * @example
 * ```ts
 * import { chi2Contingency } from 'deepbox/stats';
 * const result = chi2Contingency([[10, 20, 30], [6, 9, 17]]);
 * console.log(result.pvalue);
 * ```
 */
export const chi2Contingency = chi2_contingency;
/**
 * Two-sample Kolmogorov-Smirnov test.
 *
 * Tests whether two samples are drawn from the same continuous distribution.
 * The two-sided statistic is the maximum absolute difference between the
 * empirical distribution functions of the two samples.
 *
 * The p-value comes from the exact null distribution when both samples have at most
 * 10000 values (SciPy's `method="auto"`), and from the asymptotic distribution otherwise.
 * With tied values the exact p-value is conservative.
 *
 * @param x - First sample (any shape; the tensor is flattened)
 * @param y - Second sample
 * @param options - `alternative` (`"two-sided"` by default) and `method`
 *   (see {@link KsTestOptions})
 * @returns The statistic (`D`, or `D+` / `D-` for one-sided alternatives) and the p-value.
 *   Both are `NaN` if a sample contains `NaN`.
 * @throws {InvalidParameterError} If a sample is empty or an option is invalid
 *
 * @example
 * ```ts
 * import { ks2samp } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const result = ks2samp(tensor([1, 2, 3]), tensor([1.5, 2.5, 3.5]));
 * console.log(result.statistic, result.pvalue);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const ks2samp = ks_2samp;
/**
 * Fisher's exact test for a 2×2 contingency table.
 *
 * Computes the exact p-value for the association between two binary variables.
 * Uses the hypergeometric distribution. The two-sided p-value sums the probabilities of
 * all tables that are at most as likely as the observed one (with a relative tolerance
 * of 1e-7, as in R's `fisher.test`; SciPy 1.17 uses 1e-14, which only matters for tables
 * whose probabilities agree to more than 7 digits without being equal).
 *
 * @param table - 2×2 array of observed counts [[a, b], [c, d]] (non-negative integers)
 * @param alternative - 'two-sided' (default), 'less', or 'greater'
 * @returns Object with the sample odds ratio `a * d / (b * c)` (`Infinity` or `NaN` when the
 *   denominator is zero) and the p-value
 * @throws {InvalidParameterError} If a count is negative, not an integer or not finite, the
 *   total count exceeds 2^40, or `alternative` is invalid
 *
 * @example
 * ```ts
 * import { fisherExact } from 'deepbox/stats';
 * const result = fisherExact([[1, 9], [11, 3]]);
 * console.log(result.pvalue);
 * ```
 */
export const fisherExact = fisher_exact;
/**
 * Two-way ANOVA for balanced designs.
 *
 * Tests main effects of two factors and their interaction.
 * Expects data organized as a 3D array: data[a][b][replication].
 *
 * @param data - 3D array indexed by [factorA level][factorB level][replications]
 * @returns Object with factorA, factorB, and interaction test results
 * @throws {InvalidParameterError} If the design is unbalanced, a factor has fewer than 2
 *   levels, cells have a single replication, or a value is not finite
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const fTwoway = f_twoway;
/**
 * Mood's median test.
 *
 * Tests whether two or more samples have the same median. It builds the 2 by k contingency
 * table of counts above and below the grand median and applies the chi-square test of
 * independence. As in SciPy, Yates' continuity correction is applied by default when there
 * are two samples.
 *
 * @param args - Two or more samples (flattened), optionally followed by
 *   {@link MedianTestOptions}
 * @returns The chi-square statistic and the p-value (`NaN` if a value is `NaN`)
 * @throws {InvalidParameterError} If fewer than 2 samples are given, a sample is empty,
 *   all values fall on one side of the grand median, or `ties: "ignore"` leaves a sample empty
 *
 * @example
 * ```ts
 * import { medianTest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const result = medianTest(tensor([1, 2, 3, 4, 5]), tensor([3, 5, 7, 8, 9]));
 * console.log(result.statistic, result.pvalue);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export const medianTest = median_test;
/**
 * Wald-Wolfowitz runs test for randomness.
 *
 * Tests whether a sequence of binary (above/below median) values
 * is random by counting the number of "runs" (consecutive sequences
 * of the same type). Values equal to the median count as "above".
 *
 * @param x - Input tensor of numeric values, in observation order
 * @returns TestResult with z-statistic and two-sided p-value (`NaN` if the data contain `NaN`)
 *
 * @example
 * ```ts
 * import { runsTest } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 * const result = runsTest(tensor([1, 2, 1, 2, 1, 2, 1, 2]));
 * ```
 */
export const runsTest = runs_test;
