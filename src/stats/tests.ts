import { InvalidParameterError } from "../core";
import type { Tensor } from "../ndarray";
import {
  chiSquareCdf,
  fCdf,
  forEachIndexOffset,
  getNumberAt,
  normalCdf,
  normalPpf,
  rankData,
  studentTCdf,
} from "./_internal";

/** Result of a statistical hypothesis test. */
export type TestResult = {
  statistic: number;
  pvalue: number;
};

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
  out.sort((a, b) => a - b);
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
    // Shapiro-Wilk p-value is the upper tail: P(W <= w) = 1 - Φ(z).
    return { statistic: w, pvalue: 1 - normalCdf(z) };
  }
  m = poly(c5, lnN);
  s = Math.exp(poly(c6, lnN));
  const z = (y - m) / s;
  // Shapiro-Wilk p-value is the upper tail: P(W <= w) = 1 - Φ(z).
  return { statistic: w, pvalue: 1 - normalCdf(z) };
}

/**
 * One-sample t-test.
 *
 * Tests whether mean of sample differs from population mean.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function ttest_1samp(a: Tensor, popmean: number): TestResult {
  const x = toDenseSortedArray1D(a);
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
  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Independent two-sample t-test.
 *
 * Tests whether means of two independent samples are equal.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ttestInd}.
 */
export function ttest_ind(a: Tensor, b: Tensor, equalVar = true): TestResult {
  const xa = toDenseSortedArray1D(a);
  const xb = toDenseSortedArray1D(b);
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

  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Paired-sample t-test.
 *
 * Tests whether means of two related samples are equal.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 * @deprecated Prefer {@link ttestRel}.
 */
export function ttest_rel(a: Tensor, b: Tensor): TestResult {
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

  // Differences
  const diffs = new Float64Array(n);
  let i = 0;
  forEachIndexOffset(a, (offA) => {
    // Map to corresponding flat element in b using the same iteration order.
    // We rely on size equality and iterate both tensors to dense arrays.
    diffs[i] = getNumberAt(a, offA);
    i++;
  });
  const bd = new Float64Array(n);
  i = 0;
  forEachIndexOffset(b, (offB) => {
    bd[i] = getNumberAt(b, offB);
    i++;
  });
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
  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Chi-square goodness of fit test.
 *
 * Observed and expected frequencies must be non-negative and sum to the same total
 * (within floating-point tolerance).
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function chisquare(f_obs: Tensor, f_exp?: Tensor): TestResult {
  const obs = toDenseArray1D(f_obs);
  const n = obs.length;
  if (n < 1) {
    throw new InvalidParameterError("chisquare() requires at least one observed value", "n", n);
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

  const df = n - 1;
  if (df < 1) {
    throw new InvalidParameterError(
      "chisquare() requires at least 2 categories (df must be >= 1)",
      "df",
      df
    );
  }
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * Kolmogorov-Smirnov test for goodness of fit.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function kstest(data: Tensor, cdf: string | ((x: number) => number)): TestResult {
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

  let d = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? 0;
    const fi = F(xi);
    const dPlus = (i + 1) / n - fi;
    const dMinus = fi - i / n;
    d = Math.max(d, dPlus, dMinus);
  }

  // Asymptotic p-value (two-sided) using Kolmogorov distribution
  // p ≈ 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 d^2 n)
  let p = 0;
  for (let k = 1; k < 200; k++) {
    const term = Math.exp(-2 * k * k * d * d * n);
    p += (k % 2 === 1 ? 1 : -1) * term;
    if (term < 1e-12) break;
  }
  p = Math.max(0, Math.min(1, 2 * p));

  return { statistic: d, pvalue: p };
}

/**
 * Test for normality.
 *
 * Uses D'Agostino-Pearson's omnibus test combining skewness and kurtosis.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function normaltest(a: Tensor): TestResult {
  const x = toDenseSortedArray1D(a);
  const n = x.length;
  if (n < 8) {
    throw new InvalidParameterError("normaltest() requires at least 8 samples", "n", n);
  }
  const { mean, m2 } = meanAndM2(x);
  const m2n = m2 / n;
  const std = Math.sqrt(m2n);
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError("normaltest() is undefined for constant input", "std", std);
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
  const pvalue = 1 - chiSquareCdf(k2, 2);
  return { statistic: k2, pvalue };
}

/**
 * Shapiro-Wilk test for normality.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function shapiro(x: Tensor): TestResult {
  const sorted = toDenseSortedArray1D(x);
  return shapiroWilk(sorted);
}

/**
 * Anderson-Darling test for normality.
 *
 * Uses sample standard deviation and size-adjusted critical values.
 * For very small samples (n < 10), uses an IQR-based scale estimate to stabilize
 * the statistic.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function anderson(x: Tensor): {
  statistic: number;
  critical_values: number[];
  significance_level: number[];
} {
  const sorted = toDenseSortedArray1D(x);
  const n = sorted.length;
  if (n < 1) {
    throw new InvalidParameterError("anderson() requires at least one element", "n", n);
  }
  const { mean, m2 } = meanAndM2(sorted);
  const variance = n > 1 ? m2 / (n - 1) : NaN;
  let std = Math.sqrt(variance);
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError("anderson() is undefined for constant input", "std", std);
  }
  if (n < 10) {
    const quantile = (q: number): number => {
      if (n === 1) return sorted[0] ?? 0;
      const pos = (n - 1) * q;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      const v0 = sorted[lo] ?? 0;
      const v1 = sorted[hi] ?? v0;
      return v0 + (pos - lo) * (v1 - v0);
    };
    const q1 = quantile(0.25);
    const q3 = quantile(0.75);
    const iqr = q3 - q1;
    const robust = iqr / 1.349;
    if (Number.isFinite(robust) && robust > 0) {
      std = robust;
    }
  }

  // Anderson-Darling for normal: A^2 = -n - (1/n) sum_{i=1..n} (2i-1)[ln Phi(z_i) + ln(1-Phi(z_{n+1-i}))]
  let A2 = 0;
  for (let i = 0; i < n; i++) {
    const zi = ((sorted[i] ?? 0) - mean) / std;
    const zj = ((sorted[n - 1 - i] ?? 0) - mean) / std;
    const PhiI = Math.max(1e-300, Math.min(1 - 1e-16, normalCdf(zi)));
    const PhiJ = Math.max(1e-300, Math.min(1 - 1e-16, normalCdf(zj)));
    A2 += (2 * (i + 1) - 1) * (Math.log(PhiI) + Math.log(1 - PhiJ));
  }
  A2 = -n - A2 / n;

  const baseCritical = [0.576, 0.656, 0.787, 0.918, 1.092];
  const factor = 1 + 4 / n - 25 / (n * n);
  const critical_values = baseCritical.map((v) => Math.round((v / factor) * 1000) / 1000);

  return {
    statistic: A2,
    critical_values,
    significance_level: [0.15, 0.1, 0.05, 0.025, 0.01],
  };
}

/**
 * Mann-Whitney U test (non-parametric).
 *
 * Tests whether two independent samples come from same distribution.
 *
 * Note: Uses normal approximation for the p-value with tie correction and
 * continuity correction. No exact method selection is available.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function mannwhitneyu(x: Tensor, y: Tensor): TestResult {
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

  const { ranks, tieSum } = rankData(combined);

  // Rank sum for group 1 (first sample)
  let R1 = 0;
  for (let i = 0; i < nx; i++) {
    R1 += ranks[i] ?? 0;
  }

  const U1 = R1 - (nx * (nx + 1)) / 2;
  const U2 = nx * ny - U1;
  const U = Math.min(U1, U2);

  const meanU = (nx * ny) / 2;
  // Tie correction uses sum(t^3 - t) over tied groups.
  const tieAdj = n > 1 ? tieSum / (n * (n - 1)) : 0;
  const varU = (nx * ny * (n + 1 - tieAdj)) / 12;
  if (varU <= 0) {
    return { statistic: U, pvalue: NaN };
  }

  const stdU = Math.sqrt(varU);
  // Continuity correction for the two-sided normal approximation, applied for
  // all sample sizes (scipy's asymptotic method always applies it). Omitting
  // it for small n made the test anti-conservative rather than closer to the
  // exact distribution.
  const continuity = U < meanU ? 0.5 : U > meanU ? -0.5 : 0;
  const z = (U - meanU + continuity) / stdU;
  const pvalue = 2 * (1 - normalCdf(Math.abs(z)));

  return { statistic: U, pvalue };
}

/**
 * Wilcoxon signed-rank test (non-parametric paired test).
 *
 * Note: Uses normal approximation for the p-value with tie correction and
 * continuity correction. No exact method selection is available.
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function wilcoxon(x: Tensor, y?: Tensor): TestResult {
  const n = x.size;
  const diffs: number[] = [];

  if (y) {
    if (x.size !== y.size) {
      throw new InvalidParameterError("Paired samples must have equal length", "size", {
        x: x.size,
        y: y.size,
      });
    }
    const xd = toDenseArray1D(x);
    const yd = toDenseArray1D(y);
    for (let i = 0; i < n; i++) {
      const diff = (xd[i] ?? 0) - (yd[i] ?? 0);
      if (diff !== 0) diffs.push(diff);
    }
  } else {
    const xd = toDenseArray1D(x);
    for (let i = 0; i < n; i++) {
      const val = xd[i] ?? 0;
      if (val !== 0) diffs.push(val);
    }
  }

  if (diffs.length === 0) {
    throw new InvalidParameterError(
      "wilcoxon() is undefined when all differences are zero",
      "diffs",
      diffs.length
    );
  }

  const absDiffs = new Float64Array(diffs.length);
  for (let i = 0; i < diffs.length; i++) {
    absDiffs[i] = Math.abs(diffs[i] ?? 0);
  }

  const { ranks, tieSum } = rankData(absDiffs);

  let Wplus = 0;
  for (let i = 0; i < diffs.length; i++) {
    if ((diffs[i] ?? 0) > 0) {
      Wplus += ranks[i] ?? 0;
    }
  }

  const nEff = diffs.length;
  const meanW = (nEff * (nEff + 1)) / 4;
  // Tie-corrected variance for signed-rank statistic.
  const varW = (nEff * (nEff + 1) * (2 * nEff + 1)) / 24 - tieSum / 48;
  if (varW <= 0) {
    return { statistic: Wplus, pvalue: NaN };
  }

  const stdW = Math.sqrt(varW);
  // Continuity correction for two-sided normal approximation.
  // For small samples, omit correction to better match exact distribution behavior.
  const useContinuity = nEff > 20;
  const continuity = useContinuity ? (Wplus < meanW ? 0.5 : Wplus > meanW ? -0.5 : 0) : 0;
  const z = (Wplus - meanW + continuity) / stdW;
  const pvalue = 2 * (1 - normalCdf(Math.abs(z)));

  return { statistic: Wplus, pvalue };
}

/**
 * Kruskal-Wallis H-test (non-parametric version of ANOVA).
 *
 * Note: Uses chi-square approximation for the p-value with tie correction.
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
  const pvalue = 1 - chiSquareCdf(H, df);

  return { statistic: H, pvalue };
}

/**
 * Friedman test (non-parametric repeated measures ANOVA).
 *
 * Note: Uses chi-square approximation for the p-value with tie correction.
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
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * Levene's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances.
 * More robust than Bartlett's test for non-normal data.
 *
 * @param center - Method to use for centering: 'median' (default, most robust),
 *                 'mean' (traditional), or 'trimmed' (10% trimmed mean)
 * @param samples - Two or more sample tensors to compare
 * @returns Test result with statistic and p-value
 *
 * @example
 * ```ts
 * import { levene, tensor } from 'deepbox';
 *
 * const group1 = tensor([1, 2, 3, 4, 5]);
 * const group2 = tensor([2, 4, 6, 8, 10]);
 * const result = levene('median', group1, group2);
 * console.log(result.pvalue);  // p-value for equal variances
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
 */
export function levene(center: "mean" | "median" | "trimmed", ...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("levene() requires at least 2 groups", "k", k);
  }

  // Convert samples to arrays and compute centers
  const groups: Float64Array[] = [];
  const centers: number[] = [];

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError("levene() requires all groups to be non-empty", "groupSize", {
        group: g,
        size: sample?.size ?? 0,
      });
    }
    const arr = toDenseSortedArray1D(sample);
    if (arr.length < 2) {
      throw new InvalidParameterError(
        "levene() requires at least 2 samples per group",
        "groupSize",
        arr.length
      );
    }
    groups.push(arr);

    // Compute center based on method
    if (center === "mean") {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) sum += arr[i] ?? 0;
      centers.push(sum / arr.length);
    } else if (center === "median") {
      const mid = Math.floor(arr.length / 2);
      if (arr.length % 2 === 0) {
        centers.push(((arr[mid - 1] ?? 0) + (arr[mid] ?? 0)) / 2);
      } else {
        centers.push(arr[mid] ?? 0);
      }
    } else {
      // Trimmed mean (10%)
      const trimCount = Math.floor(arr.length * 0.1);
      let sum = 0;
      const n = arr.length - 2 * trimCount;
      for (let i = trimCount; i < arr.length - trimCount; i++) {
        sum += arr[i] ?? 0;
      }
      centers.push(sum / n);
    }
  }

  // Compute absolute deviations from center (Z_ij = |Y_ij - center_i|)
  const Z: Float64Array[] = [];
  const groupMeansZ: number[] = [];
  let N = 0;
  let grandSumZ = 0;

  for (let g = 0; g < groups.length; g++) {
    const arr = groups[g];
    if (!arr) continue;
    const c = centers[g] ?? 0;
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

  // Compute Levene's W statistic (F-test on absolute deviations)
  let SSB = 0; // Between-group sum of squares
  let SSW = 0; // Within-group sum of squares

  for (let g = 0; g < Z.length; g++) {
    const zArr = Z[g];
    if (!zArr) continue;
    const n = zArr.length;
    SSB += n * ((groupMeansZ[g] ?? 0) - grandMeanZ) ** 2;

    for (let i = 0; i < n; i++) {
      SSW += ((zArr[i] ?? 0) - (groupMeansZ[g] ?? 0)) ** 2;
    }
  }

  const dfB = k - 1;
  const dfW = N - k;
  if (dfW <= 0) {
    throw new InvalidParameterError(
      "levene() requires more total observations than groups",
      "dfW",
      dfW
    );
  }
  if (SSW === 0) {
    return { statistic: Infinity, pvalue: 0 };
  }
  const W = SSB / dfB / (SSW / dfW);

  const pvalue = 1 - fCdf(W, dfB, dfW);

  return { statistic: W, pvalue };
}

/**
 * Bartlett's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances.
 * Assumes data is normally distributed; use Levene's test for non-normal data.
 *
 * @param samples - Two or more sample tensors to compare
 * @returns Test result with statistic and p-value
 *
 * @example
 * ```ts
 * import { bartlett, tensor } from 'deepbox';
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

  // Convert samples to arrays and compute variances
  const variances: number[] = [];
  const sizes: number[] = [];
  let N = 0;

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError(
        "bartlett() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: sample?.size ?? 0 }
      );
    }
    const arr = toDenseSortedArray1D(sample);
    const n = arr.length;
    if (n < 2) {
      throw new InvalidParameterError(
        "bartlett() requires at least 2 samples per group",
        "groupSize",
        n
      );
    }

    // Compute mean
    let mean = 0;
    for (let i = 0; i < n; i++) mean += arr[i] ?? 0;
    mean /= n;

    // Compute sample variance (using n-1 denominator)
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = (arr[i] ?? 0) - mean;
      ss += d * d;
    }
    const variance = ss / (n - 1);

    variances.push(variance);
    sizes.push(n);
    N += n;
  }

  // Check for zero variances
  for (let g = 0; g < k; g++) {
    if ((variances[g] ?? 0) === 0) {
      throw new InvalidParameterError(
        "bartlett() is undefined when a group has zero variance",
        "variance",
        variances[g]
      );
    }
  }

  // Compute pooled variance
  let pooledNumerator = 0;
  for (let g = 0; g < k; g++) {
    pooledNumerator += ((sizes[g] ?? 1) - 1) * (variances[g] ?? 1);
  }
  const pooledVariance = pooledNumerator / (N - k);

  // Compute Bartlett's statistic
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

  // Chi-square statistic
  const chiSq = T / C;

  const df = k - 1;
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * One-way ANOVA.
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

  const pvalue = 1 - fCdf(F, dfB, dfW);

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
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Hypothesis Tests}
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
    const pval = 1 - fCdf(stat, df, dfE);
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
 * @param observed - 2D array of observed frequencies (rows × cols)
 * @returns Object with chi2 statistic, p-value, degrees of freedom, and expected frequencies
 *
 * @example
 * ```ts
 * import { chi2_contingency } from 'deepbox/stats';
 * const result = chi2_contingency([[10, 20, 30], [6, 9, 17]]);
 * console.log(result.pvalue);
 * ```
 * @deprecated Prefer {@link chi2Contingency}.
 */
export function chi2_contingency(observed: readonly (readonly number[])[]): ContingencyResult {
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
      if (val < 0) {
        throw new InvalidParameterError(
          "Observed frequencies must be non-negative",
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

  // Compute expected frequencies and chi-squared statistic
  const expected: number[][] = [];
  let chi2Stat = 0;

  for (let i = 0; i < nRows; i++) {
    const row: number[] = [];
    for (let j = 0; j < nCols; j++) {
      const exp = ((rowTotals[i] ?? 0) * (colTotals[j] ?? 0)) / grandTotal;
      row.push(exp);
      const obs = observed[i]?.[j] ?? 0;
      if (exp > 0) {
        chi2Stat += (obs - exp) ** 2 / exp;
      }
    }
    expected.push(row);
  }

  const dof = (nRows - 1) * (nCols - 1);
  const pvalue = 1 - chiSquareCdf(chi2Stat, dof);

  return { statistic: chi2Stat, pvalue, dof, expected };
}

/**
 * Fisher's exact test for a 2×2 contingency table.
 *
 * Computes the exact p-value for the association between two binary variables.
 * Uses the hypergeometric distribution.
 *
 * @param table - 2×2 array of observed frequencies [[a, b], [c, d]]
 * @param alternative - 'two-sided' (default), 'less', or 'greater'
 * @returns Object with odds ratio and p-value
 *
 * @example
 * ```ts
 * import { fisher_exact } from 'deepbox/stats';
 * const result = fisher_exact([[1, 9], [11, 3]]);
 * console.log(result.pvalue);
 * ```
 * @deprecated Prefer {@link fisherExact}.
 */
export function fisher_exact(
  table: readonly [readonly [number, number], readonly [number, number]],
  alternative: "two-sided" | "less" | "greater" = "two-sided"
): { oddsRatio: number; pvalue: number } {
  const a = table[0][0];
  const b = table[0][1];
  const c = table[1][0];
  const d = table[1][1];

  if (a < 0 || b < 0 || c < 0 || d < 0) {
    throw new InvalidParameterError("All values must be non-negative", "table", 0);
  }

  const n = a + b + c + d;
  const r1 = a + b; // row 1 total
  const r2 = c + d; // row 2 total
  const c1 = a + c; // col 1 total

  // Odds ratio
  const oddsRatio = b === 0 || c === 0 ? (a * d === 0 ? 0 : Infinity) : (a * d) / (b * c);

  // Log of hypergeometric PMF: P(X = k) = C(r1,k) * C(r2,c1-k) / C(n,c1)
  const logHyperPmf = (k: number): number => {
    if (k < 0 || k > r1 || k > c1 || c1 - k > r2) return -Infinity;
    return logChoose(r1, k) + logChoose(r2, c1 - k) - logChoose(n, c1);
  };

  const pObs = logHyperPmf(a);
  const kMin = Math.max(0, c1 - r2);
  const kMax = Math.min(r1, c1);

  let pvalue = 0;
  if (alternative === "less") {
    for (let k = kMin; k <= a; k++) {
      pvalue += Math.exp(logHyperPmf(k));
    }
  } else if (alternative === "greater") {
    for (let k = a; k <= kMax; k++) {
      pvalue += Math.exp(logHyperPmf(k));
    }
  } else {
    // two-sided: sum probabilities <= P(X = a)
    for (let k = kMin; k <= kMax; k++) {
      const lp = logHyperPmf(k);
      if (lp <= pObs + 1e-10) {
        pvalue += Math.exp(lp);
      }
    }
  }

  return { oddsRatio, pvalue: Math.min(1, pvalue) };
}

function logChoose(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity;
  if (k === 0 || k === n) return 0;
  // Use log-gamma: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
  return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
}

function logFactorial(n: number): number {
  if (n <= 1) return 0;
  let sum = 0;
  for (let i = 2; i <= n; i++) {
    sum += Math.log(i);
  }
  return sum;
}

// ─── Runs test ──────────────────────────────────────────────────────────────

/**
 * Wald-Wolfowitz runs test for randomness.
 *
 * Tests whether a sequence of binary (above/below median) values
 * is random by counting the number of "runs" (consecutive sequences
 * of the same type).
 *
 * @param x - Input tensor of numeric values
 * @returns TestResult with z-statistic and p-value
 *
 * @example
 * ```ts
 * import { runs_test } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 * const result = runs_test(tensor([1, 2, 1, 2, 1, 2, 1, 2]));
 * ```
 */
export function runs_test(x: Tensor): TestResult {
  const data = toDenseArray1D(x);
  const n = data.length;
  if (n < 2) {
    throw new InvalidParameterError("runs_test requires at least 2 observations", "x.size", n);
  }

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
  const pvalue = 2 * (1 - normalCdf(Math.abs(z)));

  return { statistic: z, pvalue };
}

// ─── Lilliefors test ────────────────────────────────────────────────────────

/**
 * Lilliefors test for normality.
 *
 * A variant of the Kolmogorov-Smirnov test where the mean and variance
 * are estimated from the data (rather than specified). Uses the KS
 * statistic with critical values adjusted for estimated parameters.
 *
 * @param x - Input tensor of numeric values
 * @returns TestResult with KS statistic and approximate p-value
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

  // Estimate mean and std from data
  let sum = 0;
  for (let i = 0; i < n; i++) sum += data[i] ?? 0;
  const mean = sum / n;

  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = (data[i] ?? 0) - mean;
    ss += d * d;
  }
  const std = Math.sqrt(ss / (n - 1));

  if (std < 1e-15) {
    // All values identical — cannot test
    return { statistic: 0, pvalue: 1 };
  }

  // Compute KS statistic against N(mean, std)
  let dMax = 0;
  for (let i = 0; i < n; i++) {
    const z = ((data[i] ?? 0) - mean) / std;
    const empiricalCdf = (i + 1) / n;
    const empiricalCdfPrev = i / n;
    const theoreticalCdf = normalCdf(z);

    const d1 = Math.abs(empiricalCdf - theoreticalCdf);
    const d2 = Math.abs(empiricalCdfPrev - theoreticalCdf);
    dMax = Math.max(dMax, d1, d2);
  }

  // Approximate p-value using Dallal-Wilkinson formula (1986)
  // This is a well-known approximation for the Lilliefors test
  const sqrtN = Math.sqrt(n);
  const dn = dMax * (sqrtN - 0.01 + 0.85 / sqrtN);
  let pvalue: number;
  if (dn < 0.302) {
    pvalue = 1;
  } else if (dn > 1.8) {
    pvalue = 0;
  } else {
    // Approximation formula
    pvalue = Math.exp(
      -7.01256 * dn * dn * (dn + 2.78019) +
        2.99587 * dn * (dn + 2.78019) -
        0.122119 +
        0.974598 / sqrtN +
        1.67997 / n
    );
    pvalue = Math.max(0, Math.min(1, pvalue));
  }

  return { statistic: dMax, pvalue };
}

// ─── Fligner-Killeen test ───────────────────────────────────────────────────

/**
 * Fligner-Killeen test for equality of variances.
 *
 * A non-parametric test that is robust against departures from normality.
 * Uses the ranks of absolute deviations from group medians.
 *
 * @param groups - Array of tensors, one per group (at least 2 groups)
 * @returns TestResult with chi-squared statistic and p-value
 *
 * @example
 * ```ts
 * import { fligner } from 'deepbox/stats';
 * import { tensor } from 'deepbox/ndarray';
 * const result = fligner([tensor([1, 2, 3]), tensor([4, 5, 6, 7])]);
 * ```
 */
export function fligner(groups: readonly Tensor[]): TestResult {
  if (groups.length < 2) {
    throw new InvalidParameterError(
      "fligner test requires at least 2 groups",
      "groups.length",
      groups.length
    );
  }

  // Convert groups to arrays and compute group medians
  const groupData: Float64Array[] = [];
  let N = 0;
  for (const g of groups) {
    const arr = toDenseSortedArray1D(g);
    if (arr.length < 2) {
      throw new InvalidParameterError(
        "Each group must have at least 2 observations",
        "group.size",
        arr.length
      );
    }
    groupData.push(arr);
    N += arr.length;
  }

  // Compute absolute deviations from group medians
  const allDeviations: number[] = [];
  const groupSizes: number[] = [];
  const groupLabels: number[] = []; // which group each deviation belongs to

  for (let g = 0; g < groupData.length; g++) {
    const arr = groupData[g];
    if (!arr) continue;
    const n = arr.length;
    groupSizes.push(n);

    // Median of sorted array
    const med =
      n % 2 === 0 ? ((arr[n / 2 - 1] ?? 0) + (arr[n / 2] ?? 0)) / 2 : (arr[Math.floor(n / 2)] ?? 0);

    for (let i = 0; i < n; i++) {
      allDeviations.push(Math.abs((arr[i] ?? 0) - med));
      groupLabels.push(g);
    }
  }

  // Rank the absolute deviations
  const indices = Array.from({ length: N }, (_, i) => i);
  indices.sort((a, b) => (allDeviations[a] ?? 0) - (allDeviations[b] ?? 0));

  const ranks = new Float64Array(N);
  let i = 0;
  while (i < N) {
    let j = i;
    while (
      j < N - 1 &&
      (allDeviations[indices[j + 1] ?? 0] ?? 0) === (allDeviations[indices[j] ?? 0] ?? 0)
    ) {
      j++;
    }
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      ranks[indices[k] ?? 0] = avgRank;
    }
    i = j + 1;
  }

  // Transform ranks using normal scores. Fligner-Killeen uses the half-normal
  // score a_i = Φ⁻¹(½ + rank_i / (2·(N+1))), reflecting that the absolute
  // deviations are folded about zero.
  const scores = new Float64Array(N);
  for (let idx = 0; idx < N; idx++) {
    const p = 0.5 + (ranks[idx] ?? 0) / (2 * (N + 1));
    scores[idx] = normalPpf(p);
  }

  // Compute group means of scores
  const groupScoreSums: number[] = new Array(groupData.length).fill(0);
  for (let idx = 0; idx < N; idx++) {
    const g = groupLabels[idx] ?? 0;
    groupScoreSums[g] = (groupScoreSums[g] ?? 0) + (scores[idx] ?? 0);
  }

  let grandMean = 0;
  for (let idx = 0; idx < N; idx++) {
    grandMean += scores[idx] ?? 0;
  }
  grandMean /= N;

  // Fligner-Killeen statistic (chi-squared)
  let numerator = 0;
  for (let g = 0; g < groupData.length; g++) {
    const ng = groupSizes[g] ?? 0;
    const gMean = (groupScoreSums[g] ?? 0) / ng;
    numerator += ng * (gMean - grandMean) ** 2;
  }

  let denominator = 0;
  for (let idx = 0; idx < N; idx++) {
    denominator += ((scores[idx] ?? 0) - grandMean) ** 2;
  }
  denominator /= N - 1;

  if (denominator < 1e-15) {
    return { statistic: 0, pvalue: 1 };
  }

  const stat = numerator / denominator;
  const dof = groupData.length - 1;
  const pvalue = 1 - chiSquareCdf(stat, dof);

  return { statistic: stat, pvalue };
}

/**
 * Two-sample Kolmogorov-Smirnov test.
 *
 * Tests whether two samples are drawn from the same distribution.
 * The test statistic is the maximum absolute difference between the
 * empirical CDFs of the two samples.
 *
 * The p-value uses the asymptotic Kolmogorov distribution.
 *
 * @param x - First sample (1-D tensor)
 * @param y - Second sample (1-D tensor)
 * @returns Object with `statistic` (D) and `pvalue`
 *
 * @example
 * ```ts
 * const result = ks_2samp(tensor([1, 2, 3]), tensor([1.5, 2.5, 3.5]));
 * console.log(result.statistic, result.pvalue);
 * ```
 * @deprecated Prefer {@link ks2samp}.
 */
export function ks_2samp(x: Tensor, y: Tensor): TestResult {
  const xs = toDenseSortedArray1D(x);
  const ys = toDenseSortedArray1D(y);
  const nx = xs.length;
  const ny = ys.length;
  if (nx === 0 || ny === 0) {
    throw new InvalidParameterError("ks_2samp() requires non-empty samples", "size", 0);
  }

  // Compute max |F1(t) - F2(t)| by merging sorted arrays
  let d = 0;
  let i = 0;
  let j = 0;
  while (i < nx && j < ny) {
    const xi = xs[i] ?? 0;
    const yj = ys[j] ?? 0;
    if (xi <= yj) {
      i++;
    }
    if (yj <= xi) {
      j++;
    }
    const diff = Math.abs(i / nx - j / ny);
    if (diff > d) d = diff;
  }

  // Asymptotic p-value using Kolmogorov distribution
  const en = Math.sqrt((nx * ny) / (nx + ny));
  const pvalue = kolmogorovPvalue(d, en);

  return { statistic: d, pvalue };
}

/** Kolmogorov distribution survival function (asymptotic approximation). */
function kolmogorovPvalue(d: number, sqrtN: number): number {
  const lambda = (sqrtN + 0.12 + 0.11 / sqrtN) * d;
  if (lambda < 1e-15) return 1;
  // Series expansion: P(D > d) = 2 * sum_{k=1}^{inf} (-1)^(k-1) * exp(-2*k^2*lambda^2)
  let sum = 0;
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda);
    if (k % 2 === 1) sum += term;
    else sum -= term;
    if (term < 1e-15) break;
  }
  return Math.max(0, Math.min(1, 2 * sum));
}

/**
 * Mood's median test.
 *
 * Tests whether two or more samples have the same median.
 * Computes a chi-squared statistic from the 2×k contingency table of
 * counts above/below the grand median.
 *
 * @param samples - Two or more 1-D tensors
 * @returns Object with `statistic` (chi-squared) and `pvalue`
 *
 * @example
 * ```ts
 * const result = median_test(tensor([1, 2, 3]), tensor([4, 5, 6]));
 * console.log(result.statistic, result.pvalue);
 * ```
 */
export function median_test(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("median_test() requires at least 2 groups", "groups", k);
  }

  // Collect all values to find grand median
  const groups: number[][] = [];
  const allVals: number[] = [];
  for (const s of samples) {
    const vals: number[] = [];
    forEachIndexOffset(s, (offset) => {
      const v = getNumberAt(s, offset);
      vals.push(v);
      allVals.push(v);
    });
    if (vals.length === 0) {
      throw new InvalidParameterError("median_test() requires non-empty samples", "size", 0);
    }
    groups.push(vals);
  }

  // Grand median
  allVals.sort((a, b) => a - b);
  const n = allVals.length;
  const grandMedian =
    n % 2 === 1
      ? (allVals[Math.floor(n / 2)] ?? 0)
      : ((allVals[n / 2 - 1] ?? 0) + (allVals[n / 2] ?? 0)) / 2;

  // Build 2×k contingency table: [above, at-or-below] for each group
  const above: number[] = new Array(k).fill(0);
  const below: number[] = new Array(k).fill(0);
  let totalAbove = 0;
  let totalBelow = 0;

  for (let g = 0; g < k; g++) {
    const vals = groups[g] as number[];
    for (const v of vals) {
      if (v > grandMedian) {
        above[g] = (above[g] ?? 0) + 1;
        totalAbove++;
      } else {
        below[g] = (below[g] ?? 0) + 1;
        totalBelow++;
      }
    }
  }

  // Chi-squared test on 2×k table
  let chi2Stat = 0;
  for (let g = 0; g < k; g++) {
    const ng = (above[g] ?? 0) + (below[g] ?? 0);
    const expAbove = (ng * totalAbove) / n;
    const expBelow = (ng * totalBelow) / n;
    if (expAbove > 0) {
      chi2Stat += ((above[g] ?? 0) - expAbove) ** 2 / expAbove;
    }
    if (expBelow > 0) {
      chi2Stat += ((below[g] ?? 0) - expBelow) ** 2 / expBelow;
    }
  }

  const dof = k - 1;
  const pvalue = 1 - chiSquareCdf(chi2Stat, dof);

  return { statistic: chi2Stat, pvalue };
}

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror SciPy and remain exported (marked
// `@deprecated`) for backward compatibility. These camelCase aliases are the
// recommended names on Deepbox's public surface and refer to the same function.
// ---------------------------------------------------------------------------

/** Canonical camelCase alias of {@link ttest_ind}. */
export const ttestInd = ttest_ind;
/** Canonical camelCase alias of {@link ttest_rel}. */
export const ttestRel = ttest_rel;
/** Canonical camelCase alias of {@link f_oneway}. */
export const fOneway = f_oneway;
/** Canonical camelCase alias of {@link chi2_contingency}. */
export const chi2Contingency = chi2_contingency;
/** Canonical camelCase alias of {@link ks_2samp}. */
export const ks2samp = ks_2samp;
/** Canonical camelCase alias of {@link fisher_exact}. */
export const fisherExact = fisher_exact;
