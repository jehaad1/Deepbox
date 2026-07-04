/**
 * Probability distribution objects.
 *
 * Each distribution provides: pdf/pmf, cdf, ppf, sf, rvs, mean, variance, entropy.
 *
 * @module stats/distributions
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { __random } from "../random/random";
import {
  chiSquareCdf,
  digamma,
  fCdf,
  logGamma,
  normalCdf,
  normalPpf,
  regularizedIncompleteBeta,
  studentTCdf,
} from "./_internal";

// ---- Helpers ----

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Box-Muller transform for normal random variates */
function randn(): number {
  let u = 0,
    v = 0;
  while (u === 0) u = __random();
  while (v === 0) v = __random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gamma random variate via Marsaglia-Tsang method */
/**
 * Invert a monotonically-increasing CDF for the quantile p ∈ (0,1) on the
 * support [lo, hi] (hi may be Infinity). Brackets the root by expansion, then
 * bisects — robust across the whole range, unlike an unbracketed Newton
 * iteration which diverges in the tails.
 */
function invertMonotoneCdf(
  cdf: (x: number) => number,
  p: number,
  lo: number,
  hi: number,
  guess = 1
): number {
  // Establish a finite upper bracket where cdf(hiB) >= p.
  let loB = lo;
  let hiB = Number.isFinite(hi) ? hi : Math.max(guess, 1);
  if (!Number.isFinite(hi)) {
    let steps = 0;
    while (cdf(hiB) < p && steps < 200) {
      loB = hiB;
      hiB *= 2;
      steps++;
    }
  }
  // Establish a lower bracket where cdf(loB) <= p.
  if (cdf(loB) > p) {
    let steps = 0;
    let span = hiB - loB || 1;
    while (loB > lo && cdf(loB) > p && steps < 200) {
      hiB = loB;
      loB = Math.max(lo, loB - span);
      span *= 2;
      steps++;
    }
  }
  // Bisection.
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (loB + hiB);
    const c = cdf(mid);
    if (c < p) loB = mid;
    else hiB = mid;
    if (hiB - loB <= 1e-12 * Math.max(1, Math.abs(mid))) break;
  }
  return 0.5 * (loB + hiB);
}

function gammaSample(shape: number): number {
  if (shape < 1) {
    return gammaSample(shape + 1) * __random() ** (1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = __random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// ---- Distribution Interface ----

export interface ContinuousDistribution {
  pdf(x: number): number;
  cdf(x: number): number;
  ppf(p: number): number;
  sf(x: number): number;
  rvs(size?: number): number[];
  mean(): number;
  variance(): number;
  entropy(): number;
}

export interface DiscreteDistribution {
  pmf(k: number): number;
  cdf(x: number): number;
  ppf(p: number): number;
  sf(x: number): number;
  rvs(size?: number): number[];
  mean(): number;
  variance(): number;
}

// ---- Normal Distribution ----

class NormalDist implements ContinuousDistribution {
  constructor(
    private loc: number = 0,
    private scale: number = 1
  ) {
    if (scale <= 0) throw new InvalidParameterError("scale must be > 0", "scale", scale);
  }

  pdf(x: number): number {
    const z = (x - this.loc) / this.scale;
    return Math.exp(-0.5 * z * z) / (this.scale * SQRT_2PI);
  }

  cdf(x: number): number {
    return normalCdf((x - this.loc) / this.scale);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    return this.loc + this.scale * normalPpf(p);
  }

  sf(x: number): number {
    // Compute the upper tail directly (Φ(−z)) rather than 1 − Φ(z), which
    // catastrophically cancels to exactly 0 for x more than ~8σ out.
    return normalCdf(-((x - this.loc) / this.scale));
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => this.loc + this.scale * randn());
  }

  mean(): number {
    return this.loc;
  }
  variance(): number {
    return this.scale * this.scale;
  }
  entropy(): number {
    return 0.5 * Math.log(2 * Math.PI * Math.E * this.scale * this.scale);
  }
}

// ---- Student's t Distribution ----

class TDist implements ContinuousDistribution {
  constructor(private df: number) {
    if (df <= 0) throw new InvalidParameterError("df must be > 0", "df", df);
  }

  pdf(x: number): number {
    const v = this.df;
    const coeff = Math.exp(logGamma((v + 1) / 2) - logGamma(v / 2)) / Math.sqrt(v * Math.PI);
    return coeff * (1 + (x * x) / v) ** (-(v + 1) / 2);
  }

  cdf(x: number): number {
    return studentTCdf(x, this.df);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    // Newton's method
    let x = normalPpf(p);
    for (let i = 0; i < 50; i++) {
      const f = this.cdf(x) - p;
      const fp = this.pdf(x);
      if (fp === 0) break;
      const dx = f / fp;
      x -= dx;
      if (Math.abs(dx) < 1e-12) break;
    }
    return x;
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      const z = randn();
      const chi2 = 2 * gammaSample(this.df / 2);
      return z / Math.sqrt(chi2 / this.df);
    });
  }

  mean(): number {
    return this.df > 1 ? 0 : NaN;
  }
  variance(): number {
    if (this.df > 2) return this.df / (this.df - 2);
    if (this.df > 1) return Infinity;
    return NaN;
  }
  entropy(): number {
    // Exact differential entropy of Student's t:
    //   h = ((v+1)/2)·[ψ((v+1)/2) − ψ(v/2)] + ½·ln(v) + ln B(v/2, ½)
    // where ln B(v/2, ½) = lnΓ(v/2) + lnΓ(½) − lnΓ((v+1)/2) and lnΓ(½) = ½·ln(π).
    const v = this.df;
    const lnBeta = logGamma(v / 2) + 0.5 * Math.log(Math.PI) - logGamma((v + 1) / 2);
    return ((v + 1) / 2) * (digamma((v + 1) / 2) - digamma(v / 2)) + 0.5 * Math.log(v) + lnBeta;
  }
}

// ---- Chi-squared Distribution ----

class Chi2Dist implements ContinuousDistribution {
  constructor(private df: number) {
    if (df <= 0) throw new InvalidParameterError("df must be > 0", "df", df);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const k = this.df;
    return Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.log(2) - logGamma(k / 2));
  }

  cdf(x: number): number {
    return chiSquareCdf(x, this.df);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    // Wilson–Hilferty initial guess, then a bracketed inversion (the previous
    // Newton clamped tiny quantiles at 1e-10, losing ~2 orders of magnitude).
    const guess = Math.max(
      0.01,
      this.df * (1 - 2 / (9 * this.df) + normalPpf(p) * Math.sqrt(2 / (9 * this.df))) ** 3
    );
    return invertMonotoneCdf((x) => this.cdf(x), p, 0, Infinity, guess);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => 2 * gammaSample(this.df / 2));
  }

  mean(): number {
    return this.df;
  }
  variance(): number {
    return 2 * this.df;
  }
  entropy(): number {
    // h = k/2 + ln(2) + lnΓ(k/2) + (1 − k/2)·ψ(k/2)
    const k = this.df;
    return k / 2 + Math.log(2) + logGamma(k / 2) + (1 - k / 2) * digamma(k / 2);
  }
}

// ---- F Distribution ----

class FDist implements ContinuousDistribution {
  constructor(
    private dfn: number,
    private dfd: number
  ) {
    if (dfn <= 0) throw new InvalidParameterError("dfn must be > 0", "dfn", dfn);
    if (dfd <= 0) throw new InvalidParameterError("dfd must be > 0", "dfd", dfd);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const d1 = this.dfn,
      d2 = this.dfd;
    const num = (d1 * x) ** d1 * d2 ** d2;
    const den = (d1 * x + d2) ** (d1 + d2);
    return (
      Math.sqrt(num / den) /
      (x * Math.exp(logGamma(d1 / 2) + logGamma(d2 / 2) - logGamma((d1 + d2) / 2)))
    );
  }

  cdf(x: number): number {
    return fCdf(x, this.dfn, this.dfd);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    // Bracketed inversion (unbracketed Newton diverged by orders of magnitude
    // in the lower tail for many df combinations).
    return invertMonotoneCdf((x) => this.cdf(x), p, 0, Infinity, 1);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      const x1 = 2 * gammaSample(this.dfn / 2);
      const x2 = 2 * gammaSample(this.dfd / 2);
      return x1 / this.dfn / (x2 / this.dfd);
    });
  }

  mean(): number {
    return this.dfd > 2 ? this.dfd / (this.dfd - 2) : NaN;
  }

  variance(): number {
    const d1 = this.dfn,
      d2 = this.dfd;
    if (d2 <= 4) return NaN;
    return (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4));
  }

  entropy(): number {
    // h = ln(d2/d1) + ln B(d1/2, d2/2) + (1 − d1/2)·ψ(d1/2)
    //     − (1 + d2/2)·ψ(d2/2) + ((d1+d2)/2)·ψ((d1+d2)/2)
    const d1 = this.dfn,
      d2 = this.dfd;
    const lnBeta = logGamma(d1 / 2) + logGamma(d2 / 2) - logGamma((d1 + d2) / 2);
    return (
      Math.log(d2 / d1) +
      lnBeta +
      (1 - d1 / 2) * digamma(d1 / 2) -
      (1 + d2 / 2) * digamma(d2 / 2) +
      ((d1 + d2) / 2) * digamma((d1 + d2) / 2)
    );
  }
}

// ---- Uniform Distribution ----

class UniformDist implements ContinuousDistribution {
  constructor(
    private low: number = 0,
    private high: number = 1
  ) {
    if (low >= high) throw new InvalidParameterError("low must be < high", "low", low);
  }

  pdf(x: number): number {
    return x >= this.low && x <= this.high ? 1 / (this.high - this.low) : 0;
  }

  cdf(x: number): number {
    if (x < this.low) return 0;
    if (x > this.high) return 1;
    return (x - this.low) / (this.high - this.low);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    return this.low + p * (this.high - this.low);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => this.low + __random() * (this.high - this.low));
  }

  mean(): number {
    return (this.low + this.high) / 2;
  }
  variance(): number {
    const r = this.high - this.low;
    return (r * r) / 12;
  }
  entropy(): number {
    return Math.log(this.high - this.low);
  }
}

// ---- Exponential Distribution ----

class ExponentialDist implements ContinuousDistribution {
  constructor(private rate: number = 1) {
    if (rate <= 0) throw new InvalidParameterError("rate must be > 0", "rate", rate);
  }

  pdf(x: number): number {
    return x < 0 ? 0 : this.rate * Math.exp(-this.rate * x);
  }

  cdf(x: number): number {
    return x < 0 ? 0 : 1 - Math.exp(-this.rate * x);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    return -Math.log(1 - p) / this.rate;
  }

  sf(x: number): number {
    return x < 0 ? 1 : Math.exp(-this.rate * x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      let u = __random();
      while (u === 0) u = __random();
      return -Math.log(u) / this.rate;
    });
  }

  mean(): number {
    return 1 / this.rate;
  }
  variance(): number {
    return 1 / (this.rate * this.rate);
  }
  entropy(): number {
    return 1 - Math.log(this.rate);
  }
}

// ---- Beta Distribution ----

class BetaDist implements ContinuousDistribution {
  constructor(
    private a: number,
    private b: number
  ) {
    if (a <= 0) throw new InvalidParameterError("a must be > 0", "a", a);
    if (b <= 0) throw new InvalidParameterError("b must be > 0", "b", b);
  }

  pdf(x: number): number {
    if (x <= 0 || x >= 1) return 0;
    const lnB = logGamma(this.a) + logGamma(this.b) - logGamma(this.a + this.b);
    return Math.exp((this.a - 1) * Math.log(x) + (this.b - 1) * Math.log(1 - x) - lnB);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return regularizedIncompleteBeta(this.a, this.b, x);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return 1;
    // Bracketed bisection on [0,1] (unbracketed Newton from 0.5 collapsed to
    // the wrong endpoint in the tails, returning 1 for a lower-tail quantile).
    return invertMonotoneCdf((x) => this.cdf(x), p, 0, 1, 0.5);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      const x = gammaSample(this.a);
      const y = gammaSample(this.b);
      return x / (x + y);
    });
  }

  mean(): number {
    return this.a / (this.a + this.b);
  }
  variance(): number {
    const ab = this.a + this.b;
    return (this.a * this.b) / (ab * ab * (ab + 1));
  }
  entropy(): number {
    // h = ln B(a,b) − (a−1)·ψ(a) − (b−1)·ψ(b) + (a+b−2)·ψ(a+b)
    const ab = this.a + this.b;
    const lnBeta = logGamma(this.a) + logGamma(this.b) - logGamma(ab);
    return (
      lnBeta -
      (this.a - 1) * digamma(this.a) -
      (this.b - 1) * digamma(this.b) +
      (ab - 2) * digamma(ab)
    );
  }
}

// ---- Gamma Distribution ----

class GammaDist implements ContinuousDistribution {
  constructor(
    private shape: number,
    private rate: number = 1
  ) {
    if (shape <= 0) throw new InvalidParameterError("shape must be > 0", "shape", shape);
    if (rate <= 0) throw new InvalidParameterError("rate must be > 0", "rate", rate);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const a = this.shape,
      b = this.rate;
    return Math.exp(a * Math.log(b) + (a - 1) * Math.log(x) - b * x - logGamma(a));
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return chiSquareCdf(2 * this.rate * x, 2 * this.shape);
    // chi2_cdf(x, k) = regularized_gamma(k/2, x/2), so gamma_cdf(x, a, b) = regularized_gamma(a, b*x)
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    // Bracketed inversion (the previous Newton clamped tiny quantiles at
    // 1e-10, off by orders of magnitude in the lower tail).
    return invertMonotoneCdf((x) => this.cdf(x), p, 0, Infinity, this.shape / this.rate);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => gammaSample(this.shape) / this.rate);
  }

  mean(): number {
    return this.shape / this.rate;
  }
  variance(): number {
    return this.shape / (this.rate * this.rate);
  }
  entropy(): number {
    // h = a − ln(b) + lnΓ(a) + (1 − a)·ψ(a)
    const a = this.shape,
      b = this.rate;
    return a - Math.log(b) + logGamma(a) + (1 - a) * digamma(a);
  }
}

// ---- Binomial Distribution ----

class BinomialDist implements DiscreteDistribution {
  constructor(
    private n: number,
    private p_param: number
  ) {
    if (!Number.isInteger(n) || n < 0)
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    if (p_param < 0 || p_param > 1)
      throw new InvalidParameterError("p must be in [0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0 || k > this.n) return 0;
    // Degenerate p handled directly: k·log(0) = 0·(-Inf) = NaN otherwise.
    if (this.p_param === 0) return k === 0 ? 1 : 0;
    if (this.p_param === 1) return k === this.n ? 1 : 0;
    const lnCoeff = logGamma(this.n + 1) - logGamma(k + 1) - logGamma(this.n - k + 1);
    return Math.exp(
      lnCoeff + k * Math.log(this.p_param) + (this.n - k) * Math.log(1 - this.p_param)
    );
  }

  cdf(x: number): number {
    const k = Math.floor(x);
    if (k < 0) return 0;
    if (k >= this.n) return 1;
    // Use incomplete beta: CDF_binom(k; n, p) = I_{1-p}(n-k, k+1)
    return regularizedIncompleteBeta(this.n - k, k + 1, 1 - this.p_param);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    // Binary search
    let lo = 0,
      hi = this.n;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (this.cdf(mid) < p) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      let count = 0;
      for (let i = 0; i < this.n; i++) {
        if (__random() < this.p_param) count++;
      }
      return count;
    });
  }

  mean(): number {
    return this.n * this.p_param;
  }
  variance(): number {
    return this.n * this.p_param * (1 - this.p_param);
  }
}

// ---- Poisson Distribution ----

class PoissonDist implements DiscreteDistribution {
  constructor(private mu: number) {
    if (mu <= 0) throw new InvalidParameterError("mu must be > 0", "mu", mu);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    return Math.exp(k * Math.log(this.mu) - this.mu - logGamma(k + 1));
  }

  cdf(x: number): number {
    const k = Math.floor(x);
    if (k < 0) return 0;
    // Sum PMFs
    let sum = 0;
    for (let i = 0; i <= k; i++) {
      sum += this.pmf(i);
      if (sum >= 1) return 1;
    }
    return sum;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    let k = 0;
    let cdf = 0;
    while (cdf < p && k < 10000) {
      cdf += this.pmf(k);
      if (cdf >= p) return k;
      k++;
    }
    return k;
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => samplePoisson(this.mu));
  }

  mean(): number {
    return this.mu;
  }
  variance(): number {
    return this.mu;
  }
}

/**
 * Sample a Poisson deviate. Knuth's product method is used for small lambda,
 * but exp(-lambda) underflows to 0 for lambda ≳ 745 (the loop then terminates
 * on the product underflowing, silently capping samples at ~700). For
 * lambda ≥ 30 a transformed-rejection method (Ahrens & Dieter) is used, which
 * is exact across the full range.
 */
function samplePoisson(lambda: number): number {
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= __random();
    } while (p > L);
    return k - 1;
  }
  const c = 0.767 - 3.36 / lambda;
  const beta = Math.PI / Math.sqrt(3 * lambda);
  const alpha = beta * lambda;
  const k = Math.log(c) - lambda - Math.log(beta);
  for (;;) {
    const u = __random();
    if (u === 0 || u === 1) continue;
    const x = (alpha - Math.log((1 - u) / u)) / beta;
    const n = Math.floor(x + 0.5);
    if (n < 0 || !Number.isFinite(n)) continue;
    const v = __random();
    const y = alpha - beta * x;
    const lhs = y + Math.log(v / (1 + Math.exp(y)) ** 2);
    const rhs = k + n * Math.log(lambda) - logGamma(n + 1);
    if (lhs <= rhs) return n;
  }
}

// ---- Factory Functions ----

/**
 * Normal (Gaussian) distribution.
 *
 * @param loc - Mean (default 0)
 * @param scale - Standard deviation (default 1)
 */
export function norm(loc = 0, scale = 1): ContinuousDistribution {
  return new NormalDist(loc, scale);
}

/**
 * Student's t distribution.
 *
 * @param df - Degrees of freedom
 */
export function t(df: number): ContinuousDistribution {
  return new TDist(df);
}

/**
 * Chi-squared distribution.
 *
 * @param df - Degrees of freedom
 */
export function chi2(df: number): ContinuousDistribution {
  return new Chi2Dist(df);
}

/**
 * F distribution.
 *
 * @param dfn - Numerator degrees of freedom
 * @param dfd - Denominator degrees of freedom
 */
export function f(dfn: number, dfd: number): ContinuousDistribution {
  return new FDist(dfn, dfd);
}

/**
 * Continuous uniform distribution.
 *
 * @param low - Lower bound (default 0)
 * @param high - Upper bound (default 1)
 */
export function uniform(low = 0, high = 1): ContinuousDistribution {
  return new UniformDist(low, high);
}

/**
 * Exponential distribution.
 *
 * @param rate - Rate parameter λ (default 1). Mean = 1/λ
 */
export function expon(rate = 1): ContinuousDistribution {
  return new ExponentialDist(rate);
}

/**
 * Beta distribution.
 *
 * @param a - Alpha shape parameter
 * @param b - Beta shape parameter
 */
export function beta(a: number, b: number): ContinuousDistribution {
  return new BetaDist(a, b);
}

/**
 * Gamma distribution.
 *
 * @param shape - Shape parameter (alpha)
 * @param rate - Rate parameter (beta = 1/scale). Default 1.
 */
export function gamma(shape: number, rate = 1): ContinuousDistribution {
  return new GammaDist(shape, rate);
}

/**
 * Binomial distribution.
 *
 * @param n - Number of trials
 * @param p - Probability of success per trial
 */
export function binom(n: number, p: number): DiscreteDistribution {
  return new BinomialDist(n, p);
}

/**
 * Poisson distribution.
 *
 * @param mu - Expected number of events (rate parameter λ)
 */
export function poisson(mu: number): DiscreteDistribution {
  return new PoissonDist(mu);
}

// ---- Lognormal Distribution ----

class LognormalDist implements ContinuousDistribution {
  constructor(
    private mu: number = 0,
    private sigma: number = 1
  ) {
    if (sigma <= 0) throw new InvalidParameterError("sigma must be > 0", "sigma", sigma);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const z = (Math.log(x) - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (x * this.sigma * SQRT_2PI);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return normalCdf((Math.log(x) - this.mu) / this.sigma);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return Math.exp(this.mu + this.sigma * normalPpf(p));
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => Math.exp(this.mu + this.sigma * randn()));
  }

  mean(): number {
    return Math.exp(this.mu + (this.sigma * this.sigma) / 2);
  }
  variance(): number {
    const s2 = this.sigma * this.sigma;
    return (Math.exp(s2) - 1) * Math.exp(2 * this.mu + s2);
  }
  entropy(): number {
    return this.mu + 0.5 * Math.log(2 * Math.PI * Math.E * this.sigma * this.sigma);
  }
}

// ---- Weibull Distribution ----

class WeibullDist implements ContinuousDistribution {
  constructor(
    private k: number,
    private lam: number = 1
  ) {
    if (k <= 0) throw new InvalidParameterError("k (shape) must be > 0", "k", k);
    if (lam <= 0) throw new InvalidParameterError("lambda (scale) must be > 0", "lambda", lam);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) return this.k === 1 ? this.k / this.lam : 0;
    const z = x / this.lam;
    return (this.k / this.lam) * z ** (this.k - 1) * Math.exp(-(z ** this.k));
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return 1 - Math.exp(-((x / this.lam) ** this.k));
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return this.lam * (-Math.log(1 - p)) ** (1 / this.k);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from(
      { length: size },
      () => this.lam * (-Math.log(1 - __random())) ** (1 / this.k)
    );
  }

  mean(): number {
    return this.lam * Math.exp(logGamma(1 + 1 / this.k));
  }
  variance(): number {
    const m1 = Math.exp(logGamma(1 + 1 / this.k));
    const m2 = Math.exp(logGamma(1 + 2 / this.k));
    return this.lam * this.lam * (m2 - m1 * m1);
  }
  entropy(): number {
    const euler = 0.5772156649015329;
    return euler * (1 - 1 / this.k) + Math.log(this.lam / this.k) + 1;
  }
}

// ---- Pareto Distribution ----

class ParetoDist implements ContinuousDistribution {
  constructor(
    private alpha: number,
    private xm: number = 1
  ) {
    if (alpha <= 0) throw new InvalidParameterError("alpha must be > 0", "alpha", alpha);
    if (xm <= 0) throw new InvalidParameterError("xm must be > 0", "xm", xm);
  }

  pdf(x: number): number {
    if (x < this.xm) return 0;
    return (this.alpha * this.xm ** this.alpha) / x ** (this.alpha + 1);
  }

  cdf(x: number): number {
    if (x < this.xm) return 0;
    return 1 - (this.xm / x) ** this.alpha;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return this.xm;
    if (p === 1) return Infinity;
    return this.xm / (1 - p) ** (1 / this.alpha);
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      let u = __random();
      while (u === 0) u = __random();
      return this.xm / u ** (1 / this.alpha);
    });
  }

  mean(): number {
    return this.alpha > 1 ? (this.alpha * this.xm) / (this.alpha - 1) : Infinity;
  }
  variance(): number {
    if (this.alpha <= 2) return Infinity;
    return (this.xm * this.xm * this.alpha) / ((this.alpha - 1) ** 2 * (this.alpha - 2));
  }
  entropy(): number {
    return Math.log(this.xm / this.alpha) + 1 + 1 / this.alpha;
  }
}

// ---- Cauchy Distribution ----

class CauchyDist implements ContinuousDistribution {
  constructor(
    private x0: number = 0,
    private gammaParam: number = 1
  ) {
    if (gammaParam <= 0) throw new InvalidParameterError("gamma must be > 0", "gamma", gammaParam);
  }

  pdf(x: number): number {
    const z = (x - this.x0) / this.gammaParam;
    return 1 / (Math.PI * this.gammaParam * (1 + z * z));
  }

  cdf(x: number): number {
    return 0.5 + Math.atan((x - this.x0) / this.gammaParam) / Math.PI;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return this.x0 + this.gammaParam * Math.tan(Math.PI * (p - 0.5));
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from(
      { length: size },
      () => this.x0 + this.gammaParam * Math.tan(Math.PI * (__random() - 0.5))
    );
  }

  mean(): number {
    return NaN;
  } // Cauchy has no defined mean
  variance(): number {
    return NaN;
  } // Cauchy has no defined variance
  entropy(): number {
    return Math.log(4 * Math.PI * this.gammaParam);
  }
}

// ---- Laplace Distribution ----

class LaplaceDist implements ContinuousDistribution {
  constructor(
    private loc: number = 0,
    private scale: number = 1
  ) {
    if (scale <= 0) throw new InvalidParameterError("scale must be > 0", "scale", scale);
  }

  pdf(x: number): number {
    return Math.exp(-Math.abs(x - this.loc) / this.scale) / (2 * this.scale);
  }

  cdf(x: number): number {
    const z = (x - this.loc) / this.scale;
    return z <= 0 ? 0.5 * Math.exp(z) : 1 - 0.5 * Math.exp(-z);
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return p <= 0.5
      ? this.loc + this.scale * Math.log(2 * p)
      : this.loc - this.scale * Math.log(2 * (1 - p));
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      const u = __random() - 0.5;
      return this.loc - this.scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
    });
  }

  mean(): number {
    return this.loc;
  }
  variance(): number {
    return 2 * this.scale * this.scale;
  }
  entropy(): number {
    return 1 + Math.log(2 * this.scale);
  }
}

// ---- Factory Functions (continued) ----

/**
 * Lognormal distribution.
 *
 * @param mu - Mean of the underlying normal distribution (default 0)
 * @param sigma - Std dev of the underlying normal distribution (default 1)
 */
export function lognorm(mu = 0, sigma = 1): ContinuousDistribution {
  return new LognormalDist(mu, sigma);
}

/**
 * Weibull distribution.
 *
 * @param k - Shape parameter
 * @param lam - Scale parameter (default 1)
 */
export function weibull(k: number, lam = 1): ContinuousDistribution {
  return new WeibullDist(k, lam);
}

/**
 * Pareto distribution.
 *
 * @param alpha - Shape parameter
 * @param xm - Scale parameter (minimum value, default 1)
 */
export function pareto(alpha: number, xm = 1): ContinuousDistribution {
  return new ParetoDist(alpha, xm);
}

/**
 * Cauchy distribution.
 *
 * @param x0 - Location parameter (default 0)
 * @param gammaParam - Scale parameter (default 1)
 */
export function cauchy(x0 = 0, gammaParam = 1): ContinuousDistribution {
  return new CauchyDist(x0, gammaParam);
}

/**
 * Laplace distribution.
 *
 * @param loc - Location parameter (default 0)
 * @param scale - Scale parameter (default 1)
 */
export function laplace(loc = 0, scale = 1): ContinuousDistribution {
  return new LaplaceDist(loc, scale);
}

// ---- Geometric Distribution ----

class GeometricDist implements DiscreteDistribution {
  constructor(private p_param: number) {
    if (p_param <= 0 || p_param > 1)
      throw new InvalidParameterError("p must be in (0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 1) return 0;
    return this.p_param * (1 - this.p_param) ** (k - 1);
  }

  cdf(x: number): number {
    const k = Math.floor(x);
    if (k < 1) return 0;
    return 1 - (1 - this.p_param) ** k;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    if (p === 0) return 1;
    if (p === 1) return Infinity;
    return Math.ceil(Math.log(1 - p) / Math.log(1 - this.p_param));
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () =>
      Math.ceil(Math.log(1 - __random()) / Math.log(1 - this.p_param))
    );
  }

  mean(): number {
    return 1 / this.p_param;
  }
  variance(): number {
    return (1 - this.p_param) / (this.p_param * this.p_param);
  }
}

// ---- Negative Binomial Distribution ----

class NegativeBinomialDist implements DiscreteDistribution {
  constructor(
    private r: number,
    private p_param: number
  ) {
    if (r <= 0 || !Number.isInteger(r))
      throw new InvalidParameterError("r must be a positive integer", "r", r);
    if (p_param <= 0 || p_param > 1)
      throw new InvalidParameterError("p must be in (0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    const lnCoeff = logGamma(k + this.r) - logGamma(k + 1) - logGamma(this.r);
    return Math.exp(lnCoeff + this.r * Math.log(this.p_param) + k * Math.log(1 - this.p_param));
  }

  cdf(x: number): number {
    const k = Math.floor(x);
    if (k < 0) return 0;
    let sum = 0;
    for (let i = 0; i <= k; i++) {
      sum += this.pmf(i);
      if (sum >= 1) return 1;
    }
    return sum;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    let k = 0;
    let cdf = 0;
    while (cdf < p && k < 10000) {
      cdf += this.pmf(k);
      if (cdf >= p) return k;
      k++;
    }
    return k;
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      let failures = 0;
      let successes = 0;
      while (successes < this.r) {
        if (__random() < this.p_param) successes++;
        else failures++;
      }
      return failures;
    });
  }

  mean(): number {
    return (this.r * (1 - this.p_param)) / this.p_param;
  }
  variance(): number {
    return (this.r * (1 - this.p_param)) / (this.p_param * this.p_param);
  }
}

// ---- Hypergeometric Distribution ----

class HypergeometricDist implements DiscreteDistribution {
  constructor(
    private N: number,
    private K: number,
    private nn: number
  ) {
    if (!Number.isInteger(N) || N < 0)
      throw new InvalidParameterError("N must be a non-negative integer", "N", N);
    if (!Number.isInteger(K) || K < 0 || K > N)
      throw new InvalidParameterError("K must be in [0, N]", "K", K);
    if (!Number.isInteger(nn) || nn < 0 || nn > N)
      throw new InvalidParameterError("n must be in [0, N]", "n", nn);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k)) return 0;
    const lo = Math.max(0, this.nn + this.K - this.N);
    const hi = Math.min(this.nn, this.K);
    if (k < lo || k > hi) return 0;
    const lnNum =
      logGamma(this.K + 1) -
      logGamma(k + 1) -
      logGamma(this.K - k + 1) +
      logGamma(this.N - this.K + 1) -
      logGamma(this.nn - k + 1) -
      logGamma(this.N - this.K - this.nn + k + 1);
    const lnDen = logGamma(this.N + 1) - logGamma(this.nn + 1) - logGamma(this.N - this.nn + 1);
    return Math.exp(lnNum - lnDen);
  }

  cdf(x: number): number {
    const k = Math.floor(x);
    const lo = Math.max(0, this.nn + this.K - this.N);
    if (k < lo) return 0;
    const hi = Math.min(this.nn, this.K);
    if (k >= hi) return 1;
    let sum = 0;
    for (let i = lo; i <= k; i++) {
      sum += this.pmf(i);
      if (sum >= 1) return 1;
    }
    return sum;
  }

  ppf(p: number): number {
    if (p < 0 || p > 1) throw new InvalidParameterError("p must be in [0,1]", "p", p);
    const lo = Math.max(0, this.nn + this.K - this.N);
    const hi = Math.min(this.nn, this.K);
    let cdf = 0;
    for (let k = lo; k <= hi; k++) {
      cdf += this.pmf(k);
      if (cdf >= p) return k;
    }
    return hi;
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  rvs(size = 1): number[] {
    return Array.from({ length: size }, () => {
      // Fisher-Yates sampling: draw nn from N, count how many of first K
      let successes = 0;
      let remaining = this.N;
      let good = this.K;
      for (let i = 0; i < this.nn; i++) {
        if (__random() < good / remaining) {
          successes++;
          good--;
        }
        remaining--;
      }
      return successes;
    });
  }

  mean(): number {
    return (this.nn * this.K) / this.N;
  }
  variance(): number {
    const { N, K, nn } = this;
    return (nn * K * (N - K) * (N - nn)) / (N * N * (N - 1));
  }
}

// ---- Factory Functions (continued) ----

/**
 * Geometric distribution (number of trials until first success).
 *
 * P(X = k) = p * (1-p)^(k-1) for k = 1, 2, ...
 *
 * @param p - Probability of success per trial, in (0, 1]
 */
export function geom(p: number): DiscreteDistribution {
  return new GeometricDist(p);
}

/**
 * Negative binomial distribution (number of failures before r successes).
 *
 * @param r - Number of successes required (positive integer)
 * @param p - Probability of success per trial, in (0, 1]
 */
export function nbinom(r: number, p: number): DiscreteDistribution {
  return new NegativeBinomialDist(r, p);
}

/**
 * Hypergeometric distribution.
 *
 * Models drawing n items from a population of N containing K successes,
 * without replacement.
 *
 * @param N - Population size
 * @param K - Number of success states in population
 * @param n - Number of draws
 */
export function hypergeom(N: number, K: number, n: number): DiscreteDistribution {
  return new HypergeometricDist(N, K, n);
}
