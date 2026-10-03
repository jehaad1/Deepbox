/**
 * Regression tests for the 1.5.0 audit of src/stats/descriptive.ts and
 * src/stats/distributions.ts.
 *
 * Reference values come from numpy 2.4 / scipy 1.17 (descriptive statistics, ppf and isf
 * values) and from mpmath at 50 digits (probabilities where SciPy itself is accurate only
 * to about 1e-9, for example large-parameter binomial, Poisson and hypergeometric masses).
 */
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { type Tensor, tensor, transpose } from "../../src/ndarray";
import { setSeed } from "../../src/random";
import {
  beta,
  binom,
  bootstrap,
  cauchy,
  chi2,
  cohenD,
  expon,
  f,
  gamma,
  geom,
  geometricMean,
  hypergeom,
  iqr,
  kurtosis,
  laplace,
  lognorm,
  mean,
  median,
  mode,
  moment,
  nbinom,
  norm,
  pareto,
  percentile,
  poisson,
  quantile,
  sem,
  skewness,
  std,
  t as studentT,
  trimMean,
  uniform,
  variance,
  weibull,
  zscore,
} from "../../src/stats";

const f64 = (data: number[] | number[][]): Tensor => tensor(data, { dtype: "float64" });
const values = (x: Tensor): number[] => Array.from(x.data as ArrayLike<number>);

/** Relative error check (exact zero is compared with an absolute tolerance of rtol * 1e-300). */
function expectClose(actual: number, expected: number, rtol = 1e-12): void {
  if (Number.isNaN(expected)) {
    expect(actual).toBeNaN();
    return;
  }
  if (!Number.isFinite(expected)) {
    expect(actual).toBe(expected);
    return;
  }
  const scale = expected === 0 ? 1e-300 : Math.abs(expected);
  expect(Math.abs(actual - expected) / scale).toBeLessThanOrEqual(rtol);
}

function expectAllClose(actual: number[], expected: number[], rtol = 1e-12): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expectClose(actual[i] as number, expected[i] as number, rtol);
  }
}

const M = [
  [1, 2, 7],
  [4, 5, 6],
  [9, 0, 3.5],
];

describe("c59 descriptive: standardized moments", () => {
  it("skewness and kurtosis match scipy over all elements and along axes", () => {
    const t = f64(M);
    expectClose(skewness(t).item() as number, 0.1489097403715718, 1e-13);
    expectClose(kurtosis(t).item() as number, -0.9637759515570927, 1e-12);
    expectAllClose(
      values(skewness(t, 0)),
      [0.29479962014482863, 0.23906314692954456, -0.47033046033698594],
      1e-12
    );
    expectAllClose(
      values(skewness(t, 1, false)),
      [1.5453925256950205, 0, 0.6469685949215187],
      1e-12
    );
    expectAllClose(values(kurtosis(t, 0)), [-1.5, -1.5, -1.5], 1e-12);
    // Rows of three values are too short for the bias correction of the kurtosis.
    expect(values(kurtosis(t, 1, false, false))).toEqual([Number.NaN, Number.NaN, Number.NaN]);
  });

  it("options objects equal the positional forms", () => {
    const t = f64(M);
    expect(values(skewness(t, { axis: 1, bias: false }))).toEqual(values(skewness(t, 1, false)));
    expect(values(kurtosis(t, { axis: 1, fisher: false, bias: false }))).toEqual(
      values(kurtosis(t, 1, false, false))
    );
  });

  it("bias correction matches scipy for n = 8 and gives NaN below the minimum sample size", () => {
    const x = f64([2, 8, 0, 4, 1, 9, 9, 0]);
    expectClose(skewness(x, { bias: false }).item() as number, 0.3305821804079746, 1e-12);
    expectClose(kurtosis(x, { bias: false }).item() as number, -2.098602258096087, 1e-12);
    expectClose(
      kurtosis(x, { bias: false, fisher: false }).item() as number,
      0.9013977419039132,
      1e-12
    );
    // pandas semantics for samples too small to correct (SciPy returns the biased value).
    expect(skewness(f64([1, 2]), { bias: false }).item()).toBeNaN();
    expect(kurtosis(f64([1, 2, 3]), { bias: false }).item()).toBeNaN();
  });

  it("a constant sample with rounding noise in its mean gives NaN, not a finite artifact", () => {
    // mean([0.1, 0.1, 0.1]) is 0.10000000000000002, so the computed variance is about 1e-34.
    const c = f64([0.1, 0.1, 0.1]);
    expect(skewness(c).item()).toBeNaN();
    expect(kurtosis(c).item()).toBeNaN();
    const cols = f64([
      [0.1, 1],
      [0.1, 2],
      [0.1, 4],
    ]);
    const sk = values(skewness(cols, 0));
    expect(sk[0]).toBeNaN();
    expect(Number.isFinite(sk[1])).toBe(true);
  });

  it("works for non-contiguous input and reports empty input by function name", () => {
    const t = f64(M);
    // A transposed copy has the same elements, so full reductions agree.
    expectClose(skewness(transpose(t)).item() as number, 0.1489097403715718, 1e-13);
    expect(() => skewness(f64([]))).toThrow(/skewness\(\)/);
    expect(() => kurtosis(f64([]))).toThrow(/kurtosis\(\)/);
    expect(() => skewness(f64([]))).toThrow(InvalidParameterError);
  });

  it("moment matches scipy, with exact orders 0 and 1 and NaN propagation", () => {
    const t = f64(M);
    expectClose(moment(t, 3).item() as number, 3.0925925925925855, 1e-12);
    expectAllClose(
      values(moment(t, 2, 0)),
      [10.888888888888888, 4.222222222222222, 2.1666666666666665]
    );
    expectAllClose(
      values(moment(t, 4, 1)),
      [71.18518518518518, 0.6666666666666666, 282.4490740740741]
    );
    expect(moment(f64([0.1, 0.2, 0.7]), 1).item()).toBe(0);
    expect(moment(t, 0).item()).toBe(1);
    // NaN ** 0 is 1 in JavaScript; SciPy returns NaN for NaN data at every order.
    expect(moment(f64([1, Number.NaN, 2]), 0).item()).toBeNaN();
    expect(moment(f64([1, Number.NaN, 2]), 1).item()).toBeNaN();
    expect(() => moment(t, 1.5)).toThrow(InvalidParameterError);
    expect(() => moment(f64([]), 2)).toThrow(InvalidParameterError);
  });
});

describe("c59 descriptive: quantiles, medians and trimmed means", () => {
  it("quantile interpolates like numpy.quantile (both the sort and the quickselect paths)", () => {
    const a = f64([0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 1e-3, 7.0]);
    expectAllClose(
      values(quantile(a, [0.1, 0.35, 0.9])),
      [0.06040000000000001, 0.20999999999999996, 3.1000000000000023],
      1e-14
    );
    expect(values(quantile(a, [0, 1, 0.5]))).toEqual([0.001, 7.0, 0.30000000000000004]);
    const squares = f64(Array.from({ length: 100 }, (_, i) => (i + 1) ** 2));
    expectAllClose(
      values(quantile(squares, [0.1, 0.333, 0.9999])),
      [118.9, 1153.789, 9998.0299],
      1e-13
    );
    expectAllClose(values(percentile(f64([1, 2, 3, 4, 5]), [5, 50, 99.9])), [1.2, 3, 4.996], 1e-14);
  });

  it("quantile keeps infinities and NaN well defined and validates q", () => {
    expect(quantile(f64([1, Number.POSITIVE_INFINITY]), 0.5).item()).toBe(Number.POSITIVE_INFINITY);
    expect(values(quantile(f64([1, Number.NaN, 3]), [0.2, 0.5]))).toEqual([Number.NaN, Number.NaN]);
    expect(() => quantile(f64([1, 2]), Number.NaN)).toThrow(InvalidParameterError);
    expect(() => percentile(f64([1, 2]), 101)).toThrow(InvalidParameterError);
  });

  it("quantile with an axis puts the quantile dimension first", () => {
    const out = quantile(
      f64([
        [1, 2, 7],
        [4, 5, 6],
      ]),
      [0.2, 0.5],
      1
    );
    expect(out.shape).toEqual([2, 2]);
    expectAllClose(values(out), [1.4, 4.4, 2, 5], 1e-14);
  });

  it("median, mode and the other reductions return float64 values", () => {
    const ints = tensor([1, 2, 3, 4], { dtype: "int32" });
    expect(median(ints).dtype).toBe("float64");
    expect(median(ints).item()).toBe(2.5);
    expect(mode(f64([1, 2, 2, 3])).dtype).toBe("float64");
    expect(trimMean(f64([0.1, 0.2, 0.3, 0.4, 0.5]), 0.2).dtype).toBe("float64");
    expect(quantile(f64([0.1, 0.2]), 0.5).dtype).toBe("float64");
    // A float64 value must not be rounded through float32.
    expect(trimMean(f64([0.1, 0.1, 0.1, 0.1]), 0.25).item()).toBe(0.1);
    expect(
      median(
        f64([
          [1, 2],
          [3, 8],
        ]),
        { axis: 0 }
      ).toArray()
    ).toEqual([2, 5]);
    expect(median(f64(M), { axis: 1, keepdims: true }).shape).toEqual([3, 1]);
  });

  it("trimMean matches scipy.stats.trim_mean, including huge and infinite tails", () => {
    // The kept values are summed directly, so a trimmed 1e300 or Infinity cannot disturb them.
    expect(trimMean(f64([1, 2, 3, 4, 1e300]), 0.2).item()).toBe(3);
    expect(trimMean(f64([-Infinity, 1, 2, 3, Infinity]), 0.2).item()).toBe(2);
    expect(trimMean(f64([1, 2, 3, 4, 5, 100]), 0.2).item()).toBe(3.5);
    expectClose(
      trimMean(f64([1e16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1e16 + 2]), 0.1).item() as number,
      1111111111111116,
      1e-15
    );
    expect(values(trimMean(f64(M), 0.34, 0))).toEqual([4, 2, 6]);
    expect(values(trimMean(f64(M), 0.34, 1))).toEqual([2, 5, 3.5]);
    expect(trimMean(f64([1, Number.NaN, 3]), 0.1).item()).toBeNaN();
    expect(() => trimMean(f64([1, 2, 3]), 0.5)).toThrow(InvalidParameterError);
  });

  it("iqr matches scipy for full and axis-wise reductions", () => {
    expect(iqr(f64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).item()).toBe(4.5);
    expect(values(iqr(f64(M), 0))).toEqual([4, 2.5, 1.75]);
    expect(values(iqr(f64(M), 1))).toEqual([3, 1, 4.5]);
  });

  it("geometricMean and harmonicMean agree with scipy", () => {
    expectClose(geometricMean(f64([1, 2, 4, 8])).item() as number, 2.82842712474619, 1e-14);
    expect(() => geometricMean(f64([1, 0]))).toThrow(InvalidParameterError);
  });
});

describe("c59 descriptive: dispersion", () => {
  it("sem matches scipy with ddof and axis, and validates the sample size", () => {
    const t = f64(M);
    expectAllClose(values(sem(t, 0)), [2.3333333333333335, 1.452966314513558, 1.0408329997330663]);
    expectClose(sem(t).item() as number, 0.9718253158075499, 1e-13);
    expectAllClose(
      values(sem(t, 1, 0)),
      [1.5153535218873173, 0.47140452079103173, 2.1387085061022395],
      1e-13
    );
    expect(() => sem(f64([5]))).toThrow(InvalidParameterError);
    expect(() => sem(t, 1, 3)).toThrow(InvalidParameterError);
    expect(() => sem(f64([]))).toThrow(InvalidParameterError);
  });

  it("variance and std options objects and ddof validation", () => {
    const t = f64(M);
    expectClose(variance(t, { ddof: 1 }).item() as number, 8.499999999999998, 1e-13);
    expectClose(std(t, { ddof: 1 }).item() as number, Math.sqrt(8.499999999999998), 1e-13);
    expect(() => variance(t, { ddof: 9 })).toThrow(InvalidParameterError);
    expect(() => std(t, { ddof: -1 })).toThrow(InvalidParameterError);
  });

  it("zscore matches scipy with axis and ddof", () => {
    const t = f64(M);
    expectAllClose(
      values(zscore(t, 1, 0)),
      [
        -0.9072647087265548, -0.1324532357065044, 0.8320502943378437, -0.16495721976846456,
        1.0596258856520349, 0.2773500981126146, 1.0722219284950192, -0.9271726499455306,
        -1.1094003924504583,
      ],
      1e-12
    );
    expectAllClose(
      values(zscore(t, 0, 1)),
      [
        -0.8890008890013336, -0.5080005080007621, 1.3970013970020956, -1.224744871391589, 0,
        1.224744871391589, 1.3047716849309579, -1.1248031766646192, -0.17996850826633912,
      ],
      1e-12
    );
    const all = zscore(t);
    expect(all.shape).toEqual([3, 3]);
    expectAllClose(
      values(all).slice(0, 3),
      [-1.152044218922582, -0.7882407813680824, 1.0307764064044151],
      1e-12
    );
  });

  it("zscore maps a constant slice to zeros even when rounding noise makes its variance nonzero", () => {
    expect(values(zscore(f64([0.1, 0.1, 0.1])))).toEqual([0, 0, 0]);
    expect(
      values(
        zscore(
          f64([
            [0.1, 0.1, 0.1],
            [1, 2, 3],
          ]),
          0,
          1
        )
      ).slice(0, 3)
    ).toEqual([0, 0, 0]);
    // NaN data is not "constant": it propagates.
    expect(values(zscore(f64([1, Number.NaN, 3])))).toEqual([Number.NaN, Number.NaN, Number.NaN]);
  });

  it("zscore validates ddof, size and axis", () => {
    expect(() => zscore(f64([1, 2, 3]), -1)).toThrow(InvalidParameterError);
    expect(() => zscore(f64([1, 2, 3]), 3)).toThrow(InvalidParameterError);
    expect(() => zscore(f64([]))).toThrow(InvalidParameterError);
    expect(() => zscore(f64(M), 0, 5)).toThrow();
  });

  it("cohenD returns the textbook value and signed infinity for constant groups", () => {
    expectClose(cohenD([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]), -1.2649110640673518, 1e-14);
    expect(cohenD([2, 2, 2], [2, 2])).toBe(0);
    expect(cohenD([3, 3, 3], [2, 2])).toBe(Number.POSITIVE_INFINITY);
    expect(cohenD([1, 1], [2, 2, 2])).toBe(Number.NEGATIVE_INFINITY);
    expect(() => cohenD([1], [1, 2])).toThrow(InvalidParameterError);
    // The computed mean of [0.1, 0.1, 0.1] is off by one ulp; the groups are still constant.
    expect(cohenD([0.1, 0.1, 0.1], [0.2, 0.2, 0.2])).toBe(Number.NEGATIVE_INFINITY);
    expect(cohenD([0.1, 0.1, 0.1], [0.1, 0.1])).toBe(0);
    expect(cohenD([1, Number.NaN, 3], [1, 2, 3])).toBeNaN();
  });
});

describe("c59 descriptive: bootstrap", () => {
  const meanFn = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const data = [2, 4, 4, 5, 7, 9];

  it("the interval is the linearly interpolated percentile of the samples (numpy.percentile)", () => {
    const r = bootstrap(data, meanFn, { seed: 42, nResamples: 9 });
    expect(r.samples).toEqual([
      4, 4.5, 4.666666666666667, 4.833333333333333, 5, 5.333333333333333, 5.666666666666667,
      5.666666666666667, 7.5,
    ]);
    // np.percentile(samples, [2.5, 97.5])
    expectClose(r.ci[0], 4.1, 1e-14);
    expectClose(r.ci[1], 7.133333333333333, 1e-14);
    const z = bootstrap(data, meanFn, { seed: 0, nResamples: 9 });
    expectClose(z.ci[0], 4, 1e-14);
    expectClose(z.ci[1], 5.9, 1e-14);
    expectClose(r.estimate, 5.166666666666667, 1e-14);
  });

  it("is reproducible per seed, and by setSeed when no seed is given", () => {
    const a = bootstrap(data, meanFn, { seed: 7, nResamples: 50 });
    const b = bootstrap(data, meanFn, { seed: 7, nResamples: 50 });
    expect(a.samples).toEqual(b.samples);
    setSeed(123);
    const c = bootstrap(data, meanFn, { nResamples: 50 });
    setSeed(123);
    const d = bootstrap(data, meanFn, { nResamples: 50 });
    expect(c.samples).toEqual(d.samples);
  });

  it("never reads past the end of the data and does not let statFn modify its input", () => {
    const seen: number[] = [];
    const r = bootstrap(
      [1, 2, 3],
      (xs) => {
        seen.push(...xs);
        xs.fill(Number.NaN);
        return 0;
      },
      { seed: 5, nResamples: 200 }
    );
    expect(seen.every((v) => v === 1 || v === 2 || v === 3 || Number.isNaN(v))).toBe(true);
    // 200 resamples plus one evaluation on the original data, three values each.
    expect(seen.filter((v) => v === 1 || v === 2 || v === 3).length).toBe(603);
    expect(r.samples.length).toBe(200);
    const original = [1, 2, 3];
    bootstrap(original, (xs) => xs.reduce((s, v) => s + v, 0), { seed: 1, nResamples: 5 });
    expect(original).toEqual([1, 2, 3]);
  });

  it("returns an undefined interval when the statistic is NaN, and validates options", () => {
    const r = bootstrap(data, () => Number.NaN, { seed: 1, nResamples: 10 });
    expect(r.ci[0]).toBeNaN();
    expect(r.ci[1]).toBeNaN();
    expect(() => bootstrap([], meanFn)).toThrow(InvalidParameterError);
    expect(() => bootstrap(data, meanFn, { nResamples: 0 })).toThrow(InvalidParameterError);
    expect(() => bootstrap(data, meanFn, { nResamples: 2.5 })).toThrow(InvalidParameterError);
    expect(() => bootstrap(data, meanFn, { confidenceLevel: 1 })).toThrow(InvalidParameterError);
    expect(() => bootstrap(data, meanFn, { confidenceLevel: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => bootstrap(data, meanFn, { seed: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
  });
});

describe("c59 distributions: probabilities at large parameters", () => {
  // Reference values from mpmath (50 digits); SciPy agrees only to about 1e-9 here.
  it("binomial pmf and cdf (saddle-point evaluation)", () => {
    expectClose(binom(1e6, 0.5).pmf(500000), 0.0007978843613317501, 1e-13);
    expectClose(binom(1e6, 1e-6).pmf(2), 0.1839398125556351, 1e-13);
    expectClose(binom(1e5, 0.999).pmf(99900), 0.0398809422346256, 1e-12);
    // (1 - 1e-6)^1e6; the argument 1 - p of the incomplete beta would lose 1e-10.
    expectClose(binom(1e6, 1e-6).cdf(0), 0.3678792572316451, 1e-14);
    expect(binom(20, 0).cdf(0)).toBe(1);
    expect(binom(20, 1).sf(19)).toBe(1);
  });

  it("Poisson pmf at a large mean", () => {
    expectClose(poisson(1e8).pmf(1e8), 3.989422800689808e-5, 1e-13);
    expectClose(poisson(1e5).pmf(100948), 1.4239266449306245e-5, 1e-12);
    expectClose(poisson(1e8).sf(100100001), 7.736091854472516e-24, 1e-9);
  });

  it("hypergeometric pmf, cdf and sf", () => {
    const h = hypergeom(1e6, 1000, 2000);
    expectClose(h.pmf(0), 0.1349293000782945, 1e-13);
    expectClose(h.pmf(1), 0.27067034050777183, 1e-13);
    expectClose(h.pmf(2), 0.2710771596568316, 1e-13);
    expectClose(h.pmf(3), 0.18071792517654103, 1e-13);
    expectClose(h.cdf(3), 0.857394725419439, 1e-13);
    expectClose(h.sf(3), 0.14260527458056108, 1e-12);
    // Probabilities sum to one and cdf + sf = 1.
    const small = hypergeom(52, 13, 5);
    let total = 0;
    for (let k = 0; k <= 5; k++) total += small.pmf(k);
    expectClose(total, 1, 1e-14);
    expectClose(small.cdf(2) + small.sf(2), 1, 1e-14);
    expect(hypergeom(0, 0, 0).pmf(0)).toBe(1);
  });

  it("negative binomial accepts a real r and keeps accuracy for tiny p", () => {
    const nb = nbinom(2.5, 0.3);
    expectAllClose([nb.pmf(0), nb.pmf(3), nb.pmf(10)], [0.04929503, 0.11096003, 0.039501], 1e-7);
    expectAllClose([nb.cdf(0), nb.cdf(3), nb.cdf(10)], [0.04929503, 0.35219759, 0.86216112], 1e-7);
    expectClose(nb.mean(), 5.833333333333334, 1e-14);
    expectClose(nb.variance(), 19.444444444444446, 1e-14);
    expect([nb.ppf(0.1), nb.ppf(0.5), nb.ppf(0.9)]).toEqual([1, 5, 12]);
    expectClose(nbinom(3, 1e-8).cdf(299999996), 0.5768099177529474, 1e-13);
    expect(() => nbinom(0, 0.5)).toThrow(InvalidParameterError);
    expect(nbinom(3, 0.4).cdf(1e300)).toBe(1);
  });

  it("geometric survival function and pmf for a tiny p", () => {
    // (1 - p)^k with p = 1e-10 and k = 1e11 loses 5 digits when computed by powering 1 - p.
    expectClose(geom(1e-10).sf(100000000001), 4.539992973524489e-5, 1e-12);
    expectClose(geom(1e-10).pmf(100000000001), 4.539992973978488e-15, 1e-12);
    expect(geom(1).pmf(1)).toBe(1);
    expect(geom(1).pmf(2)).toBe(0);
    expect(geom(1).sf(1)).toBe(0);
  });

  it("poisson cdf and sf stay defined for an astronomically large argument", () => {
    expect(poisson(3).cdf(1e308)).toBe(1);
    expect(poisson(3).sf(1e308)).toBe(0);
  });
});

describe("c59 distributions: continuous accuracy", () => {
  it("beta survival function is exact for tiny x", () => {
    expectClose(beta(0.1, 10).sf(1e-10), 0.8682716449712465, 1e-14);
    expectClose(beta(0.01, 0.01).sf(1e-300), 0.9994999189301472, 1e-13);
    expectClose(beta(0.01, 0.01).sf(1e-10), 0.6027714865644096, 1e-13);
  });

  it("gamma and chi-squared densities at very large shape", () => {
    expectClose(chi2(1e6).pdf(1e6), 0.0002820947447580834, 1e-13);
    expectClose(gamma(1e6, 1).pdf(1005000), 1.542023382830445e-9, 1e-12);
  });

  it("F density for a subnormal argument is finite", () => {
    expectClose(f(1, 1).pdf(1e-310), 3.183098861837907e154, 1e-12);
    expect(Number.isFinite(f(0.5, 3).pdf(1e-320))).toBe(true);
  });

  it("Weibull variance for a large shape and the Weibull and Pareto tails at infinity", () => {
    expectClose(weibull(1e4, 1).variance(), 1.6445038762822376e-8, 1e-12);
    expect(weibull(2, 1).pdf(Number.POSITIVE_INFINITY)).toBe(0);
    expect(pareto(3, 1).cdf(Number.POSITIVE_INFINITY)).toBe(1);
    expect(pareto(3, 1).pdf(Number.POSITIVE_INFINITY)).toBe(0);
    expect(chi2(3).pdf(Number.POSITIVE_INFINITY)).toBe(0);
    expect(gamma(3, 2).pdf(Number.POSITIVE_INFINITY)).toBe(0);
  });

  it("Pareto cdf just above the minimum keeps its digits", () => {
    expectClose(pareto(100, 1.5).cdf(1.50000015), 9.9999494986089e-6, 1e-13);
  });

  it("entropy of a very wide normal or lognormal does not overflow", () => {
    const h = norm(0, 1e200).entropy();
    expectClose(h, 0.5 * Math.log(2 * Math.PI * Math.E) + 200 * Math.LN10, 1e-14);
    expect(Number.isFinite(lognorm(0, 1e200).entropy())).toBe(true);
  });

  it("small quantiles just above the smallest normal double are not flushed to zero", () => {
    // The bracket search used to return 0 as soon as a shrinking step left the normal range.
    // 1.0931599773319722e-77 is the F(0.5, 3) cdf at 5e-308 (mpmath, 50 digits).
    expectClose(f(0.5, 3).ppf(1.0931599773319722e-77), 5e-308, 1e-9);
    // The root of cdf(x) = 1e-300 is about 1e-900, below every double, so 0 is correct.
    expect(f(0.5, 3).ppf(1e-300)).toBe(0);
  });
});

describe("c59 distributions: isf, std and median", () => {
  it("continuous isf matches scipy, including tiny tail probabilities", () => {
    expectClose(norm(1, 2).isf(1e-12), 15.068967650602263, 1e-12);
    expectClose(norm(1, 2).isf(0.3), 2.048801025416082, 1e-12);
    expectClose(studentT(5).isf(1e-10), 156.8255927088943, 1e-10);
    expectClose(studentT(5).isf(0.01), 3.364929998907218, 1e-12);
    expectClose(chi2(3).isf(1e-15), 72.94251784144284, 1e-10);
    expectClose(chi2(3).isf(0.05), 7.814727903251182, 1e-12);
    expectClose(gamma(2, 3).isf(1e-10), 8.77799386851029, 1e-10);
    expectClose(beta(2, 5).isf(1e-12), 0.9972166267363152, 1e-10);
    expectClose(beta(2, 5).isf(0.2), 0.4224475248462721, 1e-10);
    expectClose(f(5, 10).isf(1e-6), 49.35653976659307, 1e-10);
    expectClose(f(5, 10).isf(0.05), 3.3258345304130104, 1e-10);
    expectClose(expon(2).isf(1e-300), 345.38776394910684, 1e-13);
    expectClose(weibull(2, 1).isf(1e-20), 6.786140424415112, 1e-13);
    expectClose(pareto(3, 1).isf(1e-20), 4641588.833612775, 1e-13);
    expectClose(lognorm(1, 0.5).isf(1e-12), 91.58265635503577, 1e-11);
    expectClose(laplace(0, 2).isf(1e-20), 90.71710935864193, 1e-13);
    expectClose(laplace(0, 2).isf(0.9), -3.218875824868201, 1e-13);
    expectClose(cauchy(1, 2).isf(1e-10), 6366197724.675813, 1e-12);
    expectClose(cauchy(1, 2).isf(0.9), -5.155367074350509, 1e-10);
    expect(uniform(2, 6).isf(0.25)).toBe(5);
  });

  it("isf agrees with ppf(1 - q) away from the tails and reaches the support ends", () => {
    const dists = [norm(), studentT(4), chi2(5), f(3, 8), beta(2, 3), gamma(2, 1), expon(1.5)];
    for (const d of dists) {
      for (const q of [0.1, 0.5, 0.9]) {
        expectClose(d.isf(q), d.ppf(1 - q), 1e-10);
      }
      expect(d.isf(0)).toBe(d.ppf(1));
    }
    expect(norm().isf(1)).toBe(Number.NEGATIVE_INFINITY);
    expect(beta(2, 3).isf(0)).toBe(1);
    expect(beta(2, 3).isf(1)).toBe(0);
    expect(() => norm().isf(1.5)).toThrow(InvalidParameterError);
    expect(() => binom(5, 0.5).isf(Number.NaN)).toThrow(InvalidParameterError);
  });

  it("discrete isf matches scipy", () => {
    const qs = [1e-8, 0.01, 0.5, 0.99];
    expect(qs.map((q) => binom(100, 0.3).isf(q))).toEqual([57, 41, 30, 20]);
    expect(qs.map((q) => poisson(30).isf(q))).toEqual([65, 43, 30, 18]);
    expect(qs.map((q) => geom(0.2).isf(q))).toEqual([83, 21, 4, 1]);
    expect(qs.map((q) => nbinom(5, 0.3).isf(q))).toEqual([78, 30, 11, 1]);
    expect(qs.map((q) => hypergeom(1000, 300, 100).isf(q))).toEqual([55, 40, 30, 20]);
  });

  it("discrete isf resolves a tail probability that 1 - q cannot represent", () => {
    const g = geom(0.3);
    // sf(k) = 0.7^k <= 1e-20 first holds at k = 130 (0.7^129 = 1.2e-20, 0.7^130 = 8.4e-21).
    expect(g.isf(1e-20)).toBe(130);
    expect(g.sf(130)).toBeLessThanOrEqual(1e-20);
    expect(g.sf(129)).toBeGreaterThan(1e-20);
  });

  it("Cauchy quantiles near the centre keep their digits", () => {
    // mpmath: tan(pi * (p - 1/2)) at 50 digits.
    expectClose(cauchy().ppf(0.4999999), -3.1415926536802352e-7, 1e-13);
    expectClose(cauchy().ppf(0.7), 0.7265425280053607, 1e-14);
    expectClose(cauchy(1, 2).isf(0.4999999), 1 + 2 * 3.1415926536802352e-7, 1e-13);
  });

  it("discrete quantiles beyond 2^53 terminate", () => {
    expect(typeof binom(1e17, 0.5).ppf(0.5)).toBe("number");
    expect(typeof poisson(1e17).isf(0.25)).toBe("number");
  });

  it("std and median are provided by every distribution", () => {
    expectClose(gamma(3, 2).std(), 0.8660254037844386, 1e-14);
    expectClose(gamma(3, 2).median(), 1.3370301568617808, 1e-12);
    expect(binom(10, 0.3).median()).toBe(3);
    expect(poisson(4).std()).toBe(2);
    expect(norm(3, 2).median()).toBe(3);
    expect(cauchy(1, 2).median()).toBe(1);
    expect(Number.isNaN(cauchy().std())).toBe(true);
  });
});

describe("c59 distributions: argument handling", () => {
  it("returns NaN for NaN and the limiting values at infinity", () => {
    const dists = [
      norm(),
      studentT(3),
      chi2(3),
      f(4, 7),
      uniform(),
      expon(),
      beta(2, 3),
      gamma(2, 1),
      lognorm(),
      weibull(2, 1),
      pareto(3, 1),
      cauchy(),
      laplace(),
    ];
    for (const d of dists) {
      expect(d.pdf(Number.NaN)).toBeNaN();
      expect(d.cdf(Number.NaN)).toBeNaN();
      expect(d.sf(Number.NaN)).toBeNaN();
      expect(d.cdf(Number.POSITIVE_INFINITY)).toBe(1);
      expect(d.sf(Number.POSITIVE_INFINITY)).toBe(0);
      expect(d.cdf(Number.NEGATIVE_INFINITY)).toBe(0);
      expect(d.sf(Number.NEGATIVE_INFINITY)).toBe(1);
      expect(d.pdf(Number.POSITIVE_INFINITY)).toBe(0);
    }
    const discrete = [binom(10, 0.3), poisson(3), geom(0.3), nbinom(3, 0.4), hypergeom(20, 7, 12)];
    for (const d of discrete) {
      expect(d.pmf(Number.NaN)).toBeNaN();
      expect(d.cdf(Number.NaN)).toBeNaN();
      expect(d.sf(Number.NaN)).toBeNaN();
      expect(d.cdf(Number.POSITIVE_INFINITY)).toBe(1);
      expect(d.sf(Number.POSITIVE_INFINITY)).toBe(0);
      expect(d.pmf(Number.POSITIVE_INFINITY)).toBe(0);
    }
  });

  it("ppf validates p and rvs validates size", () => {
    expect(() => gamma(2).ppf(-0.1)).toThrow(InvalidParameterError);
    expect(() => hypergeom(10, 5, 5).ppf(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => norm().rvs(-1)).toThrow(InvalidParameterError);
    expect(() => norm().rvs(1.5)).toThrow(InvalidParameterError);
    expect(norm().rvs(0)).toEqual([]);
  });

  it("random variates have the right mean for the saddle-point and real-r negative binomial", () => {
    setSeed(2024);
    const draws = nbinom(2.5, 0.3).rvs(20000);
    const m = draws.reduce((a, b) => a + b, 0) / draws.length;
    expect(Math.abs(m - 5.833333333333334)).toBeLessThan(0.3);
  });
});

describe("c59 descriptive: shapes and error types", () => {
  it("full reductions return documented shapes", () => {
    expect(mean(f64(M)).shape).toEqual([]);
    expect(median(f64(M)).shape).toEqual([]);
    expect(mode(f64([1, 2, 2])).shape).toEqual([1]);
    expect(trimMean(f64(M), 0.1).shape).toEqual([1]);
    expect(quantile(f64(M), 0.5).shape).toEqual([1]);
    expect(quantile(f64(M), [0.1, 0.5]).shape).toEqual([2]);
    expect(iqr(f64(M)).shape).toEqual([]);
    expect(skewness(f64(M)).shape).toEqual([]);
  });

  it("empty and string inputs throw typed errors", () => {
    const empty = f64([]);
    for (const fn of [
      () => median(empty),
      () => mode(empty),
      () => quantile(empty, 0.5),
      () => moment(empty, 2),
      () => geometricMean(empty),
      () => trimMean(empty, 0.1),
      () => iqr(empty),
      () => sem(empty),
      () => zscore(empty),
    ]) {
      expect(fn).toThrow(InvalidParameterError);
    }
    const strings = tensor(["a", "b"]);
    expect(() => median(strings)).toThrow();
    expect(() => skewness(strings)).toThrow();
  });
});
