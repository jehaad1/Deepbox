import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { tensor as makeTensor, reshape, Tensor, transpose } from "../../src/ndarray";
import {
  chiSquareCdf,
  chiSquareSf,
  correlationPValue,
  digamma,
  erf,
  erfc,
  fCdf,
  fSf,
  logBeta,
  logGamma,
  normalCdf,
  normalPpf,
  normalSf,
  quickMedianF64,
  quickSelectF64,
  rankData,
  reduceMean,
  reduceVariance,
  regularizedIncompleteBeta,
  studentTCdf,
  studentTSf,
} from "../../src/stats/_internal";
import {
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
} from "../../src/stats/confidence";
import {
  corrcoef,
  cov,
  kendalltau,
  partialcorr,
  pearsonr,
  pointbiserialr,
  spearmanr,
} from "../../src/stats/correlation";

/** The default dtype of `tensor()` is float32; the references below are float64 values. */
function tensor(
  data: Parameters<typeof makeTensor>[0],
  opts: NonNullable<Parameters<typeof makeTensor>[1]> = {}
): Tensor {
  return makeTensor(data, { dtype: "float64", ...opts });
}

/** Asserts |actual - expected| <= rel * |expected|. */
function expectRel(actual: number, expected: number, rel: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const X50 = Array.from({ length: 50 }, (_, i) => i);
const Y50 = [
  0.103675, 1.246485, 2.099131, 2.609053, 4.271607, 5.133912, 5.838914, 7.174335, 8.109372, 9.08824,
  10.008527, 11.164014, 11.779064, 12.951127, 13.855364, 15.179654, 16.011917, 16.912263, 17.765427,
  18.922842, 20.002443, 20.917319, 22.388219, 23.302017, 23.186651, 24.433296, 25.947568, 26.873343,
  28.064093, 29.065197, 30.635352, 30.666394, 31.886718, 33.612831, 34.194011, 35.198919, 35.845798,
  36.505577, 38.050239, 39.032704, 39.631794, 40.795032, 41.978387, 42.716575, 43.970519, 45.028645,
  46.010676, 46.848113, 48.178124, 49.26735,
];

describe("c58 stats/_internal: selection and ranking", () => {
  it("quickMedianF64 stays linear on constant and two-valued input", () => {
    // The old Lomuto partition was O(n^2) on runs of equal values (seconds for 2e5).
    const t0 = Date.now();
    expect(quickMedianF64(new Float64Array(200_000).fill(3))).toBe(3);
    const twoValued = new Float64Array(200_000);
    for (let i = 0; i < twoValued.length; i++) twoValued[i] = i % 2;
    expect(quickMedianF64(twoValued)).toBe(0.5);
    expect(Date.now() - t0).toBeLessThan(1500);
  });

  it("quickSelectF64 partitions around the selected element", () => {
    const rng = (() => {
      let s = 12345;
      return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };
    })();
    for (const n of [1, 2, 3, 10, 101]) {
      const base = new Float64Array(n);
      for (let i = 0; i < n; i++) base[i] = Math.floor(rng() * 7); // many duplicates
      const sorted = Float64Array.from(base).sort();
      for (let k = 0; k < n; k++) {
        const a = Float64Array.from(base);
        expect(quickSelectF64(a, k)).toBe(sorted[k]);
        for (let i = 0; i < k; i++) expect(a[i] as number).toBeLessThanOrEqual(a[k] as number);
        for (let i = k + 1; i < n; i++)
          expect(a[i] as number).toBeGreaterThanOrEqual(a[k] as number);
      }
    }
  });

  it("quickMedianF64 handles empty input and NaN", () => {
    expect(quickMedianF64(new Float64Array(0))).toBeNaN();
    expect(quickMedianF64(Float64Array.of(1, Number.NaN, 3))).toBeNaN();
    expect(quickMedianF64(Float64Array.of(4, 1, 3, 2))).toBe(2.5);
  });

  it("rankData sorts NaN last instead of corrupting the order", () => {
    const { ranks, tieSum } = rankData(Float64Array.of(3, Number.NaN, 1, Number.NaN, 2));
    expect(Array.from(ranks)).toEqual([3, 4, 1, 5, 2]);
    expect(tieSum).toBe(0);
  });

  it("rankData averages tied infinities and reports tie sum", () => {
    const { ranks, tieSum } = rankData(
      Float64Array.of(
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        1
      )
    );
    expect(Array.from(ranks)).toEqual([3.5, 3.5, 1, 2]);
    expect(tieSum).toBe(6);
  });
});

describe("c58 stats/_internal: reductions", () => {
  it("full-array mean uses blocked summation (numpy: 0.1 to 1e-15)", () => {
    const n = 1_000_000;
    const data = new Float64Array(n).fill(0.1);
    const t = Tensor.fromTypedArray({ data, shape: [n], dtype: "float64", device: "cpu" });
    const m = reduceMean(t, undefined, false).toArray() as number;
    // A plain left-to-right sum is off by 1.3e-12 here.
    expect(Math.abs(m - 0.1)).toBeLessThan(1e-14);
  });

  it("variance of a large offset is exact (numpy: 4.000004999991)", () => {
    const n = 1_000_000;
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) data[i] = 1e9 + (i % 7);
    const t = Tensor.fromTypedArray({ data, shape: [n], dtype: "float64", device: "cpu" });
    expectRel(reduceVariance(t, undefined, false, 0).toArray() as number, 4.000004999991, 1e-9);
  });

  it("multi-axis reductions with keepdims match numpy", () => {
    const vals = Array.from({ length: 24 }, (_, i) => i ** 1.5);
    const t = reshape(tensor(vals), [2, 3, 4]);
    const mean = reduceMean(t, [0, 2], true);
    expect(mean.shape).toEqual([1, 3, 1]);
    const expMean = [25.992989889760523, 43.20961818157129, 64.61289334249186];
    const got = Array.from(mean.data as Float64Array);
    for (let i = 0; i < 3; i++) expectRel(got[i] as number, expMean[i] as number, 1e-14);

    const v = reduceVariance(t, [0, 2], false, 1);
    expect(v.shape).toEqual([3]);
    const expVar = [667.8451161037799, 1073.061596117513, 1464.198873042023];
    const gv = Array.from(v.data as Float64Array);
    for (let i = 0; i < 3; i++) expectRel(gv[i] as number, expVar[i] as number, 1e-13);

    const v1 = reduceVariance(t, 1, true, 0);
    expect(v1.shape).toEqual([2, 1, 4]);
    expectRel((v1.data as Float64Array)[0] as number, 87.77348089249864, 1e-13);
    expectRel((v1.data as Float64Array)[7] as number, 454.72655632758534, 1e-13);
  });

  it("reduceVariance rejects NaN and infinite ddof", () => {
    const t = tensor([1, 2, 3, 4]);
    expect(() => reduceVariance(t, undefined, false, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => reduceVariance(t, undefined, false, Number.POSITIVE_INFINITY)).toThrow(
      /non-negative/
    );
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => reduceVariance(m, 0, false, Number.NaN)).toThrow(InvalidParameterError);
  });

  it("reduceMean on a non-contiguous view and int64 agree with the contiguous result", () => {
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(reduceMean(transpose(m), undefined, false).toArray()).toBe(3.5);
    expect(reduceMean(tensor([1, 2, 3, 4], { dtype: "int64" }), undefined, false).toArray()).toBe(
      2.5
    );
  });
});

describe("c58 stats/_internal: special functions (scipy/mpmath references)", () => {
  it("logGamma: negative non-integers use |Γ|, poles are +Infinity", () => {
    expectRel(logGamma(-0.5), 1.2655121234846454, 1e-14); // ln|Γ(-0.5)| = ln(2√π)
    expectRel(logGamma(-1.5), 0.860047015376481, 1e-13);
    expectRel(logGamma(-100.5), -364.9009683094273, 1e-14);
    expect(logGamma(0)).toBe(Number.POSITIVE_INFINITY);
    expect(logGamma(-3)).toBe(Number.POSITIVE_INFINITY);
    expect(logGamma(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
    expect(logGamma(Number.NaN)).toBeNaN();
    expect(logGamma(1)).toBe(0);
    expect(logGamma(2)).toBe(0);
    expectRel(logGamma(3.5), 1.2009736023470743, 1e-14);
  });

  it("digamma: full accuracy, negative arguments and poles", () => {
    expectRel(digamma(1), -0.5772156649015329, 1e-14);
    expectRel(digamma(0.1), -10.423754940411076, 1e-14);
    expectRel(digamma(2), 0.42278433509846713, 1e-14);
    expectRel(digamma(5.5), 1.611093148581751, 1e-14);
    expectRel(digamma(-0.5), 0.03648997397857652, 1e-12);
    expectRel(digamma(-1.5), 0.7031566406452432, 1e-13);
    expect(digamma(-2)).toBeNaN();
    expect(digamma(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(digamma(Number.NEGATIVE_INFINITY)).toBeNaN(); // used to loop forever
    expect(digamma(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
    expect(digamma(Number.NaN)).toBeNaN();
  });

  it("erf / erfc reach double precision (the old fit was ~1e-7)", () => {
    expectRel(erf(1), 0.8427007929497149, 1e-14);
    expectRel(erf(1e-10), 1.1283791670955126e-10, 1e-14);
    expectRel(erf(-2), -0.9953222650189527, 1e-14);
    expectRel(erfc(3), 2.209049699858544e-5, 1e-14);
    expectRel(erfc(10), 2.088487583762545e-45, 1e-14);
    expectRel(erfc(-0.5), 1.5204998778130465, 1e-14);
    expect(erfc(30)).toBe(0);
    expect(erfc(-30)).toBe(2);
    expect(erf(Number.NaN)).toBeNaN();
    expect(erfc(Number.NaN)).toBeNaN();
  });

  it("normal cdf / sf / ppf", () => {
    expectRel(normalCdf(-1), 0.15865525393145707, 1e-14);
    expectRel(normalCdf(-5), 2.866515718791939e-7, 1e-13);
    expectRel(normalCdf(-10), 7.6198530241605e-24, 1e-12); // mpmath
    expectRel(normalSf(30), 4.906713927148187e-198, 1e-12); // mpmath
    expectRel(normalSf(5), 2.866515718791939e-7, 1e-13);
    expect(normalCdf(0)).toBe(0.5);
    expect(normalCdf(-40)).toBe(0);
    expect(normalCdf(Number.NaN)).toBeNaN();
    expectRel(normalPpf(1e-10), -6.361340902404056, 1e-14);
    expectRel(normalPpf(0.9), 1.2815515655446004, 1e-14);
    expectRel(normalPpf(0.999999999999), 7.034486910047836, 1e-14);
    expectRel(normalPpf(0.02), -2.053748910631823, 1e-13);
    expectRel(normalPpf(1e-300), -37.0470962993612, 1e-14);
    expect(normalPpf(0.5)).toBe(0);
    expect(normalPpf(0.3) + normalPpf(0.7)).toBeCloseTo(0, 15);
    expect(normalPpf(1.5)).toBeNaN();
  });

  it("special functions handle infinite arguments", () => {
    expect(normalCdf(Number.POSITIVE_INFINITY)).toBe(1);
    expect(normalCdf(Number.NEGATIVE_INFINITY)).toBe(0);
    expect(normalSf(Number.POSITIVE_INFINITY)).toBe(0);
    expect(normalSf(Number.NEGATIVE_INFINITY)).toBe(1);
    expect(erf(Number.POSITIVE_INFINITY)).toBe(1);
    expect(erf(Number.NEGATIVE_INFINITY)).toBe(-1);
    expect(erfc(Number.POSITIVE_INFINITY)).toBe(0);
    expect(erfc(Number.NEGATIVE_INFINITY)).toBe(2);
    expect(studentTCdf(1e200, 5)).toBe(1);
    expect(studentTCdf(-1e200, 5)).toBe(0);
  });

  it("incomplete beta and t / F cdf stay accurate for large shape parameters", () => {
    // The continued fraction used to stop after 200 iterations.
    expectRel(regularizedIncompleteBeta(5e5, 5e5, 0.5), 0.5, 1e-12);
    expectRel(regularizedIncompleteBeta(1e5, 1e5, 0.5005), 0.6726394155565629, 1e-12);
    expectRel(regularizedIncompleteBeta(1e6, 0.5, 0.999999), 0.15729915515583526, 1e-12);
    expectRel(studentTCdf(1, 1e6), 0.8413446250832108, 1e-13);
    expectRel(studentTCdf(0.5, 1e8), 0.6914624607239108, 1e-13);
    expectRel(fCdf(0.5, 5e4, 3), 0.11162482048641763, 1e-10);
    expect(regularizedIncompleteBeta(2, 3, 0.5)).toBeCloseTo(0.6875, 14);
    expect(regularizedIncompleteBeta(1, 1, Number.NaN)).toBeNaN();
  });

  it("chi-square cdf is correct for large degrees of freedom", () => {
    // The gamma series/fraction used to stop after 200 iterations (0.4995 instead of 0.5019).
    expectRel(chiSquareCdf(1e4, 1e4), 0.5018806340338173, 1e-13);
    expectRel(chiSquareCdf(1e5, 1e5), 0.5005947081047933, 1e-13);
    expectRel(chiSquareCdf(1e6, 1e6), 0.5001880631966055, 1e-12);
    expectRel(chiSquareCdf(3.841, 1), 0.9499863162360433, 1e-14);
  });

  it("survival functions keep relative precision in the tail", () => {
    expectRel(chiSquareSf(60, 3), 5.878230727906919e-13, 1e-13);
    expectRel(chiSquareSf(200, 30), 4.9527335290032224e-27, 1e-12);
    expectRel(studentTSf(40, 7), 7.951089992425182e-10, 1e-13);
    expectRel(studentTSf(3, 5), 0.015049623948731288, 1e-13);
    expectRel(fSf(3, 100, 1000), 3.0844090247166417e-18, 1e-12);
    expectRel(fSf(40, 5, 10), 2.703018795087396e-6, 1e-13);
    expect(chiSquareSf(0, 3)).toBe(1);
    expect(chiSquareSf(Number.POSITIVE_INFINITY, 3)).toBe(0);
    expect(fSf(Number.POSITIVE_INFINITY, 3, 4)).toBe(0);
    expect(fCdf(1e308, 5, 3)).toBe(1);
    expect(() => chiSquareSf(1, 0)).toThrow(InvalidParameterError);
    expect(() => fSf(1, 0, 1)).toThrow(InvalidParameterError);
  });

  it("logBeta does not lose digits when an argument is large", () => {
    expectRel(logBeta(1e6, 0.3), -3.048855067571199, 1e-13); // mpmath
    expectRel(logBeta(1e6, 1e6), -1386300.0033629211, 1e-14);
    expectRel(logBeta(2, 3), -2.4849066497880004, 1e-14);
    expectRel(logBeta(5, 1e9), -100.4382753643841, 1e-14);
  });

  it("correlationPValue is exact for r close to 1 and tiny p-values", () => {
    expectRel(correlationPValue(0.9999999999999862, 4), 2.8428574871902087e-28, 1e-6);
    expect(correlationPValue(1, 5)).toBe(0);
    expect(correlationPValue(-1.0000000000000002, 5)).toBe(0);
    expect(correlationPValue(0, 5)).toBe(1);
    expect(correlationPValue(Number.NaN, 5)).toBeNaN();
    expect(() => correlationPValue(0.5, 0)).toThrow(InvalidParameterError);
  });
});

describe("c58 stats/correlation", () => {
  it("pearsonr p-value does not underflow to 0 (scipy: 4.3179e-85)", () => {
    const [r, p] = pearsonr(tensor(X50), tensor(Y50));
    expectRel(r, 0.9998328861278101, 1e-13);
    expectRel(p, 4.317937294871387e-85, 1e-8);
  });

  it("pearsonr near-perfect correlation keeps a tiny positive p-value (scipy: 2.8e-28)", () => {
    // 1 - (1 - r) is conditioned on the last ulp of r, so only the magnitude is stable.
    const [r, p] = pearsonr(tensor([1, 2, 3, 4, 5, 6]), tensor([1, 2, 3, 4, 5, 6.000001]));
    expectRel(r, 0.9999999999999862, 1e-14);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1e-25);
  });

  it("pearsonr: exact |r| = 1 gives p = 0, never NaN", () => {
    const [r, p] = pearsonr(tensor([1, 2, 3, 4]), tensor([2, 4, 6, 8]));
    expect(r).toBeCloseTo(1, 14);
    expect(p).toBe(0);
  });

  it("pearsonr is scale invariant at extreme magnitudes", () => {
    const x = [1, 2, 4, 3];
    const y = [2, 1, 5, 3.5];
    for (const scale of [1e200, 1e-200]) {
      const [r, p] = pearsonr(tensor(x.map((v) => v * scale)), tensor(y.map((v) => v * scale)));
      expectRel(r, 0.8483677805978154, 1e-13);
      expectRel(p, 0.15163221940218463, 1e-11);
    }
  });

  it("pearsonr detects constant input exactly (0.1 + 0.1 + 0.1 != 3 * 0.1)", () => {
    expect(() => pearsonr(tensor([0.1, 0.1, 0.1]), tensor([1, 2, 3]))).toThrow(/constant/);
    expect(() => pearsonr(tensor([1, 2, 3]), tensor([0.7, 0.7, 0.7]))).toThrow(/constant/);
  });

  it("pearsonr NaN / Infinity input gives NaN statistic and NaN p-value", () => {
    const [r, p] = pearsonr(tensor([1, 2, 3, Number.POSITIVE_INFINITY]), tensor([1, 2, 3, 4]));
    expect(r).toBeNaN();
    expect(p).toBeNaN();
  });

  it("pearsonr on int64 and non-contiguous tensors", () => {
    const x = tensor([1, 2, 3, 4, 5, 6], { dtype: "int64" });
    const y = tensor([2, 1, 4, 3, 6, 5]);
    const [r] = pearsonr(x, y);
    expect(r).toBeCloseTo(0.8285714285714284, 12);
    const m = tensor([
      [1, 10],
      [2, 20],
      [3, 31],
    ]);
    const col0 = reshape(m.slice({ start: 0, end: 3 }, { start: 0, end: 1 }), [3]);
    expect(pearsonr(col0, tensor([5, 6, 8]))[0]).toBeCloseTo(0.9819805060619655, 12);
    // Transposed view (non-contiguous strides)
    const tr = transpose(tensor([[1, 2, 3, 4, 5, 6]]));
    expect(pearsonr(tr, y)[0]).toBeCloseTo(0.8285714285714284, 12);
  });

  it("spearmanr matches scipy with ties", () => {
    const a = tensor([3.1, 1.2, 5.5, 2.2, 2.2, 9.9, 4.0, 0.5]);
    const b = tensor([1, 3, 2, 5, 4, 8, 7, 6]);
    const [rho, p] = spearmanr(a, b);
    expectRel(rho, 0.15569141404872366, 1e-13);
    expectRel(p, 0.7127617079991629, 1e-12);
  });

  it("spearmanr with NaN returns NaN (it used to rank NaN as a value)", () => {
    const [rho, p] = spearmanr(tensor([1, 2, Number.NaN, 4, 5]), tensor([2, 4, 6, 8, 10]));
    expect(rho).toBeNaN();
    expect(p).toBeNaN();
  });

  it("spearmanr treats repeated infinities as ties", () => {
    const [rho, p] = spearmanr(
      tensor([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 1, 2, 5]),
      tensor([1, 2, 3, 4, 0])
    );
    expectRel(rho, -0.5642880936468347, 1e-12);
    expectRel(p, 0.3217233358243003, 1e-10);
  });

  it("spearmanr reports its own name for constant input", () => {
    expect(() => spearmanr(tensor([1, 1, 1]), tensor([1, 2, 3]))).toThrow(/spearmanr\(\)/);
  });

  it("kendalltau uses the exact p-value for small untied samples (scipy 'auto')", () => {
    const [tau5, p5] = kendalltau(tensor([1, 2, 3, 4, 5]), tensor([1, 3, 2, 4, 5]));
    expectRel(tau5, 0.7999999999999999, 1e-14);
    expectRel(p5, 0.08333333333333333, 1e-13); // the normal approximation gives 0.0500

    const a = tensor([3.1, 1.2, 5.5, 2.2, 7.7, 9.9, 4.0, 0.5]);
    const b = tensor([1, 3, 2, 5, 4, 8, 7, 6]);
    const [tau8, p8] = kendalltau(a, b);
    expectRel(tau8, 0.14285714285714285, 1e-13);
    expectRel(p8, 0.7195436507936508, 1e-12);

    const ten = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expectRel(
      kendalltau(ten, tensor([1, 2, 3, 4, 5, 6, 7, 8, 10, 9]))[1],
      5.5114638447971785e-6,
      1e-9
    );
    expectRel(
      kendalltau(ten, tensor([1, 2, 3, 4, 5, 6, 7, 10, 9, 8]))[1],
      0.00011518959435626102,
      1e-9
    );
  });

  it("kendalltau n = 2 is p = 1 (exact), and nearly sorted large samples use the closed form", () => {
    expect(kendalltau(tensor([1, 2]), tensor([3, 5]))[1]).toBe(1);
    const n = 40;
    const xs = Array.from({ length: n }, (_, i) => i);
    const ys = [...xs];
    [ys[n - 2], ys[n - 1]] = [ys[n - 1] as number, ys[n - 2] as number];
    const [tau, p] = kendalltau(tensor(xs), tensor(ys));
    expectRel(tau, 0.9974358974358974, 1e-13);
    expectRel(p, 9.804939513027087e-47, 1e-12); // 2 / 39!
  });

  it("kendalltau with ties still uses the tie-corrected normal approximation", () => {
    const [tau, p] = kendalltau(tensor([1, 2, 2, 3, 4, 5, 5, 6]), tensor([1, 3, 2, 2, 5, 4, 6, 6]));
    expectRel(tau, 0.7692307692307694, 1e-13);
    expectRel(p, 0.010747577580460075, 1e-10);
  });

  it("kendalltau handles repeated infinities and NaN", () => {
    // The old comparator returned NaN for Infinity - Infinity and misordered ties.
    const [tau, p] = kendalltau(
      tensor([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 1, 2]),
      tensor([1, 2, 3, 4])
    );
    expectRel(tau, -0.5477225575051662, 1e-13);
    expectRel(p, 0.2785986718379625, 1e-10);

    const [t2, p2] = kendalltau(tensor([1, Number.NaN, 3]), tensor([1, 2, 3]));
    expect(t2).toBeNaN();
    expect(p2).toBeNaN();
  });

  it("kendalltau with a constant input is NaN", () => {
    const [tau, p] = kendalltau(tensor([2, 2, 2, 2]), tensor([1, 2, 3, 4]));
    expect(tau).toBeNaN();
    expect(p).toBeNaN();
  });

  it("pointbiserialr matches scipy and uses its own name in errors", () => {
    const [r, p] = pointbiserialr(
      tensor([0, 1, 1, 0, 1, 0, 1, 0]),
      tensor([10, 20, 18, 12, 22, 11, 19, 13])
    );
    expectRel(r, 0.9530251207255478, 1e-13);
    expectRel(p, 0.0002500974528352013, 1e-10);
    expect(() => pointbiserialr(tensor([1, 1, 1]), tensor([1, 2, 3]))).toThrow(
      /pointbiserialr\(\)/
    );
  });

  it("partialcorr includes an intercept (offset confounders do not change the answer)", () => {
    const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = tensor([2, 3, 5, 4, 7, 6, 8, 9, 10, 11.5]);
    const z1 = [101, 101, 102, 102, 103, 103, 104, 104, 105, 105];
    const z2 = tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]);
    // Reference: OLS residuals with an intercept column, numpy.
    const [r1, p1] = partialcorr(x, y, tensor(z1));
    expectRel(r1, 0.2662069528248353, 1e-12);
    expectRel(p1, 0.48870133375955593, 1e-10);
    // Shifting the confounder must not matter.
    const [r1b] = partialcorr(x, y, tensor(z1.map((v) => v - 100)));
    expectRel(r1b, r1, 1e-12);

    const [r2, p2] = partialcorr(x, y, [tensor(z1), z2]);
    expectRel(r2, 0.27392909707541174, 1e-12);
    expectRel(p2, 0.5114981247331706, 1e-10);
  });

  it("partialcorr rejects a variable that is a linear function of the confounders", () => {
    const x = tensor([1, 2, 3, 4, 5, 6]);
    const y = tensor([1, 3, 2, 5, 4, 6]);
    expect(() => partialcorr(x, y, tensor([2, 4, 6, 8, 10, 12]))).toThrow(/linear combination/);
    expect(() => partialcorr(tensor([4, 4, 4, 4, 4, 4]), y, tensor([1, 5, 2, 6, 3, 9]))).toThrow(
      /constant/
    );
  });

  it("partialcorr ignores a constant or duplicate confounder", () => {
    const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = tensor([2, 3, 5, 4, 7, 6, 8, 9, 10, 11.5]);
    const z = tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]);
    const base = partialcorr(x, y, z);
    const dup = partialcorr(x, y, [z, tensor([6, 2, 8, 2, 10, 18, 4, 12, 10, 6])]);
    expectRel(dup[0], base[0], 1e-12);
    const withConst = partialcorr(x, y, [z, tensor(new Array<number>(10).fill(7))]);
    expectRel(withConst[0], base[0], 1e-12);
  });

  it("corrcoef(x, y) does not throw for constant input (numpy returns NaN)", () => {
    const c = corrcoef(tensor([1, 1, 1]), tensor([1, 2, 3]));
    const d = Array.from(c.data as Float64Array);
    expect(d[0]).toBeNaN();
    expect(d[1]).toBeNaN();
    expect(d[3]).toBe(1);
  });

  it("corrcoef example from the docs: r = 0.7746", () => {
    const c = corrcoef(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 5]));
    const d = Array.from(c.data as Float64Array);
    expect(d[0]).toBe(1);
    expect(d[1]).toBeCloseTo(0.7745966692414834, 13);
    expect(d[1]).toBe(d[2]);
  });

  it("corrcoef survives extreme magnitudes where numpy returns NaN", () => {
    for (const scale of [1e200, 1e-200]) {
      const m = tensor(
        [
          [1, 2],
          [2, 1],
          [4, 5],
          [3, 3.5],
        ].map((row) => row.map((v) => v * scale))
      );
      const d = Array.from(corrcoef(m).data as Float64Array);
      expectRel(d[1] as number, 0.8483677805978154, 1e-12);
      expect(d[0]).toBe(1);
      expect(d[3]).toBe(1);
    }
  });

  it("corrcoef: constant column or non-finite value poisons only its own row/column", () => {
    const m = tensor([
      [0.1, 1, 5],
      [0.1, 2, 7],
      [0.1, 4, 8],
    ]);
    const d = Array.from(corrcoef(m).data as Float64Array);
    expect(d[0]).toBeNaN(); // constant column, even though its computed mean is off by one ulp
    expect(d[1]).toBeNaN();
    expect(d[4]).toBe(1);
    expectRel(d[5] as number, 0.9285714285714285, 1e-12);

    const withInf = tensor([
      [Number.POSITIVE_INFINITY, 1],
      [2, 2],
      [3, 4],
    ]);
    expect(
      Array.from(corrcoef(withInf).data as Float64Array)
        .slice(0, 2)
        .every(Number.isNaN)
    ).toBe(true);
  });

  it("corrcoef / cov handle non-contiguous input and return float64", () => {
    const m = tensor([
      [1, 2, 9],
      [3, 4, 1],
      [5, 7, 2],
      [8, 9, 4],
    ]);
    // transpose(transpose(m)) is a strided view of the same data.
    const view = transpose(transpose(m));
    expect(corrcoef(view).dtype).toBe("float64");
    // The 3x4 matrix transposed: variables are now the 4 original rows.
    expect(cov(transpose(m)).shape).toEqual([4, 4]);
    // numpy.cov(m, rowvar=False)
    const expected = [
      8.916666666666666, 9.166666666666666, -4.666666666666666, 9.166666666666666,
      9.666666666666666, -5.333333333333333, -4.666666666666666, -5.333333333333333,
      12.666666666666666,
    ];
    const direct = Array.from(cov(m).data as Float64Array);
    for (let i = 0; i < 9; i++) expectRel(direct[i] as number, expected[i] as number, 1e-12);
    const viaView = Array.from(cov(view).data as Float64Array);
    for (let i = 0; i < 9; i++) expectRel(viaView[i] as number, expected[i] as number, 1e-12);
  });

  it("cov rejects non-finite ddof and keeps float64 output for y input", () => {
    expect(() => cov(tensor([1, 2, 3]), undefined, Number.NaN)).toThrow(InvalidParameterError);
    const out = cov(tensor([1, 2, 3, 4]), tensor([2, 4, 5, 9]), 0);
    expect(out.dtype).toBe("float64");
    const d = Array.from(out.data as Float64Array);
    // numpy.cov([1,2,3,4], [2,4,5,9], ddof=0)
    expect(d[0]).toBeCloseTo(1.25, 14);
    expect(d[1]).toBeCloseTo(2.75, 14);
    expect(d[2]).toBe(d[1]);
    expect(d[3]).toBeCloseTo(6.5, 14);
  });
});

describe("c58 stats/confidence", () => {
  it("meanConfidenceInterval matches scipy t.interval", () => {
    const ci = meanConfidenceInterval([2.3, 2.5, 2.1, 2.4, 2.6, 2.2], 0.95);
    expectRel(ci.lower, 2.153668569301968, 1e-12);
    expectRel(ci.upper, 2.5463314306980322, 1e-12);
  });

  it("meanConfidenceInterval is accurate for levels extremely close to 1", () => {
    // ppf(1 - alpha/2) would lose ~6 digits at alpha = 1e-10.
    const ci = meanConfidenceInterval([2.3, 2.5, 2.1, 2.4, 2.6, 2.2], 1 - 1e-10);
    expectRel(ci.lower, -11.409114622540343, 1e-10);
    expectRel(ci.upper, 16.109114622540343, 1e-10);
  });

  it("confidence level and popStd validation rejects NaN", () => {
    const d = [1, 2, 3];
    expect(() => meanConfidenceInterval(d, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => meanConfidenceIntervalZ(d, 1, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => meanConfidenceIntervalZ(d, Number.NaN)).toThrow(/popStd/);
    expect(() => meanConfidenceIntervalZ(d, Number.POSITIVE_INFINITY)).toThrow(/popStd/);
    expect(() => proportionConfidenceInterval(1, 10, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => proportionConfidenceInterval(1, Number.NaN)).toThrow(/total/);
    expect(() => proportionConfidenceInterval(Number.NaN, 10)).toThrow(/successes/);
    expect(() => meanDiffConfidenceInterval(d, d, 2)).toThrow(InvalidParameterError);
  });

  it("meanConfidenceIntervalZ z critical value (norm.isf(0.005) = 2.5758293035489)", () => {
    const ci = meanConfidenceIntervalZ([10, 12, 11, 13, 9], 2, 0.99);
    expectRel(ci.marginOfError, (2.575829303548901 * 2) / Math.sqrt(5), 1e-13);
  });

  it("meanDiffConfidenceInterval uses fractional Welch degrees of freedom (scipy)", () => {
    const a = [5.1, 4.9, 5.6, 5.8, 6.3, 5.0, 5.2];
    const b = [4.2, 4.8, 4.4, 5.0, 9.5, 3.9];
    const ci = meanDiffConfidenceInterval(a, b, 0.95);
    // scipy: df = 5.504745848369152. Flooring it to 5 gave a wider interval.
    expectRel(ci.lower, -2.0789681177554424, 1e-9);
    expectRel(ci.upper, 2.3075395463268737, 1e-9);
  });

  it("meanDiffConfidenceInterval supports a pooled-variance interval", () => {
    const a = [5.1, 4.9, 5.6, 5.8, 6.3, 5.0, 5.2];
    const b = [4.2, 4.8, 4.4, 5.0, 9.5, 3.9];
    const ci = meanDiffConfidenceInterval(a, b, 0.95, true);
    expectRel(ci.lower, -1.6758292461687725, 1e-9);
    expectRel(ci.upper, 1.9044006747402038, 1e-9);
  });

  it("meanDiffConfidenceInterval with two constant samples is a point, not NaN", () => {
    const ci = meanDiffConfidenceInterval([3, 3, 3], [1, 1], 0.9);
    expect(ci.lower).toBe(2);
    expect(ci.upper).toBe(2);
    expect(ci.mean).toBe(2);
    expect(ci.marginOfError).toBe(0);
  });

  it("proportionConfidenceInterval Wald is unchanged and Wilson matches scipy", () => {
    const wald = proportionConfidenceInterval(45, 100, 0.95);
    expectRel(wald.lower, 0.3524930229100606, 1e-12);
    expectRel(wald.upper, 0.5475069770899395, 1e-12);

    const w45 = proportionConfidenceInterval(45, 100, 0.95, "wilson");
    expectRel(w45.lower, 0.35614537979511984, 1e-12);
    expectRel(w45.upper, 0.5475539700255787, 1e-12);
    expect(w45.mean).toBe(0.45);

    const w0 = proportionConfidenceInterval(0, 20, 0.95, "wilson");
    expect(w0.lower).toBe(0);
    expectRel(w0.upper, 0.16112515805281935, 1e-12);

    const w3 = proportionConfidenceInterval(3, 10, 0.95, "wilson");
    expectRel(w3.lower, 0.10779126740630102, 1e-12);
    expectRel(w3.upper, 0.6032218525388546, 1e-12);

    const wAll = proportionConfidenceInterval(20, 20, 0.95, "wilson");
    expectRel(wAll.lower, 0.8388748419471808, 1e-12);
    expect(wAll.upper).toBe(1);
    // The Wald interval collapses at the boundary; Wilson does not.
    expect(proportionConfidenceInterval(0, 20).upper).toBe(0);
  });

  it("proportionConfidenceInterval rejects an unknown method", () => {
    expect(() => proportionConfidenceInterval(1, 10, 0.95, "exact" as unknown as "wald")).toThrow(
      /method/
    );
  });
});

describe("c58 stats/_internal: second-pass accuracy fixes", () => {
  it("logGamma keeps full relative accuracy near its zeros at 1 and 2 (mpmath)", () => {
    // The Lanczos sum alone has ~1e-16 absolute error: 1e-9 relative at 1.0000001.
    expectRel(logGamma(1.0000001), -5.772155829918507e-8, 1e-14);
    expectRel(logGamma(0.99), 0.005854806764709781, 1e-14);
    expectRel(logGamma(1.9999), -4.227520877215346e-5, 1e-14);
    expectRel(logGamma(2.0001), 4.2281658112919945e-5, 1e-14);
    expectRel(logGamma(1.2499), -0.09824908508160847, 1e-14);
    expectRel(logGamma(1.7501), -0.08437636995471517, 1e-14);
    expectRel(logGamma(2.2499), 0.1248144630324305, 1e-14);
    expectRel(logGamma(0.7501), 0.20317237805183128, 1e-14);
  });

  it("normalPpf keeps relative precision for quantiles close to 0 (scipy ndtri)", () => {
    // Phi(z) - p cancelled to 1e-16 absolute, i.e. 2.4e-13 relative at p = 0.4999.
    expectRel(normalPpf(0.4999), -0.0002506628300880075, 1e-14);
    expectRel(normalPpf(0.5001), 0.0002506628300880075, 1e-14);
    expectRel(normalPpf(0.5 - 1e-9), -2.5066283428845328e-9, 1e-14);
  });

  it("studentTCdf accepts df = Infinity as the normal limit", () => {
    expect(studentTCdf(1.5, Number.POSITIVE_INFINITY)).toBe(normalCdf(1.5));
    expect(() => studentTCdf(1, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => studentTCdf(1, 0)).toThrow(InvalidParameterError);
  });

  it("chi-square lower tail with large df is exact (mpmath: 1.431626770125612e-264)", () => {
    // d = (x - s) / s was rounded before log1p(d), costing ~6e-12 relative here.
    expectRel(
      chiSquareCdf(2 * 0.12819162200796222, 2 * 105.71319753447206),
      1.431626770125612e-264,
      1e-13
    );
  });

  it("erfc and normal tails stay near 1e-15 on [1, 1.5] (the old split lost 1e-14)", () => {
    expectRel(erfc(1.48), 0.03634593458573115, 2e-15); // mpmath
    expectRel(normalSf(2.05), 0.02018221540570441, 2e-15);
  });

  it("F cdf with large numerator df and tiny upper-tail complement", () => {
    expectRel(fCdf(0.5, 5e4, 3), 0.11162482048641761, 2e-12);
    expectRel(fSf(0.5, 5e4, 3), 0.8883751795135824, 2e-13);
  });
});

describe("c58 stats/correlation: alternatives, variants and methods", () => {
  const x = [1, 2, 3, 4, 5, 6, 7, 8];
  const yPos = [2, 1, 4, 3, 7, 5, 8, 6];
  const yNeg = [8, 6, 7, 5, 4, 3, 1, 2];

  it("pearsonr and spearmanr support one-sided alternatives (scipy)", () => {
    const [, pg] = pearsonr(tensor(x), tensor(yPos), { alternative: "greater" });
    const [, pl] = pearsonr(tensor(x), tensor(yPos), { alternative: "less" });
    expectRel(pg, 0.00508777006172842, 1e-11);
    expectRel(pl, 0.9949122299382716, 1e-13);
    const [, sg] = spearmanr(tensor(x), tensor(yPos), { alternative: "greater" });
    const [, sl] = spearmanr(tensor(x), tensor(yPos), { alternative: "less" });
    expectRel(sg, 0.005087770061728379, 1e-11);
    expectRel(sl, 0.9949122299382717, 1e-13);

    const [, ng] = pearsonr(tensor(x), tensor(yNeg), { alternative: "greater" });
    const [, nl] = pearsonr(tensor(x), tensor(yNeg), { alternative: "less" });
    expectRel(ng, 0.9998697999878063, 1e-13);
    expectRel(nl, 0.00013020001219362726, 1e-11);
    // The default is still two-sided.
    expectRel(pearsonr(tensor(x), tensor(yPos))[1], 0.01017554012345684, 1e-11);
  });

  it("one-sided p-values at r = 0 and |r| = 1", () => {
    const x6 = tensor([1, 2, 3, 4, 5, 6]);
    expect(pearsonr(x6, tensor([1, 2, 3, 4, 5, 6]), { alternative: "greater" })[1]).toBe(0);
    expect(pearsonr(x6, tensor([1, 2, 3, 4, 5, 6]), { alternative: "less" })[1]).toBe(1);
    expect(pearsonr(x6, tensor([6, 5, 4, 3, 2, 1]), { alternative: "greater" })[1]).toBe(1);
    // r = 0 exactly
    const [r, p] = pearsonr(tensor([1, 2, 3, 4, 5]), tensor([1, -1, 0, -1, 1]), {
      alternative: "less",
    });
    expect(r).toBe(0);
    expect(p).toBe(0.5);
  });

  it("invalid alternative is rejected by every test", () => {
    const a = tensor(x);
    const b = tensor(yPos);
    const bad = { alternative: "both" } as unknown as { alternative: "less" };
    expect(() => pearsonr(a, b, bad)).toThrow(/pearsonr\(\) alternative/);
    expect(() => spearmanr(a, b, bad)).toThrow(/spearmanr\(\) alternative/);
    expect(() => kendalltau(a, b, bad)).toThrow(/kendalltau\(\) alternative/);
    expect(() => pointbiserialr(tensor([0, 1, 0, 1, 1, 0, 1, 0]), b, bad)).toThrow(
      /pointbiserialr\(\) alternative/
    );
  });

  it("pearsonr with two samples returns exactly +-1 and p = 1 (scipy)", () => {
    expect(pearsonr(tensor([1, 2]), tensor([3, 5]))).toEqual([1, 1]);
    expect(pearsonr(tensor([3, 1]), tensor([3, 5]), { alternative: "less" })).toEqual([-1, 1]);
    expect(pointbiserialr(tensor([0, 1]), tensor([3, 5]))).toEqual([1, 1]);
    const [rho, p] = spearmanr(tensor([1, 2]), tensor([3, 5]));
    expect(rho).toBeCloseTo(1, 14);
    expect(p).toBeNaN();
  });

  it("pointbiserialr supports alternative (scipy pearsonr alternative)", () => {
    const [r, p] = pointbiserialr(
      tensor([0, 1, 1, 0, 1, 0, 1, 0]),
      tensor([10, 20, 18, 12, 22, 11, 19, 13]),
      { alternative: "less" }
    );
    expectRel(r, 0.9530251207255478, 1e-13);
    expectRel(p, 0.9998749512735824, 1e-13);
  });

  it("kendalltau exact one-sided p-values", () => {
    const [tau, pTwo] = kendalltau(tensor(x), tensor(yPos));
    expectRel(tau, 0.6428571428571428, 1e-13);
    expectRel(pTwo, 0.03115079365079365, 1e-12);
    expectRel(
      kendalltau(tensor(x), tensor(yPos), { alternative: "greater" })[1],
      0.015575396825396826,
      1e-12
    );
    expectRel(
      kendalltau(tensor(x), tensor(yPos), { alternative: "less" })[1],
      0.9929315476190476,
      1e-13
    );
    expectRel(
      kendalltau(tensor(x), tensor(yNeg), { alternative: "greater" })[1],
      0.9998015873015873,
      1e-13
    );
    expectRel(
      kendalltau(tensor(x), tensor(yNeg), { alternative: "less" })[1],
      0.0008680555555555555,
      1e-12
    );
  });

  it("kendalltau method 'asymptotic' and one-sided normal approximation", () => {
    expectRel(
      kendalltau(tensor(x), tensor(yPos), { method: "asymptotic" })[1],
      0.025952456022374483,
      1e-11
    );
    expectRel(
      kendalltau(tensor(x), tensor(yPos), { method: "asymptotic", alternative: "greater" })[1],
      0.012976228011187241,
      1e-11
    );
    const a = tensor([1, 2, 2, 3, 4, 5, 5, 6, 7, 8]);
    const b = tensor([1, 3, 2, 2, 5, 4, 6, 6, 8, 7]);
    expectRel(kendalltau(a, b, { alternative: "greater" })[1], 0.0007347249324091818, 1e-10);
    expectRel(kendalltau(a, b, { alternative: "less" })[1], 0.9992652750675908, 1e-12);
  });

  it("kendalltau variant 'c' (scipy)", () => {
    const [tauC, pC] = kendalltau(
      tensor([1, 1, 2, 2, 3, 3, 4, 4]),
      tensor([1, 2, 1, 2, 3, 3, 4, 3]),
      { variant: "c" }
    );
    expectRel(tauC, 0.75, 1e-13);
    expectRel(pC, 0.0165203975720026, 1e-10);
    const [tauB] = kendalltau(tensor([1, 1, 2, 2, 3, 3, 4, 4]), tensor([1, 2, 1, 2, 3, 3, 4, 3]));
    expectRel(tauB, 0.7661308776828738, 1e-13);
    expectRel(kendalltau(tensor(x), tensor(yPos), { variant: "c" })[0], 0.6428571428571429, 1e-13);
  });

  it("kendalltau exact method works for larger n without overflow and rejects ties", () => {
    const perm = [
      23, 32, 30, 17, 20, 11, 33, 2, 12, 3, 21, 29, 38, 4, 25, 22, 0, 28, 39, 35, 10, 31, 15, 24,
      34, 27, 13, 8, 9, 18, 6, 1, 26, 37, 16, 14, 36, 7, 5, 19,
    ];
    const idx = Array.from({ length: 40 }, (_, i) => i);
    const opts = { method: "exact" } as const;
    const [tau, p] = kendalltau(tensor(idx), tensor(perm), opts);
    expectRel(tau, -0.09487179487179487, 1e-13);
    expectRel(p, 0.3974648091548658, 1e-11);
    expectRel(
      kendalltau(tensor(idx), tensor(perm), { ...opts, alternative: "greater" })[1],
      0.8076909060914644,
      1e-11
    );
    expectRel(
      kendalltau(tensor(idx), tensor(perm), { ...opts, alternative: "less" })[1],
      0.1987324045774329,
      1e-11
    );
    // method "auto" switches to the normal approximation above n = 33
    expectRel(
      kendalltau(tensor(idx.slice(0, 34)), tensor(perm.slice(0, 34)))[1],
      0.645833696254254,
      1e-10
    );
    expect(() => kendalltau(tensor([1, 2, 2, 3]), tensor([1, 2, 3, 4]), opts)).toThrow(/ties/);
    expect(() => kendalltau(tensor(x), tensor(yPos), { method: "bogus" as "auto" })).toThrow(
      /method/
    );
    expect(() => kendalltau(tensor(x), tensor(yPos), { variant: "d" as "b" })).toThrow(/variant/);
  });
});
