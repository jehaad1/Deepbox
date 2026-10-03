/**
 * Tests for all v1.0.0 audit features:
 * - PolynomialFeatures, Binarizer, FunctionTransformer
 * - TimeSeriesSplit, RepeatedKFold, RepeatedStratifiedKFold
 * - Probability distributions (norm, t, chi2, f, uniform, expon, beta, gamma, binom, poisson)
 * - GridSearchCV, RandomizedSearchCV
 * - FeatureUnion
 */

import { describe, expect, it } from "vitest";
import { FeatureUnion, GridSearchCV, RandomizedSearchCV, Ridge } from "../src/ml";
import { tensor } from "../src/ndarray";
import {
  Binarizer,
  FunctionTransformer,
  PolynomialFeatures,
  RepeatedKFold,
  RepeatedStratifiedKFold,
  TimeSeriesSplit,
} from "../src/preprocess";
import {
  beta,
  binom,
  chi2,
  expon,
  f as fDist,
  gamma,
  norm,
  poisson,
  t as tDist,
  uniform,
} from "../src/stats";

// ============================================================
// PolynomialFeatures
// ============================================================

describe("PolynomialFeatures", () => {
  it("generates degree-2 features with bias", () => {
    const poly = new PolynomialFeatures({ degree: 2 });
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = poly.fitTransform(X);
    // Columns: [1, x1, x2, x1^2, x1*x2, x2^2] = 6 columns
    expect(result.shape[0]).toBe(2);
    expect(result.shape[1]).toBe(6);

    // Row 0: [1, 1, 2, 1, 2, 4]
    const row0 = [];
    for (let j = 0; j < 6; j++) row0.push(Number(result.data[result.offset + j]));
    expect(row0).toEqual([1, 1, 2, 1, 2, 4]);
  });

  it("generates degree-2 features without bias", () => {
    const poly = new PolynomialFeatures({ degree: 2, includeBias: false });
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = poly.fitTransform(X);
    // Columns: [x1, x2, x1^2, x1*x2, x2^2] = 5 columns
    expect(result.shape[1]).toBe(5);
  });

  it("generates interaction-only features", () => {
    const poly = new PolynomialFeatures({
      degree: 2,
      interactionOnly: true,
      includeBias: false,
    });
    const X = tensor([[1, 2, 3]]);
    const result = poly.fitTransform(X);
    // Columns: [x1, x2, x3, x1*x2, x1*x3, x2*x3] = 6 columns
    expect(result.shape[1]).toBe(6);
  });

  it("throws on unfitted transform", () => {
    const poly = new PolynomialFeatures();
    const X = tensor([[1, 2]]);
    expect(() => poly.transform(X)).toThrow("fitted");
  });

  it("throws on feature mismatch", () => {
    const poly = new PolynomialFeatures();
    poly.fit(tensor([[1, 2]]));
    expect(() => poly.transform(tensor([[1, 2, 3]]))).toThrow("Expected 2");
  });

  it("returns correct nOutputFeatures", () => {
    const poly = new PolynomialFeatures({ degree: 2 });
    poly.fit(tensor([[1, 2]]));
    expect(poly.nOutputFeatures).toBe(6);
  });

  it("getParams returns config", () => {
    const poly = new PolynomialFeatures({ degree: 3, interactionOnly: true });
    const params = poly.getParams();
    expect(params.degree).toBe(3);
    expect(params.interactionOnly).toBe(true);
  });
});

// ============================================================
// Binarizer
// ============================================================

describe("Binarizer", () => {
  it("binarizes with default threshold 0", () => {
    const bin = new Binarizer();
    const X = tensor([
      [-1, 0, 1],
      [2, -0.5, 0.5],
    ]);
    const result = bin.fitTransform(X);
    expect(result.shape).toEqual([2, 3]);
    const row0 = [];
    for (let j = 0; j < 3; j++) row0.push(Number(result.data[result.offset + j]));
    expect(row0).toEqual([0, 0, 1]);
  });

  it("binarizes with custom threshold", () => {
    const bin = new Binarizer({ threshold: 1.5 });
    const X = tensor([[1, 2, 3]]);
    const result = bin.fitTransform(X);
    const row0 = [];
    for (let j = 0; j < 3; j++) row0.push(Number(result.data[result.offset + j]));
    expect(row0).toEqual([0, 1, 1]);
  });

  it("throws on unfitted transform", () => {
    const bin = new Binarizer();
    expect(() => bin.transform(tensor([[1]]))).toThrow("fitted");
  });
});

// ============================================================
// FunctionTransformer
// ============================================================

describe("FunctionTransformer", () => {
  it("applies identity when no func provided", () => {
    const ft = new FunctionTransformer();
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = ft.fitTransform(X);
    expect(result).toBe(X); // Identity returns same tensor
  });

  it("applies custom function", () => {
    const ft = new FunctionTransformer({
      func: (X) => {
        const data: number[] = [];
        for (let i = 0; i < X.size; i++) {
          data.push(Number(X.data[X.offset + i]) * 2);
        }
        return tensor(data).reshape(X.shape);
      },
    });
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = ft.fitTransform(X);
    expect(Number(result.data[result.offset])).toBe(2);
    expect(Number(result.data[result.offset + 1])).toBe(4);
  });

  it("applies inverse transform", () => {
    const ft = new FunctionTransformer({
      func: (X) => X,
      inverseFunc: (X) => {
        const data: number[] = [];
        for (let i = 0; i < X.size; i++) {
          data.push(Number(X.data[X.offset + i]) + 100);
        }
        return tensor(data).reshape(X.shape);
      },
    });
    ft.fit(tensor([[1]]));
    const inv = ft.inverseTransform(tensor([[5]]));
    expect(Number(inv.data[inv.offset])).toBe(105);
  });

  it("throws on unfitted transform", () => {
    const ft = new FunctionTransformer({ func: (X) => X });
    expect(() => ft.transform(tensor([[1]]))).toThrow("fitted");
  });
});

// ============================================================
// TimeSeriesSplit
// ============================================================

describe("TimeSeriesSplit", () => {
  it("produces correct number of splits", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 3 });
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const splits = tscv.split(X);
    expect(splits.length).toBe(3);
  });

  it("train indices precede test indices (temporal ordering)", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 3 });
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const splits = tscv.split(X);
    for (const split of splits) {
      const maxTrain = Math.max(...split.trainIndex);
      const minTest = Math.min(...split.testIndex);
      expect(maxTrain).toBeLessThan(minTest);
    }
  });

  it("train size grows with each split", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 3 });
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const splits = tscv.split(X);
    for (let i = 1; i < splits.length; i++) {
      expect(splits[i]!.trainIndex.length).toBeGreaterThanOrEqual(splits[i - 1]!.trainIndex.length);
    }
  });

  it("respects maxTrainSize", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 3, maxTrainSize: 2 });
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const splits = tscv.split(X);
    for (const split of splits) {
      expect(split.trainIndex.length).toBeLessThanOrEqual(2);
    }
  });

  it("respects gap", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 2, gap: 1 });
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const splits = tscv.split(X);
    for (const split of splits) {
      if (split.trainIndex.length > 0 && split.testIndex.length > 0) {
        const maxTrain = Math.max(...split.trainIndex);
        const minTest = Math.min(...split.testIndex);
        expect(minTest - maxTrain).toBeGreaterThan(1);
      }
    }
  });

  it("getNSplits returns correct value", () => {
    const tscv = new TimeSeriesSplit({ nSplits: 4 });
    expect(tscv.getNSplits()).toBe(4);
  });
});

// ============================================================
// RepeatedKFold
// ============================================================

describe("RepeatedKFold", () => {
  it("produces nSplits * nRepeats total splits", () => {
    const rkf = new RepeatedKFold({ nSplits: 3, nRepeats: 2, randomState: 42 });
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
      [11, 12],
    ]);
    const splits = rkf.split(X);
    expect(splits.length).toBe(6);
  });

  it("getNSplits returns total splits", () => {
    const rkf = new RepeatedKFold({ nSplits: 5, nRepeats: 3 });
    expect(rkf.getNSplits()).toBe(15);
  });

  it("each fold covers all samples", () => {
    const rkf = new RepeatedKFold({ nSplits: 3, nRepeats: 2, randomState: 0 });
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
      [11, 12],
    ]);
    const splits = rkf.split(X);

    // For each repeat (group of nSplits), all indices should be covered
    for (let r = 0; r < 2; r++) {
      const allTest = new Set<number>();
      for (let f = 0; f < 3; f++) {
        const split = splits[r * 3 + f]!;
        for (const idx of split.testIndex) allTest.add(idx);
      }
      expect(allTest.size).toBe(6);
    }
  });
});

// ============================================================
// RepeatedStratifiedKFold
// ============================================================

describe("RepeatedStratifiedKFold", () => {
  it("produces nSplits * nRepeats total splits", () => {
    const rskf = new RepeatedStratifiedKFold({
      nSplits: 2,
      nRepeats: 3,
      randomState: 42,
    });
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const splits = rskf.split(X, y);
    expect(splits.length).toBe(6);
  });

  it("getNSplits returns total splits", () => {
    const rskf = new RepeatedStratifiedKFold({ nSplits: 5, nRepeats: 2 });
    expect(rskf.getNSplits()).toBe(10);
  });
});

// ============================================================
// Probability Distributions
// ============================================================

describe("Probability Distributions", () => {
  describe("Normal (norm)", () => {
    const d = norm(0, 1);

    it("pdf at 0 is ~0.3989", () => {
      expect(d.pdf(0)).toBeCloseTo(0.3989, 3);
    });

    it("cdf at 0 is 0.5", () => {
      expect(d.cdf(0)).toBeCloseTo(0.5, 3);
    });

    it("ppf at 0.5 is ~0", () => {
      expect(d.ppf(0.5)).toBeCloseTo(0, 1);
    });

    it("sf(x) = 1 - cdf(x)", () => {
      expect(d.sf(1)).toBeCloseTo(1 - d.cdf(1), 6);
    });

    it("mean is 0", () => expect(d.mean()).toBe(0));
    it("variance is 1", () => expect(d.variance()).toBe(1));

    it("rvs produces correct count", () => {
      expect(d.rvs(100).length).toBe(100);
    });

    it("entropy is correct", () => {
      expect(d.entropy()).toBeCloseTo(0.5 * Math.log(2 * Math.PI * Math.E), 4);
    });
  });

  describe("Student's t", () => {
    const d = tDist(10);

    it("pdf at 0 is symmetric peak", () => {
      expect(d.pdf(0)).toBeGreaterThan(0.3);
    });

    it("cdf at 0 is 0.5", () => {
      expect(d.cdf(0)).toBeCloseTo(0.5, 3);
    });

    it("mean is 0 for df > 1", () => {
      expect(d.mean()).toBe(0);
    });

    it("variance is df/(df-2) for df > 2", () => {
      expect(d.variance()).toBeCloseTo(10 / 8, 4);
    });

    it("rvs works", () => {
      expect(d.rvs(50).length).toBe(50);
    });
  });

  describe("Chi-squared", () => {
    const d = chi2(5);

    it("cdf at 0 is 0", () => {
      expect(d.cdf(0)).toBe(0);
    });

    it("mean is df", () => {
      expect(d.mean()).toBe(5);
    });

    it("variance is 2*df", () => {
      expect(d.variance()).toBe(10);
    });

    it("pdf at negative is 0", () => {
      expect(d.pdf(-1)).toBe(0);
    });

    it("rvs works", () => {
      expect(d.rvs(20).length).toBe(20);
    });
  });

  describe("F distribution", () => {
    const d = fDist(5, 10);

    it("cdf at 0 is 0", () => {
      expect(d.cdf(0)).toBe(0);
    });

    it("mean is dfd/(dfd-2) for dfd > 2", () => {
      expect(d.mean()).toBeCloseTo(10 / 8, 4);
    });

    it("pdf at negative is 0", () => {
      expect(d.pdf(-1)).toBe(0);
    });

    it("rvs works", () => {
      expect(d.rvs(20).length).toBe(20);
    });
  });

  describe("Uniform", () => {
    const d = uniform(0, 1);

    it("pdf inside range is 1", () => {
      expect(d.pdf(0.5)).toBe(1);
    });

    it("pdf outside range is 0", () => {
      expect(d.pdf(2)).toBe(0);
    });

    it("cdf at 0.5 is 0.5", () => {
      expect(d.cdf(0.5)).toBeCloseTo(0.5, 6);
    });

    it("ppf at 0.25 is 0.25", () => {
      expect(d.ppf(0.25)).toBeCloseTo(0.25, 6);
    });

    it("mean is 0.5", () => expect(d.mean()).toBe(0.5));
    it("variance is 1/12", () => expect(d.variance()).toBeCloseTo(1 / 12, 6));
    it("entropy is 0", () => expect(d.entropy()).toBeCloseTo(0, 6));
  });

  describe("Exponential", () => {
    const d = expon(2);

    it("pdf at 0 is rate", () => {
      expect(d.pdf(0)).toBeCloseTo(2, 6);
    });

    it("cdf at 0 is 0", () => {
      expect(d.cdf(0)).toBe(0);
    });

    it("mean is 1/rate", () => {
      expect(d.mean()).toBeCloseTo(0.5, 6);
    });

    it("ppf inverts cdf", () => {
      const x = 0.7;
      expect(d.ppf(d.cdf(x))).toBeCloseTo(x, 4);
    });

    it("rvs works", () => {
      const samples = d.rvs(1000);
      expect(samples.length).toBe(1000);
      expect(samples.every((s) => s >= 0)).toBe(true);
    });
  });

  describe("Beta", () => {
    const d = beta(2, 5);

    it("pdf at 0 and 1 is 0", () => {
      expect(d.pdf(0)).toBe(0);
      expect(d.pdf(1)).toBe(0);
    });

    it("cdf at 0 is 0, at 1 is 1", () => {
      expect(d.cdf(0)).toBe(0);
      expect(d.cdf(1)).toBe(1);
    });

    it("mean is a/(a+b)", () => {
      expect(d.mean()).toBeCloseTo(2 / 7, 6);
    });

    it("rvs produces values in [0,1]", () => {
      const samples = d.rvs(100);
      expect(samples.every((s) => s >= 0 && s <= 1)).toBe(true);
    });
  });

  describe("Gamma", () => {
    const d = gamma(2, 1);

    it("pdf at 0 is 0", () => {
      expect(d.pdf(0)).toBe(0);
    });

    it("mean is shape/rate", () => {
      expect(d.mean()).toBeCloseTo(2, 6);
    });

    it("variance is shape/rate^2", () => {
      expect(d.variance()).toBeCloseTo(2, 6);
    });

    it("rvs works", () => {
      const samples = d.rvs(100);
      expect(samples.length).toBe(100);
      expect(samples.every((s) => s >= 0)).toBe(true);
    });
  });

  describe("Binomial", () => {
    const d = binom(10, 0.5);

    it("pmf at mean is highest", () => {
      expect(d.pmf(5)).toBeGreaterThan(d.pmf(0));
      expect(d.pmf(5)).toBeGreaterThan(d.pmf(10));
    });

    it("cdf at n is 1", () => {
      expect(d.cdf(10)).toBeCloseTo(1, 6);
    });

    it("mean is n*p", () => {
      expect(d.mean()).toBe(5);
    });

    it("variance is n*p*(1-p)", () => {
      expect(d.variance()).toBe(2.5);
    });

    it("rvs produces integers in [0, n]", () => {
      const samples = d.rvs(100);
      expect(samples.every((s) => Number.isInteger(s) && s >= 0 && s <= 10)).toBe(true);
    });
  });

  describe("Poisson", () => {
    const d = poisson(3);

    it("pmf sums to ~1", () => {
      let sum = 0;
      for (let k = 0; k < 30; k++) sum += d.pmf(k);
      expect(sum).toBeCloseTo(1, 4);
    });

    it("mean equals mu", () => {
      expect(d.mean()).toBe(3);
    });

    it("variance equals mu", () => {
      expect(d.variance()).toBe(3);
    });

    it("rvs produces non-negative integers", () => {
      const samples = d.rvs(100);
      expect(samples.every((s) => Number.isInteger(s) && s >= 0)).toBe(true);
    });
  });
});

// ============================================================
// GridSearchCV
// ============================================================

describe("GridSearchCV", () => {
  it("searches over parameter grid and finds best", () => {
    // Create a simple regression dataset
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

    const gs = new GridSearchCV(new Ridge({ alpha: 1 }), { alpha: [0.01, 0.1, 1, 10] }, { cv: 2 });
    gs.fit(X, y);

    expect(gs.bestParams).toBeDefined();
    expect(gs.bestScore).toBeGreaterThan(-Infinity);
    expect(gs.cvResults.length).toBe(4);
    expect(gs.bestEstimator).toBeDefined();
  });

  it("predict works after fit", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

    const gs = new GridSearchCV(new Ridge({ alpha: 1 }), { alpha: [0.1, 1] }, { cv: 2 });
    gs.fit(X, y);
    const pred = gs.predict(tensor([[5, 6]]));
    expect(pred.shape[0]).toBe(1);
  });

  it("throws on predict before fit", () => {
    const gs = new GridSearchCV(new Ridge(), { alpha: [1] });
    expect(() => gs.predict(tensor([[1, 2]]))).toThrow("fitted");
  });
});

// ============================================================
// RandomizedSearchCV
// ============================================================

describe("RandomizedSearchCV", () => {
  it("samples nIter combinations", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

    const rs = new RandomizedSearchCV(
      new Ridge({ alpha: 1 }),
      { alpha: [0.001, 0.01, 0.1, 1, 10, 100] },
      { nIter: 3, cv: 2, randomState: 42 }
    );
    rs.fit(X, y);

    expect(rs.cvResults.length).toBe(3);
    expect(rs.bestScore).toBeGreaterThan(-Infinity);
  });
});

// ============================================================
// FeatureUnion
// ============================================================

describe("FeatureUnion", () => {
  it("concatenates features from multiple transformers", () => {
    const union = new FeatureUnion([
      ["poly", new PolynomialFeatures({ degree: 1, includeBias: false })],
      ["bin", new Binarizer({ threshold: 1.5 })],
    ]);

    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = union.fitTransform(X);

    // poly degree=1 noBias: [x1, x2] = 2 cols
    // binarizer threshold=1.5: [0/1, 0/1] = 2 cols
    // total = 4 cols
    expect(result.shape[0]).toBe(2);
    expect(result.shape[1]).toBe(4);
  });

  it("throws on duplicate names", () => {
    expect(
      () =>
        new FeatureUnion([
          ["a", new Binarizer()],
          ["a", new Binarizer()],
        ])
    ).toThrow("Duplicate");
  });

  it("throws on empty transformers", () => {
    expect(() => new FeatureUnion([])).toThrow("at least one");
  });

  it("throws on unfitted transform", () => {
    const union = new FeatureUnion([["bin", new Binarizer()]]);
    expect(() => union.transform(tensor([[1]]))).toThrow("fitted");
  });

  it("transformerNames returns names", () => {
    const union = new FeatureUnion([
      ["alpha", new Binarizer()],
      ["beta", new Binarizer()],
    ]);
    expect(union.transformerNames).toEqual(["alpha", "beta"]);
  });
});
