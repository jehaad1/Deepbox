import { describe, expect, it } from "vitest";
import { catchWarnings, filterWarnings, resetWarnings, warn } from "../src/core";
import {
  detCurve,
  meanGammaDeviance,
  meanPoissonDeviance,
  multilabelConfusionMatrix,
} from "../src/metrics";
import { tensor } from "../src/ndarray";
import { f_classif, f_regression, SelectKBest, StratifiedShuffleSplit } from "../src/preprocess";
import { categorical, gumbel_softmax, setSeed } from "../src/random";

// ─── Metrics: meanPoissonDeviance ──────────────────────────────
describe("meanPoissonDeviance", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1, 2, 3, 4]);
    expect(meanPoissonDeviance(yt, yp)).toBeCloseTo(0, 10);
  });

  it("returns positive value for imperfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 3.5]);
    expect(meanPoissonDeviance(yt, yp)).toBeGreaterThan(0);
  });

  it("handles y_true = 0 correctly", () => {
    const yt = tensor([0, 0, 0]);
    const yp = tensor([1, 2, 3]);
    // When y_true=0, deviance_i = 2*y_pred
    const expected = (2 * 1 + 2 * 2 + 2 * 3) / 3;
    expect(meanPoissonDeviance(yt, yp)).toBeCloseTo(expected, 10);
  });

  it("throws for negative y_true", () => {
    const yt = tensor([-1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(() => meanPoissonDeviance(yt, yp)).toThrow();
  });

  it("throws for non-positive y_pred", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([0, 2, 3]);
    expect(() => meanPoissonDeviance(yt, yp)).toThrow();
  });

  it("returns 0 for empty tensors", () => {
    const yt = tensor([]);
    const yp = tensor([]);
    expect(meanPoissonDeviance(yt, yp)).toBe(0);
  });
});

// ─── Metrics: meanGammaDeviance ────────────────────────────────
describe("meanGammaDeviance", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1, 2, 3, 4]);
    expect(meanGammaDeviance(yt, yp)).toBeCloseTo(0, 10);
  });

  it("returns positive value for imperfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 3.5]);
    expect(meanGammaDeviance(yt, yp)).toBeGreaterThan(0);
  });

  it("throws for non-positive y_true", () => {
    const yt = tensor([0, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(() => meanGammaDeviance(yt, yp)).toThrow();
  });

  it("throws for non-positive y_pred", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([0, 2, 3]);
    expect(() => meanGammaDeviance(yt, yp)).toThrow();
  });
});

// ─── Metrics: multilabelConfusionMatrix ────────────────────────
describe("multilabelConfusionMatrix", () => {
  it("computes per-class confusion matrices", () => {
    const yt = tensor([0, 0, 1, 1, 2, 2]);
    const yp = tensor([0, 1, 1, 1, 2, 0]);
    const result = multilabelConfusionMatrix(yt, yp);

    expect(result.length).toBe(3);

    // Class 0: TP=1, FP=1, FN=1, TN=3
    const c0 = result.find((r) => r.label === 0)!;
    expect(c0.tp).toBe(1);
    expect(c0.fp).toBe(1);
    expect(c0.fn).toBe(1);
    expect(c0.tn).toBe(3);

    // Class 1: TP=2, FP=1, FN=0, TN=3
    const c1 = result.find((r) => r.label === 1)!;
    expect(c1.tp).toBe(2);
    expect(c1.fp).toBe(1);
    expect(c1.fn).toBe(0);
    expect(c1.tn).toBe(3);
  });

  it("supports explicit labels parameter", () => {
    const yt = tensor([0, 1, 2]);
    const yp = tensor([0, 1, 2]);
    const result = multilabelConfusionMatrix(yt, yp, [0, 1]);
    expect(result.length).toBe(2);
  });

  it("returns sorted labels", () => {
    const yt = tensor([2, 0, 1]);
    const yp = tensor([2, 0, 1]);
    const result = multilabelConfusionMatrix(yt, yp);
    expect(result.map((r) => r.label)).toEqual([0, 1, 2]);
  });
});

// ─── Metrics: detCurve ─────────────────────────────────────────
describe("detCurve", () => {
  it("returns FPR and FNR at thresholds", () => {
    const yt = tensor([0, 0, 1, 1]);
    const ys = tensor([0.1, 0.4, 0.6, 0.9]);
    const { fpr, fnr, thresholds } = detCurve(yt, ys);

    expect(fpr.length).toBeGreaterThan(0);
    expect(fnr.length).toBe(fpr.length);
    expect(thresholds.length).toBe(fpr.length);
    // At the most permissive threshold, FNR should be 0
    expect(fnr[fnr.length - 1]).toBe(0);
  });

  it("throws for non-binary labels", () => {
    const yt = tensor([0, 1, 2]);
    const ys = tensor([0.1, 0.5, 0.9]);
    expect(() => detCurve(yt, ys)).toThrow();
  });

  it("returns empty for all-same labels", () => {
    const yt = tensor([1, 1, 1]);
    const ys = tensor([0.1, 0.5, 0.9]);
    const { fpr } = detCurve(yt, ys);
    expect(fpr.length).toBe(0);
  });

  it("handles empty input", () => {
    const yt = tensor([]);
    const ys = tensor([]);
    const { fpr } = detCurve(yt, ys);
    expect(fpr.length).toBe(0);
  });
});

// ─── Preprocess: SelectKBest ───────────────────────────────────
describe("SelectKBest", () => {
  it("selects top-k features with f_classif", () => {
    // Feature 0: constant (no discriminative power)
    // Feature 1: perfectly separates classes
    // Feature 2: some noise
    const X = tensor([
      [1, 0, 0.5],
      [1, 0, 0.6],
      [1, 1, 0.4],
      [1, 1, 0.3],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const skb = new SelectKBest({ scoreFunc: f_classif, k: 2 });
    skb.fit(X, y);
    const Xt = skb.transform(X);

    expect(Xt.shape).toEqual([4, 2]);
    // Feature 0 (constant) should be excluded
    const support = skb.getSupport();
    expect(support[0]).toBe(false); // constant feature removed
  });

  it("scores are accessible after fit", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const skb = new SelectKBest({ k: 1 });
    skb.fit(X, y);
    expect(skb.scores.length).toBe(2);
  });

  it("throws if not fitted", () => {
    const skb = new SelectKBest({ k: 1 });
    expect(() => skb.transform(tensor([[1, 2]]))).toThrow();
    expect(() => skb.getSupport()).toThrow();
    expect(() => skb.scores).toThrow();
  });

  it("throws if k exceeds features", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1]);
    const skb = new SelectKBest({ k: 5 });
    expect(() => skb.fit(X, y)).toThrow();
  });

  it("fitTransform works", () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const skb = new SelectKBest({ k: 2 });
    const Xt = skb.fitTransform(X, y);
    expect(Xt.shape).toEqual([4, 2]);
  });
});

describe("f_regression", () => {
  it("scores features by correlation with target", () => {
    // Feature 0: perfect correlation with y
    // Feature 1: no correlation (constant)
    const X = tensor([
      [1, 5],
      [2, 5],
      [3, 5],
      [4, 5],
    ]);
    const y = tensor([1, 2, 3, 4]);
    const scores = f_regression(X, y);
    expect(scores[0]).toBeGreaterThan(scores[1]!);
  });
});

// ─── Preprocess: StratifiedShuffleSplit ─────────────────────────
describe("StratifiedShuffleSplit", () => {
  it("preserves class proportions in splits", () => {
    // 60% class 0, 40% class 1
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);

    const sss = new StratifiedShuffleSplit({
      nSplits: 3,
      testSize: 0.3,
      randomState: 42,
    });
    const splits = sss.split(X, y);

    expect(splits.length).toBe(3);
    for (const split of splits) {
      expect(split.trainIndex.length).toBeGreaterThan(0);
      expect(split.testIndex.length).toBeGreaterThan(0);

      // Check that both classes appear in test set
      const testLabels = split.testIndex.map((i) => Number(y.data[y.offset + i]));
      const uniqueTest = new Set(testLabels);
      expect(uniqueTest.size).toBe(2);
    }
  });

  it("is deterministic with randomState", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    const sss1 = new StratifiedShuffleSplit({ nSplits: 2, testSize: 0.3, randomState: 123 });
    const sss2 = new StratifiedShuffleSplit({ nSplits: 2, testSize: 0.3, randomState: 123 });

    const splits1 = sss1.split(X, y);
    const splits2 = sss2.split(X, y);

    expect(splits1[0]!.trainIndex).toEqual(splits2[0]!.trainIndex);
    expect(splits1[0]!.testIndex).toEqual(splits2[0]!.testIndex);
  });

  it("getNSplits returns correct value", () => {
    const sss = new StratifiedShuffleSplit({ nSplits: 7 });
    expect(sss.getNSplits()).toBe(7);
  });

  it("throws for mismatched X and y", () => {
    const X = tensor([[1], [2], [3]]);
    const y = tensor([0, 1]);
    const sss = new StratifiedShuffleSplit({ nSplits: 1, testSize: 0.5 });
    expect(() => sss.split(X, y)).toThrow();
  });
});

// ─── Core: Warning System ──────────────────────────────────────
describe("Warning System", () => {
  it("warn() issues a warning that can be caught", () => {
    const warnings = catchWarnings(() => {
      warn("test warning", "ConvergenceWarning");
    });
    expect(warnings.length).toBe(1);
    expect(warnings[0]!.category).toBe("ConvergenceWarning");
    expect(warnings[0]!.message).toBe("test warning");
  });

  it("filterWarnings('ignore') silences warnings", () => {
    resetWarnings();
    filterWarnings("ignore", { category: "UserWarning" });
    const warnings = catchWarnings(() => {
      warn("should be ignored", "UserWarning");
    });
    expect(warnings.length).toBe(0);
    resetWarnings();
  });

  it("filterWarnings('error') converts to error", () => {
    resetWarnings();
    filterWarnings("error", { category: "DataConversionWarning" });
    expect(() => {
      warn("bad data", "DataConversionWarning");
    }).toThrow("[DataConversionWarning] bad data");
    resetWarnings();
  });

  it("filterWarnings('once') emits only first occurrence", () => {
    resetWarnings();
    filterWarnings("once", { category: "UndefinedMetricWarning" });
    const warnings = catchWarnings(() => {
      warn("metric undefined", "UndefinedMetricWarning");
      warn("metric undefined", "UndefinedMetricWarning");
      warn("metric undefined", "UndefinedMetricWarning");
    });
    expect(warnings.length).toBe(1);
    resetWarnings();
  });

  it("catchWarnings collects multiple warnings", () => {
    resetWarnings();
    const warnings = catchWarnings(() => {
      warn("w1", "ConvergenceWarning");
      warn("w2", "FitFailedWarning");
      warn("w3", "UserWarning");
    });
    expect(warnings.length).toBe(3);
    expect(warnings[0]!.category).toBe("ConvergenceWarning");
    expect(warnings[1]!.category).toBe("FitFailedWarning");
    expect(warnings[2]!.category).toBe("UserWarning");
    resetWarnings();
  });

  it("resetWarnings clears all filters", () => {
    filterWarnings("ignore");
    resetWarnings();
    const warnings = catchWarnings(() => {
      warn("should appear", "UserWarning");
    });
    expect(warnings.length).toBe(1);
  });
});

// ─── Random: categorical ───────────────────────────────────────
describe("categorical", () => {
  it("samples from uniform distribution", () => {
    setSeed(42);
    const probs = tensor([1, 1, 1, 1]);
    const samples = categorical(probs, 1000, true);
    expect(samples.shape).toEqual([1000]);
    expect(samples.dtype).toBe("int32");

    // All samples should be in [0, 3]
    for (let i = 0; i < 1000; i++) {
      const v = Number(samples.data[i]);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(3);
    }
  });

  it("respects probability weights", () => {
    setSeed(42);
    // Category 0 has 99% probability
    const probs = tensor([99, 1]);
    const samples = categorical(probs, 500, true);
    let count0 = 0;
    for (let i = 0; i < 500; i++) {
      if (Number(samples.data[i]) === 0) count0++;
    }
    // Should be heavily biased toward 0
    expect(count0).toBeGreaterThan(400);
  });

  it("samples without replacement", () => {
    setSeed(42);
    const probs = tensor([1, 1, 1, 1, 1]);
    const samples = categorical(probs, 5, false);
    const vals = new Set<number>();
    for (let i = 0; i < 5; i++) vals.add(Number(samples.data[i]));
    // All 5 distinct values
    expect(vals.size).toBe(5);
  });

  it("throws for too many samples without replacement", () => {
    const probs = tensor([1, 1, 1]);
    expect(() => categorical(probs, 5, false)).toThrow();
  });

  it("throws for negative probs", () => {
    const probs = tensor([-1, 1, 1]);
    expect(() => categorical(probs, 1)).toThrow();
  });

  it("throws for non-1D input", () => {
    const probs = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => categorical(probs, 1)).toThrow();
  });
});

// ─── Random: gumbel_softmax ────────────────────────────────────
describe("gumbel_softmax", () => {
  it("returns valid probability distribution for 1D logits", () => {
    setSeed(42);
    const logits = tensor([1.0, 2.0, 3.0]);
    const result = gumbel_softmax(logits, 1.0, false);
    expect(result.shape).toEqual([3]);

    // Should sum to ~1
    let sum = 0;
    for (let i = 0; i < 3; i++) sum += Number(result.data[i]);
    expect(sum).toBeCloseTo(1.0, 5);

    // All values should be positive
    for (let i = 0; i < 3; i++) {
      expect(Number(result.data[i])).toBeGreaterThan(0);
    }
  });

  it("hard=true returns one-hot vectors", () => {
    setSeed(42);
    const logits = tensor([1.0, 2.0, 3.0]);
    const result = gumbel_softmax(logits, 1.0, true);

    let sum = 0;
    let numOnes = 0;
    for (let i = 0; i < 3; i++) {
      const v = Number(result.data[i]);
      sum += v;
      if (v === 1) numOnes++;
    }
    expect(sum).toBe(1);
    expect(numOnes).toBe(1);
  });

  it("handles 2D batched logits", () => {
    setSeed(42);
    const logits = tensor([
      [1, 2, 3],
      [3, 2, 1],
    ]);
    const result = gumbel_softmax(logits, 0.5, false);
    expect(result.shape).toEqual([2, 3]);

    // Each row should sum to ~1
    for (let b = 0; b < 2; b++) {
      let rowSum = 0;
      for (let j = 0; j < 3; j++) rowSum += Number(result.data[b * 3 + j]);
      expect(rowSum).toBeCloseTo(1.0, 5);
    }
  });

  it("lower temperature produces sharper distributions", () => {
    setSeed(42);
    const logits = tensor([0, 0, 10]);
    const soft = gumbel_softmax(logits, 10.0, false);
    setSeed(42);
    const sharp = gumbel_softmax(logits, 0.01, false);

    // With very low temperature, the max logit should dominate
    expect(Number(sharp.data[2])).toBeGreaterThan(Number(soft.data[2]!));
  });

  it("throws for invalid tau", () => {
    const logits = tensor([1, 2, 3]);
    expect(() => gumbel_softmax(logits, 0)).toThrow();
    expect(() => gumbel_softmax(logits, -1)).toThrow();
  });
});
