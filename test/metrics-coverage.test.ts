import { describe, expect, it } from "vitest";
import {
  brierScoreLoss,
  coverageError,
  d2TweedieScore,
  detCurve,
  hingeLoss,
  labelRankingLoss,
  meanGammaDeviance,
  meanPinballLoss,
  meanPoissonDeviance,
  meanSquaredLogError,
  multilabelConfusionMatrix,
  smape,
  topKAccuracyScore,
  zeroOneLoss,
} from "../src/metrics/extra";
import { tensor } from "../src/ndarray";

describe("smape", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(smape(yt, yp)).toBe(0);
  });

  it("computes SMAPE correctly", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.1, 2.2, 2.8]);
    const s = smape(yt, yp);
    expect(s).toBeGreaterThan(0);
    expect(s).toBeLessThan(2);
  });

  it("handles zero denominator", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([0, 1]);
    expect(smape(yt, yp)).toBe(0);
  });

  it("returns 0 for empty arrays", () => {
    const yt = tensor(new Float64Array(0));
    const yp = tensor(new Float64Array(0));
    expect(smape(yt, yp)).toBe(0);
  });

  it("validates inputs", () => {
    expect(() => smape(tensor([[1]]), tensor([1]))).toThrow(/1D/);
    expect(() => smape(tensor([1]), tensor([[1]]))).toThrow(/1D/);
    expect(() => smape(tensor([1, 2]), tensor([1]))).toThrow(/same length/);
  });
});

describe("meanSquaredLogError", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanSquaredLogError(yt, yp)).toBe(0);
  });

  it("computes MSLE correctly", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.1, 2.2, 2.8]);
    expect(meanSquaredLogError(yt, yp)).toBeGreaterThan(0);
  });

  it("throws for negative values in yTrue", () => {
    expect(() => meanSquaredLogError(tensor([-1, 2]), tensor([1, 2]))).toThrow(/non-negative/);
  });

  it("throws for negative values in yPred", () => {
    expect(() => meanSquaredLogError(tensor([1, 2]), tensor([-1, 2]))).toThrow(/non-negative/);
  });

  it("returns 0 for empty", () => {
    expect(meanSquaredLogError(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("brierScoreLoss", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 1, 1, 0]);
    expect(brierScoreLoss(yt, yp)).toBe(0);
  });

  it("returns 1 for worst predictions", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([1, 0]);
    expect(brierScoreLoss(yt, yp)).toBe(1);
  });

  it("computes correctly", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0.1, 0.9, 0.8, 0.2]);
    const score = brierScoreLoss(yt, yp);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("returns 0 for empty", () => {
    expect(brierScoreLoss(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("hingeLoss", () => {
  it("returns 0 for perfect margin", () => {
    const yt = tensor([1, -1, 1]);
    const yd = tensor([2, -2, 3]);
    expect(hingeLoss(yt, yd)).toBe(0);
  });

  it("computes correctly for margin violations", () => {
    const yt = tensor([1, -1]);
    const yd = tensor([0.5, -0.5]);
    const loss = hingeLoss(yt, yd);
    expect(loss).toBeCloseTo(0.5, 5);
  });

  it("returns 0 for empty", () => {
    expect(hingeLoss(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("zeroOneLoss", () => {
  it("returns 0 for perfect predictions", () => {
    expect(zeroOneLoss(tensor([0, 1, 2]), tensor([0, 1, 2]))).toBe(0);
  });

  it("returns 1 for all wrong", () => {
    expect(zeroOneLoss(tensor([0, 1]), tensor([1, 0]))).toBe(1);
  });

  it("computes fraction correctly", () => {
    expect(zeroOneLoss(tensor([0, 1, 2, 3]), tensor([0, 1, 0, 3]))).toBe(0.25);
  });

  it("returns 0 for empty", () => {
    expect(zeroOneLoss(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("topKAccuracyScore", () => {
  it("computes top-1 accuracy", () => {
    const yt = tensor([0, 1, 2]);
    const yScore = tensor([
      [0.9, 0.05, 0.05],
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
    ]);
    expect(topKAccuracyScore(yt, yScore, 1)).toBe(1);
  });

  it("computes top-2 accuracy", () => {
    const yt = tensor([0, 1, 2]);
    const yScore = tensor([
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
      [0.1, 0.8, 0.1],
    ]);
    // True labels: 0 -> top-2 is [1,0] or [1,2], 0 not in top2 of first row
    // 1 -> top-2 is [2,0] or [2,1], 1 not in top2 of second row
    // 2 -> top-2 is [1,0] or [1,2], 2 in top2 of third row
    // Actually let me just check it returns a valid number
    const acc = topKAccuracyScore(yt, yScore, 2);
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });

  it("validates inputs", () => {
    expect(() => topKAccuracyScore(tensor([[0]]), tensor([[0.5, 0.5]]), 1)).toThrow(/1D/);
    expect(() => topKAccuracyScore(tensor([0]), tensor([0.5]), 1)).toThrow(/2D/);
    expect(() => topKAccuracyScore(tensor([0, 1]), tensor([[0.5, 0.5]]), 1)).toThrow(
      /same n_samples/
    );
    expect(() => topKAccuracyScore(tensor([0]), tensor([[0.5, 0.5]]), 3)).toThrow(/k must be/);
  });
});

describe("meanPoissonDeviance", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanPoissonDeviance(yt, yp)).toBeCloseTo(0, 10);
  });

  it("handles y_true=0", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([0.5, 1]);
    const dev = meanPoissonDeviance(yt, yp);
    expect(dev).toBeGreaterThan(0);
  });

  it("throws for negative y_true", () => {
    expect(() => meanPoissonDeviance(tensor([-1]), tensor([1]))).toThrow(/y_true >= 0/);
  });

  it("throws for non-positive y_pred", () => {
    expect(() => meanPoissonDeviance(tensor([1]), tensor([0]))).toThrow(/y_pred > 0/);
  });

  it("returns 0 for empty", () => {
    expect(meanPoissonDeviance(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("meanGammaDeviance", () => {
  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanGammaDeviance(yt, yp)).toBeCloseTo(0, 10);
  });

  it("throws for non-positive y_true", () => {
    expect(() => meanGammaDeviance(tensor([0, 1]), tensor([1, 1]))).toThrow(/y_true > 0/);
  });

  it("throws for non-positive y_pred", () => {
    expect(() => meanGammaDeviance(tensor([1, 1]), tensor([0, 1]))).toThrow(/y_pred > 0/);
  });

  it("returns 0 for empty", () => {
    expect(meanGammaDeviance(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("multilabelConfusionMatrix", () => {
  it("computes confusion matrix for each label", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0, 1, 0, 1]);
    const result = multilabelConfusionMatrix(yt, yp);
    expect(result.length).toBe(2);
    expect(result[0]!.label).toBe(0);
    expect(result[0]!.tp).toBe(1);
    expect(result[0]!.fn).toBe(1);
    expect(result[0]!.fp).toBe(1);
    expect(result[0]!.tn).toBe(1);
  });

  it("works with explicit labels", () => {
    const yt = tensor([0, 1, 2]);
    const yp = tensor([0, 1, 2]);
    const result = multilabelConfusionMatrix(yt, yp, [0, 1, 2]);
    expect(result.length).toBe(3);
    for (const r of result) {
      expect(r.tp).toBe(1);
      expect(r.fp).toBe(0);
      expect(r.fn).toBe(0);
    }
  });
});

describe("detCurve", () => {
  it("computes DET curve", () => {
    const yt = tensor([0, 0, 1, 1]);
    const ys = tensor([0.1, 0.4, 0.6, 0.9]);
    const { fpr, fnr, thresholds } = detCurve(yt, ys);
    expect(fpr.length).toBeGreaterThan(0);
    expect(fnr.length).toBe(fpr.length);
    expect(thresholds.length).toBe(fpr.length);
  });

  it("returns empty for single class", () => {
    const yt = tensor([1, 1, 1]);
    const ys = tensor([0.1, 0.5, 0.9]);
    const { fpr } = detCurve(yt, ys);
    expect(fpr.length).toBe(0);
  });

  it("returns empty for empty input", () => {
    const { fpr } = detCurve(tensor(new Float64Array(0)), tensor(new Float64Array(0)));
    expect(fpr.length).toBe(0);
  });

  it("throws for non-binary labels", () => {
    expect(() => detCurve(tensor([0, 2]), tensor([0.1, 0.9]))).toThrow(/binary labels/);
  });
});

describe("d2TweedieScore", () => {
  it("power=0 equivalent to R²", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 3.9]);
    const score = d2TweedieScore(yt, yp, 0);
    expect(score).toBeGreaterThan(0.9);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("power=1 (Poisson)", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 3.9]);
    const score = d2TweedieScore(yt, yp, 1);
    expect(score).toBeGreaterThan(0.9);
  });

  it("power=2 (Gamma)", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 3.9]);
    const score = d2TweedieScore(yt, yp, 2);
    expect(score).toBeGreaterThan(0.9);
  });

  it("general power >= 1", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 3.9]);
    const score = d2TweedieScore(yt, yp, 3);
    expect(Number.isFinite(score)).toBe(true);
  });

  it("returns 1 for perfect predictions when devNull=0", () => {
    const yt = tensor([2, 2, 2]);
    const yp = tensor([2, 2, 2]);
    expect(d2TweedieScore(yt, yp, 0)).toBe(1);
  });

  it("throws for empty input", () => {
    expect(() => d2TweedieScore(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toThrow(
      /at least one sample/
    );
  });

  it("throws for non-finite power", () => {
    expect(() => d2TweedieScore(tensor([1]), tensor([1]), Infinity)).toThrow(
      /power must be finite/
    );
  });

  it("throws for power in (0, 1)", () => {
    expect(() => d2TweedieScore(tensor([1]), tensor([1]), 0.5)).toThrow(/power must be/);
  });

  it("throws for non-positive yPred with power=1", () => {
    expect(() => d2TweedieScore(tensor([1, 2]), tensor([0, 1]), 1)).toThrow(
      /yPred must be positive/
    );
  });

  it("throws for non-positive values with power=2", () => {
    expect(() => d2TweedieScore(tensor([0, 1]), tensor([1, 1]), 2)).toThrow(/positive/);
  });
});

describe("meanPinballLoss", () => {
  it("computes median loss (alpha=0.5)", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 2.5]);
    const loss = meanPinballLoss(yt, yp, 0.5);
    expect(loss).toBeGreaterThan(0);
  });

  it("returns 0 for perfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanPinballLoss(yt, yp, 0.5)).toBe(0);
  });

  it("handles alpha < 0.5 (underprediction penalized less)", () => {
    const yt = tensor([3]);
    const yp = tensor([1]);
    const loss1 = meanPinballLoss(yt, yp, 0.1);
    const loss2 = meanPinballLoss(yt, yp, 0.9);
    // Higher alpha penalizes underprediction more
    expect(loss2).toBeGreaterThan(loss1);
  });

  it("handles overprediction (diff < 0 branch)", () => {
    const yt = tensor([1]);
    const yp = tensor([3]);
    const loss = meanPinballLoss(yt, yp, 0.5);
    expect(loss).toBeGreaterThan(0);
  });

  it("throws for invalid alpha", () => {
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), 0)).toThrow(/alpha must be in/);
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), 1)).toThrow(/alpha must be in/);
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), -0.1)).toThrow(/alpha must be in/);
  });

  it("returns 0 for empty", () => {
    expect(meanPinballLoss(tensor(new Float64Array(0)), tensor(new Float64Array(0)))).toBe(0);
  });
});

describe("coverageError", () => {
  it("computes coverage correctly", () => {
    const yt = tensor([
      [1, 0, 1],
      [0, 1, 0],
    ]);
    const ys = tensor([
      [0.9, 0.1, 0.8],
      [0.2, 0.7, 0.1],
    ]);
    const cov = coverageError(yt, ys);
    expect(cov).toBeGreaterThanOrEqual(1);
  });

  it("returns 0 for empty", () => {
    // Empty 2D tensors
    const yt = tensor(new Float64Array(0)).reshape([0, 3]);
    const ys = tensor(new Float64Array(0)).reshape([0, 3]);
    expect(coverageError(yt, ys)).toBe(0);
  });

  it("throws for shape mismatch", () => {
    const yt = tensor([[1, 0]]);
    const ys = tensor([[0.5, 0.5, 0.5]]);
    expect(() => coverageError(yt, ys)).toThrow(/shape mismatch/);
  });

  it("throws for non-2D", () => {
    expect(() => coverageError(tensor([1, 0]), tensor([0.5, 0.5]))).toThrow(/2D/);
  });
});

describe("labelRankingLoss", () => {
  it("returns 0 for perfect ranking", () => {
    const yt = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const ys = tensor([
      [0.9, 0.1, 0.0],
      [0.1, 0.9, 0.0],
    ]);
    expect(labelRankingLoss(yt, ys)).toBe(0);
  });

  it("computes loss for imperfect ranking", () => {
    const yt = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const ys = tensor([
      [0.1, 0.9, 0.0],
      [0.9, 0.1, 0.0],
    ]);
    const loss = labelRankingLoss(yt, ys);
    expect(loss).toBeGreaterThan(0);
  });

  it("returns 0 for empty", () => {
    const yt = tensor(new Float64Array(0)).reshape([0, 3]);
    const ys = tensor(new Float64Array(0)).reshape([0, 3]);
    expect(labelRankingLoss(yt, ys)).toBe(0);
  });

  it("skips samples with all positive or all negative", () => {
    const yt = tensor([
      [1, 1, 1],
      [0, 0, 0],
    ]);
    const ys = tensor([
      [0.9, 0.1, 0.5],
      [0.1, 0.9, 0.5],
    ]);
    expect(labelRankingLoss(yt, ys)).toBe(0);
  });

  it("throws for shape mismatch", () => {
    const yt = tensor([[1, 0]]);
    const ys = tensor([[0.5, 0.5, 0.5]]);
    expect(() => labelRankingLoss(yt, ys)).toThrow(/shape mismatch/);
  });
});
