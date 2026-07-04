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
  it("perfect predictions return 0", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(smape(yt, yp)).toBeCloseTo(0);
  });

  it("returns value in [0, 2]", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([3, 1, 2]);
    const s = smape(yt, yp);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(2);
  });

  it("handles zero denominator gracefully", () => {
    const yt = tensor([0, 1, 2]);
    const yp = tensor([0, 1, 2]);
    expect(smape(yt, yp)).toBeCloseTo(0);
  });

  it("empty input returns 0", () => {
    const yt = tensor([] as number[]);
    const yp = tensor([] as number[]);
    expect(smape(yt, yp)).toBe(0);
  });

  it("rejects different length inputs", () => {
    expect(() => smape(tensor([1, 2]), tensor([1]))).toThrow(/length/i);
  });

  it("rejects non-1D inputs", () => {
    expect(() => smape(tensor([[1, 2]]), tensor([1, 2]))).toThrow(/1D/i);
  });
});

describe("meanSquaredLogError", () => {
  it("perfect predictions return 0", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanSquaredLogError(yt, yp)).toBeCloseTo(0);
  });

  it("returns positive value for imperfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 3.5]);
    expect(meanSquaredLogError(yt, yp)).toBeGreaterThan(0);
  });

  it("rejects negative values in yTrue", () => {
    expect(() => meanSquaredLogError(tensor([-1, 2]), tensor([1, 2]))).toThrow(/non-negative/i);
  });

  it("rejects negative values in yPred", () => {
    expect(() => meanSquaredLogError(tensor([1, 2]), tensor([-1, 2]))).toThrow(/non-negative/i);
  });

  it("empty input returns 0", () => {
    expect(meanSquaredLogError(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("brierScoreLoss", () => {
  it("perfect binary predictions return 0", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 1, 1, 0]);
    expect(brierScoreLoss(yt, yp)).toBeCloseTo(0);
  });

  it("worst predictions return 1", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([1, 0]);
    expect(brierScoreLoss(yt, yp)).toBeCloseTo(1);
  });

  it("returns value between 0 and 1 for probabilities", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0.1, 0.9, 0.8, 0.2]);
    const score = brierScoreLoss(yt, yp);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("empty input returns 0", () => {
    expect(brierScoreLoss(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("hingeLoss", () => {
  it("correct margin returns 0", () => {
    const yt = tensor([1, -1, 1]);
    const yd = tensor([2, -2, 2]);
    expect(hingeLoss(yt, yd)).toBeCloseTo(0);
  });

  it("margin < 1 returns positive loss", () => {
    const yt = tensor([1, -1]);
    const yd = tensor([0.5, -0.5]);
    expect(hingeLoss(yt, yd)).toBeGreaterThan(0);
  });

  it("empty returns 0", () => {
    expect(hingeLoss(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("zeroOneLoss", () => {
  it("perfect predictions return 0", () => {
    expect(zeroOneLoss(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBe(0);
  });

  it("all wrong returns 1", () => {
    expect(zeroOneLoss(tensor([0, 1]), tensor([1, 0]))).toBe(1);
  });

  it("partial returns fraction", () => {
    expect(zeroOneLoss(tensor([0, 0, 1, 1]), tensor([0, 1, 1, 0]))).toBeCloseTo(0.5);
  });

  it("empty returns 0", () => {
    expect(zeroOneLoss(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("topKAccuracyScore", () => {
  it("top-1 accuracy", () => {
    const yt = tensor([0, 1, 2]);
    const ys = tensor([
      [0.9, 0.05, 0.05],
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
    ]);
    expect(topKAccuracyScore(yt, ys, 1)).toBeCloseTo(1.0);
  });

  it("top-2 accuracy", () => {
    const yt = tensor([0, 1, 2]);
    const ys = tensor([
      [0.1, 0.5, 0.4],
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
    ]);
    expect(topKAccuracyScore(yt, ys, 2)).toBeGreaterThanOrEqual(0.66);
  });

  it("rejects non-1D yTrue", () => {
    expect(() => topKAccuracyScore(tensor([[0, 1]]), tensor([[0.5, 0.5]]))).toThrow(/1D/i);
  });

  it("rejects non-2D yScore", () => {
    expect(() => topKAccuracyScore(tensor([0]), tensor([0.5]))).toThrow(/2D/i);
  });

  it("rejects mismatched sample counts", () => {
    expect(() => topKAccuracyScore(tensor([0, 1]), tensor([[0.5, 0.5]]))).toThrow(/n_samples/i);
  });

  it("rejects invalid k", () => {
    expect(() => topKAccuracyScore(tensor([0]), tensor([[0.5, 0.5]]), 0)).toThrow();
    expect(() => topKAccuracyScore(tensor([0]), tensor([[0.5, 0.5]]), 3)).toThrow();
  });
});

describe("meanPoissonDeviance", () => {
  it("perfect predictions return 0", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanPoissonDeviance(yt, yp)).toBeCloseTo(0, 5);
  });

  it("positive value for imperfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 3.5]);
    expect(meanPoissonDeviance(yt, yp)).toBeGreaterThan(0);
  });

  it("handles yTrue=0", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([1, 1]);
    expect(meanPoissonDeviance(yt, yp)).toBeGreaterThan(0);
  });

  it("rejects negative yTrue", () => {
    expect(() => meanPoissonDeviance(tensor([-1, 2]), tensor([1, 2]))).toThrow(/y_true >= 0/i);
  });

  it("rejects non-positive yPred", () => {
    expect(() => meanPoissonDeviance(tensor([1, 2]), tensor([0, 2]))).toThrow(/y_pred > 0/i);
  });

  it("empty returns 0", () => {
    expect(meanPoissonDeviance(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("meanGammaDeviance", () => {
  it("perfect predictions return 0", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanGammaDeviance(yt, yp)).toBeCloseTo(0, 5);
  });

  it("positive value for imperfect predictions", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1.5, 2.5, 3.5]);
    expect(meanGammaDeviance(yt, yp)).toBeGreaterThan(0);
  });

  it("rejects non-positive yTrue", () => {
    expect(() => meanGammaDeviance(tensor([0, 2]), tensor([1, 2]))).toThrow(/y_true > 0/i);
  });

  it("rejects non-positive yPred", () => {
    expect(() => meanGammaDeviance(tensor([1, 2]), tensor([0, 2]))).toThrow(/y_pred > 0/i);
  });

  it("empty returns 0", () => {
    expect(meanGammaDeviance(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("multilabelConfusionMatrix", () => {
  it("computes confusion matrix per label", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0, 1, 1, 0]);
    const result = multilabelConfusionMatrix(yt, yp);
    expect(result.length).toBe(2);
    // Label 0: TP=1, FP=1, FN=1, TN=1
    expect(result[0]!.tp).toBe(1);
    expect(result[0]!.fp).toBe(1);
    expect(result[0]!.fn).toBe(1);
    expect(result[0]!.tn).toBe(1);
  });

  it("with custom labels", () => {
    const yt = tensor([0, 0, 1, 1, 2, 2]);
    const yp = tensor([0, 1, 1, 2, 2, 0]);
    const result = multilabelConfusionMatrix(yt, yp, [0, 1, 2]);
    expect(result.length).toBe(3);
  });

  it("perfect predictions", () => {
    const yt = tensor([0, 1, 0, 1]);
    const yp = tensor([0, 1, 0, 1]);
    const result = multilabelConfusionMatrix(yt, yp);
    expect(result[0]!.tp).toBe(2);
    expect(result[0]!.fp).toBe(0);
    expect(result[0]!.fn).toBe(0);
    expect(result[0]!.tn).toBe(2);
  });
});

describe("detCurve", () => {
  it("computes FPR and FNR", () => {
    const yt = tensor([0, 0, 1, 1]);
    const ys = tensor([0.1, 0.4, 0.6, 0.9]);
    const { fpr, fnr, thresholds } = detCurve(yt, ys);
    expect(fpr.length).toBeGreaterThan(0);
    expect(fnr.length).toBe(fpr.length);
    expect(thresholds.length).toBe(fpr.length);
  });

  it("empty returns empty", () => {
    const { fpr, fnr, thresholds } = detCurve(tensor([] as number[]), tensor([] as number[]));
    expect(fpr).toEqual([]);
    expect(fnr).toEqual([]);
    expect(thresholds).toEqual([]);
  });

  it("all positive returns empty", () => {
    const { fpr } = detCurve(tensor([1, 1, 1]), tensor([0.5, 0.6, 0.7]));
    expect(fpr).toEqual([]);
  });

  it("all negative returns empty", () => {
    const { fpr } = detCurve(tensor([0, 0, 0]), tensor([0.5, 0.6, 0.7]));
    expect(fpr).toEqual([]);
  });

  it("rejects non-binary labels", () => {
    expect(() => detCurve(tensor([0, 2, 1]), tensor([0.5, 0.6, 0.7]))).toThrow(/binary/i);
  });
});

describe("d2TweedieScore", () => {
  it("power=0 (Normal) equivalent to R²", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 4.1]);
    const score = d2TweedieScore(yt, yp, 0);
    expect(score).toBeGreaterThan(0.9);
    expect(score).toBeLessThanOrEqual(1.0);
  });

  it("power=1 (Poisson)", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 4.1]);
    const score = d2TweedieScore(yt, yp, 1);
    expect(score).toBeGreaterThan(0.9);
  });

  it("power=2 (Gamma)", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 4.1]);
    const score = d2TweedieScore(yt, yp, 2);
    expect(score).toBeGreaterThan(0.9);
  });

  it("general power (e.g. 3)", () => {
    const yt = tensor([1, 2, 3, 4]);
    const yp = tensor([1.1, 2.1, 2.9, 4.1]);
    const score = d2TweedieScore(yt, yp, 3);
    expect(Number.isFinite(score)).toBe(true);
  });

  it("rejects empty input", () => {
    expect(() => d2TweedieScore(tensor([] as number[]), tensor([] as number[]))).toThrow(
      /at least one/i
    );
  });

  it("rejects power in (0, 1)", () => {
    expect(() => d2TweedieScore(tensor([1]), tensor([1]), 0.5)).toThrow();
  });

  it("rejects non-finite power", () => {
    expect(() => d2TweedieScore(tensor([1]), tensor([1]), NaN)).toThrow(/finite/i);
  });

  it("constant predictions (devNull=0)", () => {
    const yt = tensor([3, 3, 3]);
    const yp = tensor([3, 3, 3]);
    const score = d2TweedieScore(yt, yp, 0);
    expect(score).toBe(1.0);
  });
});

describe("meanPinballLoss", () => {
  it("perfect predictions return 0", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([1, 2, 3]);
    expect(meanPinballLoss(yt, yp, 0.5)).toBeCloseTo(0);
  });

  it("asymmetric loss for alpha=0.9", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([0, 1, 2]);
    const loss = meanPinballLoss(yt, yp, 0.9);
    expect(loss).toBeGreaterThan(0);
  });

  it("under-prediction penalized more for high alpha", () => {
    const yt = tensor([2]);
    const yp = tensor([1]);
    const lossHigh = meanPinballLoss(yt, yp, 0.9);
    const lossLow = meanPinballLoss(yt, yp, 0.1);
    expect(lossHigh).toBeGreaterThan(lossLow);
  });

  it("over-prediction penalized more for low alpha", () => {
    const yt = tensor([1]);
    const yp = tensor([2]);
    const lossLow = meanPinballLoss(yt, yp, 0.1);
    const lossHigh = meanPinballLoss(yt, yp, 0.9);
    expect(lossLow).toBeGreaterThan(lossHigh);
  });

  it("rejects invalid alpha", () => {
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), 0)).toThrow();
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), 1)).toThrow();
    expect(() => meanPinballLoss(tensor([1]), tensor([1]), NaN)).toThrow();
  });

  it("empty returns 0", () => {
    expect(meanPinballLoss(tensor([] as number[]), tensor([] as number[]))).toBe(0);
  });
});

describe("coverageError", () => {
  it("perfect ranking", () => {
    const yt = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const ys = tensor([
      [0.9, 0.1, 0.05],
      [0.1, 0.9, 0.05],
    ]);
    const score = coverageError(yt, ys);
    expect(score).toBe(1);
  });

  it("worst ranking", () => {
    const yt = tensor([
      [1, 0, 0],
      [0, 0, 1],
    ]);
    const ys = tensor([
      [0.1, 0.5, 0.9],
      [0.9, 0.5, 0.1],
    ]);
    const score = coverageError(yt, ys);
    expect(score).toBe(3);
  });

  it("rejects shape mismatch", () => {
    expect(() =>
      coverageError(
        tensor([[1, 0]]),
        tensor([
          [0.5, 0.5],
          [0.3, 0.7],
        ])
      )
    ).toThrow(/shape/i);
  });

  it("empty returns 0", () => {
    // 0 x 2 tensor
    const yt = tensor([] as number[]).reshape([0, 2]);
    const ys = tensor([] as number[]).reshape([0, 2]);
    expect(coverageError(yt, ys)).toBe(0);
  });
});

describe("labelRankingLoss", () => {
  it("perfect ranking returns 0", () => {
    const yt = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const ys = tensor([
      [0.9, 0.1, 0.05],
      [0.1, 0.9, 0.05],
    ]);
    expect(labelRankingLoss(yt, ys)).toBeCloseTo(0);
  });

  it("worst ranking returns positive value", () => {
    const yt = tensor([
      [1, 0],
      [0, 1],
    ]);
    const ys = tensor([
      [0.1, 0.9],
      [0.9, 0.1],
    ]);
    expect(labelRankingLoss(yt, ys)).toBeGreaterThan(0);
  });

  it("empty returns 0", () => {
    const yt = tensor([] as number[]).reshape([0, 2]);
    const ys = tensor([] as number[]).reshape([0, 2]);
    expect(labelRankingLoss(yt, ys)).toBe(0);
  });

  it("all positive labels — no negative => loss 0 for that sample", () => {
    const yt = tensor([
      [1, 1],
      [0, 1],
    ]);
    const ys = tensor([
      [0.9, 0.1],
      [0.1, 0.9],
    ]);
    const loss = labelRankingLoss(yt, ys);
    expect(loss).toBeGreaterThanOrEqual(0);
  });
});
