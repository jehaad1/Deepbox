import { describe, expect, it } from "vitest";
import { d2TweedieScore, meanPinballLoss } from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("d2TweedieScore", () => {
  it("with power=0 matches R² for normal distribution", () => {
    const yTrue = tensor([3, -0.5, 2, 7]);
    const yPred = tensor([2.5, 0.0, 2, 8]);
    const score = d2TweedieScore(yTrue, yPred, 0);
    // D² with power=0 uses sum of squared deviations (unit deviance for Normal)
    // deviance_model = Σ(y-ŷ)² = 0.25+0.25+0+1 = 1.5
    // deviance_null = Σ(y-ȳ)² where ȳ = mean(yTrue) = 2.875
    // = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
    // D² = 1 - 1.5/29.1875 ≈ 0.9486
    expect(score).toBeCloseTo(0.9486, 2);
  });

  it("returns 1 for perfect predictions", () => {
    const y = tensor([1, 2, 3, 4]);
    expect(d2TweedieScore(y, y, 0)).toBeCloseTo(1.0, 10);
  });

  it("returns 0 for mean prediction (power=0)", () => {
    const yTrue = tensor([1, 2, 3, 4]);
    const yMean = tensor([2.5, 2.5, 2.5, 2.5]);
    expect(d2TweedieScore(yTrue, yMean, 0)).toBeCloseTo(0.0, 10);
  });

  it("works with power=1 (Poisson)", () => {
    const yTrue = tensor([1, 2, 3, 4]);
    const yPred = tensor([1.1, 1.9, 3.2, 3.8]);
    const score = d2TweedieScore(yTrue, yPred, 1);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("works with power=2 (Gamma)", () => {
    const yTrue = tensor([1, 2, 3, 4]);
    const yPred = tensor([1.1, 1.9, 3.2, 3.8]);
    const score = d2TweedieScore(yTrue, yPred, 2);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws for power in (0, 1)", () => {
    const y = tensor([1, 2, 3]);
    expect(() => d2TweedieScore(y, y, 0.5)).toThrow();
  });

  it("throws for empty inputs", () => {
    expect(() => d2TweedieScore(tensor([]), tensor([]))).toThrow();
  });

  it("throws for mismatched lengths", () => {
    expect(() => d2TweedieScore(tensor([1, 2]), tensor([1, 2, 3]))).toThrow();
  });
});

describe("meanPinballLoss", () => {
  it("computes median loss (alpha=0.5)", () => {
    const yTrue = tensor([1, 2, 3]);
    const yPred = tensor([1.5, 2.5, 2.5]);
    // diffs: -0.5, -0.5, 0.5
    // losses: 0.5*0.5, 0.5*0.5, 0.5*0.5 = 0.25, 0.25, 0.25
    // mean = 0.25
    expect(meanPinballLoss(yTrue, yPred, 0.5)).toBeCloseTo(0.25, 10);
  });

  it("returns 0 for perfect predictions", () => {
    const y = tensor([1, 2, 3, 4]);
    expect(meanPinballLoss(y, y, 0.5)).toBe(0);
  });

  it("penalizes under-predictions more with high alpha", () => {
    const yTrue = tensor([5, 5, 5]);
    const yPredLow = tensor([3, 3, 3]); // under-predicting by 2
    const yPredHigh = tensor([7, 7, 7]); // over-predicting by 2

    const lossLow = meanPinballLoss(yTrue, yPredLow, 0.9);
    const lossHigh = meanPinballLoss(yTrue, yPredHigh, 0.9);
    // alpha=0.9 penalizes under-prediction (positive diff) more
    expect(lossLow).toBeGreaterThan(lossHigh);
  });

  it("penalizes over-predictions more with low alpha", () => {
    const yTrue = tensor([5, 5, 5]);
    const yPredLow = tensor([3, 3, 3]);
    const yPredHigh = tensor([7, 7, 7]);

    const lossLow = meanPinballLoss(yTrue, yPredLow, 0.1);
    const lossHigh = meanPinballLoss(yTrue, yPredHigh, 0.1);
    expect(lossHigh).toBeGreaterThan(lossLow);
  });

  it("throws for alpha outside (0, 1)", () => {
    const y = tensor([1, 2]);
    expect(() => meanPinballLoss(y, y, 0)).toThrow();
    expect(() => meanPinballLoss(y, y, 1)).toThrow();
    expect(() => meanPinballLoss(y, y, -0.5)).toThrow();
  });

  it("throws for mismatched lengths", () => {
    expect(() => meanPinballLoss(tensor([1, 2]), tensor([1, 2, 3]))).toThrow();
  });
});
