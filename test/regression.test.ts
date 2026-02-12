import { describe, expect, it } from "vitest";
import {
  adjustedR2Score,
  explainedVarianceScore,
  mae,
  mape,
  maxError,
  medianAbsoluteError,
  mse,
  r2Score,
  rmse,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("Regression Metrics", () => {
  const yTrue = tensor([3, -0.5, 2, 7]);
  const yPred = tensor([2.5, 0.0, 2, 8]);

  it("should calculate MSE", () => {
    const error = mse(yTrue, yPred);
    expect(error).toBeCloseTo(0.375);
  });

  it("should calculate RMSE", () => {
    const error = rmse(yTrue, yPred);
    expect(error).toBeCloseTo(Math.sqrt(0.375));
  });

  it("should calculate MAE", () => {
    const error = mae(yTrue, yPred);
    expect(error).toBeCloseTo(0.5);
  });

  it("should calculate R² score", () => {
    const score = r2Score(yTrue, yPred);
    expect(score).toBeCloseTo(0.9486301369863014);
  });

  it("should calculate adjusted R² score", () => {
    const score = adjustedR2Score(yTrue, yPred, 2);
    expect(score).toBeCloseTo(0.845890410958904);
  });

  it("should calculate MAPE", () => {
    const error = mape(yTrue, yPred);
    expect(error).toBeCloseTo(32.73809523809524);
  });

  it("should calculate median absolute error", () => {
    const error = medianAbsoluteError(yTrue, yPred);
    expect(error).toBeCloseTo(0.5);
  });

  it("should calculate max error", () => {
    const error = maxError(yTrue, yPred);
    expect(error).toBeCloseTo(1);
  });

  it("should calculate explained variance score", () => {
    const score = explainedVarianceScore(yTrue, yPred);
    expect(score).toBeCloseTo(0.9571734475374732);
  });

  it("should handle empty inputs", () => {
    const empty = tensor([]);
    expect(mse(empty, empty)).toBe(0);
    expect(mae(empty, empty)).toBe(0);
  });

  it("should throw on size mismatch", () => {
    expect(() => mse(tensor([1, 2]), tensor([1]))).toThrow();
    expect(() => mae(tensor([1, 2]), tensor([1]))).toThrow();
  });
});
