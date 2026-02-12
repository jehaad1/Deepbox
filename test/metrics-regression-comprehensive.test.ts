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

describe("Regression Metrics - Comprehensive Tests", () => {
  describe("mse (Mean Squared Error)", () => {
    it("should calculate MSE correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(mse(yTrue, yPred)).toBeCloseTo(0.375);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(mse(yTrue, yPred)).toBe(0);
    });

    it("should handle large errors", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yPred = tensor([10, 10, 10, 10]);
      expect(mse(yTrue, yPred)).toBe(100);
    });

    it("should handle negative values", () => {
      const yTrue = tensor([-5, -3, -1]);
      const yPred = tensor([-4, -2, 0]);
      expect(mse(yTrue, yPred)).toBeCloseTo(1);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(mse(empty, empty)).toBe(0);
    });

    it("should throw on size mismatch", () => {
      expect(() => mse(tensor([1, 2]), tensor([1]))).toThrow();
    });

    it("should handle single element", () => {
      expect(mse(tensor([5]), tensor([3]))).toBe(4);
    });

    it("should handle floating point precision", () => {
      const yTrue = tensor([0.1, 0.2, 0.3]);
      const yPred = tensor([0.1, 0.2, 0.3]);
      expect(mse(yTrue, yPred)).toBeCloseTo(0, 10);
    });

    it("should handle large datasets efficiently", () => {
      const size = 100000;
      const yTrue = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i)
      );
      const yPred = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i + 1)
      );

      const start = Date.now();
      const error = mse(yTrue, yPred);
      const duration = Date.now() - start;

      expect(error).toBe(1);
      expect(duration).toBeLessThan(1000);
    });

    it("should handle mixed positive and negative errors", () => {
      const yTrue = tensor([1, 2, 3, 4]);
      const yPred = tensor([2, 1, 4, 3]);
      expect(mse(yTrue, yPred)).toBe(1);
    });
  });

  describe("rmse (Root Mean Squared Error)", () => {
    it("should calculate RMSE correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(rmse(yTrue, yPred)).toBeCloseTo(Math.sqrt(0.375));
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(rmse(yTrue, yPred)).toBe(0);
    });

    it("should be square root of MSE", () => {
      const yTrue = tensor([1, 2, 3, 4]);
      const yPred = tensor([2, 3, 4, 5]);
      const mseVal = mse(yTrue, yPred);
      const rmseVal = rmse(yTrue, yPred);
      expect(rmseVal).toBeCloseTo(Math.sqrt(mseVal));
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(rmse(empty, empty)).toBe(0);
    });

    it("should handle large errors", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yPred = tensor([10, 10, 10, 10]);
      expect(rmse(yTrue, yPred)).toBe(10);
    });
  });

  describe("mae (Mean Absolute Error)", () => {
    it("should calculate MAE correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(mae(yTrue, yPred)).toBeCloseTo(0.5);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(mae(yTrue, yPred)).toBe(0);
    });

    it("should handle negative values", () => {
      const yTrue = tensor([-5, -3, -1]);
      const yPred = tensor([-4, -2, 0]);
      expect(mae(yTrue, yPred)).toBe(1);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(mae(empty, empty)).toBe(0);
    });

    it("should throw on size mismatch", () => {
      expect(() => mae(tensor([1, 2]), tensor([1]))).toThrow();
    });

    it("should be less sensitive to outliers than MSE", () => {
      const yTrue = tensor([1, 2, 3, 4, 100]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      const maeVal = mae(yTrue, yPred);
      const mseVal = mse(yTrue, yPred);
      expect(maeVal).toBeLessThan(mseVal);
    });

    it("should handle symmetric errors", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yPred = tensor([1, -1, 2, -2]);
      expect(mae(yTrue, yPred)).toBe(1.5);
    });
  });

  describe("r2Score (R² Score)", () => {
    it("should calculate R² correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(r2Score(yTrue, yPred)).toBeCloseTo(0.9486301369863014);
    });

    it("should return 1 for perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(r2Score(yTrue, yPred)).toBeCloseTo(1.0);
    });

    it("should handle constant predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const mean = 3;
      const yPred = tensor([mean, mean, mean, mean, mean]);
      expect(r2Score(yTrue, yPred)).toBeCloseTo(0);
    });

    it("should handle negative R²", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([5, 4, 3, 2, 1]);
      const r2 = r2Score(yTrue, yPred);
      expect(r2).toBeLessThan(0);
    });

    it("should handle single unique value", () => {
      const yTrue = tensor([5, 5, 5, 5]);
      const yPred = tensor([5, 5, 5, 5]);
      const r2 = r2Score(yTrue, yPred);
      expect(Number.isNaN(r2) || r2 === 1).toBe(true);
    });

    it("should handle large datasets", () => {
      const size = 10000;
      const yTrue = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i)
      );
      const yPred = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i + Math.random() * 0.1)
      );
      const r2 = r2Score(yTrue, yPred);
      expect(r2).toBeGreaterThan(0.99);
    });

    it("should reject empty input", () => {
      const empty = tensor([]);
      expect(() => r2Score(empty, empty)).toThrow(/at least one sample/i);
    });
  });

  describe("adjustedR2Score", () => {
    it("should calculate adjusted R² correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(adjustedR2Score(yTrue, yPred, 2)).toBeCloseTo(0.845890410958904);
    });

    it("should penalize more features", () => {
      const yTrue = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const yPred = tensor([1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1]);

      const adjR2_1 = adjustedR2Score(yTrue, yPred, 1);
      const adjR2_5 = adjustedR2Score(yTrue, yPred, 5);

      expect(adjR2_1).toBeGreaterThan(adjR2_5);
    });

    it("should handle perfect fit", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      const adjR2 = adjustedR2Score(yTrue, yPred, 1);
      expect(adjR2).toBeCloseTo(1.0);
    });

    it("should handle zero features", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      const adjR2 = adjustedR2Score(yTrue, yPred, 0);
      expect(Number.isFinite(adjR2)).toBe(true);
    });
  });

  describe("mape (Mean Absolute Percentage Error)", () => {
    it("should calculate MAPE correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(mape(yTrue, yPred)).toBeCloseTo(32.73809523809524);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(mape(yTrue, yPred)).toBe(0);
    });

    it("should handle zero true values gracefully", () => {
      const yTrue = tensor([0, 1, 2, 3]);
      const yPred = tensor([1, 1, 2, 3]);
      const mapeVal = mape(yTrue, yPred);
      expect(Number.isFinite(mapeVal)).toBe(true);
    });

    it("should return percentage", () => {
      const yTrue = tensor([100, 200, 300]);
      const yPred = tensor([110, 190, 310]);
      const mapeVal = mape(yTrue, yPred);
      expect(mapeVal).toBeGreaterThan(0);
      expect(mapeVal).toBeLessThan(100);
    });

    it("should handle negative values", () => {
      const yTrue = tensor([-10, -20, -30]);
      const yPred = tensor([-11, -19, -31]);
      const mapeVal = mape(yTrue, yPred);
      expect(Number.isFinite(mapeVal)).toBe(true);
    });
  });

  describe("medianAbsoluteError", () => {
    it("should calculate median absolute error correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(medianAbsoluteError(yTrue, yPred)).toBeCloseTo(0.5);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(medianAbsoluteError(yTrue, yPred)).toBe(0);
    });

    it("should be robust to outliers", () => {
      const yTrue = tensor([1, 2, 3, 4, 100]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      const medianErr = medianAbsoluteError(yTrue, yPred);
      expect(medianErr).toBe(0);
    });

    it("should handle odd number of samples", () => {
      const yTrue = tensor([1, 2, 3]);
      const yPred = tensor([2, 3, 4]);
      expect(medianAbsoluteError(yTrue, yPred)).toBe(1);
    });

    it("should handle even number of samples", () => {
      const yTrue = tensor([1, 2, 3, 4]);
      const yPred = tensor([2, 3, 4, 5]);
      expect(medianAbsoluteError(yTrue, yPred)).toBe(1);
    });

    it("should handle single element", () => {
      expect(medianAbsoluteError(tensor([5]), tensor([3]))).toBe(2);
    });

    it("should handle negative errors", () => {
      const yTrue = tensor([10, 20, 30]);
      const yPred = tensor([5, 15, 25]);
      expect(medianAbsoluteError(yTrue, yPred)).toBe(5);
    });
  });

  describe("maxError", () => {
    it("should calculate max error correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(maxError(yTrue, yPred)).toBeCloseTo(1);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(maxError(yTrue, yPred)).toBe(0);
    });

    it("should find maximum error", () => {
      const yTrue = tensor([1, 2, 3, 100]);
      const yPred = tensor([1, 2, 3, 5]);
      expect(maxError(yTrue, yPred)).toBe(95);
    });

    it("should handle negative errors", () => {
      const yTrue = tensor([10, 20, 30]);
      const yPred = tensor([5, 15, 25]);
      expect(maxError(yTrue, yPred)).toBe(5);
    });

    it("should handle single element", () => {
      expect(maxError(tensor([10]), tensor([5]))).toBe(5);
    });

    it("should handle symmetric errors", () => {
      const yTrue = tensor([0, 0, 0]);
      const yPred = tensor([5, -5, 3]);
      expect(maxError(yTrue, yPred)).toBe(5);
    });
  });

  describe("explainedVarianceScore", () => {
    it("should calculate explained variance correctly", () => {
      const yTrue = tensor([3, -0.5, 2, 7]);
      const yPred = tensor([2.5, 0.0, 2, 8]);
      expect(explainedVarianceScore(yTrue, yPred)).toBeCloseTo(0.9571734475374732);
    });

    it("should return 1 for perfect predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1, 2, 3, 4, 5]);
      expect(explainedVarianceScore(yTrue, yPred)).toBeCloseTo(1.0);
    });

    it("should handle constant predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const mean = 3;
      const yPred = tensor([mean, mean, mean, mean, mean]);
      const evs = explainedVarianceScore(yTrue, yPred);
      expect(evs).toBeCloseTo(0);
    });

    it("should handle negative values", () => {
      const yTrue = tensor([-5, -3, -1, 1, 3]);
      const yPred = tensor([-4, -2, 0, 2, 4]);
      const evs = explainedVarianceScore(yTrue, yPred);
      expect(evs).toBeGreaterThan(0);
    });

    it("should be similar to R² for unbiased predictions", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1.1, 2.1, 2.9, 4.1, 4.9]);
      const evs = explainedVarianceScore(yTrue, yPred);
      const r2 = r2Score(yTrue, yPred);
      expect(Math.abs(evs - r2)).toBeLessThan(0.1);
    });

    it("should reject empty input", () => {
      const empty = tensor([]);
      expect(() => explainedVarianceScore(empty, empty)).toThrow(/at least one sample/i);
    });
  });

  describe("Edge Cases and Error Handling", () => {
    it("should throw on size mismatch for all metrics", () => {
      const short = tensor([1, 2]);
      const long = tensor([1, 2, 3]);

      expect(() => mse(short, long)).toThrow();
      expect(() => rmse(short, long)).toThrow();
      expect(() => mae(short, long)).toThrow();
      expect(() => r2Score(short, long)).toThrow();
      expect(() => adjustedR2Score(short, long, 1)).toThrow();
      expect(() => mape(short, long)).toThrow();
      expect(() => medianAbsoluteError(short, long)).toThrow();
      expect(() => maxError(short, long)).toThrow();
      expect(() => explainedVarianceScore(short, long)).toThrow();
    });

    it("should reject non-numeric tensors", () => {
      const yTrue = tensor(["a", "b", "c"]);
      const yPred = tensor([1, 2, 3]);
      expect(() => mse(yTrue, yPred)).toThrow(/numeric/);
      expect(() => mae(yTrue, yPred)).toThrow(/numeric/);
      expect(() => r2Score(yTrue, yPred)).toThrow(/numeric/);
    });

    it("should reject int64 tensors", () => {
      const yTrue = tensor([1, 2, 3], { dtype: "int64" });
      const yPred = tensor([1, 2, 4], { dtype: "int64" });
      expect(() => mse(yTrue, yPred)).toThrow(/int64/);
      expect(() => mae(yTrue, yPred)).toThrow(/int64/);
      expect(() => r2Score(yTrue, yPred)).toThrow(/int64/);
    });

    it("should handle very small differences", () => {
      const yTrue = tensor([1.0000001, 2.0000001, 3.0000001]);
      const yPred = tensor([1.0, 2.0, 3.0]);

      const mseVal = mse(yTrue, yPred);
      const maeVal = mae(yTrue, yPred);

      expect(mseVal).toBeGreaterThan(0);
      expect(mseVal).toBeLessThan(1e-10);
      expect(maeVal).toBeGreaterThan(0);
      expect(maeVal).toBeLessThan(1e-6);
    });

    it("should handle very large values", () => {
      const yTrue = tensor([1e10, 2e10, 3e10]);
      const yPred = tensor([1.1e10, 2.1e10, 2.9e10]);

      const mseVal = mse(yTrue, yPred);
      const r2 = r2Score(yTrue, yPred);

      expect(Number.isFinite(mseVal)).toBe(true);
      expect(Number.isFinite(r2)).toBe(true);
    });

    it("should handle mixed scale values", () => {
      const yTrue = tensor([0.001, 1, 1000]);
      const yPred = tensor([0.002, 1.1, 1100]);

      const mseVal = mse(yTrue, yPred);
      const maeVal = mae(yTrue, yPred);

      expect(Number.isFinite(mseVal)).toBe(true);
      expect(Number.isFinite(maeVal)).toBe(true);
    });

    it("should handle all zeros", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yPred = tensor([0, 0, 0, 0]);

      expect(mse(yTrue, yPred)).toBe(0);
      expect(mae(yTrue, yPred)).toBe(0);
      expect(maxError(yTrue, yPred)).toBe(0);
    });

    it("should handle alternating signs", () => {
      const yTrue = tensor([1, -1, 1, -1]);
      const yPred = tensor([1.1, -0.9, 1.1, -0.9]);

      const mseVal = mse(yTrue, yPred);
      const maeVal = mae(yTrue, yPred);

      expect(mseVal).toBeCloseTo(0.01);
      expect(maeVal).toBeCloseTo(0.1);
    });

    it("should maintain numerical stability", () => {
      const yTrue = tensor([1e-10, 2e-10, 3e-10]);
      const yPred = tensor([1.1e-10, 2.1e-10, 2.9e-10]);

      const r2 = r2Score(yTrue, yPred);
      expect(Number.isFinite(r2)).toBe(true);
    });
  });

  describe("Performance Tests", () => {
    it("should handle large datasets efficiently", () => {
      const size = 200000;
      const yTrue = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i)
      );
      const yPred = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i + 0.5)
      );

      const start = Date.now();
      mse(yTrue, yPred);
      mae(yTrue, yPred);
      r2Score(yTrue, yPred);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds
    }, 20000);
  });

  describe("Consistency Tests", () => {
    it("should have MSE = RMSE²", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1.5, 2.5, 2.5, 4.5, 4.5]);

      const mseVal = mse(yTrue, yPred);
      const rmseVal = rmse(yTrue, yPred);

      expect(rmseVal * rmseVal).toBeCloseTo(mseVal);
    });

    it("should have MAE <= RMSE", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1.5, 2.5, 2.5, 4.5, 4.5]);

      const maeVal = mae(yTrue, yPred);
      const rmseVal = rmse(yTrue, yPred);

      expect(maeVal).toBeLessThanOrEqual(rmseVal);
    });

    it("should have maxError >= MAE", () => {
      const yTrue = tensor([1, 2, 3, 4, 5]);
      const yPred = tensor([1.5, 2.5, 2.5, 4.5, 4.5]);

      const maeVal = mae(yTrue, yPred);
      const maxErr = maxError(yTrue, yPred);

      expect(maxErr).toBeGreaterThanOrEqual(maeVal);
    });
  });

  describe("P0 Edge Cases and Error Handling", () => {
    describe("r2Score with constant targets", () => {
      it("should return 1.0 for perfect predictions on constant targets", () => {
        const yTrue = tensor([5, 5, 5, 5]);
        const yPred = tensor([5, 5, 5, 5]);

        const score = r2Score(yTrue, yPred);
        expect(score).toBe(1.0);
      });

      it("should return 0.0 for non-perfect predictions on constant targets", () => {
        const yTrue = tensor([5, 5, 5, 5]);
        const yPred = tensor([3, 4, 6, 7]);

        const score = r2Score(yTrue, yPred);
        expect(score).toBe(0.0);
      });
    });

    describe("explainedVarianceScore with constant targets", () => {
      it("should return 1.0 for perfect predictions on constant targets", () => {
        const yTrue = tensor([5, 5, 5, 5]);
        const yPred = tensor([5, 5, 5, 5]);

        const score = explainedVarianceScore(yTrue, yPred);
        expect(score).toBe(1.0);
      });

      it("should return 0.0 for non-perfect predictions on constant targets", () => {
        const yTrue = tensor([5, 5, 5, 5]);
        const yPred = tensor([3, 4, 6, 7]);

        const score = explainedVarianceScore(yTrue, yPred);
        expect(score).toBe(0.0);
      });
    });

    describe("adjustedR2Score validation", () => {
      it("should throw InvalidParameterError when n <= p + 1", () => {
        const yTrue = tensor([1, 2, 3]);
        const yPred = tensor([1.1, 2.1, 2.9]);
        const nFeatures = 2; // n=3, p=2, so n <= p+1

        expect(() => adjustedR2Score(yTrue, yPred, nFeatures)).toThrow("Adjusted R²");
      });

      it("should work when n > p + 1", () => {
        const yTrue = tensor([1, 2, 3, 4, 5]);
        const yPred = tensor([1.1, 2.1, 2.9, 4.2, 4.8]);
        const nFeatures = 2; // n=5, p=2, so n > p+1

        const score = adjustedR2Score(yTrue, yPred, nFeatures);
        expect(Number.isFinite(score)).toBe(true);
      });
    });

    describe("mape with zero values", () => {
      it("should skip zero values in yTrue", () => {
        const yTrue = tensor([1, 0, 2, 3]); // Contains zero
        const yPred = tensor([1.1, 0.5, 2.1, 3.2]);

        // Should compute MAPE only for non-zero values
        const error = mape(yTrue, yPred);
        expect(error).toBeGreaterThan(0);
        expect(Number.isFinite(error)).toBe(true);
      });

      it("should handle all zeros gracefully", () => {
        const yTrue = tensor([0, 0, 0, 0]);
        const yPred = tensor([1, 2, 3, 4]);

        // All zeros skipped, denominator becomes 0
        const error = mape(yTrue, yPred);
        expect(error).toBe(0); // (0 / 4) * 100 = 0
      });

      it("uses only non-zero targets in the denominator", () => {
        const yTrue = tensor([0, 2]);
        const yPred = tensor([10, 4]);
        expect(mape(yTrue, yPred)).toBe(100);
      });
    });

    describe("error type validation", () => {
      it("should throw ShapeError for size mismatch", () => {
        const yTrue = tensor([1, 2, 3]);
        const yPred = tensor([1, 2]);

        expect(() => mse(yTrue, yPred)).toThrow("size");
      });

      it("should throw DTypeError for string tensors", () => {
        const yTrue = tensor(["a", "b", "c"]);
        const yPred = tensor(["a", "b", "c"]);

        expect(() => mse(yTrue, yPred)).toThrow("numeric tensors");
      });
    });
  });
});
