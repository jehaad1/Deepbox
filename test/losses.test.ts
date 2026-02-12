import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { binaryCrossEntropyLoss, huberLoss, maeLoss, mseLoss, rmseLoss } from "../src/nn";

describe("deepbox/nn - Loss Functions", () => {
  describe("mseLoss", () => {
    it("should compute MSE for identical predictions and targets", () => {
      const predictions = tensor([1, 2, 3, 4]);
      const targets = tensor([1, 2, 3, 4]);
      const loss = mseLoss(predictions, targets);

      expect(loss.size).toBe(1);
      expect(loss.data[0]).toBeCloseTo(0, 5);
    });

    it("should compute MSE for different predictions and targets", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = mseLoss(predictions, targets);

      // MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1 + 1 + 1) = 1
      expect(loss.data[0]).toBeCloseTo(1, 5);
    });

    it("should handle 2D tensors", () => {
      const predictions = tensor([
        [1, 2],
        [3, 4],
      ]);
      const targets = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loss = mseLoss(predictions, targets);

      expect(loss.data[0]).toBeCloseTo(0, 5);
    });

    it("should support reduction='none'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = mseLoss(predictions, targets, "none");

      expect(loss.shape).toEqual([3]);
      expect(loss.data[0]).toBeCloseTo(1, 5); // (1-2)^2 = 1
      expect(loss.data[1]).toBeCloseTo(1, 5); // (2-3)^2 = 1
      expect(loss.data[2]).toBeCloseTo(1, 5); // (3-4)^2 = 1
    });

    it("should support reduction='sum'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = mseLoss(predictions, targets, "sum");

      // Sum = (1-2)^2 + (2-3)^2 + (3-4)^2 = 3
      expect(loss.data[0]).toBeCloseTo(3, 5);
    });

    it("should throw error for mismatched shapes", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);

      expect(() => mseLoss(predictions, targets)).toThrow("Shape mismatch");
    });

    it("should penalize large errors more heavily", () => {
      const pred1 = tensor([1]);
      const target1 = tensor([2]);
      const loss1 = mseLoss(pred1, target1);

      const pred2 = tensor([1]);
      const target2 = tensor([3]);
      const loss2 = mseLoss(pred2, target2);

      // Error of 2 should have 4x the loss of error of 1
      const ratio = Number(loss2.data[0]) / Number(loss1.data[0]);
      expect(ratio).toBeCloseTo(4, 1);
    });
  });

  describe("maeLoss", () => {
    it("should compute MAE for identical predictions and targets", () => {
      const predictions = tensor([1, 2, 3, 4]);
      const targets = tensor([1, 2, 3, 4]);
      const loss = maeLoss(predictions, targets);

      expect(loss.data[0]).toBeCloseTo(0, 5);
    });

    it("should compute MAE for different predictions and targets", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = maeLoss(predictions, targets);

      // MAE = mean(|1-2| + |2-3| + |3-4|) = mean(1 + 1 + 1) = 1
      expect(loss.data[0]).toBeCloseTo(1, 5);
    });

    it("should handle negative differences", () => {
      const predictions = tensor([5, 6, 7]);
      const targets = tensor([2, 3, 4]);
      const loss = maeLoss(predictions, targets);

      // MAE = mean(|5-2| + |6-3| + |7-4|) = mean(3 + 3 + 3) = 3
      expect(loss.data[0]).toBeCloseTo(3, 5);
    });

    it("should support reduction='none'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = maeLoss(predictions, targets, "none");

      expect(loss.shape).toEqual([3]);
    });

    it("should support reduction='sum'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = maeLoss(predictions, targets, "sum");

      // Sum = |1-2| + |2-3| + |3-4| = 3
      expect(loss.data[0]).toBeCloseTo(3, 5);
    });

    it("should throw error for mismatched shapes", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);

      expect(() => maeLoss(predictions, targets)).toThrow("Shape mismatch");
    });

    it("should have linear penalty for errors", () => {
      const pred1 = tensor([1]);
      const target1 = tensor([2]);
      const loss1 = maeLoss(pred1, target1);

      const pred2 = tensor([1]);
      const target2 = tensor([3]);
      const loss2 = maeLoss(pred2, target2);

      // Error of 2 should have 2x the loss of error of 1 (linear)
      const ratio = Number(loss2.data[0]) / Number(loss1.data[0]);
      expect(ratio).toBeCloseTo(2, 1);
    });
  });

  describe("binaryCrossEntropyLoss", () => {
    it("should compute BCE for perfect predictions", () => {
      const predictions = tensor([0.99, 0.01]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets);

      // Should be close to 0 for perfect predictions
      expect(loss.data[0]).toBeLessThan(0.1);
    });

    it("should compute BCE for wrong predictions", () => {
      const predictions = tensor([0.01, 0.99]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets);

      // Should be high for wrong predictions
      expect(loss.data[0]).toBeGreaterThan(2);
    });

    it("should handle probabilities of 0.5", () => {
      const predictions = tensor([0.5, 0.5]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets);

      // BCE at 0.5 should be -log(0.5) ≈ 0.693
      expect(loss.data[0]).toBeCloseTo(Math.LN2, 1);
    });

    it("should support reduction='none'", () => {
      const predictions = tensor([0.9, 0.1]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets, "none");

      expect(loss.shape).toEqual([2]);
    });

    it("should support reduction='sum'", () => {
      const predictions = tensor([0.9, 0.1]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets, "sum");

      expect(loss.size).toBe(1);
    });

    it("should throw error for mismatched shapes", () => {
      const predictions = tensor([0.5, 0.5]);
      const targets = tensor([1]);

      expect(() => binaryCrossEntropyLoss(predictions, targets)).toThrow("Shape mismatch");
    });

    it("should handle batch predictions", () => {
      const predictions = tensor([
        [0.9, 0.1],
        [0.2, 0.8],
      ]);
      const targets = tensor([
        [1, 0],
        [0, 1],
      ]);
      const loss = binaryCrossEntropyLoss(predictions, targets);

      expect(loss.size).toBe(1);
    });
  });

  describe("rmseLoss", () => {
    it("should compute RMSE for identical predictions and targets", () => {
      const predictions = tensor([1, 2, 3, 4]);
      const targets = tensor([1, 2, 3, 4]);
      const loss = rmseLoss(predictions, targets);

      expect(loss.data[0]).toBeCloseTo(0, 5);
    });

    it("should compute RMSE correctly", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = rmseLoss(predictions, targets);

      // RMSE = sqrt(MSE) = sqrt(1) = 1
      expect(loss.data[0]).toBeCloseTo(1, 5);
    });

    it("should be square root of MSE", () => {
      const predictions = tensor([1, 2, 3, 4]);
      const targets = tensor([2, 4, 6, 8]);

      const mse = mseLoss(predictions, targets);
      const rmse = rmseLoss(predictions, targets);

      const mseVal = Number(mse.data[0]);
      const rmseVal = Number(rmse.data[0]);

      expect(rmseVal).toBeCloseTo(Math.sqrt(mseVal), 5);
    });

    it("should handle 2D tensors", () => {
      const predictions = tensor([
        [1, 2],
        [3, 4],
      ]);
      const targets = tensor([
        [2, 3],
        [4, 5],
      ]);
      const loss = rmseLoss(predictions, targets);

      expect(loss.size).toBe(1);
    });
  });

  describe("huberLoss", () => {
    it("should compute Huber loss for identical predictions and targets", () => {
      const predictions = tensor([1, 2, 3, 4]);
      const targets = tensor([1, 2, 3, 4]);
      const loss = huberLoss(predictions, targets);

      expect(loss.data[0]).toBeCloseTo(0, 5);
    });

    it("should compute Huber loss with default delta", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = huberLoss(predictions, targets);

      expect(loss.size).toBe(1);
    });

    it("should compute Huber loss with custom delta", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = huberLoss(predictions, targets, 0.5);

      expect(loss.size).toBe(1);
    });

    it("should support reduction='none'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = huberLoss(predictions, targets, 1.0, "none");

      expect(loss.shape).toEqual([3]);
    });

    it("should support reduction='sum'", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([2, 3, 4]);
      const loss = huberLoss(predictions, targets, 1.0, "sum");

      expect(loss.size).toBe(1);
    });

    it("should throw error for mismatched shapes", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);

      expect(() => huberLoss(predictions, targets)).toThrow("Shape mismatch");
    });

    it("should handle 2D tensors", () => {
      const predictions = tensor([
        [1, 2],
        [3, 4],
      ]);
      const targets = tensor([
        [2, 3],
        [4, 5],
      ]);
      const loss = huberLoss(predictions, targets);

      expect(loss.size).toBe(1);
    });
  });

  describe("edge cases", () => {
    it("should handle single element tensors", () => {
      const predictions = tensor([1]);
      const targets = tensor([2]);

      expect(() => mseLoss(predictions, targets)).not.toThrow();
      expect(() => maeLoss(predictions, targets)).not.toThrow();
      expect(() => rmseLoss(predictions, targets)).not.toThrow();
      expect(() => huberLoss(predictions, targets)).not.toThrow();
    });

    it("should handle large tensors", () => {
      const size = 1000;
      const predictions = tensor(new Array(size).fill(1));
      const targets = tensor(new Array(size).fill(2));

      expect(() => mseLoss(predictions, targets)).not.toThrow();
      expect(() => maeLoss(predictions, targets)).not.toThrow();
    });

    it("should handle zero predictions and targets", () => {
      const predictions = tensor([0, 0, 0]);
      const targets = tensor([0, 0, 0]);

      const mse = mseLoss(predictions, targets);
      const mae = maeLoss(predictions, targets);

      expect(mse.data[0]).toBe(0);
      expect(mae.data[0]).toBe(0);
    });
  });
});
