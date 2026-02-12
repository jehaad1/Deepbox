import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  binaryCrossEntropyLoss,
  huberLoss,
  maeLoss,
  mseLoss,
  rmseLoss,
} from "../src/nn/losses/index";
import { expectNumber, expectNumberArray } from "./nn-test-utils";

describe("deepbox/nn - Loss Functions", () => {
  describe("mseLoss", () => {
    it("should compute MSE correctly", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2, 3]);
      const loss = mseLoss(predictions, targets);
      // mean reduction returns scalar
      expect(expectNumber(loss.toArray(), "mseLoss")).toBeCloseTo(0, 5);
    });

    it("should compute MSE for non-zero error", () => {
      const predictions = tensor([0, 0, 0]);
      const targets = tensor([1, 2, 3]);
      const loss = mseLoss(predictions, targets);
      // MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
      expect(expectNumber(loss.toArray(), "mseLoss")).toBeCloseTo(14 / 3, 5);
    });

    it("should support sum reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([1, 2]);
      const loss = mseLoss(predictions, targets, "sum");
      // Sum = 1 + 4 = 5
      expect(expectNumber(loss.toArray(), "mseLoss")).toBeCloseTo(5, 5);
    });

    it("should support none reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([1, 2]);
      const loss = mseLoss(predictions, targets, "none");
      const arr = expectNumberArray(loss.toArray(), "mseLoss");
      expect(arr[0] ?? 0).toBeCloseTo(1, 5);
      expect(arr[1] ?? 0).toBeCloseTo(4, 5);
    });

    it("should throw on shape mismatch", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);
      expect(() => mseLoss(predictions, targets)).toThrow(/shape/i);
    });
  });

  describe("maeLoss", () => {
    it("should compute MAE correctly", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2, 3]);
      const loss = maeLoss(predictions, targets);
      // mean reduction returns scalar
      expect(expectNumber(loss.toArray(), "maeLoss")).toBeCloseTo(0, 5);
    });

    it("should compute MAE for non-zero error", () => {
      const predictions = tensor([0, 0, 0]);
      const targets = tensor([1, 2, 3]);
      const loss = maeLoss(predictions, targets);
      // MAE = (1 + 2 + 3) / 3 = 2
      expect(expectNumber(loss.toArray(), "maeLoss")).toBeCloseTo(2, 5);
    });

    it("should support sum reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([1, 2]);
      const loss = maeLoss(predictions, targets, "sum");
      // Sum = 1 + 2 = 3
      expect(expectNumber(loss.toArray(), "maeLoss")).toBeCloseTo(3, 5);
    });

    it("should support none reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([1, -2]);
      const loss = maeLoss(predictions, targets, "none");
      const arr = expectNumberArray(loss.toArray(), "maeLoss");
      expect(arr[0] ?? 0).toBeCloseTo(1, 5);
      expect(arr[1] ?? 0).toBeCloseTo(2, 5);
    });

    it("should throw on shape mismatch", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);
      expect(() => maeLoss(predictions, targets)).toThrow(/shape/i);
    });
  });

  describe("binaryCrossEntropyLoss", () => {
    it("should compute BCE for perfect predictions", () => {
      const predictions = tensor([0.999, 0.001]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets);
      // mean reduction returns scalar
      expect(expectNumber(loss.toArray(), "binaryCrossEntropyLoss")).toBeLessThan(0.01);
    });

    it("should compute BCE for wrong predictions", () => {
      const predictions = tensor([0.001, 0.999]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets);
      // mean reduction returns scalar, wrong predictions have high loss
      expect(expectNumber(loss.toArray(), "binaryCrossEntropyLoss")).toBeGreaterThan(5);
    });

    it("should support sum reduction", () => {
      const predictions = tensor([0.5, 0.5]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets, "sum");
      // -log(0.5) * 2 ≈ 1.386
      expect(expectNumber(loss.toArray(), "binaryCrossEntropyLoss")).toBeCloseTo(
        Math.log(2) * 2,
        3
      );
    });

    it("should support none reduction", () => {
      const predictions = tensor([0.5, 0.5]);
      const targets = tensor([1, 0]);
      const loss = binaryCrossEntropyLoss(predictions, targets, "none");
      const arr = expectNumberArray(loss.toArray(), "binaryCrossEntropyLoss");
      expect(arr[0] ?? 0).toBeCloseTo(Math.log(2), 3);
      expect(arr[1] ?? 0).toBeCloseTo(Math.log(2), 3);
    });

    it("should throw on shape mismatch", () => {
      const predictions = tensor([0.5, 0.5, 0.5]);
      const targets = tensor([1, 0]);
      expect(() => binaryCrossEntropyLoss(predictions, targets)).toThrow(/shape/i);
    });
  });

  describe("rmseLoss", () => {
    it("should compute RMSE correctly", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2, 3]);
      const loss = rmseLoss(predictions, targets);
      // RMSE returns a scalar tensor
      expect(expectNumber(loss.toArray(), "rmseLoss")).toBeCloseTo(0, 5);
    });

    it("should compute RMSE for non-zero error", () => {
      const predictions = tensor([0, 0, 0]);
      const targets = tensor([1, 2, 3]);
      const loss = rmseLoss(predictions, targets);
      // RMSE = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16
      expect(expectNumber(loss.toArray(), "rmseLoss")).toBeCloseTo(Math.sqrt(14 / 3), 5);
    });
  });

  describe("huberLoss", () => {
    it("should compute Huber loss with default delta", () => {
      const predictions = tensor([0]);
      const targets = tensor([0.5]);
      const loss = huberLoss(predictions, targets);
      // |error| = 0.5 <= delta=1, so quadratic: 0.5 * 0.5^2 = 0.125
      // mean reduction returns scalar
      expect(expectNumber(loss.toArray(), "huberLoss")).toBeCloseTo(0.125, 5);
    });

    it("should use linear region for large errors", () => {
      const predictions = tensor([0]);
      const targets = tensor([2]);
      const loss = huberLoss(predictions, targets, 1.0);
      // |error| = 2 > delta=1, so linear: 1 * (2 - 0.5 * 1) = 1.5
      expect(expectNumber(loss.toArray(), "huberLoss")).toBeCloseTo(1.5, 5);
    });

    it("should support custom delta", () => {
      const predictions = tensor([0]);
      const targets = tensor([1]);
      const loss = huberLoss(predictions, targets, 2.0);
      // |error| = 1 <= delta=2, so quadratic: 0.5 * 1^2 = 0.5
      expect(expectNumber(loss.toArray(), "huberLoss")).toBeCloseTo(0.5, 5);
    });

    it("should support sum reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([0.5, 0.5]);
      const loss = huberLoss(predictions, targets, 1.0, "sum");
      // 2 * 0.125 = 0.25
      expect(expectNumber(loss.toArray(), "huberLoss")).toBeCloseTo(0.25, 5);
    });

    it("should support none reduction", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([0.5, 2]);
      const loss = huberLoss(predictions, targets, 1.0, "none");
      const arr = expectNumberArray(loss.toArray(), "huberLoss");
      expect(arr[0] ?? 0).toBeCloseTo(0.125, 5);
      expect(arr[1] ?? 0).toBeCloseTo(1.5, 5);
    });

    it("should throw on invalid delta", () => {
      const predictions = tensor([1]);
      const targets = tensor([1]);
      expect(() => huberLoss(predictions, targets, 0)).toThrow(/delta/i);
      expect(() => huberLoss(predictions, targets, -1)).toThrow(/delta/i);
    });

    it("should throw on shape mismatch", () => {
      const predictions = tensor([1, 2, 3]);
      const targets = tensor([1, 2]);
      expect(() => huberLoss(predictions, targets)).toThrow(/shape/i);
    });
  });
});
