import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  binaryCrossEntropyLoss,
  cosineEmbeddingLoss,
  ctcLoss,
  huberLoss,
  klDivLoss,
  maeLoss,
  mseLoss,
  nllLoss,
  rmseLoss,
  smoothL1Loss,
  tripletMarginLoss,
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

  describe("nllLoss", () => {
    it("should compute negative log likelihood for class targets", () => {
      const logProbs = tensor([
        [Math.log(0.1), Math.log(0.9)],
        [Math.log(0.8), Math.log(0.2)],
      ]);
      const targets = tensor([1, 0], { dtype: "int32" });

      const loss = nllLoss(logProbs, targets);

      expect(expectNumber(loss.toArray(), "nllLoss")).toBeCloseTo(
        (-Math.log(0.9) - Math.log(0.8)) / 2,
        5
      );
    });

    it("should reject out-of-range targets", () => {
      const logProbs = tensor([[Math.log(0.5), Math.log(0.5)]]);
      const targets = tensor([2], { dtype: "int32" });

      expect(() => nllLoss(logProbs, targets)).toThrow(/out of range/i);
    });
  });

  describe("klDivLoss", () => {
    it("should be near zero for matching distributions", () => {
      const probs = tensor([0.25, 0.75]);
      const input = tensor([Math.log(0.25), Math.log(0.75)]);

      const loss = klDivLoss(input, probs);

      expect(expectNumber(loss.toArray(), "klDivLoss")).toBeCloseTo(0, 6);
    });
  });

  describe("smoothL1Loss", () => {
    it("should compute Smooth L1 loss across quadratic and linear regions", () => {
      const predictions = tensor([0, 0]);
      const targets = tensor([0.5, 2]);

      const loss = smoothL1Loss(predictions, targets);

      expect(expectNumber(loss.toArray(), "smoothL1Loss")).toBeCloseTo((0.125 + 1.5) / 2, 5);
    });
  });

  describe("cosineEmbeddingLoss", () => {
    it("should return zero for identical positive pairs", () => {
      const x1 = tensor([[1, 0]]);
      const x2 = tensor([[1, 0]]);
      const y = tensor([1], { dtype: "int32" });

      const loss = cosineEmbeddingLoss(x1, x2, y);

      expect(expectNumber(loss.toArray(), "cosineEmbeddingLoss")).toBeCloseTo(0, 6);
    });
  });

  describe("tripletMarginLoss", () => {
    it("should compute positive margin violations", () => {
      const anchor = tensor([[0, 0]]);
      const positive = tensor([[1, 0]]);
      const negative = tensor([[1.5, 0]]);

      const loss = tripletMarginLoss(anchor, positive, negative);

      expect(expectNumber(loss.toArray(), "tripletMarginLoss")).toBeCloseTo(0.5, 6);
    });
  });

  describe("ctcLoss", () => {
    it("should compute loss for a simple single-target sequence", () => {
      const logProbs = tensor([
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
      ]);
      const targets = tensor([1], { dtype: "int32" });
      const inputLengths = tensor([2], { dtype: "int32" });
      const targetLengths = tensor([1], { dtype: "int32" });

      const loss = ctcLoss(logProbs, targets, inputLengths, targetLengths);

      expect(expectNumber(loss.toArray(), "ctcLoss")).toBeCloseTo(-Math.log(0.56), 5);
    });

    it("should support reduction none", () => {
      const logProbs = tensor([
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
      ]);
      const targets = tensor([1], { dtype: "int32" });
      const inputLengths = tensor([2], { dtype: "int32" });
      const targetLengths = tensor([1], { dtype: "int32" });

      const loss = ctcLoss(logProbs, targets, inputLengths, targetLengths, {
        reduction: "none",
      });

      const arr = expectNumberArray(loss.toArray(), "ctcLoss");
      expect(arr[0] ?? 0).toBeCloseTo(-Math.log(0.56), 5);
    });

    it("should return Infinity when the input is shorter than the target", () => {
      const logProbs = tensor([[[Math.log(0.5), Math.log(0.4), Math.log(0.1)]]]);
      const targets = tensor([1, 2], { dtype: "int32" });
      const inputLengths = tensor([1], { dtype: "int32" });
      const targetLengths = tensor([2], { dtype: "int32" });

      const loss = ctcLoss(logProbs, targets, inputLengths, targetLengths, {
        reduction: "none",
      });

      const arr = expectNumberArray(loss.toArray(), "ctcLoss");
      expect(arr[0]).toBe(Infinity);
    });

    it("should reject targets equal to the blank index", () => {
      const logProbs = tensor([
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
        [[Math.log(0.5), Math.log(0.4), Math.log(0.1)]],
      ]);
      const targets = tensor([0], { dtype: "int32" });
      const inputLengths = tensor([2], { dtype: "int32" });
      const targetLengths = tensor([1], { dtype: "int32" });

      expect(() => ctcLoss(logProbs, targets, inputLengths, targetLengths)).toThrow(/blank/i);
    });
  });
});
