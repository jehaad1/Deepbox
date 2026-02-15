import { describe, expect, it } from "vitest";
import { GradTensor, parameter, tensor } from "../src/ndarray";
import { binaryCrossEntropyWithLogitsLoss, crossEntropyLoss } from "../src/nn/losses/crossEntropy";
import { expectNumberArray2D } from "./nn-test-utils";

describe("deepbox/nn - Cross Entropy Losses", () => {
  it("computes cross entropy loss for class indices", () => {
    const yPred = tensor([
      [2, 1, 0],
      [0, 2, 1],
    ]);
    const yTrue = tensor([0, 1]);
    const loss = crossEntropyLoss(yPred, yTrue);
    expect(loss).toBeGreaterThan(0);
  });

  it("computes binary cross entropy with logits", () => {
    const yPred = tensor([0.5, -0.3, 1.2]);
    const yTrue = tensor([1, 0, 1]);
    const loss = binaryCrossEntropyWithLogitsLoss(yPred, yTrue);
    expect(loss).toBeGreaterThan(0);
  });

  it("returns GradTensor for crossEntropyLoss", () => {
    const yPred = parameter([
      [2, 1, 0],
      [0, 2, 1],
    ]);
    const yTrue = GradTensor.fromTensor(
      tensor(
        [
          [1, 0, 0],
          [0, 1, 0],
        ],
        { dtype: yPred.tensor.dtype }
      ),
      { requiresGrad: false }
    );
    const loss = crossEntropyLoss(yPred, yTrue);
    expect(loss.tensor.size).toBe(1);
    expect(loss.requiresGrad).toBe(true);
  });

  it("validates input shapes", () => {
    const yPred = tensor([1, 2, 3]);
    const yTrue = tensor([0, 1]);
    expect(() => crossEntropyLoss(yPred, yTrue)).toThrow();
  });

  it("throws on mismatched or invalid dimensions", () => {
    const yPred = tensor([
      [0.2, 0.8],
      [0.7, 0.3],
    ]);
    const yTrueBad = tensor([[0, 1]]);
    expect(() => crossEntropyLoss(yPred, yTrueBad)).toThrow(/1-dimensional/i);
    expect(() => crossEntropyLoss(yPred, tensor([0]))).toThrow(/same number of samples/i);

    expect(() => binaryCrossEntropyWithLogitsLoss(tensor([[[1, 2, 3]]]), tensor([1]))).toThrow(
      /1 or 2-dimensional/i
    );
    expect(() => binaryCrossEntropyWithLogitsLoss(tensor([1, 2, 3]), tensor([[1, 0, 1]]))).toThrow(
      /1-dimensional/i
    );
  });

  it("throws on invalid class indices", () => {
    const yPred = tensor([
      [2, 1, 0],
      [0, 2, 1],
    ]);
    expect(() => crossEntropyLoss(yPred, tensor([3, 0]))).toThrow(/out of range/i);
    expect(() => crossEntropyLoss(yPred, tensor([0.5, 1]))).toThrow(/integer/i);
  });

  it("supports binary cross entropy with 2D logits", () => {
    const yPred = tensor([[0.5], [-0.3], [1.2]]);
    const yTrue = tensor([1, 0, 1]);
    const loss = binaryCrossEntropyWithLogitsLoss(yPred, yTrue);
    expect(loss).toBeGreaterThan(0);
  });

  it("throws on invalid binary logits shape", () => {
    const yPred = tensor([
      [0.5, -0.3],
      [1.2, 0.7],
    ]);
    const yTrue = tensor([1, 0]);
    expect(() => binaryCrossEntropyWithLogitsLoss(yPred, yTrue)).toThrow(/shape/i);
  });

  it("validates crossEntropyLoss inputs", () => {
    const yPred = parameter([
      [1, 2],
      [3, 4],
    ]);
    // 1D GradTensor targets (class indices) are now accepted — no longer throws
    const yTrue1D = GradTensor.fromTensor(tensor([0, 1], { dtype: yPred.tensor.dtype }), {
      requiresGrad: false,
    });
    expect(() => crossEntropyLoss(yPred, yTrue1D)).not.toThrow();

    const yTrueWrongShape = GradTensor.fromTensor(tensor([[1, 0]], { dtype: yPred.tensor.dtype }), {
      requiresGrad: false,
    });
    expect(() => crossEntropyLoss(yPred, yTrueWrongShape)).toThrow(/same shape/i);
  });

  it("propagates gradients in crossEntropyLoss", () => {
    const yPred = parameter([
      [2, 1],
      [0, 3],
    ]);
    const yTrue = GradTensor.fromTensor(
      tensor(
        [
          [1, 0],
          [0, 1],
        ],
        { dtype: yPred.tensor.dtype }
      ),
      { requiresGrad: false }
    );
    const loss = crossEntropyLoss(yPred, yTrue);
    loss.backward();
    const grad = yPred.grad;
    expect(grad).not.toBeNull();
    if (grad) {
      expect(grad.size).toBe(4);
      const values = expectNumberArray2D(grad.toArray(), "crossEntropyLoss");
      expect(values[0]?.[0]).not.toBe(0);
      expect(values[1]?.[1]).not.toBe(0);
    }
  });
});
