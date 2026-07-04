import { describe, expect, it } from "vitest";
import { type Tensor, tensor } from "../src/ndarray";
import { gaussianNLLLoss, poissonNLLLoss } from "../src/nn/losses";

function scalarVal(t: Tensor): number {
  const d = t.data;
  if (Array.isArray(d)) throw new Error("string tensor");
  return Number(d[0]);
}

describe("gaussianNLLLoss", () => {
  it("computes basic loss", () => {
    const input = tensor([0.5, 1.0, 1.5]);
    const target = tensor([0.5, 1.0, 1.5]);
    const variance = tensor([1.0, 1.0, 1.0]);
    const loss = gaussianNLLLoss(input, target, variance);
    // When input == target, loss = 0.5 * log(var) = 0
    expect(scalarVal(loss)).toBeCloseTo(0, 4);
  });

  it("loss increases with larger error", () => {
    const variance = tensor([1.0, 1.0]);
    const loss1 = gaussianNLLLoss(tensor([1.0, 1.0]), tensor([1.0, 1.0]), variance);
    const loss2 = gaussianNLLLoss(tensor([2.0, 2.0]), tensor([1.0, 1.0]), variance);
    expect(scalarVal(loss2) > scalarVal(loss1)).toBe(true);
  });

  it("supports reduction=none", () => {
    const input = tensor([0.5, 1.0]);
    const target = tensor([0.0, 0.0]);
    const variance = tensor([1.0, 1.0]);
    const loss = gaussianNLLLoss(input, target, variance, {
      reduction: "none",
    });
    expect(loss.shape).toEqual([2]);
  });

  it("supports reduction=sum", () => {
    const input = tensor([0.5, 1.0]);
    const target = tensor([0.0, 0.0]);
    const variance = tensor([1.0, 1.0]);
    const loss = gaussianNLLLoss(input, target, variance, {
      reduction: "sum",
    });
    expect(loss.shape).toEqual([]);
  });

  it("includes log(2π) term when full=true", () => {
    const input = tensor([1.0]);
    const target = tensor([1.0]);
    const variance = tensor([1.0]);
    const lossPartial = gaussianNLLLoss(input, target, variance, {
      full: false,
    });
    const lossFull = gaussianNLLLoss(input, target, variance, { full: true });
    const diff = scalarVal(lossFull) - scalarVal(lossPartial);
    expect(diff).toBeCloseTo(0.5 * Math.log(2 * Math.PI), 4);
  });

  it("clamps small variance to eps", () => {
    const input = tensor([1.0]);
    const target = tensor([0.0]);
    const variance = tensor([0.0]); // would cause log(0) without clamping
    expect(() => gaussianNLLLoss(input, target, variance)).not.toThrow();
    const loss = gaussianNLLLoss(input, target, variance);
    expect(Number.isFinite(scalarVal(loss))).toBe(true);
  });

  it("throws on shape mismatch", () => {
    expect(() => gaussianNLLLoss(tensor([1, 2]), tensor([1]), tensor([1, 2]))).toThrow(/Shape/);
  });
});

describe("poissonNLLLoss", () => {
  it("computes basic loss with logInput=true", () => {
    // loss = exp(input) - target * input
    const input = tensor([0.0]); // exp(0) = 1
    const target = tensor([1.0]);
    const loss = poissonNLLLoss(input, target, { logInput: true });
    // exp(0) - 1*0 = 1
    expect(scalarVal(loss)).toBeCloseTo(1.0, 4);
  });

  it("computes loss with logInput=false", () => {
    // loss = input - target * log(input + eps)
    const input = tensor([1.0]);
    const target = tensor([1.0]);
    const loss = poissonNLLLoss(input, target, { logInput: false });
    // 1 - 1 * log(1 + eps) ≈ 1 - 0 = 1 (approx since eps is tiny)
    expect(scalarVal(loss)).toBeCloseTo(1.0, 2);
  });

  it("supports reduction=none", () => {
    const input = tensor([0.5, 1.0, 1.5]);
    const target = tensor([1.0, 2.0, 3.0]);
    const loss = poissonNLLLoss(input, target, { reduction: "none" });
    expect(loss.shape).toEqual([3]);
  });

  it("supports reduction=sum", () => {
    const input = tensor([0.5, 1.0]);
    const target = tensor([1.0, 2.0]);
    const loss = poissonNLLLoss(input, target, { reduction: "sum" });
    expect(loss.shape).toEqual([]);
  });

  it("full option adds Stirling approximation", () => {
    const input = tensor([1.0]);
    const target = tensor([5.0]); // > 1 to trigger Stirling
    const lossPartial = poissonNLLLoss(input, target, { full: false });
    const lossFull = poissonNLLLoss(input, target, { full: true });
    expect(scalarVal(lossFull) > scalarVal(lossPartial)).toBe(true);
  });

  it("loss is non-negative for reasonable inputs", () => {
    // For target=0 and logInput=true: loss = exp(input) >= 0
    const input = tensor([0.0, 1.0, 2.0]);
    const target = tensor([0.0, 0.0, 0.0]);
    const loss = poissonNLLLoss(input, target, { reduction: "none" });
    for (let i = 0; i < 3; i++) {
      expect(loss.at(i) as number).toBeGreaterThanOrEqual(0);
    }
  });

  it("throws on shape mismatch", () => {
    expect(() => poissonNLLLoss(tensor([1, 2]), tensor([1]))).toThrow(/Shape/);
  });
});
