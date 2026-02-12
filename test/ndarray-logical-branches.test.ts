import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { logicalAnd, logicalNot, logicalOr, logicalXor } from "../src/ndarray/ops/logical";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray logical branch coverage", () => {
  it("handles scalar broadcasting and basic logical ops", () => {
    const a = tensor([1, 0, 1]);
    const b = tensor(1);
    expect(logicalAnd(a, b).toArray()).toEqual([1, 0, 1]);
    expect(logicalOr(a, b).toArray()).toEqual([1, 1, 1]);
    expect(logicalXor(a, b).toArray()).toEqual([0, 1, 0]);
    expect(logicalNot(a).toArray()).toEqual([0, 1, 0]);
  });

  it("handles non-scalar broadcasting", () => {
    const a = tensor([
      [1, 0, 1],
      [0, 1, 0],
    ]);
    const b = tensor([1, 0, 1]);
    expect(logicalAnd(a, b).shape).toEqual([2, 3]);
    expect(logicalOr(a, b).shape).toEqual([2, 3]);
    expect(logicalXor(a, b).shape).toEqual([2, 3]);
    expect(logicalAnd(a, b).toArray()).toEqual([
      [1, 0, 1],
      [0, 0, 0],
    ]);
  });

  it("throws for invalid shapes and string dtype", () => {
    const a = tensor([1, 2]);
    const b = tensor([1, 2, 3]);
    expect(() => logicalAnd(a, b)).toThrow(/broadcast/i);
    expect(() => logicalOr(a, b)).toThrow(/broadcast/i);
    expect(() => logicalXor(a, b)).toThrow(/broadcast/i);

    const s = tensor(["a", "b"]);
    expect(() => logicalAnd(s, tensor([1, 0]))).toThrow(/string dtype/i);
    expect(() => logicalOr(s, tensor([1, 0]))).toThrow(/string dtype/i);
    expect(() => logicalXor(s, tensor([1, 0]))).toThrow(/string dtype/i);
    expect(() => logicalNot(s)).toThrow(/string dtype/i);
  });

  it("broadcasts zero-length dimensions correctly", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const b = tensor([1, 0, 1], { dtype: "float32" });
    const out = logicalAnd(a, b);
    expect(out.shape).toEqual([0, 3]);
    expect(out.size).toBe(0);
  });

  it("rejects incompatible zero-length broadcasting", () => {
    const zeroRows = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const twoRows = tensor(
      [
        [1, 0, 1],
        [0, 1, 0],
      ],
      { dtype: "float32" }
    );
    expect(() => logicalAnd(zeroRows, twoRows)).toThrow(/broadcast/i);
    expect(() => logicalOr(zeroRows, twoRows)).toThrow(/broadcast/i);
    expect(() => logicalXor(zeroRows, twoRows)).toThrow(/broadcast/i);
  });
});
