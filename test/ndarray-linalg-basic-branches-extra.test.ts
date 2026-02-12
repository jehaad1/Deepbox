import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { dot, matmul } from "../src/ndarray/linalg/basic";

describe("ndarray linalg basic extra branches", () => {
  it("matmul handles BigInt and shape errors", () => {
    const a = tensor([[1, 2]], { dtype: "int64" });
    const b = tensor([[3], [4]], { dtype: "int64" });
    const out = matmul(a, b);
    expect(out.toArray()).toEqual([[11n]]);

    expect(() => matmul(tensor([1, 2]), tensor([1, 2]))).toThrow(/2D/);
    expect(() => matmul(tensor([[1, 2]]), tensor([[1, 2]]))).toThrow(/matmul/);
  });

  it("dot handles BigInt and dtype errors", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([3, 4], { dtype: "int64" });
    const out = dot(a, b);
    expect(out.toArray()).toEqual(11n);

    expect(() => dot(tensor([[1, 2]]), tensor([[1, 2]]))).toThrow(/1D/);
    expect(() => dot(tensor([1, 2]), tensor([1, 2, 3]))).toThrow(/dot/);
  });

  it("throws on dtype mismatch", () => {
    const a = tensor([[1, 2]], { dtype: "float32" });
    const b = tensor([[1], [2]], { dtype: "int32" });
    expect(() => matmul(a, b)).toThrow(/matching dtypes/i);

    const v1 = tensor([1, 2], { dtype: "float32" });
    const v2 = tensor([1, 2], { dtype: "int32" });
    expect(() => dot(v1, v2)).toThrow(/matching dtypes/i);
  });
});
