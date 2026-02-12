import { describe, expect, it } from "vitest";
import { arange, full, geomspace, linspace, tensor } from "../src/ndarray";

describe("tensor creation error branches", () => {
  it("validates tensor data for dtype compatibility", () => {
    expect(() => tensor([1, 2], { dtype: "string" })).toThrow(/string tensor data/i);
    expect(() => tensor(["a", ["b"]])).toThrow(/string tensor data leaf values/i);
    expect(() => tensor([["a"], ["b", "c"]])).toThrow(/Ragged tensor/i);
    // @ts-expect-error - mixed number/string data should be rejected at runtime
    expect(() => tensor([1, "b"])).toThrow(/numbers/i);
    expect(() => tensor([[1], [2, 3]])).toThrow(/Ragged tensor/i);
    expect(() => tensor([1.2], { dtype: "int64" })).toThrow(/finite integers/i);
  });

  it("validates filled tensors and ranges", () => {
    expect(() => full([2, 2], 1, { dtype: "string" })).toThrow(/string fill value/i);
    expect(() => full([2, 2], "x", { dtype: "float32" })).toThrow(/Expected number fill value/i);

    expect(() => arange(0, 1, 0)).toThrow(/non-zero/i);
    expect(() => linspace(0, 1, -1)).toThrow(/non-negative/i);

    expect(() => geomspace(0, 1, 5)).toThrow(/non-zero/i);
    expect(() => geomspace(-1, 1, 5)).toThrow(/same sign/i);
  });
});
