import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { gather, slice } from "../src/ndarray/tensor/indexing";
import { expandDims, squeeze, unsqueeze } from "../src/ndarray/tensor/shapeOps";

describe("deepbox/ndarray - Tensor Extra", () => {
  it("creates string tensors", () => {
    const t = tensor(["a", "b"]);
    expect(t.dtype).toBe("string");
    expect(t.toArray()).toEqual(["a", "b"]);
  });

  it("rejects ragged arrays", () => {
    expect(() => tensor([[1, 2], [3]])).toThrow();
  });

  it("rejects non-integer values for int64", () => {
    expect(() => tensor([1.5], { dtype: "int64" })).toThrow();
  });

  it("squeezes and unsqueezes dimensions", () => {
    const t = tensor([[[1], [2], [3]]]);
    const s = squeeze(t);
    expect(s.shape).toEqual([3]);
    const u = unsqueeze(s, 0);
    expect(u.shape).toEqual([1, 3]);
    const e = expandDims(s, -1);
    expect(e.shape).toEqual([3, 1]);
  });

  it("squeeze validates axis", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => squeeze(t, 0)).toThrow();
  });

  it("slice supports ranges and scalar indexing", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const sliced = slice(t, { start: 0, end: 2 }, { start: 1, end: 3 });
    expect(sliced.toArray()).toEqual([
      [2, 3],
      [5, 6],
    ]);

    const scalar = slice(t, 1, 2);
    expect(scalar.shape).toEqual([]);
    expect(scalar.toArray()).toEqual(6);
  });

  it("gather indexes along axis", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const idx = tensor([0, 2]);
    const gathered = gather(t, idx, 0);
    expect(gathered.toArray()).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });
});
