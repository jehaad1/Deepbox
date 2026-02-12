import { describe, expect, it } from "vitest";
import { slice, squeeze, tensor, unsqueeze } from "../src/ndarray";

describe("deepbox/ndarray - Indexing & Shape Ops", () => {
  it("slice() should return a scalar when all axes are indexed", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);

    const s = slice(t, 0, 0);
    expect(s.shape).toEqual([]);
    expect(s.size).toBe(1);
    expect(Number(s.data[s.offset])).toBe(1);
  });

  it("squeeze() should return a scalar shape [] when all dims are removed", () => {
    const t = tensor([[[5]]]);
    const s = squeeze(t);
    expect(s.shape).toEqual([]);
    expect(s.size).toBe(1);
    expect(Number(s.data[s.offset])).toBe(5);
  });

  it("unsqueeze() should insert a dimension of size 1", () => {
    const t = tensor([1, 2, 3]);
    const u = unsqueeze(t, 0);
    expect(u.shape).toEqual([1, 3]);
    expect(u.size).toBe(3);
  });
});
