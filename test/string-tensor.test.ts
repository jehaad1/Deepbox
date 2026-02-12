import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";

describe("String Tensor", () => {
  it("should create a 1D string tensor", () => {
    const t = tensor(["cat", "dog", "cat"]);
    expect(t.dtype).toBe("string");
    expect(t.shape).toEqual([3]);
    expect(String(t.data[t.offset + 0])).toBe("cat");
  });

  it("should create a 2D string tensor", () => {
    const t = tensor([
      ["a", "b"],
      ["c", "d"],
    ]);
    expect(t.dtype).toBe("string");
    expect(t.shape).toEqual([2, 2]);
    expect(String(t.data[t.offset + 3])).toBe("d");
  });
});
