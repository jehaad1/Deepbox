import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";

describe("deepbox/ndarray - Tensor core helpers", () => {
  it("should support at() indexing with negative indices", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(t.at(0, 1)).toBe(2);
    expect(t.at(-1, -1)).toBe(4);
  });

  it("should convert to nested arrays with toArray()", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(t.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});
