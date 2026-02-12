import { describe, expect, it } from "vitest";
import { repeat, tensor } from "../src/ndarray";

describe("ndarray manipulation extra branches (repeat)", () => {
  it("repeats string tensors along axis", () => {
    const s = tensor([
      ["a", "b"],
      ["c", "d"],
    ]);
    const out = repeat(s, 2, 0);
    expect(out.toArray()).toEqual([
      ["a", "b"],
      ["a", "b"],
      ["c", "d"],
      ["c", "d"],
    ]);
  });

  it("repeats BigInt tensors along axis", () => {
    const t = tensor([1, 2], { dtype: "int64" });
    const out = repeat(t, 3, 0);
    expect(out.toArray()).toEqual([1n, 1n, 1n, 2n, 2n, 2n]);
  });
});
