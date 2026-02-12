import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { argsort, sort } from "../src/ndarray/ops/sorting";

describe("ndarray sorting extra branches", () => {
  it("sorts descending with axis undefined", () => {
    const t = tensor([3, 1, 2]);
    expect(sort(t, undefined, true).toArray()).toEqual([3, 2, 1]);
    expect(argsort(t, undefined, true).toArray()).toEqual([0, 2, 1]);
  });

  it("sorts BigInt tensors (ascending and descending)", () => {
    const t = tensor([3, 1, 2], { dtype: "int64" });
    expect(sort(t).toArray()).toEqual([1n, 2n, 3n]);
    expect(sort(t, -1, true).toArray()).toEqual([3n, 2n, 1n]);
    expect(argsort(t).toArray()).toEqual([1, 2, 0]);
    expect(argsort(t, -1, true).toArray()).toEqual([0, 2, 1]);
  });
});
