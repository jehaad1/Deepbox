import { describe, expect, it } from "vitest";
import { inv, pinv } from "../src/linalg";
import { tensor, zeros } from "../src/ndarray";

describe("linalg inverse extra branches", () => {
  it("inv validates shapes and empty matrices", () => {
    expect(() => inv(tensor([1, 2, 3]))).toThrow(/2D/);
    expect(() => inv(tensor([[1, 2, 3]]))).toThrow(/square/);

    const empty = zeros([0, 0]);
    const out = inv(empty);
    expect(out.shape).toEqual([0, 0]);
  });

  it("pinv validates rcond and handles empty matrices", () => {
    expect(() => pinv(tensor([[1, 2]]), -1)).toThrow(/rcond/);
    const empty = zeros([0, 3]);
    const out = pinv(empty);
    expect(out.shape).toEqual([3, 0]);
  });
});
