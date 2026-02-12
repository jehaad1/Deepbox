import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  allclose,
  arrayEqual,
  isclose,
  isfinite,
  isinf,
  isnan,
} from "../src/ndarray/ops/comparison";

describe("ndarray comparison extra branches", () => {
  it("handles scalar broadcasting for isclose", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor(2);
    expect(isclose(a, b).toArray()).toEqual([0, 1, 0]);
  });

  it("allclose returns false on size mismatch", () => {
    const a = tensor([1, 2]);
    const b = tensor([1, 2, 3]);
    expect(allclose(a, b)).toBe(false);
  });

  it("arrayEqual covers dtype/shape mismatch and string equality", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1, 2, 3], { dtype: "int32" });
    expect(arrayEqual(a, b)).toBe(false);

    const c = tensor([[1, 2, 3]]);
    expect(arrayEqual(a, c)).toBe(false);

    const s1 = tensor(["a", "b"]);
    const s2 = tensor(["a", "b"]);
    const s3 = tensor(["a", "c"]);
    expect(arrayEqual(s1, s2)).toBe(true);
    expect(arrayEqual(s1, s3)).toBe(false);
  });

  it("finite/inf/nan checks on BigInt tensors", () => {
    const big = tensor([1, 2, 3], { dtype: "int64" });
    expect(isnan(big).toArray()).toEqual([0, 0, 0]);
    expect(isinf(big).toArray()).toEqual([0, 0, 0]);
    expect(isfinite(big).toArray()).toEqual([1, 1, 1]);
  });
});
