import { describe, expect, it } from "vitest";
import { gcd, intersect1d, lcm, setdiff1d, tensor, union1d } from "../src/ndarray";

describe("gcd", () => {
  it("computes element-wise GCD", () => {
    const a = tensor([12, 15, 20]);
    const b = tensor([8, 10, 25]);
    const result = gcd(a, b);
    expect(result.toArray()).toEqual([4, 5, 5]);
    expect(result.dtype).toBe("int32");
  });

  it("handles zeros", () => {
    const a = tensor([0, 6, 0]);
    const b = tensor([5, 0, 0]);
    const result = gcd(a, b);
    expect(result.toArray()).toEqual([5, 6, 0]);
  });

  it("handles negatives", () => {
    const a = tensor([-12, 15]);
    const b = tensor([8, -10]);
    const result = gcd(a, b);
    expect(result.toArray()).toEqual([4, 5]);
  });

  it("works on 2D tensors", () => {
    const a = tensor([
      [12, 18],
      [24, 30],
    ]);
    const b = tensor([
      [8, 12],
      [16, 20],
    ]);
    const result = gcd(a, b);
    expect(result.toArray()).toEqual([
      [4, 6],
      [8, 10],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("throws on string dtype", () => {
    const a = tensor(["a"]);
    const b = tensor(["b"]);
    expect(() => gcd(a, b)).toThrow();
  });

  it("throws on size mismatch", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1, 2]);
    expect(() => gcd(a, b)).toThrow();
  });
});

describe("lcm", () => {
  it("computes element-wise LCM", () => {
    const a = tensor([4, 6, 12]);
    const b = tensor([6, 8, 15]);
    const result = lcm(a, b);
    expect(result.toArray()).toEqual([12, 24, 60]);
    expect(result.dtype).toBe("int32");
  });

  it("handles zeros", () => {
    const a = tensor([0, 6]);
    const b = tensor([5, 0]);
    const result = lcm(a, b);
    expect(result.toArray()).toEqual([0, 0]);
  });

  it("handles negatives", () => {
    const a = tensor([-4, 6]);
    const b = tensor([6, -8]);
    const result = lcm(a, b);
    expect(result.toArray()).toEqual([12, 24]);
  });

  it("throws on string dtype", () => {
    const a = tensor(["a"]);
    const b = tensor(["b"]);
    expect(() => lcm(a, b)).toThrow();
  });
});

describe("union1d", () => {
  it("computes sorted unique union", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([2, 3, 4]);
    const result = union1d(a, b);
    expect(result.toArray()).toEqual([1, 2, 3, 4]);
  });

  it("handles disjoint sets", () => {
    const a = tensor([1, 3, 5]);
    const b = tensor([2, 4, 6]);
    const result = union1d(a, b);
    expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("handles duplicates in input", () => {
    const a = tensor([1, 1, 2, 2]);
    const b = tensor([2, 2, 3, 3]);
    const result = union1d(a, b);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it("handles empty tensor", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([] as number[]).reshape([0]);
    const result = union1d(a, b);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it("throws on non-1D input", () => {
    const a = tensor([[1, 2]]);
    const b = tensor([3, 4]);
    expect(() => union1d(a, b)).toThrow();
  });
});

describe("intersect1d", () => {
  it("computes sorted unique intersection", () => {
    const a = tensor([1, 2, 3, 4]);
    const b = tensor([2, 4, 6]);
    const result = intersect1d(a, b);
    expect(result.toArray()).toEqual([2, 4]);
  });

  it("handles no overlap", () => {
    const a = tensor([1, 3, 5]);
    const b = tensor([2, 4, 6]);
    const result = intersect1d(a, b);
    expect(result.toArray()).toEqual([]);
    expect(result.shape).toEqual([0]);
  });

  it("handles duplicates", () => {
    const a = tensor([1, 1, 2, 2, 3]);
    const b = tensor([2, 2, 3, 3, 4]);
    const result = intersect1d(a, b);
    expect(result.toArray()).toEqual([2, 3]);
  });

  it("throws on non-1D input", () => {
    const a = tensor([[1, 2]]);
    const b = tensor([3, 4]);
    expect(() => intersect1d(a, b)).toThrow();
  });
});

describe("setdiff1d", () => {
  it("computes sorted set difference", () => {
    const a = tensor([1, 2, 3, 4]);
    const b = tensor([2, 4]);
    const result = setdiff1d(a, b);
    expect(result.toArray()).toEqual([1, 3]);
  });

  it("handles complete overlap", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1, 2, 3, 4]);
    const result = setdiff1d(a, b);
    expect(result.toArray()).toEqual([]);
  });

  it("handles no overlap", () => {
    const a = tensor([1, 3, 5]);
    const b = tensor([2, 4, 6]);
    const result = setdiff1d(a, b);
    expect(result.toArray()).toEqual([1, 3, 5]);
  });

  it("handles duplicates", () => {
    const a = tensor([1, 1, 2, 3, 3]);
    const b = tensor([2]);
    const result = setdiff1d(a, b);
    expect(result.toArray()).toEqual([1, 3]);
  });

  it("throws on non-1D input", () => {
    const a = tensor([[1, 2]]);
    const b = tensor([3]);
    expect(() => setdiff1d(a, b)).toThrow();
  });
});
