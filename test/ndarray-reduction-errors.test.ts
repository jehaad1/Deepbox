import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  all,
  any,
  cumprod,
  cumsum,
  diff,
  max,
  median,
  min,
  prod,
  sum,
  variance,
} from "../src/ndarray/ops/reduction";

describe("ndarray reduction error branches", () => {
  it("throws for string dtype reductions", () => {
    const s = tensor(["a", "b"]);
    expect(() => sum(s)).toThrow(/string/i);
    expect(() => prod(s)).toThrow(/string/i);
    expect(() => min(s)).toThrow(/string/i);
    expect(() => max(s)).toThrow(/string/i);
    expect(() => median(s)).toThrow(/string/i);
    expect(() => cumsum(s)).toThrow(/string/i);
    expect(() => cumprod(s)).toThrow(/string/i);
    expect(() => diff(s)).toThrow(/string/i);
    expect(() => any(s)).toThrow(/string/i);
    expect(() => all(s)).toThrow(/string/i);
    expect(() => variance(s)).toThrow(/string/i);
  });

  it("throws for invalid axes", () => {
    const t = tensor([1, 2, 3]);
    expect(() => prod(t, 2)).toThrow(/axis/i);
    expect(() => min(t, -2)).toThrow(/axis/i);
    expect(() => max(t, 4)).toThrow(/axis/i);
    expect(() => cumsum(t, 5)).toThrow(/axis/i);
    expect(() => cumprod(t, -5)).toThrow(/axis/i);
    expect(() => diff(t, 1, 3)).toThrow(/axis/i);
    expect(() => any(t, 5)).toThrow(/axis/i);
    expect(() => all(t, -5)).toThrow(/axis/i);
  });

  it("covers variance and diff edge cases", () => {
    const t = tensor([1]);
    expect(() => variance(t, undefined, false, 1)).toThrow(/ddof=1/);

    const t2 = tensor([[1, 2]]);
    expect(() => variance(t2, 1, false, 3)).toThrow(/ddof=3/);

    expect(diff(t).toArray()).toEqual([]);
    expect(diff(tensor([1, 2, 3]), 2).toArray()).toEqual([0]);
  });
});
