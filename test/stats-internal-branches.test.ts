import { describe, expect, it } from "vitest";
import { tensor, zeros } from "../src/ndarray";
import {
  assertSameSize,
  chiSquareCdf,
  computeStrides,
  fCdf,
  forEachIndexOffset,
  getNumberAt,
  normalCdf,
  normalizeAxes,
  reducedShape,
  reduceMean,
  reduceVariance,
  regularizedIncompleteBeta,
  studentTCdf,
} from "../src/stats/_internal";

describe("stats internal branch coverage", () => {
  it("normalizes axes and computes reduced shapes", () => {
    expect(normalizeAxes(undefined, 3)).toEqual([]);
    expect(normalizeAxes([-1, 0, 0], 3)).toEqual([0, 2]);
    expect(() => normalizeAxes([3], 3)).toThrow(/out of bounds/);

    expect(reducedShape([2, 3, 4], [], false)).toEqual([]);
    expect(reducedShape([2, 3], [], true)).toEqual([1, 1]);
    expect(reducedShape([2, 3, 4], [1], false)).toEqual([2, 4]);
    expect(reducedShape([2, 3, 4], [1], true)).toEqual([2, 1, 4]);
  });

  it("iterates indices and reads numbers (BigInt and numeric)", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const seen: number[] = [];
    forEachIndexOffset(t, (off) => seen.push(getNumberAt(t, off)));
    expect(seen).toEqual([1, 2, 3, 4]);

    const big = tensor([1, 2, 3], { dtype: "int64" });
    const vals: number[] = [];
    forEachIndexOffset(big, (off) => vals.push(getNumberAt(big, off)));
    expect(vals).toEqual([1, 2, 3]);
  });

  it("computes strides and rejects size mismatches", () => {
    expect(computeStrides([3, 4, 5])).toEqual([20, 5, 1]);
    expect(() => assertSameSize(tensor([1]), tensor([1, 2]), "pearsonr")).toThrow(
      /same number of elements/
    );
  });

  it("reduceMean handles keepdims and empty reduction errors", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const full = reduceMean(t, undefined, true);
    expect(full.shape).toEqual([1, 1]);

    const axis = reduceMean(t, 0, false);
    expect(axis.shape).toEqual([2]);

    const empty = zeros([0, 2]);
    expect(() => reduceMean(empty, 0, false)).toThrow(/empty axis/i);
    expect(() => reduceMean(tensor([]), undefined, false)).toThrow(/at least one element/i);
  });

  it("reduceVariance handles ddof and axis reductions", () => {
    const t = tensor([1, 2, 3, 4]);
    const v = reduceVariance(t, undefined, false, 0);
    expect(v.shape).toEqual([]);
    expect(() => reduceVariance(t, undefined, false, 10)).toThrow(/ddof/i);

    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    const vAxis = reduceVariance(m, 0, true, 0);
    expect(vAxis.shape).toEqual([1, 2]);
    const s = tensor(["a", "b"]);
    expect(() => reduceVariance(s, undefined, false, 0)).toThrow(/string/);
  });

  it("covers special-function branches", () => {
    expect(() => studentTCdf(1, 0)).toThrow(/df/);
    expect(studentTCdf(Number.NEGATIVE_INFINITY, 5)).toBe(0);
    expect(studentTCdf(Number.POSITIVE_INFINITY, 5)).toBe(1);

    expect(() => chiSquareCdf(1, 0)).toThrow(/degrees of freedom/);
    expect(chiSquareCdf(-1, 2)).toBe(0);

    expect(() => fCdf(1, 0, 1)).toThrow(/degrees of freedom/);
    expect(fCdf(0, 2, 3)).toBe(0);

    expect(() => regularizedIncompleteBeta(1, 1, -0.1)).toThrow(/x must be/);
    expect(() => regularizedIncompleteBeta(0, 1, 0.5)).toThrow(/a must be/i);
    expect(() => regularizedIncompleteBeta(1, 0, 0.5)).toThrow(/b must be/i);
    expect(regularizedIncompleteBeta(1, 1, 0)).toBe(0);
    expect(regularizedIncompleteBeta(1, 1, 1)).toBe(1);
    // Exercise both symmetry branches
    expect(regularizedIncompleteBeta(2, 3, 0.2)).toBeGreaterThan(0);
    expect(regularizedIncompleteBeta(2, 3, 0.9)).toBeGreaterThan(0);

    // chiSquareCdf path: x<=0 already covered, now small/large x for gamma branches
    expect(chiSquareCdf(0.5, 4)).toBeGreaterThan(0);
    expect(chiSquareCdf(20, 4)).toBeGreaterThan(0);

    expect(normalCdf(0)).toBeCloseTo(0.5, 6);
    expect(studentTCdf(2, 5)).toBeGreaterThan(0.5);
    expect(studentTCdf(-2, 5)).toBeLessThan(0.5);
  });
});
