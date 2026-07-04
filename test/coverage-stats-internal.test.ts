import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  assertSameSize,
  chiSquareCdf,
  computeStrides,
  fCdf,
  forEachIndexOffset,
  getNumberAt,
  logGamma,
  normalCdf,
  normalizeAxes,
  rankData,
  reducedShape,
  reduceMean,
  reduceVariance,
  regularizedIncompleteBeta,
  studentTCdf,
} from "../src/stats/_internal";

describe("normalizeAxes", () => {
  it("returns empty for undefined", () => {
    expect(normalizeAxes(undefined, 3)).toEqual([]);
  });

  it("normalizes single axis", () => {
    expect(normalizeAxes(0, 3)).toEqual([0]);
    expect(normalizeAxes(-1, 3)).toEqual([2]);
  });

  it("normalizes array of axes", () => {
    expect(normalizeAxes([1, 0], 3)).toEqual([0, 1]);
  });

  it("deduplicates", () => {
    expect(normalizeAxes([1, 1, 0], 3)).toEqual([0, 1]);
  });
});

describe("reducedShape", () => {
  it("empty axes without keepdims", () => {
    expect(reducedShape([3, 4, 5], [], false)).toEqual([]);
  });

  it("empty axes with keepdims", () => {
    expect(reducedShape([3, 4, 5], [], true)).toEqual([1, 1, 1]);
  });

  it("single axis without keepdims", () => {
    expect(reducedShape([3, 4, 5], [1], false)).toEqual([3, 5]);
  });

  it("single axis with keepdims", () => {
    expect(reducedShape([3, 4, 5], [1], true)).toEqual([3, 1, 5]);
  });

  it("multiple axes", () => {
    expect(reducedShape([3, 4, 5], [0, 2], false)).toEqual([4]);
  });
});

describe("computeStrides", () => {
  it("computes strides for 2D", () => {
    expect(computeStrides([3, 4])).toEqual([4, 1]);
  });

  it("computes strides for 3D", () => {
    expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
  });

  it("computes strides for 1D", () => {
    expect(computeStrides([5])).toEqual([1]);
  });
});

describe("assertSameSize", () => {
  it("passes for same size", () => {
    expect(() => assertSameSize(tensor([1, 2, 3]), tensor([4, 5, 6]), "test")).not.toThrow();
  });

  it("throws for different sizes", () => {
    expect(() => assertSameSize(tensor([1, 2]), tensor([3, 4, 5]), "test")).toThrow(/same number/i);
  });
});

describe("getNumberAt", () => {
  it("reads numeric value", () => {
    const t = tensor([10, 20, 30]);
    expect(getNumberAt(t, t.offset + 1)).toBe(20);
  });

  it("throws for string dtype", () => {
    const t = tensor(["a", "b"]);
    expect(() => getNumberAt(t, 0)).toThrow(/string/i);
  });
});

describe("rankData", () => {
  it("ranks unique values", () => {
    const { ranks } = rankData(new Float64Array([3, 1, 2]));
    expect(ranks[0]).toBe(3); // 3 is rank 3
    expect(ranks[1]).toBe(1); // 1 is rank 1
    expect(ranks[2]).toBe(2); // 2 is rank 2
  });

  it("handles ties with average ranks", () => {
    const { ranks, tieSum } = rankData(new Float64Array([1, 2, 2, 3]));
    expect(ranks[1]).toBe(2.5); // tied values get average
    expect(ranks[2]).toBe(2.5);
    expect(tieSum).toBeGreaterThan(0);
  });

  it("handles empty array", () => {
    const { ranks, tieSum } = rankData(new Float64Array([]));
    expect(ranks.length).toBe(0);
    expect(tieSum).toBe(0);
  });
});

describe("forEachIndexOffset", () => {
  it("iterates over all elements of 2D tensor", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const values: number[] = [];
    forEachIndexOffset(t, (off) => {
      values.push(Number(t.data[off]));
    });
    expect(values).toEqual([1, 2, 3, 4]);
  });

  it("handles scalar tensor", () => {
    const t = tensor(42);
    const values: number[] = [];
    forEachIndexOffset(t, (off) => {
      values.push(Number(t.data[off]));
    });
    expect(values).toEqual([42]);
  });

  it("handles empty tensor", () => {
    const t = tensor([] as number[]);
    const values: number[] = [];
    forEachIndexOffset(t, (off) => {
      values.push(Number(t.data[off]));
    });
    expect(values).toEqual([]);
  });
});

describe("reduceMean", () => {
  it("full reduction", () => {
    const t = tensor([1, 2, 3, 4]);
    const result = reduceMean(t, undefined, false);
    expect(Number(result.data[0])).toBeCloseTo(2.5);
  });

  it("full reduction with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceMean(t, undefined, true);
    expect(result.shape).toEqual([1, 1]);
    expect(Number(result.data[0])).toBeCloseTo(2.5);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceMean(t, 0, false);
    expect(result.shape).toEqual([2]);
    expect(Number(result.data[0])).toBeCloseTo(2);
    expect(Number(result.data[1])).toBeCloseTo(3);
  });

  it("axis reduction with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceMean(t, 0, true);
    expect(result.shape).toEqual([1, 2]);
  });

  it("throws for empty tensor", () => {
    const t = tensor([] as number[]);
    expect(() => reduceMean(t, undefined, false)).toThrow(/at least one/i);
  });
});

describe("reduceVariance", () => {
  it("population variance (ddof=0)", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const result = reduceVariance(t, undefined, false, 0);
    expect(Number(result.data[0])).toBeCloseTo(2.0);
  });

  it("sample variance (ddof=1)", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const result = reduceVariance(t, undefined, false, 1);
    expect(Number(result.data[0])).toBeCloseTo(2.5);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceVariance(t, 0, false, 0);
    expect(result.shape).toEqual([2]);
  });

  it("axis reduction with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceVariance(t, 0, true, 0);
    expect(result.shape).toEqual([1, 2]);
  });

  it("full with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = reduceVariance(t, undefined, true, 0);
    expect(result.shape).toEqual([1, 1]);
  });

  it("throws for empty tensor", () => {
    const t = tensor([] as number[]);
    expect(() => reduceVariance(t, undefined, false, 0)).toThrow(/at least one/i);
  });

  it("throws for ddof >= size", () => {
    const t = tensor([1, 2]);
    expect(() => reduceVariance(t, undefined, false, 2)).toThrow(/ddof/i);
  });

  it("throws for negative ddof", () => {
    const t = tensor([1, 2, 3]);
    expect(() => reduceVariance(t, undefined, false, -1)).toThrow(/non-negative/i);
  });

  it("throws for string dtype", () => {
    const t = tensor(["a", "b"]);
    expect(() => reduceVariance(t, undefined, false, 0)).toThrow(/string/i);
  });
});

describe("logGamma", () => {
  it("z >= 0.5 (Lanczos)", () => {
    expect(logGamma(5)).toBeCloseTo(Math.log(24), 5);
  });

  it("z < 0.5 (reflection)", () => {
    const result = logGamma(0.25);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  it("z = 1", () => {
    expect(logGamma(1)).toBeCloseTo(0, 5);
  });
});

describe("regularizedIncompleteBeta", () => {
  it("x=0 returns 0", () => {
    expect(regularizedIncompleteBeta(2, 3, 0)).toBe(0);
  });

  it("x=1 returns 1", () => {
    expect(regularizedIncompleteBeta(2, 3, 1)).toBe(1);
  });

  it("intermediate value", () => {
    const result = regularizedIncompleteBeta(2, 3, 0.5);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("symmetry branch (x >= (a+1)/(a+b+2))", () => {
    const result = regularizedIncompleteBeta(1, 5, 0.8);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(1);
  });

  it("rejects invalid a", () => {
    expect(() => regularizedIncompleteBeta(0, 1, 0.5)).toThrow();
    expect(() => regularizedIncompleteBeta(-1, 1, 0.5)).toThrow();
  });

  it("rejects invalid b", () => {
    expect(() => regularizedIncompleteBeta(1, 0, 0.5)).toThrow();
  });

  it("rejects invalid x", () => {
    expect(() => regularizedIncompleteBeta(1, 1, -0.1)).toThrow();
    expect(() => regularizedIncompleteBeta(1, 1, 1.1)).toThrow();
  });
});

describe("normalCdf", () => {
  it("Φ(0) = 0.5", () => {
    expect(normalCdf(0)).toBeCloseTo(0.5);
  });

  it("Φ(1.96) ≈ 0.975", () => {
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 2);
  });

  it("Φ(-1.96) ≈ 0.025", () => {
    expect(normalCdf(-1.96)).toBeCloseTo(0.025, 2);
  });

  it("Φ(large) ≈ 1", () => {
    expect(normalCdf(10)).toBeCloseTo(1.0, 5);
  });

  it("Φ(-large) ≈ 0", () => {
    expect(normalCdf(-10)).toBeCloseTo(0.0, 5);
  });
});

describe("studentTCdf", () => {
  it("t=0 gives 0.5", () => {
    expect(studentTCdf(0, 10)).toBeCloseTo(0.5);
  });

  it("positive t gives > 0.5", () => {
    expect(studentTCdf(2, 10)).toBeGreaterThan(0.5);
  });

  it("negative t gives < 0.5", () => {
    expect(studentTCdf(-2, 10)).toBeLessThan(0.5);
  });

  it("rejects invalid df", () => {
    expect(() => studentTCdf(0, 0)).toThrow();
    expect(() => studentTCdf(0, -1)).toThrow();
  });

  it("handles NaN t", () => {
    expect(Number.isNaN(studentTCdf(NaN, 10))).toBe(true);
  });

  it("handles infinite t", () => {
    expect(studentTCdf(Infinity, 10)).toBe(1);
    expect(studentTCdf(-Infinity, 10)).toBe(0);
  });
});

describe("chiSquareCdf", () => {
  it("x=0 gives 0", () => {
    expect(chiSquareCdf(0, 5)).toBe(0);
  });

  it("reasonable value", () => {
    const p = chiSquareCdf(3.841, 1);
    expect(p).toBeGreaterThan(0.9);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("rejects invalid k", () => {
    expect(() => chiSquareCdf(1, 0)).toThrow();
    expect(() => chiSquareCdf(1, -1)).toThrow();
  });

  it("handles NaN x", () => {
    expect(Number.isNaN(chiSquareCdf(NaN, 5))).toBe(true);
  });

  it("handles Infinity x", () => {
    expect(chiSquareCdf(Infinity, 5)).toBe(1);
  });

  it("negative x gives 0", () => {
    expect(chiSquareCdf(-1, 5)).toBe(0);
  });
});

describe("fCdf", () => {
  it("x=0 gives 0", () => {
    expect(fCdf(0, 5, 10)).toBe(0);
  });

  it("reasonable value", () => {
    const p = fCdf(4, 5, 10);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("rejects invalid dfn", () => {
    expect(() => fCdf(1, 0, 10)).toThrow();
  });

  it("rejects invalid dfd", () => {
    expect(() => fCdf(1, 5, 0)).toThrow();
  });

  it("handles NaN x", () => {
    expect(Number.isNaN(fCdf(NaN, 5, 10))).toBe(true);
  });

  it("handles Infinity x", () => {
    expect(fCdf(Infinity, 5, 10)).toBe(1);
  });

  it("negative x gives 0", () => {
    expect(fCdf(-1, 5, 10)).toBe(0);
  });
});
