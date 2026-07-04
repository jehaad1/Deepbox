import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe/DataFrame";
import {
  makeFriedman1,
  makeFriedman2,
  makeFriedman3,
  makeSCurve,
  makeSwissRoll,
} from "../src/datasets";
import { tensor } from "../src/ndarray";
import { einsum } from "../src/ndarray/ops/einsum";
import { f_twoway } from "../src/stats";

// ─── DataFrame P1 Utilities ───────────────────────────────────────────────────

describe("DataFrame.abs()", () => {
  it("computes element-wise absolute value", () => {
    const df = new DataFrame({ a: [-1, 2, -3], b: [4, -5, 6] });
    const result = df.abs();
    expect(result.get("a").data).toEqual([1, 2, 3]);
    expect(result.get("b").data).toEqual([4, 5, 6]);
  });

  it("preserves non-numeric values", () => {
    const df = new DataFrame({ a: [-1, "hello", -3] });
    const result = df.abs();
    expect(result.get("a").data).toEqual([1, "hello", 3]);
  });
});

describe("DataFrame.round()", () => {
  it("rounds to 0 decimal places by default", () => {
    const df = new DataFrame({ a: [1.5, 2.3, 3.7] });
    const result = df.round();
    expect(result.get("a").data).toEqual([2, 2, 4]);
  });

  it("rounds to specified decimal places", () => {
    const df = new DataFrame({ a: [1.456, 2.789] });
    const result = df.round(2);
    expect(result.get("a").data).toEqual([1.46, 2.79]);
  });
});

describe("DataFrame.nunique()", () => {
  it("counts unique values per column", () => {
    const df = new DataFrame({ a: [1, 2, 2, 3], b: ["x", "x", "y", "y"] });
    const result = df.nunique();
    expect(result.data[0]).toBe(3);
    expect(result.data[1]).toBe(2);
  });
});

describe("DataFrame.nlargest()", () => {
  it("returns top n rows by column value", () => {
    const df = new DataFrame({ a: [10, 30, 20, 40], b: [1, 2, 3, 4] });
    const result = df.nlargest(2, "a");
    expect(result.get("a").data).toEqual([40, 30]);
    expect(result.get("b").data).toEqual([4, 2]);
  });
});

describe("DataFrame.nsmallest()", () => {
  it("returns bottom n rows by column value", () => {
    const df = new DataFrame({ a: [10, 30, 20, 40], b: [1, 2, 3, 4] });
    const result = df.nsmallest(2, "a");
    expect(result.get("a").data).toEqual([10, 20]);
    expect(result.get("b").data).toEqual([1, 3]);
  });
});

describe("DataFrame.idxmin()", () => {
  it("returns index of min value per column", () => {
    const df = new DataFrame({ a: [3, 1, 2] }, { index: ["x", "y", "z"] });
    const result = df.idxmin();
    expect(result.data[0]).toBe("y");
  });
});

describe("DataFrame.idxmax()", () => {
  it("returns index of max value per column", () => {
    const df = new DataFrame({ a: [3, 1, 2] }, { index: ["x", "y", "z"] });
    const result = df.idxmax();
    expect(result.data[0]).toBe("x");
  });
});

describe("DataFrame.between()", () => {
  it("filters rows where column is between bounds (inclusive)", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.between("a", 2, 4);
    expect(result.get("a").data).toEqual([2, 3, 4]);
  });

  it("supports exclusive bounds", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.between("a", 2, 4, false);
    expect(result.get("a").data).toEqual([3]);
  });
});

describe("DataFrame.assign()", () => {
  it("adds a new column from array", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.assign({ b: [4, 5, 6] });
    expect(result.columns).toContain("b");
    expect(result.get("b").data).toEqual([4, 5, 6]);
  });

  it("adds computed column from function", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.assign({
      doubled: (row: Record<string, unknown>) => (row.a as number) * 2,
    });
    expect(result.get("doubled").data).toEqual([2, 4, 6]);
  });
});

describe("DataFrame.where()", () => {
  it("replaces values where condition is false", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.where([true, false, true], 0);
    expect(result.get("a").data).toEqual([1, 0, 3]);
  });

  it("works with function condition", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.where((v: unknown) => (v as number) > 1, -1);
    expect(result.get("a").data).toEqual([-1, 2, 3]);
  });
});

describe("DataFrame.mask()", () => {
  it("replaces values where condition is true", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.mask([false, true, false], 0);
    expect(result.get("a").data).toEqual([1, 0, 3]);
  });
});

describe("DataFrame.astype()", () => {
  it("casts column to number", () => {
    const df = new DataFrame({ a: ["1", "2", "3"] });
    const result = df.astype("a", "number");
    expect(result.get("a").data).toEqual([1, 2, 3]);
  });

  it("casts column to string", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.astype("a", "string");
    expect(result.get("a").data).toEqual(["1", "2", "3"]);
  });

  it("casts column to boolean", () => {
    const df = new DataFrame({ a: [0, 1, 2] });
    const result = df.astype("a", "boolean");
    expect(result.get("a").data).toEqual([false, true, true]);
  });
});

describe("DataFrame.info()", () => {
  it("returns summary string", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: ["x", "y", "z"] });
    const info = df.info();
    expect(info).toContain("<DataFrame>");
    expect(info).toContain("RangeIndex: 3 entries");
    expect(info).toContain("Data columns (total 2 columns)");
  });
});

describe("DataFrame.any()", () => {
  it("checks if any value is truthy per column", () => {
    const df = new DataFrame({ a: [0, 0, 1], b: [0, 0, 0] });
    const result = df.any();
    const arr = Array.isArray(result) ? result : result.data;
    expect(arr).toEqual([true, false]);
  });
});

describe("DataFrame.all()", () => {
  it("checks if all values are truthy per column", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [1, 0, 3] });
    const result = df.all();
    const arr = Array.isArray(result) ? result : result.data;
    expect(arr).toEqual([true, false]);
  });
});

describe("DataFrame.value_counts()", () => {
  it("counts unique values", () => {
    const df = new DataFrame({
      fruit: ["apple", "banana", "apple", "cherry", "banana", "apple"],
    });
    const result = df.value_counts("fruit");
    expect(result.columns).toContain("fruit");
    expect(result.columns).toContain("count");
    // apple = 3 should be first
    expect(result.get("fruit").data[0]).toBe("apple");
    expect(result.get("count").data[0]).toBe(3);
  });
});

// ─── Rolling Window ──────────────────────────────────────────────────────────

describe("DataFrame.rolling()", () => {
  const df = new DataFrame({ a: [1, 2, 3, 4, 5] });

  it("mean()", () => {
    const result = df.rolling(3).mean();
    expect(result.get("a").data[0]).toBeNull();
    expect(result.get("a").data[1]).toBeNull();
    expect(result.get("a").data[2]).toBe(2);
    expect(result.get("a").data[3]).toBe(3);
    expect(result.get("a").data[4]).toBe(4);
  });

  it("sum()", () => {
    const result = df.rolling(3).sum();
    expect(result.get("a").data[2]).toBe(6);
    expect(result.get("a").data[3]).toBe(9);
    expect(result.get("a").data[4]).toBe(12);
  });

  it("min()", () => {
    const result = df.rolling(3).min();
    expect(result.get("a").data[2]).toBe(1);
    expect(result.get("a").data[3]).toBe(2);
    expect(result.get("a").data[4]).toBe(3);
  });

  it("max()", () => {
    const result = df.rolling(3).max();
    expect(result.get("a").data[2]).toBe(3);
    expect(result.get("a").data[3]).toBe(4);
    expect(result.get("a").data[4]).toBe(5);
  });

  it("std()", () => {
    const result = df.rolling(3).std();
    expect(result.get("a").data[0]).toBeNull();
    expect(result.get("a").data[1]).toBeNull();
    expect(result.get("a").data[2]).toBeCloseTo(1.0, 5);
  });

  it("var()", () => {
    const result = df.rolling(3).var();
    expect(result.get("a").data[2]).toBeCloseTo(1.0, 5);
  });

  it("apply()", () => {
    const result = df.rolling(3).apply((vals) => vals.reduce((a, b) => a + b, 0));
    expect(result.get("a").data[2]).toBe(6);
    expect(result.get("a").data[4]).toBe(12);
  });
});

// ─── Dataset Generators ──────────────────────────────────────────────────────

describe("makeFriedman1", () => {
  it("generates correct shapes", () => {
    const [X, y] = makeFriedman1({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    expect(X.shape).toEqual([50, 10]);
    expect(y.shape).toEqual([50]);
  });

  it("throws for nFeatures < 5", () => {
    expect(() => makeFriedman1({ nFeatures: 3 })).toThrow();
  });
});

describe("makeFriedman2", () => {
  it("generates correct shapes", () => {
    const [X, y] = makeFriedman2({ nSamples: 30, randomState: 42 });
    expect(X.shape).toEqual([30, 4]);
    expect(y.shape).toEqual([30]);
  });
});

describe("makeFriedman3", () => {
  it("generates correct shapes", () => {
    const [X, y] = makeFriedman3({ nSamples: 30, randomState: 42 });
    expect(X.shape).toEqual([30, 4]);
    expect(y.shape).toEqual([30]);
  });
});

describe("makeSwissRoll", () => {
  it("generates 3D manifold data", () => {
    const [X, t] = makeSwissRoll({ nSamples: 100, randomState: 42 });
    expect(X.shape).toEqual([100, 3]);
    expect(t.shape).toEqual([100]);
  });
});

describe("makeSCurve", () => {
  it("generates 3D S-curve data", () => {
    const [X, t] = makeSCurve({ nSamples: 100, randomState: 42 });
    expect(X.shape).toEqual([100, 3]);
    expect(t.shape).toEqual([100]);
  });
});

// ─── Two-way ANOVA ───────────────────────────────────────────────────────────

describe("f_twoway", () => {
  it("computes two-way ANOVA for balanced design", () => {
    // 2 levels of factor A, 2 levels of factor B, 3 replications each
    const data = [
      [
        [6, 8, 7],
        [4, 5, 3],
      ],
      [
        [8, 9, 10],
        [7, 6, 8],
      ],
    ];
    const result = f_twoway(data);
    expect(result.factorA.statistic).toBeGreaterThan(0);
    expect(result.factorB.statistic).toBeGreaterThan(0);
    expect(result.interaction.statistic).toBeGreaterThanOrEqual(0);
    expect(result.factorA.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.factorA.pvalue).toBeLessThanOrEqual(1);
    expect(result.factorB.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.factorB.pvalue).toBeLessThanOrEqual(1);
  });

  it("throws for less than 2 factor levels", () => {
    expect(() => f_twoway([[[1, 2]]])).toThrow();
  });

  it("detects clear factor effect", () => {
    // Factor A has big effect, factor B has small/no effect
    const data = [
      [
        [2, 3, 1],
        [3, 2, 1],
      ],
      [
        [100, 101, 99],
        [100, 102, 98],
      ],
    ];
    const result = f_twoway(data);
    expect(result.factorA.pvalue).toBeLessThan(0.001);
    // Factor B should not be significant (high p-value)
    expect(result.factorB.pvalue).toBeGreaterThan(0.05);
  });
});

// ─── einsum ──────────────────────────────────────────────────────────────────

describe("einsum", () => {
  it("computes matrix multiplication (ij,jk->ik)", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    const result = einsum("ij,jk->ik", a, b);
    expect(result.shape).toEqual([2, 2]);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    const d = result.data;
    expect(d[0]).toBe(19);
    expect(d[1]).toBe(22);
    expect(d[2]).toBe(43);
    expect(d[3]).toBe(50);
  });

  it("computes dot product (i,i->)", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const result = einsum("i,i->", a, b);
    expect(result.shape).toEqual([]);
    expect(result.data[0]).toBe(32); // 1*4 + 2*5 + 3*6
  });

  it("computes trace (ii->)", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = einsum("ii->", a);
    expect(result.data[0]).toBe(5); // 1 + 4
  });

  it("computes transpose (ij->ji)", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = einsum("ij->ji", a);
    expect(result.shape).toEqual([3, 2]);
    expect(result.data[0]).toBe(1);
    expect(result.data[1]).toBe(4);
    expect(result.data[2]).toBe(2);
    expect(result.data[3]).toBe(5);
  });

  it("computes matrix-vector product (ij,j->i)", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([5, 6]);
    const result = einsum("ij,j->i", a, b);
    expect(result.shape).toEqual([2]);
    expect(result.data[0]).toBe(17); // 1*5 + 2*6
    expect(result.data[1]).toBe(39); // 3*5 + 4*6
  });

  it("computes element-wise product sum (ij,ij->)", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    const result = einsum("ij,ij->", a, b);
    expect(result.data[0]).toBe(70); // 1*5 + 2*6 + 3*7 + 4*8
  });
});
