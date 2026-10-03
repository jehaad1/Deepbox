import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("Rolling.corr()", () => {
  it("computes rolling Pearson correlation", () => {
    const df = new DataFrame({
      A: [1, 2, 3, 4, 5],
      B: [2, 4, 6, 8, 10],
    });
    const result = df.rolling(3).corr("A", "B");
    const col = result.get("A_B").data;
    // First two values should be null (window not full)
    expect(col[0]).toBeNull();
    expect(col[1]).toBeNull();
    // Perfect positive correlation for linear relationship
    expect(col[2]).toBeCloseTo(1.0, 10);
    expect(col[3]).toBeCloseTo(1.0, 10);
    expect(col[4]).toBeCloseTo(1.0, 10);
  });

  it("computes negative correlation", () => {
    const df = new DataFrame({
      A: [1, 2, 3, 4, 5],
      B: [10, 8, 6, 4, 2],
    });
    const result = df.rolling(3).corr("A", "B");
    const col = result.get("A_B").data;
    expect(col[2]).toBeCloseTo(-1.0, 10);
    expect(col[3]).toBeCloseTo(-1.0, 10);
    expect(col[4]).toBeCloseTo(-1.0, 10);
  });

  it("handles NaN/null values gracefully", () => {
    const df = new DataFrame({
      A: [1, null, 3, 4, 5],
      B: [2, 4, 6, 8, 10],
    });
    const result = df.rolling(3).corr("A", "B");
    const col = result.get("A_B").data;
    expect(col[0]).toBeNull();
    expect(col[1]).toBeNull();
    // Window [1, null, 3] vs [2, 4, 6]: only 2 valid pairs (idx 0 and 2)
    // corr([1,3], [2,6]) = 1.0
    expect(col[2]).toBeCloseTo(1.0, 10);
  });

  it("returns null when fewer than 2 valid pairs in window", () => {
    const df = new DataFrame({
      A: [1, null, null, 4, 5],
      B: [null, null, 6, 8, 10],
    });
    const result = df.rolling(3).corr("A", "B");
    const col = result.get("A_B").data;
    // Window at index 2: pairs (0,0)→(1,null) skip, (1,1)→(null,null) skip, (2,2)→(null,6) skip → 0 valid
    expect(col[2]).toBeNull();
  });

  it("returns NaN for constant values in window", () => {
    const df = new DataFrame({
      A: [5, 5, 5, 4, 3],
      B: [1, 2, 3, 4, 5],
    });
    const result = df.rolling(3).corr("A", "B");
    const col = result.get("A_B").data;
    // First window A=[5,5,5] is constant → denominator is 0 → NaN
    expect(col[2]).toBeNaN();
  });

  it("preserves index", () => {
    const df = new DataFrame({ A: [1, 2, 3], B: [4, 5, 6] }, { index: ["x", "y", "z"] });
    const result = df.rolling(2).corr("A", "B");
    expect(result.index).toEqual(["x", "y", "z"]);
  });

  it("window=2 produces results from index 1", () => {
    const df = new DataFrame({
      A: [1, 2, 3, 4],
      B: [10, 20, 30, 40],
    });
    const result = df.rolling(2).corr("A", "B");
    const col = result.get("A_B").data;
    expect(col[0]).toBeNull();
    expect(col[1]).toBeCloseTo(1.0, 10);
    expect(col[2]).toBeCloseTo(1.0, 10);
    expect(col[3]).toBeCloseTo(1.0, 10);
  });
});

describe("Rolling.cov()", () => {
  it("computes rolling sample covariance", () => {
    const df = new DataFrame({
      A: [1, 2, 3, 4, 5],
      B: [2, 4, 6, 8, 10],
    });
    const result = df.rolling(3).cov("A", "B");
    const col = result.get("A_B").data;
    expect(col[0]).toBeNull();
    expect(col[1]).toBeNull();
    // Window [1,2,3] vs [2,4,6]: cov = sum((xi-mx)(yi-my))/(n-1)
    // mx=2, my=4, sum = (1-2)(2-4)+(2-2)(4-4)+(3-2)(6-4) = 2+0+2 = 4, cov = 4/2 = 2
    expect(col[2]).toBeCloseTo(2.0, 10);
    // Window [2,3,4] vs [4,6,8]: mx=3, my=6, sum = (-1)(-2)+(0)(0)+(1)(2) = 4, cov=2
    expect(col[3]).toBeCloseTo(2.0, 10);
    expect(col[4]).toBeCloseTo(2.0, 10);
  });

  it("computes negative covariance", () => {
    const df = new DataFrame({
      A: [1, 2, 3],
      B: [6, 4, 2],
    });
    const result = df.rolling(3).cov("A", "B");
    const col = result.get("A_B").data;
    // mx=2, my=4, sum = (1-2)(6-4)+(2-2)(4-4)+(3-2)(2-4) = -2+0-2 = -4, cov = -4/2 = -2
    expect(col[2]).toBeCloseTo(-2.0, 10);
  });

  it("handles NaN/null values", () => {
    const df = new DataFrame({
      A: [1, null, 3, 4, 5],
      B: [2, 4, 6, 8, 10],
    });
    const result = df.rolling(3).cov("A", "B");
    const col = result.get("A_B").data;
    // Window at idx 2: valid pairs (1,2) and (3,6) → 2 pairs
    // mx=2, my=4, cov = ((1-2)(2-4)+(3-2)(6-4))/(2-1) = (2+2)/1 = 4
    expect(col[2]).toBeCloseTo(4.0, 10);
  });

  it("returns null when fewer than 2 valid pairs", () => {
    const df = new DataFrame({
      A: [null, null, 3],
      B: [1, null, 3],
    });
    const result = df.rolling(3).cov("A", "B");
    const col = result.get("A_B").data;
    // Only 1 valid pair at idx 2: (3,3) → not enough
    expect(col[2]).toBeNull();
  });

  it("preserves index", () => {
    const df = new DataFrame({ A: [1, 2, 3], B: [4, 5, 6] }, { index: ["a", "b", "c"] });
    const result = df.rolling(2).cov("A", "B");
    expect(result.index).toEqual(["a", "b", "c"]);
  });

  it("covariance of a column with itself equals variance", () => {
    const df = new DataFrame({
      A: [1, 2, 3, 4, 5],
    });
    // Copy A as B
    const df2 = new DataFrame({
      A: [1, 2, 3, 4, 5],
      B: [1, 2, 3, 4, 5],
    });
    const covResult = df2.rolling(3).cov("A", "B");
    const varResult = df.rolling(3).var();
    const covCol = covResult.get("A_B").data;
    const varCol = varResult.get("A").data;
    for (let i = 0; i < 5; i++) {
      if (covCol[i] === null || varCol[i] === null) {
        expect(covCol[i]).toEqual(varCol[i]);
      } else {
        expect(covCol[i] as number).toBeCloseTo(varCol[i] as number, 10);
      }
    }
  });
});
