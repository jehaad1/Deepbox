import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame regression contracts", () => {
  it("builds sparse pivot tables with null fill", () => {
    const sparse = new DataFrame({
      i: ["A", "B"],
      c: ["x", "y"],
      v: [1, 2],
    });
    const out = sparse.pivot("i", "c", "v");
    expect(out.index).toEqual(["A", "B"]);
    expect(out.columns).toEqual(["x", "y"]);
    expect(out.toArray()).toEqual([
      [1, null],
      [null, 2],
    ]);
  });

  it("handles pivot tables with missing combinations", () => {
    const df = new DataFrame({
      i: ["A", "A", "B"],
      c: ["x", "y", "x"],
      v: [1, 2, 3],
    });
    const out = df.pivot("i", "c", "v");
    expect(out.toArray()).toEqual([
      [1, 2],
      [3, null],
    ]);
  });

  it("rejects duplicate pivot index-column entries", () => {
    const dup = new DataFrame({
      i: ["A", "A"],
      c: ["x", "x"],
      v: [1, 2],
    });
    expect(() => dup.pivot("i", "c", "v")).toThrow(/Duplicate pivot entry/);
  });

  it("reindexes concat(axis=0) output to avoid duplicate index labels", () => {
    const left = new DataFrame({ a: [1, 2] });
    const right = new DataFrame({ a: [3, 4] });
    const out = left.concat(right, 0);
    expect(out.index).toEqual([0, 1, 2, 3]);
    expect(out.toArray()).toEqual([[1], [2], [3], [4]]);
  });

  it("preserves overlapping columns on concat(axis=1) via suffixing", () => {
    const left = new DataFrame({ x: [1, 2] });
    const right = new DataFrame({ x: [3, 4] });
    const out = left.concat(right, 1);
    expect(out.columns).toEqual(["x_left", "x_right"]);
    expect(out.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("preserves overlapping non-key columns on left join via suffixing", () => {
    const left = new DataFrame({
      id: [1, 2],
      x: ["a", "b"],
    });
    const right = new DataFrame({
      id: [1],
      x: ["c"],
    });
    const out = left.join(right, "id", "left");
    expect(out.columns).toEqual(["id", "x_left", "x_right"]);
    expect(out.toArray()).toEqual([
      [1, "a", "c"],
      [2, "b", null],
    ]);
  });

  it("keeps row count stable for large absolute shift periods", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(df.shift(5).get("a").data).toEqual([null, null, null]);
    expect(df.shift(-5, 0).get("a").data).toEqual([0, 0, 0]);
  });

  it("validates shift periods and rolling window parameters", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.shift(1.5)).toThrow(/periods must be a finite integer/);
    expect(() => df.rolling(0)).toThrow(/window must be a positive integer/);
    expect(() => df.rolling(-1)).toThrow(/window must be a positive integer/);
    expect(() => df.rolling(1.2)).toThrow(/window must be a positive integer/);
  });

  it("validates sample and diff/pct_change parameters", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.sample(1.5)).toThrow(/n must be a finite integer/);
    expect(() => df.sample(1, 1.2)).toThrow(/random_state must be a finite integer/);
    expect(() => df.diff(-1)).toThrow(/periods must be a non-negative integer/);
    expect(() => df.diff(1.5)).toThrow(/periods must be a non-negative integer/);
    expect(() => df.pct_change(-1)).toThrow(/periods must be a non-negative integer/);
    expect(() => df.pct_change(1.2)).toThrow(/periods must be a non-negative integer/);
  });

  it("handles diff and pct_change when periods exceed length", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(df.diff(5).get("a").data).toEqual([null, null, null]);
    expect(df.pct_change(5).get("a").data).toEqual([null, null, null]);
  });
});
