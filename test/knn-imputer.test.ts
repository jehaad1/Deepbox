import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { KNNImputer, MissingIndicator } from "../src/preprocess";

describe("KNNImputer", () => {
  it("imputes missing values using nearest neighbors", () => {
    const X = tensor([1, 2, 3, 4, 5, 6, 7, 8, NaN, 4]).reshape([5, 2]);

    const imp = new KNNImputer({ nNeighbors: 2 });
    const result = imp.fitTransform(X);

    expect(result.shape).toEqual([5, 2]);
    // The NaN in row 4 col 0 should be imputed from neighbors
    const val = Number(result.data[result.offset + 4 * 2 + 0]);
    expect(Number.isNaN(val)).toBe(false);
    expect(Number.isFinite(val)).toBe(true);
    // Non-NaN values should be preserved
    expect(Number(result.data[result.offset + 0])).toBe(1);
    expect(Number(result.data[result.offset + 1])).toBe(2);
  });

  it("preserves rows without missing values", () => {
    const X = tensor([1, 2, 3, 4, NaN, 6]).reshape([3, 2]);

    const imp = new KNNImputer({ nNeighbors: 2 });
    const result = imp.fitTransform(X);

    expect(Number(result.data[result.offset + 0])).toBe(1);
    expect(Number(result.data[result.offset + 1])).toBe(2);
    expect(Number(result.data[result.offset + 2])).toBe(3);
    expect(Number(result.data[result.offset + 3])).toBe(4);
  });

  it("supports distance weighting", () => {
    const X = tensor([1, 2, 3, 4, 5, 6, NaN, 4]).reshape([4, 2]);

    const imp = new KNNImputer({ nNeighbors: 3, weights: "distance" });
    const result = imp.fitTransform(X);

    const val = Number(result.data[result.offset + 3 * 2 + 0]);
    expect(Number.isNaN(val)).toBe(false);
    expect(Number.isFinite(val)).toBe(true);
  });

  it("handles multiple missing values in same row", () => {
    const X = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, NaN, NaN, 6]).reshape([4, 3]);

    const imp = new KNNImputer({ nNeighbors: 2 });
    const result = imp.fitTransform(X);

    expect(result.shape).toEqual([4, 3]);
    const v0 = Number(result.data[result.offset + 3 * 3 + 0]);
    const v1 = Number(result.data[result.offset + 3 * 3 + 1]);
    expect(Number.isNaN(v0)).toBe(false);
    expect(Number.isNaN(v1)).toBe(false);
  });

  it("throws if not fitted", () => {
    const imp = new KNNImputer();
    expect(() => imp.transform(tensor([[1, 2]]).reshape([1, 2]))).toThrow("fitted");
  });

  it("throws for non-2D input", () => {
    const imp = new KNNImputer();
    expect(() => imp.fit(tensor([1, 2, 3]))).toThrow("2D");
  });

  it("throws for feature mismatch on transform", () => {
    const imp = new KNNImputer({ nNeighbors: 1 });
    imp.fit(tensor([1, 2, 3, 4]).reshape([2, 2]));
    expect(() => imp.transform(tensor([1, 2, 3]).reshape([1, 3]))).toThrow("features");
  });

  it("throws for invalid nNeighbors", () => {
    expect(() => new KNNImputer({ nNeighbors: 0 })).toThrow();
    expect(() => new KNNImputer({ nNeighbors: -1 })).toThrow();
    expect(() => new KNNImputer({ nNeighbors: 1.5 })).toThrow();
  });

  it("returns params", () => {
    const imp = new KNNImputer({ nNeighbors: 3, weights: "distance" });
    expect(imp.getParams()).toMatchObject({ nNeighbors: 3, weights: "distance" });
  });

  it("defaults to nNeighbors=5, weights='uniform'", () => {
    const imp = new KNNImputer();
    expect(imp.getParams()).toMatchObject({ nNeighbors: 5, weights: "uniform" });
  });
});

describe("MissingIndicator", () => {
  it("creates binary indicator for missing values (missing-only)", () => {
    const X = tensor([1, NaN, NaN, 3, 7, 6]).reshape([3, 2]);

    const mi = new MissingIndicator();
    const result = mi.fitTransform(X);

    // Both columns have missing values
    expect(result.shape).toEqual([3, 2]);
    // Row 0: col 0 present, col 1 missing
    expect(Number(result.data[result.offset + 0])).toBe(0);
    expect(Number(result.data[result.offset + 1])).toBe(1);
    // Row 1: col 0 missing, col 1 present
    expect(Number(result.data[result.offset + 2])).toBe(1);
    expect(Number(result.data[result.offset + 3])).toBe(0);
    // Row 2: both present
    expect(Number(result.data[result.offset + 4])).toBe(0);
    expect(Number(result.data[result.offset + 5])).toBe(0);
  });

  it("only includes columns with missing values (missing-only mode)", () => {
    const X = tensor([1, NaN, 3, 4, 5, 6, 7, NaN, 9]).reshape([3, 3]);

    const mi = new MissingIndicator();
    mi.fit(X);

    // Only column 1 has missing values
    expect(mi.features_).toEqual([1]);

    const result = mi.transform(X);
    expect(result.shape).toEqual([3, 1]);
    expect(Number(result.data[result.offset + 0])).toBe(1); // row 0 col 1 is NaN
    expect(Number(result.data[result.offset + 1])).toBe(0); // row 1 col 1 is present
    expect(Number(result.data[result.offset + 2])).toBe(1); // row 2 col 1 is NaN
  });

  it("includes all columns when features='all'", () => {
    const X = tensor([1, NaN, 4, 5]).reshape([2, 2]);

    const mi = new MissingIndicator({ features: "all" });
    const result = mi.fitTransform(X);

    expect(result.shape).toEqual([2, 2]);
    expect(Number(result.data[result.offset + 0])).toBe(0); // row 0 col 0
    expect(Number(result.data[result.offset + 1])).toBe(1); // row 0 col 1
    expect(Number(result.data[result.offset + 2])).toBe(0); // row 1 col 0
    expect(Number(result.data[result.offset + 3])).toBe(0); // row 1 col 1
  });

  it("throws if not fitted", () => {
    const mi = new MissingIndicator();
    expect(() => mi.transform(tensor([[1, 2]]).reshape([1, 2]))).toThrow("fitted");
  });

  it("throws for non-2D input", () => {
    const mi = new MissingIndicator();
    expect(() => mi.fit(tensor([1, 2, 3]))).toThrow("2D");
  });

  it("throws for feature mismatch on transform", () => {
    const mi = new MissingIndicator();
    mi.fit(tensor([1, 2, 3, 4]).reshape([2, 2]));
    expect(() => mi.transform(tensor([1, 2, 3]).reshape([1, 3]))).toThrow("features");
  });

  it("returns params", () => {
    const mi = new MissingIndicator({ features: "all" });
    expect(mi.getParams()).toMatchObject({ features: "all" });
  });

  it("throws accessing features_ before fit", () => {
    const mi = new MissingIndicator();
    expect(() => mi.features_).toThrow("fitted");
  });
});
