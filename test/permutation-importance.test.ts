import { describe, expect, it } from "vitest";
import { KNeighborsClassifier, permutationImportance } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("permutationImportance", () => {
  // Simple dataset where feature 0 is informative and feature 1 is noise
  const X = tensor([
    [0, 10],
    [1, 20],
    [2, 30],
    [3, 40],
    [10, 50],
    [11, 60],
    [12, 70],
    [13, 80],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("returns correct shapes", () => {
    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    const result = permutationImportance(knn, X, y, { nRepeats: 3 });
    expect(result.importancesMean.shape).toEqual([2]);
    expect(result.importancesStd.shape).toEqual([2]);
    expect(result.importances.shape).toEqual([3, 2]);
  });

  it("produces non-negative importance for informative features", () => {
    // Use a dataset where feature 0 perfectly separates classes and feature 1 is constant
    const Xclean = tensor([
      [0, 1],
      [1, 1],
      [2, 1],
      [3, 1],
      [10, 1],
      [11, 1],
      [12, 1],
      [13, 1],
    ]);
    const yClean = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(Xclean, yClean);

    const result = permutationImportance(knn, Xclean, yClean, {
      nRepeats: 10,
      randomState: 42,
    });

    const imp0 = Number(result.importancesMean.data[result.importancesMean.offset]);
    const imp1 = Number(result.importancesMean.data[result.importancesMean.offset + 1]);

    // Feature 0 (informative) should have positive importance
    expect(imp0).toBeGreaterThan(0);
    // Feature 1 (constant) should have zero importance
    expect(imp1).toBeCloseTo(0, 5);
  });

  it("validates nRepeats parameter", () => {
    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    expect(() => permutationImportance(knn, X, y, { nRepeats: 0 })).toThrow(/nRepeats/);
  });

  it("produces reproducible results with same randomState", () => {
    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    const r1 = permutationImportance(knn, X, y, {
      nRepeats: 5,
      randomState: 123,
    });
    const r2 = permutationImportance(knn, X, y, {
      nRepeats: 5,
      randomState: 123,
    });

    for (let i = 0; i < 2; i++) {
      expect(Number(r1.importancesMean.data[r1.importancesMean.offset + i])).toBeCloseTo(
        Number(r2.importancesMean.data[r2.importancesMean.offset + i]),
        10
      );
    }
  });
});
