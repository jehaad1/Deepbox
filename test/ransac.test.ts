import { describe, expect, it } from "vitest";
import { RANSACRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("RANSACRegressor", () => {
  // Linear data with outliers
  const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
  const y = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 100]); // last is outlier

  it("fit and predict", () => {
    const ransac = new RANSACRegressor({ minSamples: 3, maxTrials: 50, randomState: 42 });
    ransac.fit(X, y);
    const pred = ransac.predict(X);
    expect(pred.size).toBe(10);
  });

  it("is robust to outliers", () => {
    const ransac = new RANSACRegressor({ minSamples: 3, maxTrials: 100, randomState: 42 });
    ransac.fit(X, y);
    const pred = ransac.predict(tensor([[5]]));
    // Should predict ~10 (y=2x) not influenced by outlier
    const predVal = Number(pred.data[pred.offset]);
    expect(predVal).toBeGreaterThan(7);
    expect(predVal).toBeLessThan(13);
  });

  it("inlierMask identifies outliers", () => {
    const ransac = new RANSACRegressor({ minSamples: 3, maxTrials: 100, randomState: 42 });
    ransac.fit(X, y);
    const mask = ransac.inlierMask;
    expect(mask.length).toBe(10);
    // Last point (outlier) should typically not be an inlier
    // Most of the first 9 should be inliers
    let nInliers = 0;
    for (let i = 0; i < 9; i++) {
      if (mask[i]) nInliers++;
    }
    expect(nInliers).toBeGreaterThanOrEqual(7);
  });

  it("score on clean data is high", () => {
    const Xclean = tensor([[1], [2], [3], [4], [5]]);
    const yClean = tensor([2, 4, 6, 8, 10]);
    const ransac = new RANSACRegressor({ minSamples: 3, maxTrials: 50, randomState: 42 });
    ransac.fit(Xclean, yClean);
    const r2 = ransac.score(Xclean, yClean);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("throws when not fitted", () => {
    const ransac = new RANSACRegressor();
    expect(() => ransac.predict(X)).toThrow();
    expect(() => ransac.inlierMask).toThrow();
  });

  it("throws for invalid minSamples", () => {
    expect(() => new RANSACRegressor({ minSamples: 0 })).toThrow();
  });

  it("throws for invalid maxTrials", () => {
    expect(() => new RANSACRegressor({ maxTrials: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const ransac = new RANSACRegressor({ minSamples: 10, maxTrials: 200, randomState: 7 });
    const params = ransac.getParams();
    expect(params.minSamples).toBe(10);
    expect(params.maxTrials).toBe(200);
    expect(params.randomState).toBe(7);
  });

  it("custom residualThreshold works", () => {
    const ransac = new RANSACRegressor({
      minSamples: 3,
      maxTrials: 50,
      residualThreshold: 5,
      randomState: 42,
    });
    ransac.fit(X, y);
    const pred = ransac.predict(X);
    expect(pred.size).toBe(10);
  });
});
