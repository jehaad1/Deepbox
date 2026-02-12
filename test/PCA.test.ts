import { describe, expect, it } from "vitest";
import { PCA } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("PCA", () => {
  it("should reduce dimensionality", () => {
    const X = tensor([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
      [1.9, 2.2],
      [3.1, 3.0],
    ]);

    const pca = new PCA({ nComponents: 1 });
    pca.fit(X);

    const XTransformed = pca.transform(X);
    expect(XTransformed.shape[0]).toBe(5);
    expect(XTransformed.shape[1]).toBe(1);
  });

  it("should have explained variance", () => {
    const X = tensor([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
      [1.9, 2.2],
      [3.1, 3.0],
    ]);

    const pca = new PCA({ nComponents: 2 });
    pca.fit(X);

    const explainedVariance = pca.explainedVariance;
    expect(explainedVariance.shape[0]).toBe(2);

    const explainedVarianceRatio = pca.explainedVarianceRatio;
    expect(explainedVarianceRatio.shape[0]).toBe(2);
  });

  it("should support inverse transform", () => {
    const X = tensor([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
    ]);

    const pca = new PCA({ nComponents: 1 });
    pca.fit(X);

    const XTransformed = pca.transform(X);
    const XReconstructed = pca.inverseTransform(XTransformed);

    expect(XReconstructed.shape).toEqual([3, 2]);
  });

  it("should support fitTransform", () => {
    const X = tensor([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
    ]);

    const pca = new PCA({ nComponents: 1 });
    const XTransformed = pca.fitTransform(X);

    expect(XTransformed.shape).toEqual([3, 1]);
  });
});
