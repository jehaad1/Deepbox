import { describe, expect, it } from "vitest";
import { PCA } from "../src/ml/decomposition";
import { tensor } from "../src/ndarray";
import { toNum2D, toNumArr } from "./_helpers";

describe("PCA branches", () => {
  it("validates constructor and fit inputs", () => {
    expect(() => new PCA({ nComponents: 0 })).toThrow(/>= 1/);

    const pca = new PCA();
    expect(() => pca.fit(tensor([1, 2]))).toThrow(/2-dimensional/i);
  });

  it("fits, transforms, and inverse transforms", () => {
    const X = tensor([
      [2, 0],
      [0, 2],
      [2, 2],
    ]);

    const pca = new PCA({ nComponents: 1 });
    pca.fit(X);

    const transformed = pca.transform(X);
    expect(transformed.shape).toEqual([3, 1]);

    const reconstructed = pca.inverseTransform(transformed);
    expect(reconstructed.shape).toEqual([3, 2]);

    const ratio = toNumArr(pca.explainedVarianceRatio.toArray());
    expect(ratio.length).toBe(1);
    expect(ratio[0]).toBeGreaterThan(0);
    expect(ratio[0]).toBeLessThan(1);

    const pcaFull = new PCA({ nComponents: 2 });
    pcaFull.fit(X);
    const ratioFull = toNumArr(pcaFull.explainedVarianceRatio.toArray());
    expect(ratio[0]).toBeCloseTo(ratioFull[0] ?? 0, 6);
    const sumFull = ratioFull.reduce((a, b) => a + b, 0);
    expect(sumFull).toBeCloseTo(1, 6);

    expect(pca.components.shape).toEqual([1, 2]);
    expect(pca.explainedVariance.shape).toEqual([1]);
  });

  it("handles fitTransform and accessors errors", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const pca = new PCA({ nComponents: 2, whiten: true });
    const transformed = pca.fitTransform(X);
    expect(transformed.shape).toEqual([2, 2]);

    const pca2 = new PCA();
    expect(() => pca2.transform(X)).toThrow(/fitted/i);
    expect(() => pca2.inverseTransform(X)).toThrow(/fitted/i);
    expect(() => pca2.components).toThrow(/fitted/i);
    expect(() => pca2.explainedVariance).toThrow(/fitted/i);
    expect(() => pca2.explainedVarianceRatio).toThrow(/fitted/i);
  });

  it("applies whitening scaling when enabled", () => {
    const X = tensor([
      [2, 0],
      [0, 2],
      [2, 2],
    ]);

    const pcaNoWhite = new PCA({ nComponents: 1, whiten: false });
    const pcaWhite = new PCA({ nComponents: 1, whiten: true });
    pcaNoWhite.fit(X);
    pcaWhite.fit(X);

    const tNo = toNum2D(pcaNoWhite.transform(X).toArray());
    const tWh = toNum2D(pcaWhite.transform(X).toArray());

    const variance = Number(pcaNoWhite.explainedVariance.data[pcaNoWhite.explainedVariance.offset]);
    const scale = Math.sqrt(variance + 1e-12);

    for (let i = 0; i < tNo.length; i++) {
      expect(tWh[i]?.[0]).toBeCloseTo((tNo[i]?.[0] ?? 0) / scale, 6);
    }
  });

  it("rejects too many components", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const pca = new PCA({ nComponents: 3 });
    expect(() => pca.fit(X)).toThrow(/must be <= min/);
  });
});
