import { describe, expect, it } from "vitest";
import { GaussianRandomProjection, IsotonicRegression, KernelRidge } from "../src/ml";
import { tensor } from "../src/ndarray";

const f64 = { dtype: "float64" as const };

describe("KernelRidge", () => {
  it("should fit and predict with linear kernel", () => {
    // Simple linear data: y = 2*x1 + 3*x2
    const X = tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
      ],
      f64
    );
    const y = tensor([2, 3, 5, 7], f64);

    const model = new KernelRidge({ alpha: 0.01, kernel: "linear" });
    model.fit(X, y);
    const pred = model.predict(X);

    expect(pred.shape).toEqual([4]);
    // Should fit training data closely
    for (let i = 0; i < 4; i++) {
      expect(Number(pred.data[i])).toBeCloseTo(Number(y.data[i]), 0);
    }
  });

  it("should fit with rbf kernel", () => {
    const X = tensor(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      f64
    );
    const y = tensor([0, 1, 1, 0], f64); // XOR-like

    const model = new KernelRidge({ alpha: 0.1, kernel: "rbf", gamma: 1.0 });
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("should fit with polynomial kernel", () => {
    const X = tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      f64
    );
    const y = tensor([1, 2, 3], f64);

    const model = new KernelRidge({ kernel: "polynomial", degree: 2, gamma: 1, coef0: 1 });
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([3]);
  });

  it("should compute R^2 score", () => {
    const X = tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
      ],
      f64
    );
    const y = tensor([2, 3, 5, 7], f64);

    const model = new KernelRidge({ alpha: 0.01, kernel: "linear" });
    model.fit(X, y);
    const r2 = model.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("should throw when not fitted", () => {
    const model = new KernelRidge();
    const X = tensor([[1, 0]], f64);
    expect(() => model.predict(X)).toThrow();
  });

  it("should get and set params", () => {
    const model = new KernelRidge({ alpha: 2.0 });
    expect(model.getParams()).toHaveProperty("alpha", 2.0);
    model.setParams({ alpha: 0.5 });
    expect(model.getParams()).toHaveProperty("alpha", 0.5);
  });
});

describe("IsotonicRegression", () => {
  it("should fit monotone increasing data", () => {
    const X = tensor([1, 2, 3, 4, 5], f64);
    const y = tensor([1, 2, 3, 4, 5], f64);

    const model = new IsotonicRegression({ increasing: true });
    model.fit(X, y);
    const pred = model.predict(X);

    expect(pred.shape).toEqual([5]);
    for (let i = 0; i < 5; i++) {
      expect(Number(pred.data[i])).toBeCloseTo(i + 1, 4);
    }
  });

  it("should enforce monotonicity on non-monotone data", () => {
    const X = tensor([1, 2, 3, 4, 5], f64);
    const y = tensor([1, 3, 2, 4, 5], f64); // 3 > 2 violates monotonicity

    const model = new IsotonicRegression({ increasing: true });
    model.fit(X, y);
    const pred = model.predict(X);

    // Predictions should be non-decreasing
    for (let i = 1; i < 5; i++) {
      expect(Number(pred.data[i])).toBeGreaterThanOrEqual(Number(pred.data[i - 1]) - 1e-10);
    }
  });

  it("should handle decreasing mode", () => {
    const X = tensor([1, 2, 3, 4, 5], f64);
    const y = tensor([5, 4, 3, 2, 1], f64);

    const model = new IsotonicRegression({ increasing: false });
    model.fit(X, y);
    const pred = model.predict(X);

    // Predictions should be non-increasing
    for (let i = 1; i < 5; i++) {
      expect(Number(pred.data[i])).toBeLessThanOrEqual(Number(pred.data[i - 1]) + 1e-10);
    }
  });

  it("should compute R^2 score", () => {
    const X = tensor([1, 2, 3, 4, 5], f64);
    const y = tensor([1, 2, 3, 4, 5], f64);

    const model = new IsotonicRegression();
    model.fit(X, y);
    const r2 = model.score(X, y);
    expect(r2).toBeGreaterThan(0.99);
  });

  it("should throw when not fitted", () => {
    const model = new IsotonicRegression();
    expect(() => model.predict(tensor([1, 2], f64))).toThrow();
  });

  it("should accept 2-D single-column X", () => {
    const X = tensor([[1], [2], [3], [4]], f64);
    const y = tensor([1, 2, 3, 4], f64);

    const model = new IsotonicRegression();
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);
  });
});

describe("GaussianRandomProjection", () => {
  it("should reduce dimensionality", () => {
    const X = tensor(
      [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
      ],
      f64
    );

    const proj = new GaussianRandomProjection({ nComponents: 2, seed: 42 });
    proj.fit(X);
    const Xr = proj.transform(X);

    expect(Xr.shape).toEqual([3, 2]);
  });

  it("should produce deterministic results with seed", () => {
    const X = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f64
    );

    const proj1 = new GaussianRandomProjection({ nComponents: 2, seed: 123 });
    const proj2 = new GaussianRandomProjection({ nComponents: 2, seed: 123 });

    proj1.fit(X);
    proj2.fit(X);

    const r1 = proj1.transform(X);
    const r2 = proj2.transform(X);

    for (let i = 0; i < r1.size; i++) {
      expect(Number(r1.data[i])).toBeCloseTo(Number(r2.data[i]), 10);
    }
  });

  it("should work with fitTransform", () => {
    const X = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      f64
    );

    const proj = new GaussianRandomProjection({ nComponents: 2, seed: 42 });
    const Xr = proj.fitTransform(X);
    expect(Xr.shape).toEqual([3, 2]);
  });

  it("should throw when not fitted", () => {
    const proj = new GaussianRandomProjection({ nComponents: 2 });
    const X = tensor([[1, 2]], f64);
    expect(() => proj.transform(X)).toThrow();
  });

  it("should throw for invalid nComponents", () => {
    expect(() => new GaussianRandomProjection({ nComponents: 0 })).toThrow();
    expect(() => new GaussianRandomProjection({ nComponents: -1 })).toThrow();
  });

  it("should get and set params", () => {
    const proj = new GaussianRandomProjection({ nComponents: 5 });
    expect(proj.getParams()).toHaveProperty("nComponents", 5);
    proj.setParams({ nComponents: 10 });
    expect(proj.getParams()).toHaveProperty("nComponents", 10);
  });
});
