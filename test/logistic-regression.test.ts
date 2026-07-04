import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError, NotFittedError } from "../src/core";
import { LogisticRegression } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("deepbox/ml - LogisticRegression", () => {
  it("should throw if predict is called before fit", () => {
    const model = new LogisticRegression();
    expect(() => model.predict(tensor([[1, 2]]))).toThrow("LogisticRegression must be fitted");
  });

  it("should handle multiclass inputs (OVR)", () => {
    const model = new LogisticRegression();
    const X = tensor([[0], [1], [2]]);
    const y = tensor([0, 2, 1]);
    expect(() => model.fit(X, y)).not.toThrow();
  });

  it("should fit and predict on a simple separable dataset", () => {
    const model = new LogisticRegression({ maxIter: 2000, learningRate: 0.2 });
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([0, 0, 1, 1]);

    model.fit(X, y);

    const pred = model.predict(X);
    expect(pred.shape).toEqual([4]);

    const acc = model.score(X, y);
    expect(acc).toBeGreaterThan(0.75);
  });

  it("predictProba should return shape (n_samples, 2) and rows sum to ~1", () => {
    const model = new LogisticRegression({ maxIter: 2000, learningRate: 0.2 });
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([0, 0, 1, 1]);

    model.fit(X, y);

    const proba = model.predictProba(X);
    expect(proba.shape).toEqual([4, 2]);

    for (let i = 0; i < 4; i++) {
      const p0 = Number(proba.data[proba.offset + i * 2]);
      const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
      expect(p0 + p1).toBeCloseTo(1, 6);
      expect(p0).toBeGreaterThanOrEqual(0);
      expect(p0).toBeLessThanOrEqual(1);
      expect(p1).toBeGreaterThanOrEqual(0);
      expect(p1).toBeLessThanOrEqual(1);
    }
  });

  it("should handle multiclass labels via OVR", () => {
    const model = new LogisticRegression();
    const X = tensor([[0], [1], [2]]);
    const y = tensor([0, 2, 1]);
    expect(() => model.fit(X, y)).not.toThrow();
  });

  it("setParams should reject unknown keys", () => {
    const model = new LogisticRegression();
    expect(() => model.setParams({ unknown: 123 })).toThrow(InvalidParameterError);
  });
});

describe("LogisticRegression edge cases", () => {
  describe("empty data validation", () => {
    it("should throw DataValidationError for zero samples", () => {
      const model = new LogisticRegression();
      const X = tensor([]).reshape([0, 2]);
      const y = tensor([]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("at least one sample");
    });
  });

  describe("NaN/Inf validation", () => {
    it("should throw DataValidationError for NaN in X", () => {
      const model = new LogisticRegression();
      const X = tensor([
        [1, NaN],
        [2, 3],
      ]);
      const y = tensor([0, 1]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("non-finite");
    });

    it("should throw DataValidationError for NaN in y", () => {
      const model = new LogisticRegression();
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([0, NaN]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
    });
  });

  describe("numerical stability", () => {
    it("should handle large logits without overflow", () => {
      const model = new LogisticRegression({ maxIter: 10, learningRate: 0.01 });
      const X = tensor([[100], [-100], [50], [-50]]);
      const y = tensor([1, 0, 1, 0]);
      expect(() => model.fit(X, y)).not.toThrow();

      const proba = model.predictProba(X);
      for (let i = 0; i < proba.size; i++) {
        const val = Number(proba.data[proba.offset + i]);
        expect(Number.isFinite(val)).toBe(true);
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });

    it("should produce valid probabilities that sum to 1", () => {
      const model = new LogisticRegression({
        maxIter: 1000,
        learningRate: 0.2,
      });
      const X = tensor([[0], [1], [2], [3]]);
      const y = tensor([0, 0, 1, 1]);
      model.fit(X, y);

      const proba = model.predictProba(X);
      for (let i = 0; i < 4; i++) {
        const p0 = Number(proba.data[proba.offset + i * 2]);
        const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
        expect(p0 + p1).toBeCloseTo(1, 6);
        expect(p0).toBeGreaterThanOrEqual(0);
        expect(p1).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe("error type assertions", () => {
    it("should throw NotFittedError when predicting before fit", () => {
      const model = new LogisticRegression();
      const X = tensor([[1, 2]]);
      expect(() => model.predict(X)).toThrow(NotFittedError);
    });

    it("should throw InvalidParameterError for C <= 0", () => {
      expect(() => new LogisticRegression({ C: 0 })).toThrow(InvalidParameterError);
      expect(() => new LogisticRegression({ C: 0 })).toThrow("C must be > 0");
    });

    it("should handle non-binary labels (multiclass)", () => {
      const model = new LogisticRegression();
      const X = tensor([[1], [2], [3]]);
      const y = tensor([0, 1, 2]);
      expect(() => model.fit(X, y)).not.toThrow();
    });
  });

  describe("convergence and accuracy", () => {
    it("should achieve high accuracy on separable data", () => {
      const model = new LogisticRegression({
        maxIter: 2000,
        learningRate: 0.2,
      });
      const X = tensor([[0], [1], [2], [3], [4], [5]]);
      const y = tensor([0, 0, 0, 1, 1, 1]);
      model.fit(X, y);

      const acc = model.score(X, y);
      expect(acc).toBeGreaterThan(0.95);
    });
  });
});
