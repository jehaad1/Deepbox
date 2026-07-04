import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../src/core";
import { LinearRegression } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("deepbox/ml - LinearRegression", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const model = new LinearRegression();
      expect(model).toBeDefined();
      expect(model.getParams()).toEqual({
        fitIntercept: true,
        normalize: false,
        copyX: true,
      });
    });

    it("should create with custom options", () => {
      const model = new LinearRegression({
        fitIntercept: false,
        normalize: true,
      });
      expect(model.getParams()).toEqual({
        fitIntercept: false,
        normalize: true,
        copyX: true,
      });
    });
  });

  describe("fit", () => {
    it("should validate input dimensions", () => {
      const model = new LinearRegression();
      const X_1d = tensor([1, 2, 3]);
      const y = tensor([1, 2, 3]);

      expect(() => model.fit(X_1d, y)).toThrow("X must be 2-dimensional");
    });

    it("should validate y dimensions", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y_2d = tensor([[1], [2]]);

      expect(() => model.fit(X, y_2d)).toThrow("y must be 1-dimensional");
    });

    it("should validate X and y have same number of samples", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([1, 2, 3]);

      expect(() => model.fit(X, y)).toThrow("X and y must have the same number of samples");
    });

    it("should fit simple linear relationship", () => {
      const model = new LinearRegression();
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);

      model.fit(X, y);
      expect(model.coef).toBeDefined();
      expect(model.coef.shape).toEqual([1]);
    });

    it("should fit with multiple features", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 3],
      ]);
      const y = tensor([6, 8, 9, 11]);

      model.fit(X, y);
      expect(model.coef).toBeDefined();
      expect(model.coef.shape).toEqual([2]);
    });

    it("should fit with normalization enabled", () => {
      const model = new LinearRegression({ normalize: true });
      const X = tensor([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
      ]);
      const y = tensor([6, 12, 18, 24]);
      model.fit(X, y);
      const preds = model.predict(X);
      expect(preds.shape).toEqual([4]);
      expect(Number(preds.data[preds.offset])).toBeCloseTo(6, 4);
    });

    it("should fit without intercept", () => {
      const model = new LinearRegression({ fitIntercept: false });
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);

      model.fit(X, y);
      expect(model.coef).toBeDefined();
      expect(model.intercept).toBeUndefined();
    });

    it("should return this for method chaining", () => {
      const model = new LinearRegression();
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([1, 2, 3, 4]);

      const result = model.fit(X, y);
      expect(result).toBe(model);
    });
  });

  describe("predict", () => {
    it("should throw if not fitted", () => {
      const model = new LinearRegression();
      const X = tensor([[1, 2]]);

      expect(() => model.predict(X)).toThrow("LinearRegression must be fitted");
    });

    it("should predict after fitting", () => {
      const model = new LinearRegression();
      const X_train = tensor([[1], [2], [3], [4]]);
      const y_train = tensor([2, 4, 6, 8]);

      model.fit(X_train, y_train);

      const X_test = tensor([[5], [6]]);
      const predictions = model.predict(X_test);

      expect(predictions).toBeDefined();
      expect(predictions.shape[0]).toBe(2);
    });

    it("should validate feature count matches training", () => {
      const model = new LinearRegression();
      const X_train = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y_train = tensor([1, 2]);

      model.fit(X_train, y_train);

      const X_test = tensor([[1, 2, 3]]);
      expect(() => model.predict(X_test)).toThrow("features");
    });

    it("should validate X is 2D", () => {
      const model = new LinearRegression();
      const X_train = tensor([[1], [2]]);
      const y_train = tensor([1, 2]);

      model.fit(X_train, y_train);

      const X_test = tensor([1, 2, 3]);
      expect(() => model.predict(X_test)).toThrow("2-dimensional");
    });
  });

  describe("score", () => {
    it("should throw if not fitted", () => {
      const model = new LinearRegression();
      const X = tensor([[1, 2]]);
      const y = tensor([1]);

      expect(() => model.score(X, y)).toThrow("LinearRegression must be fitted");
    });

    it("should compute R^2 score after fitting", () => {
      const model = new LinearRegression();
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);

      model.fit(X, y);
      const r2 = model.score(X, y);

      expect(typeof r2).toBe("number");
      // R^2 should be between 0 and 1 for a good fit
      expect(r2).toBeGreaterThanOrEqual(-1);
      expect(r2).toBeLessThanOrEqual(1);
    });

    it("should return 1.0 for perfect fit on constant y", () => {
      const model = new LinearRegression();
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([5, 5, 5, 5]);

      model.fit(X, y);
      const r2 = model.score(X, y);

      // For constant y, if predictions are also constant, R^2 should be 1.0
      expect(r2).toBeCloseTo(1.0, 5);
    });
  });

  describe("getParams/setParams", () => {
    it("should get parameters", () => {
      const model = new LinearRegression({ fitIntercept: false });
      const params = model.getParams();

      expect(params.fitIntercept).toBe(false);
      expect(params.normalize).toBe(false);
      expect(params.copyX).toBe(true);
    });

    it("should support method chaining", () => {
      const model = new LinearRegression();
      const result = model.setParams({ fitIntercept: false });

      expect(result).toBe(model);
    });
  });
});

describe("LinearRegression integration", () => {
  it("should recover a simple linear relationship with intercept", () => {
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([1, 3, 5, 7]); // y = 2x + 1

    const model = new LinearRegression({ fitIntercept: true });
    model.fit(X, y);

    const coef = model.coef;
    const intercept = model.intercept;

    expect(Number(coef.data[coef.offset])).toBeCloseTo(2, 5);
    expect(intercept).not.toBeUndefined();
    expect(Number(intercept?.data[intercept.offset])).toBeCloseTo(1, 5);

    const preds = model.predict(X);
    const values = [];
    for (let i = 0; i < preds.size; i++) {
      values.push(Number(preds.data[preds.offset + i]));
    }
    expect(values).toEqual([1, 3, 5, 7]);
  });

  it("should fit without intercept when disabled", () => {
    const X = tensor([[1], [2], [3]]);
    const y = tensor([2, 4, 6]); // y = 2x

    const model = new LinearRegression({ fitIntercept: false });
    model.fit(X, y);

    expect(model.intercept).toBeUndefined();
    const coef = model.coef;
    expect(Number(coef.data[coef.offset])).toBeCloseTo(2, 5);
  });

  it("should handle collinear features via least squares", () => {
    const X = tensor([
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
    const y = tensor([2, 4, 6, 8]); // y = 2 * x1 = 2 * x2

    const model = new LinearRegression({ fitIntercept: true });
    expect(() => model.fit(X, y)).not.toThrow();
    const coef = model.coef;
    for (let i = 0; i < coef.size; i++) {
      const v = Number(coef.data[coef.offset + i]);
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

describe("LinearRegression edge cases", () => {
  describe("empty data validation", () => {
    it("should throw DataValidationError for zero samples", () => {
      const model = new LinearRegression();
      const X = tensor([]).reshape([0, 2]);
      const y = tensor([]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("at least one sample");
    });

    it("should throw DataValidationError for zero features", () => {
      const model = new LinearRegression();
      const X = tensor([[], []]);
      const y = tensor([1, 2]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("at least one feature");
    });
  });

  describe("NaN/Inf validation", () => {
    it("should throw DataValidationError for NaN in X", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, NaN],
        [2, 3],
      ]);
      const y = tensor([1, 2]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("non-finite");
    });

    it("should throw DataValidationError for Inf in X", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, Infinity],
        [2, 3],
      ]);
      const y = tensor([1, 2]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("non-finite");
    });

    it("should throw DataValidationError for NaN in y", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([1, NaN]);
      expect(() => model.fit(X, y)).toThrow(DataValidationError);
      expect(() => model.fit(X, y)).toThrow("non-finite");
    });

    it("should throw DataValidationError for NaN in predict X", () => {
      const model = new LinearRegression();
      const X_train = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y_train = tensor([1, 2]);
      model.fit(X_train, y_train);

      const X_test = tensor([[1, NaN]]);
      expect(() => model.predict(X_test)).toThrow(DataValidationError);
    });
  });

  describe("single sample/feature", () => {
    it("should handle single sample", () => {
      const model = new LinearRegression();
      const X = tensor([[1, 2]]);
      const y = tensor([3]);
      expect(() => model.fit(X, y)).not.toThrow();
      const pred = model.predict(X);
      expect(pred.shape).toEqual([1]);
    });

    it("should handle single feature", () => {
      const model = new LinearRegression();
      const X = tensor([[1], [2], [3]]);
      const y = tensor([2, 4, 6]);
      model.fit(X, y);
      const pred = model.predict(X);
      expect(pred.shape).toEqual([3]);
      const r2 = model.score(X, y);
      expect(r2).toBeGreaterThan(0.99);
    });
  });

  describe("error type assertions", () => {
    it("should throw NotFittedError when predicting before fit", () => {
      const model = new LinearRegression();
      const X = tensor([[1, 2]]);
      expect(() => model.predict(X)).toThrow(NotFittedError);
    });

    it("should throw ShapeError for wrong X dimensions", () => {
      const model = new LinearRegression();
      const X = tensor([1, 2, 3]);
      const y = tensor([1, 2, 3]);
      expect(() => model.fit(X, y)).toThrow(ShapeError);
    });

    it("should throw InvalidParameterError for unknown parameter", () => {
      const model = new LinearRegression();
      expect(() => model.setParams({ unknown: 123 })).toThrow(InvalidParameterError);
    });
  });

  describe("numerical stability", () => {
    it("should handle large values without overflow", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1e8, 2e8],
        [3e8, 4e8],
      ]);
      const y = tensor([1e8, 2e8]);
      expect(() => model.fit(X, y)).not.toThrow();
      const coef = model.coef;
      for (let i = 0; i < coef.size; i++) {
        expect(Number.isFinite(Number(coef.data[coef.offset + i]))).toBe(true);
      }
    });

    it("should handle small values without underflow", () => {
      const model = new LinearRegression();
      const X = tensor([
        [1e-8, 2e-8],
        [3e-8, 4e-8],
      ]);
      const y = tensor([1e-8, 2e-8]);
      expect(() => model.fit(X, y)).not.toThrow();
      const pred = model.predict(X);
      expect(pred.shape).toEqual([2]);
    });
  });
});
