import { describe, expect, it } from "vitest";
import { Lasso } from "../src/ml/linear/Lasso";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { Ridge } from "../src/ml/linear/Ridge";
import { tensor } from "../src/ndarray";

describe("ML Linear Coverage & Edge Cases", () => {
  describe("Lasso", () => {
    it("should respect positive=true constraint", () => {
      // Create data where the coefficient would naturally be negative
      // y = -2 * x
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([-2, -4, -6, -8]);

      const model = new Lasso({ positive: true, alpha: 0.1 });
      model.fit(X, y);

      const coef = model.coef;
      // All coefficients should be >= 0 (likely 0 in this case)
      for (let i = 0; i < coef.size; i++) {
        expect(coef.data[coef.offset + i]).toBeGreaterThanOrEqual(0);
      }
    });

    it("should validate randomState", () => {
      expect(() => new Lasso({ randomState: Infinity })).toThrow(
        /randomState must be a finite number/
      );
      expect(() => new Lasso({ randomState: NaN })).toThrow(/randomState must be a finite number/);
    });

    it("should support warmStart", () => {
      const X = tensor([
        [1, 0],
        [0, 1],
        [1, 1],
      ]);
      const y = tensor([1, 1, 2]);

      const model = new Lasso({ warmStart: true, maxIter: 5 });
      model.fit(X, y);

      // We can't easily check internal state without mocking, but we can verify it runs
      // and produces different results if we continue fitting (simulating more iterations)
      // or simply that it doesn't crash and returns a valid model.

      // Let's verify it reuses coefficients by checking if we can fit incrementally
      const coef1 = model.coef; // Get reference

      // Fit again. If warmStart works, it should use coef1 as init.
      // Since maxIter is small, it might not have converged.
      model.fit(X, y);
      const coef2 = model.coef;

      expect(coef2.shape).toEqual(coef1.shape);
      expect(model.nIter).toBeGreaterThan(0);
    });

    it("should support random selection", () => {
      const X = tensor([
        [1, 0],
        [0, 1],
        [1, 1],
      ]);
      const y = tensor([1, 1, 2]);

      const model = new Lasso({ selection: "random", randomState: 42 });
      model.fit(X, y);
      expect(model.coef.size).toBe(2);
    });
  });

  describe("Ridge", () => {
    it("should validate solver selection", () => {
      const model = new Ridge({ solver: "auto" });
      const params = model.getParams();
      expect(params.solver).toBe("auto");
    });

    it("should support fitIntercept=false", () => {
      const X = tensor([[1], [2], [3]]);
      const y = tensor([1, 2, 3]);
      const model = new Ridge({ fitIntercept: false });
      model.fit(X, y);
      expect(model.intercept).toBe(0);
      expect(model.score(X, y)).toBeGreaterThan(0.9);
    });
  });

  describe("LogisticRegression", () => {
    it("should validate C parameter", () => {
      expect(() => new LogisticRegression({ C: -1 })).toThrow(/C must be > 0/);
      expect(() => new LogisticRegression({ C: 0 })).toThrow(/C must be > 0/);
    });

    it("should support penalty='none'", () => {
      const X = tensor([[0], [1], [2], [3]]);
      const y = tensor([0, 0, 1, 1], { dtype: "int32" });

      const model = new LogisticRegression({ penalty: "none" });
      model.fit(X, y);
      expect(model.score(X, y)).toBeGreaterThan(0.5);
    });

    it("should handle fitIntercept=false", () => {
      const X = tensor([[0], [1]]);
      const y = tensor([0, 1], { dtype: "int32" });
      const model = new LogisticRegression({ fitIntercept: false });
      model.fit(X, y);
      expect(model.intercept).toBe(0);
    });
  });
});
