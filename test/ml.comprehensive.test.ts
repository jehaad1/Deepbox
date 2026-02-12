import { describe, expect, it } from "vitest";
import { Lasso, LinearRegression, LogisticRegression, Ridge } from "../src/ml";
import { tensor } from "../src/ndarray";

/**
 * Comprehensive test suite for ML package
 *
 * This file contains extensive tests covering:
 * - Edge cases (empty data, single sample, NaN, Infinity)
 * - Numerical stability
 * - Parameter validation
 * - Convergence behavior
 * - Mathematical correctness
 * - Performance characteristics
 * - API compliance
 */

describe("deepbox/ml - Comprehensive ML Package Tests", () => {
  describe("LinearRegression - Comprehensive", () => {
    describe("Edge Cases", () => {
      it("should handle single sample", () => {
        const model = new LinearRegression();
        const X = tensor([[1, 2]]);
        const y = tensor([3]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([1]);
      });

      it("should handle single feature", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2], [3]]);
        const y = tensor([2, 4, 6]);

        model.fit(X, y);
        const coef = model.coef;

        expect(coef.shape).toEqual([1]);
        expect(Number(coef.data[0])).toBeCloseTo(2.0, 2);
      });

      it("should handle zero features gracefully", () => {
        const model = new LinearRegression();
        const X = tensor([[], []]);
        const y = tensor([1, 2]);

        // Zero features should throw DataValidationError
        expect(() => model.fit(X, y)).toThrow();
      });

      it("should handle constant y", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2], [3]]);
        const y = tensor([5, 5, 5]);

        model.fit(X, y);
        const score = model.score(X, y);

        expect(score).toBeCloseTo(1.0, 5);
      });

      it("should handle constant X column", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 1],
          [1, 2],
          [1, 3],
        ]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle negative values", () => {
        const model = new LinearRegression();
        const X = tensor([
          [-1, -2],
          [-3, -4],
        ]);
        const y = tensor([-1, -2]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([2]);
      });

      it("should handle large values", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1e6, 1e6],
          [2e6, 2e6],
        ]);
        const y = tensor([1e6, 2e6]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([2]);
      });

      it("should handle small values", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1e-6, 1e-6],
          [2e-6, 2e-6],
        ]);
        const y = tensor([1e-6, 2e-6]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([2]);
      });
    });

    describe("Mathematical Correctness", () => {
      it("should fit perfect linear relationship y = 2x", () => {
        const model = new LinearRegression({ fitIntercept: false });
        const X = tensor([[1], [2], [3], [4], [5]]);
        const y = tensor([2, 4, 6, 8, 10]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeCloseTo(2.0, 5);
      });

      it("should fit perfect linear relationship y = 3x + 2", () => {
        const model = new LinearRegression({ fitIntercept: true });
        const X = tensor([[1], [2], [3], [4], [5]]);
        const y = tensor([5, 8, 11, 14, 17]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);
        const intercept = Number(model.intercept?.data[0]);

        expect(coef).toBeCloseTo(3.0, 3);
        expect(intercept).toBeCloseTo(2.0, 3);
      });

      it("should fit multivariate linear relationship", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 0],
          [0, 1],
          [1, 1],
          [2, 1],
        ]);
        const y = tensor([1, 2, 3, 4]);

        model.fit(X, y);
        const pred = model.predict(X);

        for (let i = 0; i < y.size; i++) {
          expect(Math.abs(Number(pred.data[i]) - Number(y.data[i]))).toBeLessThan(0.5);
        }
      });

      it("should achieve R² close to 1 for perfect fit", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2], [3], [4]]);
        const y = tensor([2, 4, 6, 8]);

        model.fit(X, y);
        const r2 = model.score(X, y);

        expect(r2).toBeGreaterThan(0.99);
      });

      it("should handle orthogonal features", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 0],
          [0, 1],
          [1, 0],
          [0, 1],
        ]);
        const y = tensor([1, 2, 1, 2]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([4]);
      });
    });

    describe("Parameter Validation", () => {
      it("should validate X dimensionality in fit", () => {
        const model = new LinearRegression();
        const X = tensor([1, 2, 3]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).toThrow("2-dimensional");
      });

      it("should validate y dimensionality in fit", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2]]);
        const y = tensor([[1], [2]]);

        expect(() => model.fit(X, y)).toThrow("1-dimensional");
      });

      it("should validate sample count match", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2]]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).toThrow("same number of samples");
      });

      it("should validate X dimensionality in predict", () => {
        const model = new LinearRegression();
        model.fit(tensor([[1], [2]]), tensor([1, 2]));

        expect(() => model.predict(tensor([1, 2]))).toThrow("2-dimensional");
      });

      it("should validate feature count in predict", () => {
        const model = new LinearRegression();
        model.fit(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([1, 2])
        );

        expect(() => model.predict(tensor([[1, 2, 3]]))).toThrow("features");
      });

      it("should throw when predicting before fit", () => {
        const model = new LinearRegression();
        expect(() => model.predict(tensor([[1]]))).toThrow("fitted");
      });

      it("should throw when scoring before fit", () => {
        const model = new LinearRegression();
        expect(() => model.score(tensor([[1]]), tensor([1]))).toThrow("fitted");
      });

      it("should throw when accessing coef before fit", () => {
        const model = new LinearRegression();
        expect(() => model.coef).toThrow("fitted");
      });

      it("should throw when accessing intercept before fit", () => {
        const model = new LinearRegression();
        expect(() => model.intercept).toThrow("fitted");
      });
    });

    describe("API Compliance", () => {
      it("should support method chaining on fit", () => {
        const model = new LinearRegression();
        const result = model.fit(tensor([[1], [2]]), tensor([1, 2]));

        expect(result).toBe(model);
      });

      it("should support method chaining on setParams", () => {
        const model = new LinearRegression();
        const result = model.setParams({ fitIntercept: false });

        expect(result).toBe(model);
      });

      it("should get default parameters", () => {
        const model = new LinearRegression();
        const params = model.getParams();

        expect(params.fitIntercept).toBe(true);
        expect(params.normalize).toBe(false);
        expect(params.copyX).toBe(true);
      });

      it("should set and get parameters", () => {
        const model = new LinearRegression();
        model.setParams({ fitIntercept: false, normalize: true });
        const params = model.getParams();

        expect(params.fitIntercept).toBe(false);
        expect(params.normalize).toBe(true);
      });

      it("should reject invalid parameter types", () => {
        const model = new LinearRegression();
        expect(() => model.setParams({ fitIntercept: "true" })).toThrow();
      });

      it("should reject unknown parameters", () => {
        const model = new LinearRegression();
        expect(() => model.setParams({ unknown: 123 })).toThrow("Unknown parameter");
      });
    });

    describe("Intercept Behavior", () => {
      it("should fit intercept by default", () => {
        const model = new LinearRegression();
        const X = tensor([[1], [2], [3]]);
        const y = tensor([3, 5, 7]);

        model.fit(X, y);

        expect(model.intercept).toBeDefined();
      });

      it("should not fit intercept when disabled", () => {
        const model = new LinearRegression({ fitIntercept: false });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([2, 4, 6]);

        model.fit(X, y);

        expect(model.intercept).toBeUndefined();
      });

      it("should handle centered data without intercept", () => {
        const model = new LinearRegression({ fitIntercept: false });
        const X = tensor([[-1], [0], [1]]);
        const y = tensor([-2, 0, 2]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeCloseTo(2.0, 5);
      });
    });

    describe("Numerical Stability", () => {
      it("should handle collinear features", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 2],
          [2, 4],
          [3, 6],
        ]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle near-zero variance features", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 1.0001],
          [2, 1.0002],
          [3, 1.0003],
        ]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle mixed scale features", () => {
        const model = new LinearRegression();
        const X = tensor([
          [1, 1000],
          [2, 2000],
          [3, 3000],
        ]);
        const y = tensor([1, 2, 3]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([3]);
      });
    });
  });

  describe("Ridge - Comprehensive", () => {
    describe("Regularization Behavior", () => {
      it("should reduce coefficient magnitudes with regularization", () => {
        const X = tensor([
          [1, 0],
          [0, 1],
          [1, 1],
        ]);
        const y = tensor([1, 1, 2]);

        const model1 = new Ridge({ alpha: 0.0 });
        const model2 = new Ridge({ alpha: 10.0 });

        model1.fit(X, y);
        model2.fit(X, y);

        const coef1 = model1.coef;
        const coef2 = model2.coef;

        let norm1 = 0,
          norm2 = 0;
        for (let i = 0; i < coef1.size; i++) {
          norm1 += Number(coef1.data[i]) ** 2;
          norm2 += Number(coef2.data[i]) ** 2;
        }

        expect(norm2).toBeLessThan(norm1);
      });

      it("should accept alpha = 0", () => {
        const model = new Ridge({ alpha: 0.0 });
        const X = tensor([[1], [2]]);
        const y = tensor([1, 2]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should reject negative alpha", () => {
        const model = new Ridge({ alpha: -1.0 });
        const X = tensor([[1], [2]]);
        const y = tensor([1, 2]);

        expect(() => model.fit(X, y)).toThrow("alpha must be >= 0");
      });

      it("should handle very large alpha", () => {
        const model = new Ridge({ alpha: 1e10 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        model.fit(X, y);
        const coef = model.coef;

        expect(Math.abs(Number(coef.data[0]))).toBeLessThan(0.01);
      });

      it("should handle very small alpha", () => {
        const model = new Ridge({ alpha: 1e-10 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([2, 4, 6]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeCloseTo(2.0, 3);
      });
    });

    describe("Edge Cases", () => {
      it("should handle single sample", () => {
        const model = new Ridge();
        const X = tensor([[1, 2]]);
        const y = tensor([3]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(pred.shape).toEqual([1]);
      });

      it("should handle constant y", () => {
        const model = new Ridge();
        const X = tensor([[1], [2], [3]]);
        const y = tensor([5, 5, 5]);

        model.fit(X, y);
        const score = model.score(X, y);

        expect(score).toBeCloseTo(1.0, 5);
      });

      it("should handle large feature count", () => {
        const model = new Ridge({ alpha: 1.0 });
        const X = tensor([
          [1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6],
        ]);
        const y = tensor([1, 2]);

        expect(() => model.fit(X, y)).not.toThrow();
      });
    });

    describe("API Compliance", () => {
      it("should support method chaining", () => {
        const model = new Ridge();
        const result = model.fit(tensor([[1], [2]]), tensor([1, 2]));

        expect(result).toBe(model);
      });

      it("should get parameters", () => {
        const model = new Ridge({ alpha: 0.5 });
        const params = model.getParams();

        expect(params.alpha).toBe(0.5);
      });

      it("should set parameters", () => {
        const model = new Ridge();
        model.setParams({ alpha: 2.0, fitIntercept: false });
        const params = model.getParams();

        expect(params.alpha).toBe(2.0);
        expect(params.fitIntercept).toBe(false);
      });

      it("should reject invalid parameter types", () => {
        const model = new Ridge();
        expect(() => model.setParams({ alpha: "1.0" })).toThrow();
      });

      it("should throw when predicting before fit", () => {
        const model = new Ridge();
        expect(() => model.predict(tensor([[1]]))).toThrow("fitted");
      });

      it("should throw when accessing coef before fit", () => {
        const model = new Ridge();
        expect(() => model.coef).toThrow("fitted");
      });

      it("should throw when accessing intercept before fit", () => {
        const model = new Ridge();
        expect(() => model.intercept).toThrow("fitted");
      });
    });

    describe("Mathematical Correctness", () => {
      it("should converge to OLS when alpha approaches 0", () => {
        const X = tensor([[1], [2], [3], [4]]);
        const y = tensor([2, 4, 6, 8]);

        const model = new Ridge({ alpha: 1e-10 });
        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeCloseTo(2.0, 3);
      });

      it("should shrink coefficients towards zero with high alpha", () => {
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        const model = new Ridge({ alpha: 100.0 });
        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(Math.abs(coef)).toBeLessThan(1.0);
      });
    });
  });

  describe("Lasso - Comprehensive", () => {
    describe("Sparsity Behavior", () => {
      it("should produce sparse solutions with high alpha", () => {
        const model = new Lasso({ alpha: 10.0, maxIter: 2000 });
        const X = tensor([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 1],
        ]);
        const y = tensor([1, 1, 1, 3]);

        model.fit(X, y);
        const coef = model.coef;

        let zeroCount = 0;
        for (let i = 0; i < coef.size; i++) {
          if (Math.abs(Number(coef.data[i])) < 0.01) {
            zeroCount++;
          }
        }

        expect(zeroCount).toBeGreaterThan(0);
      });

      it("should have fewer non-zero coefficients than Ridge", () => {
        const X = tensor([
          [1, 2, 3],
          [2, 3, 4],
          [3, 4, 5],
        ]);
        const y = tensor([1, 2, 3]);

        const lasso = new Lasso({ alpha: 1.0, maxIter: 2000 });
        const ridge = new Ridge({ alpha: 1.0 });

        lasso.fit(X, y);
        ridge.fit(X, y);

        let lassoNonZero = 0,
          ridgeNonZero = 0;
        for (let i = 0; i < lasso.coef.size; i++) {
          if (Math.abs(Number(lasso.coef.data[i])) > 0.01) lassoNonZero++;
          if (Math.abs(Number(ridge.coef.data[i])) > 0.01) ridgeNonZero++;
        }

        expect(lassoNonZero).toBeLessThanOrEqual(ridgeNonZero);
      });

      it("should set coefficients exactly to zero", () => {
        const model = new Lasso({ alpha: 5.0, maxIter: 2000 });
        const X = tensor([
          [1, 0],
          [0, 1],
          [1, 1],
        ]);
        const y = tensor([1, 0, 1]);

        model.fit(X, y);
        const coef = model.coef;

        let hasExactZero = false;
        for (let i = 0; i < coef.size; i++) {
          if (Number(coef.data[i]) === 0) {
            hasExactZero = true;
          }
        }

        expect(hasExactZero).toBe(true);
      });
    });

    describe("Convergence Behavior", () => {
      it("should converge within maxIter", () => {
        const model = new Lasso({ alpha: 0.1, maxIter: 1000, tol: 1e-4 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        model.fit(X, y);

        expect(model.nIter).toBeDefined();
        expect(model.nIter).toBeLessThanOrEqual(1000);
      });

      it("should converge faster with larger tolerance", () => {
        const X = tensor([[1], [2], [3], [4]]);
        const y = tensor([1, 2, 3, 4]);

        const model1 = new Lasso({ alpha: 0.1, maxIter: 1000, tol: 1e-1 });
        const model2 = new Lasso({ alpha: 0.1, maxIter: 1000, tol: 1e-8 });

        model1.fit(X, y);
        model2.fit(X, y);

        expect(model1.nIter).toBeDefined();
        expect(model2.nIter).toBeDefined();
        if (model1.nIter === undefined || model2.nIter === undefined) return;

        expect(model1.nIter).toBeLessThanOrEqual(model2.nIter);
      });

      it("should handle maxIter = 1", () => {
        const model = new Lasso({ maxIter: 1 });
        const X = tensor([[1], [2]]);
        const y = tensor([1, 2]);

        expect(() => model.fit(X, y)).not.toThrow();
      });
    });

    describe("Selection Strategy", () => {
      it("should support cyclic selection", () => {
        const model = new Lasso({ selection: "cyclic", maxIter: 100 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should support random selection", () => {
        const model = new Lasso({ selection: "random", maxIter: 100 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        expect(() => model.fit(X, y)).not.toThrow();
      });
    });

    describe("Positive Constraint", () => {
      it("should enforce non-negative coefficients when positive=true", () => {
        const model = new Lasso({ positive: true, alpha: 0.1, maxIter: 2000 });
        const X = tensor([
          [1, -1],
          [2, -2],
          [3, -3],
        ]);
        const y = tensor([1, 2, 3]);

        model.fit(X, y);
        const coef = model.coef;

        for (let i = 0; i < coef.size; i++) {
          expect(Number(coef.data[i])).toBeGreaterThanOrEqual(0);
        }
      });

      it("should allow negative coefficients when positive=false", () => {
        const model = new Lasso({ positive: false, alpha: 0.1, maxIter: 2000 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([-1, -2, -3]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeLessThan(0);
      });
    });

    describe("Warm Start", () => {
      it("should support warm start", () => {
        const X = tensor([[1], [2], [3]]);
        const y = tensor([1, 2, 3]);

        const model = new Lasso({ warmStart: true, maxIter: 100 });

        model.fit(X, y);
        const iter1 = model.nIter;

        model.fit(X, y);
        const iter2 = model.nIter;

        expect(iter1).toBeDefined();
        expect(iter2).toBeDefined();
        if (iter1 === undefined || iter2 === undefined) return;
        expect(iter2).toBeLessThanOrEqual(iter1);
      });
    });

    describe("Edge Cases", () => {
      it("should handle alpha = 0", () => {
        const model = new Lasso({ alpha: 0.0, maxIter: 2000 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([2, 4, 6]);

        model.fit(X, y);
        const coef = Number(model.coef.data[0]);

        expect(coef).toBeCloseTo(2.0, 1);
      });

      it("should reject negative alpha", () => {
        const model = new Lasso({ alpha: -1.0 });
        const X = tensor([[1], [2]]);
        const y = tensor([1, 2]);

        expect(() => model.fit(X, y)).toThrow("alpha must be >= 0");
      });

      it("should handle constant y", () => {
        const model = new Lasso({ alpha: 0.1, maxIter: 1000 });
        const X = tensor([[1], [2], [3]]);
        const y = tensor([5, 5, 5]);

        model.fit(X, y);
        const score = model.score(X, y);

        expect(score).toBeCloseTo(1.0, 5);
      });
    });

    describe("API Compliance", () => {
      it("should throw when predicting before fit", () => {
        const model = new Lasso();
        expect(() => model.predict(tensor([[1]]))).toThrow("fitted");
      });

      it("should throw when accessing coef before fit", () => {
        const model = new Lasso();
        expect(() => model.coef).toThrow("fitted");
      });

      it("should throw when accessing intercept before fit", () => {
        const model = new Lasso();
        expect(() => model.intercept).toThrow("fitted");
      });

      it("should throw when accessing nIter before fit", () => {
        const model = new Lasso();
        expect(() => model.nIter).toThrow("fitted");
      });

      it("should support method chaining", () => {
        const model = new Lasso();
        const result = model.fit(tensor([[1], [2]]), tensor([1, 2]));

        expect(result).toBe(model);
      });

      it("should reject invalid parameter types", () => {
        const model = new Lasso();
        expect(() => model.setParams({ alpha: "1.0" })).toThrow();
      });

      it("should reject unknown parameters", () => {
        const model = new Lasso();
        expect(() => model.setParams({ unknown: 123 })).toThrow("Unknown parameter");
      });
    });
  });

  describe("LogisticRegression - Comprehensive", () => {
    describe("Binary Classification", () => {
      it("should classify linearly separable data", () => {
        const model = new LogisticRegression({
          maxIter: 2000,
          learningRate: 0.3,
        });
        const X = tensor([[0], [1], [2], [3]]);
        const y = tensor([0, 0, 1, 1]);

        model.fit(X, y);
        const pred = model.predict(X);

        expect(Number(pred.data[0])).toBe(0);
        expect(Number(pred.data[1])).toBe(0);
        expect(Number(pred.data[2])).toBe(1);
        expect(Number(pred.data[3])).toBe(1);
      });

      it("should output probabilities that sum to 1", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.2,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        model.fit(X, y);
        const proba = model.predictProba(X);

        for (let i = 0; i < X.shape[0]; i++) {
          const p0 = Number(proba.data[i * 2]);
          const p1 = Number(proba.data[i * 2 + 1]);
          expect(p0 + p1).toBeCloseTo(1.0, 6);
        }
      });

      it("should output probabilities in [0, 1]", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.2,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        model.fit(X, y);
        const proba = model.predictProba(X);

        for (let i = 0; i < proba.size; i++) {
          const p = Number(proba.data[i]);
          expect(p).toBeGreaterThanOrEqual(0);
          expect(p).toBeLessThanOrEqual(1);
        }
      });

      it("should achieve high accuracy on separable data", () => {
        const model = new LogisticRegression({
          maxIter: 3000,
          learningRate: 0.3,
        });
        const X = tensor([[0], [1], [2], [3], [4], [5]]);
        const y = tensor([0, 0, 0, 1, 1, 1]);

        model.fit(X, y);
        const acc = model.score(X, y);

        expect(acc).toBeGreaterThan(0.8);
      });
    });

    describe("Regularization", () => {
      it("should support L2 regularization", () => {
        const model = new LogisticRegression({
          penalty: "l2",
          C: 1.0,
          maxIter: 1000,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should support no regularization", () => {
        const model = new LogisticRegression({
          penalty: "none",
          maxIter: 1000,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should reject invalid penalty", () => {
        const model = new LogisticRegression();
        expect(() => model.setParams({ penalty: "l1" })).toThrow();

        const X = tensor([[0], [1]]);
        const y = tensor([0, 1]);

        // Fit remains valid with supported defaults.
        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should reduce coefficient magnitude with stronger regularization", () => {
        // Use more samples for better regularization effect
        const X = tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]);
        const y = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        // C is inverse regularization: higher C = less regularization
        const model1 = new LogisticRegression({
          C: 10.0,
          maxIter: 1000,
          learningRate: 0.01,
        });
        const model2 = new LogisticRegression({
          C: 0.1,
          maxIter: 1000,
          learningRate: 0.01,
        });

        model1.fit(X, y);
        model2.fit(X, y);

        let norm1 = 0,
          norm2 = 0;
        for (let i = 0; i < model1.coef.size; i++) {
          norm1 += Number(model1.coef.data[i]) ** 2;
          norm2 += Number(model2.coef.data[i]) ** 2;
        }

        // model2 has stronger regularization (lower C), so should have smaller norm
        expect(norm2).toBeLessThan(norm1);
      });
    });

    describe("Edge Cases", () => {
      it("should handle all samples in one class", () => {
        const model = new LogisticRegression({ maxIter: 1000 });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([1, 1, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle non-binary labels (multiclass)", () => {
        const model = new LogisticRegression();
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 1, 2]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle single feature", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.2,
        });
        const X = tensor([[0], [1]]);
        const y = tensor([0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle multiple features", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.1,
        });
        const X = tensor([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]);
        const y = tensor([0, 0, 0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });
    });

    describe("API Compliance", () => {
      it("should throw when predicting before fit", () => {
        const model = new LogisticRegression();
        expect(() => model.predict(tensor([[1]]))).toThrow("fitted");
      });

      it("should throw when calling predictProba before fit", () => {
        const model = new LogisticRegression();
        expect(() => model.predictProba(tensor([[1]]))).toThrow("fitted");
      });

      it("should support method chaining", () => {
        const model = new LogisticRegression();
        const result = model.fit(tensor([[0], [1]]), tensor([0, 1]));

        expect(result).toBe(model);
      });

      it("should get parameters", () => {
        const model = new LogisticRegression({ C: 2.0, maxIter: 500 });
        const params = model.getParams();

        expect(params.C).toBe(2.0);
        expect(params.maxIter).toBe(500);
      });

      it("should set parameters", () => {
        const model = new LogisticRegression();
        model.setParams({ C: 0.5, maxIter: 200 });
        const params = model.getParams();

        expect(params.C).toBe(0.5);
        expect(params.maxIter).toBe(200);
      });

      it("should reject invalid parameter types", () => {
        const model = new LogisticRegression();
        expect(() => model.setParams({ C: "1.0" })).toThrow();
      });

      it("should reject unknown parameters", () => {
        const model = new LogisticRegression();
        expect(() => model.setParams({ unknown: 123 })).toThrow("Unknown parameter");
      });

      it("should have classes attribute after fit", () => {
        const model = new LogisticRegression({ maxIter: 1000 });
        const X = tensor([[0], [1]]);
        const y = tensor([0, 1]);

        model.fit(X, y);

        expect(model.classes).toBeDefined();
        expect(model.classes?.size).toBe(2);
      });
    });

    describe("Intercept Behavior", () => {
      it("should fit intercept by default", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.2,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        model.fit(X, y);

        expect(model.intercept).toBeDefined();
      });

      it("should not fit intercept when disabled", () => {
        const model = new LogisticRegression({
          fitIntercept: false,
          maxIter: 1000,
        });
        const X = tensor([[0], [1]]);
        const y = tensor([0, 1]);

        model.fit(X, y);

        expect(model.intercept).toBe(0);
      });
    });

    describe("Convergence", () => {
      it("should converge within maxIter", () => {
        const model = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.2,
        });
        const X = tensor([[0], [1], [2]]);
        const y = tensor([0, 0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should handle maxIter = 1", () => {
        const model = new LogisticRegression({ maxIter: 1 });
        const X = tensor([[0], [1]]);
        const y = tensor([0, 1]);

        expect(() => model.fit(X, y)).not.toThrow();
      });

      it("should improve with more iterations", () => {
        const X = tensor([[0], [1], [2], [3]]);
        const y = tensor([0, 0, 1, 1]);

        const model1 = new LogisticRegression({
          maxIter: 10,
          learningRate: 0.1,
        });
        const model2 = new LogisticRegression({
          maxIter: 1000,
          learningRate: 0.1,
        });

        model1.fit(X, y);
        model2.fit(X, y);

        const acc1 = model1.score(X, y);
        const acc2 = model2.score(X, y);

        expect(acc2).toBeGreaterThanOrEqual(acc1);
      });
    });
  });

  describe("Cross-Model Comparisons", () => {
    it("Ridge should approach LinearRegression as alpha approaches 0", () => {
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);

      const lr = new LinearRegression();
      const ridge = new Ridge({ alpha: 1e-10 });

      lr.fit(X, y);
      ridge.fit(X, y);

      const lrCoef = Number(lr.coef.data[0]);
      const ridgeCoef = Number(ridge.coef.data[0]);

      expect(Math.abs(lrCoef - ridgeCoef)).toBeLessThan(0.01);
    });

    it("Lasso should approach LinearRegression as alpha approaches 0", () => {
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);

      const lr = new LinearRegression();
      const lasso = new Lasso({ alpha: 1e-10, maxIter: 5000 });

      lr.fit(X, y);
      lasso.fit(X, y);

      const lrCoef = Number(lr.coef.data[0]);
      const lassoCoef = Number(lasso.coef.data[0]);

      expect(Math.abs(lrCoef - lassoCoef)).toBeLessThan(0.1);
    });

    it("Lasso should produce sparser solutions than Ridge", () => {
      const X = tensor([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
      ]);
      const y = tensor([1, 2, 3, 4]);

      const ridge = new Ridge({ alpha: 1.0 });
      const lasso = new Lasso({ alpha: 1.0, maxIter: 3000 });

      ridge.fit(X, y);
      lasso.fit(X, y);

      let ridgeZeros = 0,
        lassoZeros = 0;
      for (let i = 0; i < ridge.coef.size; i++) {
        if (Math.abs(Number(ridge.coef.data[i])) < 0.01) ridgeZeros++;
        if (Math.abs(Number(lasso.coef.data[i])) < 0.01) lassoZeros++;
      }

      expect(lassoZeros).toBeGreaterThanOrEqual(ridgeZeros);
    });
  });

  describe("Performance and Scalability", () => {
    it("LinearRegression should handle 100 samples", () => {
      const X_data = Array.from({ length: 100 }, (_, i) => [i, i * 2]);
      const y_data = Array.from({ length: 100 }, (_, i) => i * 3);

      const model = new LinearRegression();
      const X = tensor(X_data);
      const y = tensor(y_data);

      const start = Date.now();
      model.fit(X, y);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(1000);
    });

    it("Ridge should handle 100 samples", () => {
      const X_data = Array.from({ length: 100 }, (_, i) => [i, i * 2]);
      const y_data = Array.from({ length: 100 }, (_, i) => i * 3);

      const model = new Ridge({ alpha: 1.0 });
      const X = tensor(X_data);
      const y = tensor(y_data);

      const start = Date.now();
      model.fit(X, y);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(1000);
    });

    it("Lasso should handle 50 samples with reasonable iterations", () => {
      const X_data = Array.from({ length: 50 }, (_, i) => [i, i * 2]);
      const y_data = Array.from({ length: 50 }, (_, i) => i * 3);

      const model = new Lasso({ alpha: 0.1, maxIter: 1000 });
      const X = tensor(X_data);
      const y = tensor(y_data);

      const start = Date.now();
      model.fit(X, y);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(2000);
    });

    it("LogisticRegression should handle 100 samples", () => {
      const X_data = Array.from({ length: 100 }, (_, i) => [i]);
      const y_data = Array.from({ length: 100 }, (_, i) => (i < 50 ? 0 : 1));

      const model = new LogisticRegression({ maxIter: 500, learningRate: 0.1 });
      const X = tensor(X_data);
      const y = tensor(y_data);

      const start = Date.now();
      model.fit(X, y);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(2000);
    });
  });
});
