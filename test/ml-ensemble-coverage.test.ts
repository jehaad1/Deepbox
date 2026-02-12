import { describe, expect, it } from "vitest";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { tensor } from "../src/ndarray";

describe("ML Ensemble Coverage", () => {
  describe("GradientBoostingRegressor", () => {
    it("should validate constructor parameters", () => {
      expect(() => new GradientBoostingRegressor({ nEstimators: -1 })).toThrow(/nEstimators/);
      expect(() => new GradientBoostingRegressor({ learningRate: 0 })).toThrow(/learningRate/);
      expect(() => new GradientBoostingRegressor({ maxDepth: 0 })).toThrow(/maxDepth/);
      expect(() => new GradientBoostingRegressor({ minSamplesSplit: 1 })).toThrow(
        /minSamplesSplit/
      );
    });

    it("should throw on setParams", () => {
      const model = new GradientBoostingRegressor();
      expect(() => model.setParams({ nEstimators: 10 })).toThrow(/does not support setParams/);
    });

    it("should throw if predict called before fit", () => {
      const model = new GradientBoostingRegressor();
      const X = tensor([[1]]);
      expect(() => model.predict(X)).toThrow(/must be fitted/);
    });

    it("should validate score inputs", () => {
      const model = new GradientBoostingRegressor({ nEstimators: 2 });
      const X = tensor([[1], [2]]);
      const y = tensor([1, 2]);
      model.fit(X, y);

      const yBadDim = tensor([[1], [2]]);
      const yBadSize = tensor([1, 2, 3]);

      expect(() => model.score(X, yBadDim)).toThrow(/y must be 1-dimensional/);
      expect(() => model.score(X, yBadSize)).toThrow(
        /X and y must have the same number of samples/
      );
    });

    it("getParams should return options", () => {
      const model = new GradientBoostingRegressor({ nEstimators: 50 });
      expect(model.getParams().nEstimators).toBe(50);
    });
  });

  describe("GradientBoostingClassifier", () => {
    it("should validate constructor parameters", () => {
      expect(() => new GradientBoostingClassifier({ nEstimators: -1 })).toThrow(/nEstimators/);
    });

    it("should handle binary classification edge cases", () => {
      const X = tensor([[1], [2], [3]]);
      const y = tensor([0, 1, 0], { dtype: "int32" });
      const model = new GradientBoostingClassifier({ nEstimators: 5 });
      model.fit(X, y);

      const preds = model.predict(X);
      expect(preds.shape).toEqual([3]);
    });
  });
});
