import { describe, expect, it } from "vitest";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
} from "../src/ml";
import { tensor } from "../src/ndarray";

describe("Feature Importances", () => {
  const X = tensor([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50],
    [6, 60],
    [7, 70],
    [8, 80],
    [9, 90],
    [10, 100],
  ]);
  const yReg = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
  const yClf = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

  describe("DecisionTreeClassifier", () => {
    it("returns feature importances summing to 1", () => {
      const clf = new DecisionTreeClassifier({ maxDepth: 5 });
      clf.fit(X, yClf);
      const imp = clf.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const clf = new DecisionTreeClassifier();
      expect(() => clf.featureImportances).toThrow(/fitted/i);
    });
  });

  describe("DecisionTreeRegressor", () => {
    it("returns feature importances summing to 1", () => {
      const reg = new DecisionTreeRegressor({ maxDepth: 5 });
      reg.fit(X, yReg);
      const imp = reg.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const reg = new DecisionTreeRegressor();
      expect(() => reg.featureImportances).toThrow(/fitted/i);
    });
  });

  describe("RandomForestClassifier", () => {
    it("returns feature importances summing to 1", () => {
      const clf = new RandomForestClassifier({ nEstimators: 5, maxDepth: 3, randomState: 42 });
      clf.fit(X, yClf);
      const imp = clf.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const clf = new RandomForestClassifier();
      expect(() => clf.featureImportances).toThrow(/fitted/i);
    });
  });

  describe("RandomForestRegressor", () => {
    it("returns feature importances summing to 1", () => {
      const reg = new RandomForestRegressor({ nEstimators: 5, maxDepth: 3, randomState: 42 });
      reg.fit(X, yReg);
      const imp = reg.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const reg = new RandomForestRegressor();
      expect(() => reg.featureImportances).toThrow(/fitted/i);
    });
  });

  describe("GradientBoostingRegressor", () => {
    it("returns feature importances summing to 1", () => {
      const reg = new GradientBoostingRegressor({ nEstimators: 10, maxDepth: 2 });
      reg.fit(X, yReg);
      const imp = reg.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const reg = new GradientBoostingRegressor();
      expect(() => reg.featureImportances).toThrow(/fitted/i);
    });
  });

  describe("GradientBoostingClassifier", () => {
    it("returns feature importances summing to 1", () => {
      const clf = new GradientBoostingClassifier({ nEstimators: 10, maxDepth: 2 });
      clf.fit(X, yClf);
      const imp = clf.featureImportances;
      expect(imp.shape).toEqual([2]);
      let sum = 0;
      for (let i = 0; i < imp.size; i++) {
        const v = Number(imp.data[imp.offset + i]);
        expect(v).toBeGreaterThanOrEqual(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    });

    it("throws NotFittedError before fitting", () => {
      const clf = new GradientBoostingClassifier();
      expect(() => clf.featureImportances).toThrow(/fitted/i);
    });
  });
});
