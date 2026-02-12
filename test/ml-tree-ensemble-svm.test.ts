import { describe, expect, it } from "vitest";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { LinearSVC, LinearSVR } from "../src/ml/svm/SVM";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree/DecisionTree";
import { RandomForestClassifier, RandomForestRegressor } from "../src/ml/tree/RandomForest";
import { tensor } from "../src/ndarray";

describe("deepbox/ml - Trees, Ensembles, SVM", () => {
  describe("DecisionTreeClassifier", () => {
    it("fits and predicts correctly on a simple dataset", () => {
      const X = tensor([[0], [1], [2], [3]]);
      const y = tensor([0, 0, 1, 1], { dtype: "int32" });
      const clf = new DecisionTreeClassifier({ maxDepth: 2 });
      clf.fit(X, y);
      const preds = clf.predict(X);
      expect(preds.toArray()).toEqual([0, 0, 1, 1]);
      const proba = clf.predictProba(X);
      expect(proba.shape).toEqual([4, 2]);
      const score = clf.score(X, y);
      expect(score).toBe(1);
    });

    it("predictProba reflects leaf class distribution", () => {
      const X = tensor([[0], [0], [1], [1]]);
      const y = tensor([0, 1, 0, 1], { dtype: "int32" });
      const clf = new DecisionTreeClassifier({ maxDepth: 1 });
      clf.fit(X, y);

      const proba = clf.predictProba(X);
      for (let i = 0; i < 4; i++) {
        const p0 = Number(proba.data[proba.offset + i * 2]);
        const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
        expect(p0).toBeCloseTo(0.5, 6);
        expect(p1).toBeCloseTo(0.5, 6);
      }
    });

    it("throws when predicting before fitting", () => {
      const clf = new DecisionTreeClassifier();
      const X = tensor([[1], [2]]);
      expect(() => clf.predict(X)).toThrow();
    });
  });

  describe("DecisionTreeRegressor", () => {
    it("fits and predicts on simple linear data", () => {
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([2, 4, 6, 8]);
      const reg = new DecisionTreeRegressor({ maxDepth: 3 });
      reg.fit(X, y);
      const preds = reg.predict(X);
      expect(preds.shape).toEqual([4]);
      const score = reg.score(X, y);
      expect(score).toBeGreaterThan(0.8);
    });
  });

  describe("RandomForestClassifier", () => {
    it("trains and predicts with deterministic random state", () => {
      const X = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);
      const y = tensor([0, 1, 1, 0], { dtype: "int32" });
      const rf = new RandomForestClassifier({
        nEstimators: 5,
        maxDepth: 4,
        randomState: 42,
        bootstrap: false,
      });
      rf.fit(X, y);
      const preds = rf.predict(X);
      expect(preds.shape).toEqual([4]);
      const proba = rf.predictProba(X);
      expect(proba.shape).toEqual([4, 2]);
      const score = rf.score(X, y);
      expect(score).toBeGreaterThanOrEqual(0.5);
    });

    it("averages tree probabilities in predictProba", () => {
      const X = tensor([[0], [0], [1], [1]]);
      const y = tensor([0, 1, 0, 1], { dtype: "int32" });
      const rf = new RandomForestClassifier({
        nEstimators: 1,
        maxDepth: 1,
        maxFeatures: 1,
        bootstrap: false,
        randomState: 7,
      });
      rf.fit(X, y);

      const proba = rf.predictProba(X);
      for (let i = 0; i < 4; i++) {
        const p0 = Number(proba.data[proba.offset + i * 2]);
        const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
        expect(p0).toBeCloseTo(0.5, 6);
        expect(p1).toBeCloseTo(0.5, 6);
      }
    });
  });

  describe("RandomForestRegressor", () => {
    it("predicts averaged outputs", () => {
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([1.1, 1.9, 3.1, 3.9]);
      const rf = new RandomForestRegressor({
        nEstimators: 5,
        maxDepth: 4,
        randomState: 7,
        bootstrap: false,
      });
      rf.fit(X, y);
      const preds = rf.predict(X);
      expect(preds.shape).toEqual([4]);
      const score = rf.score(X, y);
      expect(score).toBeGreaterThan(0.7);
    });
  });

  describe("GradientBoostingRegressor", () => {
    it("learns a simple trend", () => {
      const X = tensor([[1], [2], [3], [4], [5]]);
      const y = tensor([1.2, 2.0, 3.1, 4.1, 4.9]);
      const gbr = new GradientBoostingRegressor({
        nEstimators: 20,
        learningRate: 0.1,
      });
      gbr.fit(X, y);
      const preds = gbr.predict(X);
      expect(preds.shape).toEqual([5]);
      const score = gbr.score(X, y);
      expect(score).toBeGreaterThan(0.7);
    });
  });

  describe("GradientBoostingClassifier", () => {
    it("fits binary classification data", () => {
      const X = tensor([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
      ]);
      const y = tensor([0, 0, 1, 1], { dtype: "int32" });
      const gbc = new GradientBoostingClassifier({
        nEstimators: 25,
        learningRate: 0.2,
      });
      gbc.fit(X, y);
      const preds = gbc.predict(X);
      expect(preds.shape).toEqual([4]);
      const proba = gbc.predictProba(X);
      expect(proba.shape).toEqual([4, 2]);
      const score = gbc.score(X, y);
      expect(score).toBeGreaterThan(0.5);
    });

    it("throws for non-binary labels", () => {
      const X = tensor([[1], [2], [3]]);
      const y = tensor([0, 1, 2], { dtype: "int32" });
      const gbc = new GradientBoostingClassifier({ nEstimators: 3 });
      expect(() => gbc.fit(X, y)).toThrow();
    });
  });

  describe("LinearSVC", () => {
    it("trains a linear SVM and predicts", () => {
      const X = tensor([[0], [1], [2], [3]]);
      const y = tensor([0, 0, 1, 1], { dtype: "int32" });
      const svm = new LinearSVC({ C: 1.0, maxIter: 200 });
      svm.fit(X, y);
      const preds = svm.predict(X);
      expect(preds.shape).toEqual([4]);
      const proba = svm.predictProba(X);
      expect(proba.shape).toEqual([4, 2]);
      const coef = svm.coef;
      expect(coef.shape).toEqual([1, 1]);
      const intercept = svm.intercept;
      expect(typeof intercept).toBe("number");
      const score = svm.score(X, y);
      expect(score).toBeGreaterThanOrEqual(0.5);
    });

    it("rejects non-binary labels", () => {
      const X = tensor([[0], [1], [2]]);
      const y = tensor([0, 1, 2], { dtype: "int32" });
      const svm = new LinearSVC();
      expect(() => svm.fit(X, y)).toThrow();
    });
  });

  describe("LinearSVR", () => {
    it("fits a simple regression", () => {
      const X = tensor([[1], [2], [3], [4]]);
      const y = tensor([1.2, 2.2, 2.8, 4.1]);
      const svr = new LinearSVR({ C: 1.0, epsilon: 0.2, maxIter: 200 });
      svr.fit(X, y);
      const preds = svr.predict(X);
      expect(preds.shape).toEqual([4]);
      const score = svr.score(X, y);
      expect(score).toBeGreaterThan(0.4);
    });
  });
});
