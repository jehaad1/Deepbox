import { describe, expect, it } from "vitest";
import { BaggingClassifier, BaggingRegressor } from "../src/ml/ensemble/Bagging";
import { StackingClassifier, StackingRegressor } from "../src/ml/ensemble/Stacking";
import { VotingClassifier, VotingRegressor } from "../src/ml/ensemble/Voting";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree/DecisionTree";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
  [7, 8],
  [9, 10],
  [11, 12],
]);
const yC = tensor([0, 0, 1, 1, 0, 1]);
const yR = tensor([1, 2, 3, 4, 5, 6]);

// ────── BaggingClassifier ──────
describe("BaggingClassifier", () => {
  it("fits and predicts", () => {
    const clf = new BaggingClassifier({ nEstimators: 3, randomState: 42 });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("predictProba", () => {
    const clf = new BaggingClassifier({ nEstimators: 3, randomState: 42 });
    clf.fit(X, yC);
    const proba = clf.predictProba(X);
    expect(proba.shape[0]).toBe(6);
  });

  it("score", () => {
    const clf = new BaggingClassifier({ nEstimators: 3, randomState: 42 });
    clf.fit(X, yC);
    const s = clf.score(X, yC);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it("bootstrap=false", () => {
    const clf = new BaggingClassifier({
      nEstimators: 2,
      bootstrap: false,
      randomState: 42,
    });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("maxFeatures < 1.0", () => {
    const clf = new BaggingClassifier({
      nEstimators: 2,
      maxFeatures: 0.5,
      randomState: 42,
    });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("maxSamples < 1.0", () => {
    const clf = new BaggingClassifier({
      nEstimators: 2,
      maxSamples: 0.5,
      randomState: 42,
    });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("throws for not fitted predict", () => {
    expect(() => new BaggingClassifier().predict(X)).toThrow(/fitted/i);
  });

  it("throws for not fitted predictProba", () => {
    expect(() => new BaggingClassifier().predictProba(X)).toThrow(/fitted/i);
  });

  it("throws for not fitted score", () => {
    expect(() => new BaggingClassifier().score(X, yC)).toThrow(/fitted/i);
  });

  it("validates nEstimators", () => {
    expect(() => new BaggingClassifier({ nEstimators: 0 })).toThrow();
    expect(() => new BaggingClassifier({ nEstimators: -1 })).toThrow();
  });

  it("getParams / setParams", () => {
    const clf = new BaggingClassifier({ nEstimators: 5 });
    expect(clf.getParams().nEstimators).toBe(5);
    clf.setParams({ nEstimators: 10 });
    expect(clf.getParams().nEstimators).toBe(10);
  });
});

// ────── BaggingRegressor ──────
describe("BaggingRegressor", () => {
  it("fits and predicts", () => {
    const reg = new BaggingRegressor({ nEstimators: 3, randomState: 42 });
    reg.fit(X, yR);
    const pred = reg.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("score", () => {
    const reg = new BaggingRegressor({ nEstimators: 3, randomState: 42 });
    reg.fit(X, yR);
    const s = reg.score(X, yR);
    expect(typeof s).toBe("number");
  });

  it("bootstrap=false", () => {
    const reg = new BaggingRegressor({
      nEstimators: 2,
      bootstrap: false,
      randomState: 42,
    });
    reg.fit(X, yR);
    const pred = reg.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("maxFeatures < 1.0", () => {
    const reg = new BaggingRegressor({
      nEstimators: 2,
      maxFeatures: 0.5,
      randomState: 42,
    });
    reg.fit(X, yR);
    expect(reg.predict(X).shape).toEqual([6]);
  });

  it("throws for not fitted", () => {
    expect(() => new BaggingRegressor().predict(X)).toThrow(/fitted/i);
  });

  it("getParams / setParams", () => {
    const reg = new BaggingRegressor({ nEstimators: 5 });
    expect(reg.getParams().nEstimators).toBe(5);
  });
});

// Helper: create base estimators for Voting / Stacking
function makeClassifiers() {
  return [new DecisionTreeClassifier({ maxDepth: 2 }), new DecisionTreeClassifier({ maxDepth: 3 })];
}

function makeRegressors() {
  return [new DecisionTreeRegressor({ maxDepth: 2 }), new DecisionTreeRegressor({ maxDepth: 3 })];
}

// ────── VotingClassifier ──────
describe("VotingClassifier", () => {
  it("fits and predicts with hard voting", () => {
    const clf = new VotingClassifier({
      estimators: makeClassifiers(),
      voting: "hard",
    });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("fits and predicts with soft voting", () => {
    const clf = new VotingClassifier({
      estimators: makeClassifiers(),
      voting: "soft",
    });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("predictProba", () => {
    const clf = new VotingClassifier({ estimators: makeClassifiers() });
    clf.fit(X, yC);
    const proba = clf.predictProba(X);
    expect(proba.shape[0]).toBe(6);
  });

  it("score", () => {
    const clf = new VotingClassifier({ estimators: makeClassifiers() });
    clf.fit(X, yC);
    const s = clf.score(X, yC);
    expect(s).toBeGreaterThanOrEqual(0);
  });

  it("throws for not fitted", () => {
    const clf = new VotingClassifier({ estimators: makeClassifiers() });
    expect(() => clf.predict(X)).toThrow(/fitted/i);
  });

  it("throws for empty estimators", () => {
    expect(() => new VotingClassifier({ estimators: [] })).toThrow();
  });

  it("throws for mismatched weights", () => {
    expect(() => new VotingClassifier({ estimators: makeClassifiers(), weights: [1] })).toThrow();
  });

  it("getParams / setParams", () => {
    const clf = new VotingClassifier({ estimators: makeClassifiers() });
    const params = clf.getParams();
    expect(params).toBeDefined();
  });
});

// ────── VotingRegressor ──────
describe("VotingRegressor", () => {
  it("fits and predicts", () => {
    const reg = new VotingRegressor({ estimators: makeRegressors() });
    reg.fit(X, yR);
    const pred = reg.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("score", () => {
    const reg = new VotingRegressor({ estimators: makeRegressors() });
    reg.fit(X, yR);
    const s = reg.score(X, yR);
    expect(typeof s).toBe("number");
  });

  it("throws for not fitted", () => {
    const reg = new VotingRegressor({ estimators: makeRegressors() });
    expect(() => reg.predict(X)).toThrow(/fitted/i);
  });

  it("throws for empty estimators", () => {
    expect(() => new VotingRegressor({ estimators: [] })).toThrow();
  });
});

// ────── StackingClassifier ──────
describe("StackingClassifier", () => {
  it("fits and predicts", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    clf.fit(X, yC);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("predictProba", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    clf.fit(X, yC);
    const proba = clf.predictProba(X);
    expect(proba.shape[0]).toBe(6);
  });

  it("score", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    clf.fit(X, yC);
    const s = clf.score(X, yC);
    expect(typeof s).toBe("number");
  });

  it("throws for not fitted", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    expect(() => clf.predict(X)).toThrow(/fitted/i);
  });

  it("throws for empty estimators", () => {
    expect(() => new StackingClassifier({ estimators: [] })).toThrow();
  });

  it("getParams / setParams", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    const params = clf.getParams();
    expect(params).toBeDefined();
    clf.setParams({ passthrough: true });
    expect(clf.getParams().passthrough).toBe(true);
  });

  it("setParams rejects invalid passthrough", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    expect(() => clf.setParams({ passthrough: "bad" })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const clf = new StackingClassifier({ estimators: makeClassifiers() });
    expect(() => clf.setParams({ unknown: 1 })).toThrow();
  });

  it("passthrough=true", () => {
    const clf = new StackingClassifier({
      estimators: makeClassifiers(),
      passthrough: true,
    });
    clf.fit(X, yC);
    expect(clf.predict(X).shape).toEqual([6]);
  });
});

// ────── StackingRegressor ──────
describe("StackingRegressor", () => {
  it("fits and predicts", () => {
    const reg = new StackingRegressor({ estimators: makeRegressors() });
    reg.fit(X, yR);
    const pred = reg.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("score", () => {
    const reg = new StackingRegressor({ estimators: makeRegressors() });
    reg.fit(X, yR);
    const s = reg.score(X, yR);
    expect(typeof s).toBe("number");
  });

  it("throws for not fitted", () => {
    const reg = new StackingRegressor({ estimators: makeRegressors() });
    expect(() => reg.predict(X)).toThrow(/fitted/i);
  });

  it("throws for empty estimators", () => {
    expect(() => new StackingRegressor({ estimators: [] })).toThrow();
  });

  it("passthrough=true", () => {
    const reg = new StackingRegressor({
      estimators: makeRegressors(),
      passthrough: true,
    });
    reg.fit(X, yR);
    expect(reg.predict(X).shape).toEqual([6]);
  });
});
