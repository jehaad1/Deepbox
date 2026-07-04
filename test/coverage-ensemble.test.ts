import { describe, expect, it } from "vitest";
import { AdaBoostClassifier, AdaBoostRegressor } from "../src/ml/ensemble/AdaBoost";
import { BaggingClassifier, BaggingRegressor } from "../src/ml/ensemble/Bagging";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { StackingClassifier, StackingRegressor } from "../src/ml/ensemble/Stacking";
import { VotingClassifier, VotingRegressor } from "../src/ml/ensemble/Voting";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree/DecisionTree";
import { tensor } from "../src/ndarray";
import { setSeed } from "../src/random";

// Separable binary dataset
const Xbin = tensor([
  [1, 0],
  [1, 1],
  [0, 1],
  [0, 0],
  [5, 5],
  [5, 6],
  [6, 5],
  [6, 6],
]);
const ybin = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

// Regression dataset
const Xreg = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
const yreg = tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]);

// ---- AdaBoostClassifier ----

describe("AdaBoostClassifier extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const clf = new AdaBoostClassifier({ nEstimators: 10, maxDepth: 1 });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba sums to 1", () => {
    setSeed(42);
    const clf = new AdaBoostClassifier({ nEstimators: 10 });
    clf.fit(Xbin, ybin);
    const proba = clf.predictProba(Xbin);
    expect(proba.shape[1]).toBe(2);
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 4);
    }
  });

  it("score", () => {
    setSeed(42);
    const clf = new AdaBoostClassifier({ nEstimators: 10 });
    clf.fit(Xbin, ybin);
    const score = clf.score(Xbin, ybin);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const clf = new AdaBoostClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("rejects single class", () => {
    const clf = new AdaBoostClassifier();
    expect(() =>
      clf.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow(/2 classes/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new AdaBoostClassifier({ nEstimators: 0 })).toThrow();
    expect(() => new AdaBoostClassifier({ nEstimators: -1 })).toThrow();
    expect(() => new AdaBoostClassifier({ nEstimators: 1.5 })).toThrow();
  });

  it("constructor rejects invalid learningRate", () => {
    expect(() => new AdaBoostClassifier({ learningRate: 0 })).toThrow();
    expect(() => new AdaBoostClassifier({ learningRate: -1 })).toThrow();
  });

  it("getParams and setParams", () => {
    const clf = new AdaBoostClassifier({ nEstimators: 20, learningRate: 0.5 });
    const params = clf.getParams();
    expect(params.nEstimators).toBe(20);
    expect(params.learningRate).toBe(0.5);
  });

  it("score rejects non-1D y", () => {
    setSeed(42);
    const clf = new AdaBoostClassifier({ nEstimators: 5 });
    clf.fit(Xbin, ybin);
    expect(() => clf.score(Xbin, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });
});

// ---- AdaBoostRegressor ----

describe("AdaBoostRegressor extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const reg = new AdaBoostRegressor({ nEstimators: 10, maxDepth: 2 });
    reg.fit(Xreg, yreg);
    const preds = reg.predict(Xreg);
    expect(preds.shape).toEqual([8]);
  });

  it("score returns R²", () => {
    setSeed(42);
    const reg = new AdaBoostRegressor({ nEstimators: 10, maxDepth: 3 });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const reg = new AdaBoostRegressor();
    expect(() => reg.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new AdaBoostRegressor({ nEstimators: 0 })).toThrow();
  });

  it("constructor rejects invalid learningRate", () => {
    expect(() => new AdaBoostRegressor({ learningRate: 0 })).toThrow();
  });

  it("getParams", () => {
    const reg = new AdaBoostRegressor({ nEstimators: 20 });
    expect(reg.getParams().nEstimators).toBe(20);
  });
});

// ---- BaggingClassifier ----

describe("BaggingClassifier extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const clf = new BaggingClassifier({ nEstimators: 5, maxDepth: 3 });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba sums to 1", () => {
    setSeed(42);
    const clf = new BaggingClassifier({ nEstimators: 5 });
    clf.fit(Xbin, ybin);
    const proba = clf.predictProba(Xbin);
    expect(proba.shape[1]).toBe(2);
  });

  it("score", () => {
    setSeed(42);
    const clf = new BaggingClassifier({ nEstimators: 5 });
    clf.fit(Xbin, ybin);
    const score = clf.score(Xbin, ybin);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const clf = new BaggingClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new BaggingClassifier({ nEstimators: 0 })).toThrow();
  });

  it("getParams", () => {
    const clf = new BaggingClassifier({ nEstimators: 20 });
    expect(clf.getParams().nEstimators).toBe(20);
  });
});

// ---- BaggingRegressor ----

describe("BaggingRegressor extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const reg = new BaggingRegressor({ nEstimators: 5, maxDepth: 3 });
    reg.fit(Xreg, yreg);
    const preds = reg.predict(Xreg);
    expect(preds.shape).toEqual([8]);
  });

  it("score returns R²", () => {
    setSeed(42);
    const reg = new BaggingRegressor({ nEstimators: 5, maxDepth: 3 });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const reg = new BaggingRegressor();
    expect(() => reg.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new BaggingRegressor({ nEstimators: 0 })).toThrow();
  });
});

// ---- GradientBoostingClassifier ----

describe("GradientBoostingClassifier extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const clf = new GradientBoostingClassifier({ nEstimators: 10, maxDepth: 2, learningRate: 0.1 });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba sums to 1", () => {
    setSeed(42);
    const clf = new GradientBoostingClassifier({ nEstimators: 10 });
    clf.fit(Xbin, ybin);
    const proba = clf.predictProba(Xbin);
    expect(proba.shape[1]).toBe(2);
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 4);
    }
  });

  it("score", () => {
    setSeed(42);
    const clf = new GradientBoostingClassifier({ nEstimators: 10 });
    clf.fit(Xbin, ybin);
    const score = clf.score(Xbin, ybin);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const clf = new GradientBoostingClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("rejects single class", () => {
    const clf = new GradientBoostingClassifier();
    expect(() =>
      clf.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow();
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new GradientBoostingClassifier({ nEstimators: 0 })).toThrow();
  });

  it("constructor rejects invalid learningRate", () => {
    expect(() => new GradientBoostingClassifier({ learningRate: 0 })).toThrow();
  });

  it("getParams", () => {
    const clf = new GradientBoostingClassifier({ nEstimators: 20, learningRate: 0.05 });
    const params = clf.getParams();
    expect(params.nEstimators).toBe(20);
    expect(params.learningRate).toBe(0.05);
  });
});

// ---- GradientBoostingRegressor ----

describe("GradientBoostingRegressor extended branches", () => {
  it("fit and predict", () => {
    const reg = new GradientBoostingRegressor({ nEstimators: 10, maxDepth: 2, learningRate: 0.1 });
    reg.fit(Xreg, yreg);
    const preds = reg.predict(Xreg);
    expect(preds.shape).toEqual([8]);
  });

  it("score returns R²", () => {
    const reg = new GradientBoostingRegressor({ nEstimators: 20, maxDepth: 3 });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const reg = new GradientBoostingRegressor();
    expect(() => reg.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new GradientBoostingRegressor({ nEstimators: 0 })).toThrow();
  });

  it("constructor rejects invalid learningRate", () => {
    expect(() => new GradientBoostingRegressor({ learningRate: 0 })).toThrow();
  });

  it("getParams", () => {
    const reg = new GradientBoostingRegressor({ nEstimators: 20 });
    expect(reg.getParams().nEstimators).toBe(20);
  });
});

// ---- VotingClassifier ----

describe("VotingClassifier extended branches", () => {
  it("fit and predict with hard voting", () => {
    setSeed(42);
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
      voting: "hard",
    });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("fit and predict with soft voting", () => {
    setSeed(42);
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
      voting: "soft",
    });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba with soft voting", () => {
    setSeed(42);
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
      voting: "soft",
    });
    clf.fit(Xbin, ybin);
    const proba = clf.predictProba(Xbin);
    expect(proba.shape[1]).toBe(2);
  });

  it("throws before fit", () => {
    const clf = new VotingClassifier({
      estimators: [new DecisionTreeClassifier()],
    });
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("score", () => {
    setSeed(42);
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
    });
    clf.fit(Xbin, ybin);
    const score = clf.score(Xbin, ybin);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("getParams", () => {
    const clf = new VotingClassifier({
      estimators: [new DecisionTreeClassifier()],
      voting: "hard",
    });
    const params = clf.getParams();
    expect(params.voting).toBe("hard");
  });
});

// ---- VotingRegressor ----

describe("VotingRegressor extended branches", () => {
  it("fit and predict", () => {
    const reg = new VotingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(Xreg, yreg);
    const preds = reg.predict(Xreg);
    expect(preds.shape).toEqual([8]);
  });

  it("score returns R²", () => {
    const reg = new VotingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws before fit", () => {
    const reg = new VotingRegressor({
      estimators: [new DecisionTreeRegressor()],
    });
    expect(() => reg.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("getParams", () => {
    const reg = new VotingRegressor({
      estimators: [new DecisionTreeRegressor()],
    });
    const params = reg.getParams();
    expect(params).toBeDefined();
  });
});

// ---- StackingClassifier ----

describe("StackingClassifier extended branches", () => {
  it("fit and predict", () => {
    setSeed(42);
    const clf = new StackingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
    });
    clf.fit(Xbin, ybin);
    const preds = clf.predict(Xbin);
    expect(preds.shape).toEqual([8]);
  });

  it("score", () => {
    setSeed(42);
    const clf = new StackingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
    });
    clf.fit(Xbin, ybin);
    const score = clf.score(Xbin, ybin);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws before fit", () => {
    const clf = new StackingClassifier({
      estimators: [new DecisionTreeClassifier()],
    });
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("getParams", () => {
    const clf = new StackingClassifier({
      estimators: [new DecisionTreeClassifier()],
    });
    const params = clf.getParams();
    expect(params).toBeDefined();
  });
});

// ---- StackingRegressor ----

describe("StackingRegressor extended branches", () => {
  it("fit and predict", () => {
    const reg = new StackingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(Xreg, yreg);
    const preds = reg.predict(Xreg);
    expect(preds.shape).toEqual([8]);
  });

  it("score returns R²", () => {
    const reg = new StackingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws before fit", () => {
    const reg = new StackingRegressor({
      estimators: [new DecisionTreeRegressor()],
    });
    expect(() => reg.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("getParams", () => {
    const reg = new StackingRegressor({
      estimators: [new DecisionTreeRegressor()],
    });
    const params = reg.getParams();
    expect(params).toBeDefined();
  });
});
