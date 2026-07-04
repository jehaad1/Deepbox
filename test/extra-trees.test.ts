import { describe, expect, it } from "vitest";
import { ExtraTreesClassifier, ExtraTreesRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("ExtraTreesClassifier", () => {
  const X = tensor([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 1],
    [6, 2],
    [7, 3],
    [8, 4],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1], { dtype: "int32" });

  it("fits and predicts", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 10, randomState: 42 });
    clf.fit(X, y);
    const preds = clf.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("achieves reasonable training accuracy", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 50, maxDepth: 5, randomState: 42 });
    clf.fit(X, y);
    const acc = clf.score(X, y);
    expect(acc).toBeGreaterThanOrEqual(0.75);
  });

  it("predictProba returns correct shape", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 10, randomState: 42 });
    clf.fit(X, y);
    const proba = clf.predictProba(X);
    expect(proba.shape).toEqual([8, 2]);
  });

  it("predictProba rows sum to ~1", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 10, randomState: 42 });
    clf.fit(X, y);
    const proba = clf.predictProba(X);
    for (let i = 0; i < 8; i++) {
      const p0 = Number(proba.data[proba.offset + i * 2]);
      const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
      expect(p0 + p1).toBeCloseTo(1.0, 5);
    }
  });

  it("is deterministic with same randomState", () => {
    const clf1 = new ExtraTreesClassifier({ nEstimators: 10, randomState: 123 });
    clf1.fit(X, y);
    const p1 = clf1.predict(X);

    const clf2 = new ExtraTreesClassifier({ nEstimators: 10, randomState: 123 });
    clf2.fit(X, y);
    const p2 = clf2.predict(X);

    for (let i = 0; i < 8; i++) {
      expect(Number(p1.data[p1.offset + i])).toBe(Number(p2.data[p2.offset + i]));
    }
  });

  it("classes getter returns class labels", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 5, randomState: 42 });
    clf.fit(X, y);
    const classes = clf.classes;
    expect(classes).toBeDefined();
    expect(classes!.shape).toEqual([2]);
  });

  it("featureImportances returns correct shape", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 10, randomState: 42 });
    clf.fit(X, y);
    const imp = clf.featureImportances;
    expect(imp.shape).toEqual([2]);
  });

  it("throws before fitting", () => {
    const clf = new ExtraTreesClassifier();
    expect(() => clf.predict(X)).toThrow();
    expect(() => clf.score(X, y)).toThrow();
  });

  it("getParams returns constructor options", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 50, maxDepth: 5 });
    const params = clf.getParams();
    expect(params.nEstimators).toBe(50);
    expect(params.maxDepth).toBe(5);
    expect(params.bootstrap).toBe(false); // ExtraTrees default
  });

  it("supports bootstrap=true", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 10, bootstrap: true, randomState: 42 });
    clf.fit(X, y);
    const acc = clf.score(X, y);
    expect(acc).toBeGreaterThanOrEqual(0.5);
  });
});

describe("ExtraTreesRegressor", () => {
  const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
  const y = tensor([2, 4, 6, 8, 10, 12, 14, 16]);

  it("fits and predicts", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 10, randomState: 42 });
    reg.fit(X, y);
    const preds = reg.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("achieves reasonable R² on training data", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 50, maxDepth: 5, randomState: 42 });
    reg.fit(X, y);
    const r2 = reg.score(X, y);
    expect(r2).toBeGreaterThanOrEqual(0.5);
  });

  it("is deterministic with same randomState", () => {
    const reg1 = new ExtraTreesRegressor({ nEstimators: 10, randomState: 99 });
    reg1.fit(X, y);
    const p1 = reg1.predict(X);

    const reg2 = new ExtraTreesRegressor({ nEstimators: 10, randomState: 99 });
    reg2.fit(X, y);
    const p2 = reg2.predict(X);

    for (let i = 0; i < 8; i++) {
      expect(Number(p1.data[p1.offset + i])).toBeCloseTo(Number(p2.data[p2.offset + i]), 10);
    }
  });

  it("featureImportances returns correct shape", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 10, randomState: 42 });
    reg.fit(X, y);
    const imp = reg.featureImportances;
    expect(imp.shape).toEqual([1]);
  });

  it("throws before fitting", () => {
    const reg = new ExtraTreesRegressor();
    expect(() => reg.predict(X)).toThrow();
    expect(() => reg.score(X, y)).toThrow();
  });

  it("getParams returns constructor options", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 20, maxDepth: 3 });
    const params = reg.getParams();
    expect(params.nEstimators).toBe(20);
    expect(params.maxDepth).toBe(3);
    expect(params.bootstrap).toBe(false);
  });

  it("handles multi-feature regression", () => {
    const X2 = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
      [5, 50],
      [6, 60],
      [7, 70],
      [8, 80],
    ]);
    const y2 = tensor([11, 22, 33, 44, 55, 66, 77, 88]);
    const reg = new ExtraTreesRegressor({ nEstimators: 30, randomState: 42 });
    reg.fit(X2, y2);
    const r2 = reg.score(X2, y2);
    expect(r2).toBeGreaterThanOrEqual(0.5);
  });
});
