import { describe, expect, it } from "vitest";
import { CategoricalNB } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("CategoricalNB", () => {
  const X = tensor([
    [0, 0],
    [1, 1],
    [2, 0],
    [0, 1],
    [1, 0],
    [2, 1],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("should classify categorical data", () => {
    const clf = new CategoricalNB();
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("should predict probabilities summing to 1", () => {
    const clf = new CategoricalNB();
    clf.fit(X, y);
    const proba = clf.predictProba(tensor([[1, 0]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
    expect(p0).toBeGreaterThanOrEqual(0);
    expect(p1).toBeGreaterThanOrEqual(0);
  });

  it("should compute accuracy score", () => {
    const clf = new CategoricalNB();
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should expose classes", () => {
    const clf = new CategoricalNB();
    clf.fit(X, y);
    expect(clf.classes).toBeDefined();
    expect(clf.classes!.shape).toEqual([2]);
  });

  it("should support fitPrior=false", () => {
    const clf = new CategoricalNB({ fitPrior: false });
    clf.fit(X, y);
    const pred = clf.predict(tensor([[0, 0]]));
    expect(pred.shape).toEqual([1]);
  });

  it("should support custom alpha", () => {
    const clf = new CategoricalNB({ alpha: 0.5 });
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
  });

  it("should throw on invalid alpha", () => {
    expect(() => new CategoricalNB({ alpha: -1 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const clf = new CategoricalNB();
    expect(() => clf.predict(tensor([[1, 0]]))).toThrow();
  });

  it("should get and set params", () => {
    const clf = new CategoricalNB({ alpha: 2.0 });
    expect(clf.getParams().alpha).toBe(2.0);
    clf.setParams({ alpha: 0.5 });
    expect(clf.getParams().alpha).toBe(0.5);
  });

  it("should throw on unknown param", () => {
    const clf = new CategoricalNB();
    expect(() => clf.setParams({ badParam: 1 })).toThrow();
  });

  it("should handle unseen categories gracefully", () => {
    const clf = new CategoricalNB();
    clf.fit(X, y);
    // Category 3 was not in training data
    const pred = clf.predict(tensor([[3, 0]]));
    expect(pred.shape).toEqual([1]);
  });
});
