import { describe, expect, it } from "vitest";
import { LabelPropagation, NearestCentroid } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("LabelPropagation", () => {
  // Two clusters with partial labels
  const X = tensor([
    [0, 0],
    [0.1, 0],
    [0, 0.1],
    [10, 10],
    [10.1, 10],
    [10, 10.1],
  ]);
  const y = tensor([0, -1, -1, 1, -1, -1]); // only first of each cluster labeled

  it("fit and predict propagates labels", () => {
    const lp = new LabelPropagation({ gamma: 1 });
    lp.fit(X, y);
    const pred = lp.predict(X);
    expect(pred.size).toBe(6);
    // Nearby unlabeled points should get same label as labeled neighbor
    expect(Number(pred.data[pred.offset + 1])).toBe(0); // near [0,0]
    expect(Number(pred.data[pred.offset + 4])).toBe(1); // near [10,10]
  });

  it("predictProba returns valid probabilities", () => {
    const lp = new LabelPropagation({ gamma: 1 });
    lp.fit(X, y);
    const proba = lp.predictProba(X);
    expect(proba.shape[0]).toBe(6);
    expect(proba.shape[1]).toBe(2);
    for (let i = 0; i < 6; i++) {
      let sum = 0;
      for (let c = 0; c < 2; c++) {
        const p = Number(proba.data[proba.offset + i * 2 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        sum += p;
      }
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });

  it("score on labeled points", () => {
    const lp = new LabelPropagation({ gamma: 1 });
    const yFull = tensor([0, 0, 0, 1, 1, 1]);
    lp.fit(X, y);
    const acc = lp.score(X, yFull);
    expect(acc).toBeGreaterThan(0.8);
  });

  it("classes accessible after fit", () => {
    const lp = new LabelPropagation({ gamma: 1 });
    lp.fit(X, y);
    expect(lp.classes.size).toBe(2);
  });

  it("throws when not fitted", () => {
    const lp = new LabelPropagation();
    expect(() => lp.predict(X)).toThrow();
    expect(() => lp.classes).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new LabelPropagation({ maxIter: 0 })).toThrow();
  });

  it("throws for invalid gamma", () => {
    expect(() => new LabelPropagation({ gamma: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const lp = new LabelPropagation({ maxIter: 50, gamma: 5 });
    const params = lp.getParams();
    expect(params.maxIter).toBe(50);
    expect(params.gamma).toBe(5);
  });
});

describe("NearestCentroid", () => {
  const X = tensor([
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 1],
    [0, 2],
    [0, 3],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("fit and predict", () => {
    const nc = new NearestCentroid();
    nc.fit(X, y);
    const pred = nc.predict(X);
    expect(pred.size).toBe(6);
    // Should classify training data correctly
    for (let i = 0; i < 6; i++) {
      expect(Number(pred.data[pred.offset + i])).toBe(Number(y.data[y.offset + i]));
    }
  });

  it("score is 1.0 on linearly separable data", () => {
    const nc = new NearestCentroid();
    nc.fit(X, y);
    expect(nc.score(X, y)).toBe(1);
  });

  it("predictProba returns valid probabilities", () => {
    const nc = new NearestCentroid();
    nc.fit(X, y);
    const proba = nc.predictProba(X);
    expect(proba.shape[0]).toBe(6);
    expect(proba.shape[1]).toBe(2);
  });

  it("classes accessible after fit", () => {
    const nc = new NearestCentroid();
    nc.fit(X, y);
    expect(nc.classes.size).toBe(2);
  });

  it("throws when not fitted", () => {
    const nc = new NearestCentroid();
    expect(() => nc.predict(X)).toThrow();
    expect(() => nc.predictProba(X)).toThrow();
    expect(() => nc.classes).toThrow();
  });

  it("handles multiclass", () => {
    const X3 = tensor([
      [1, 0],
      [2, 0],
      [0, 1],
      [0, 2],
      [1, 1],
      [2, 2],
    ]);
    const y3 = tensor([0, 0, 1, 1, 2, 2]);
    const nc = new NearestCentroid();
    nc.fit(X3, y3);
    const pred = nc.predict(X3);
    expect(pred.size).toBe(6);
  });
});
