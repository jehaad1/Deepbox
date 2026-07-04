import { describe, expect, it } from "vitest";
import { RadiusNeighborsClassifier, RadiusNeighborsRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("RadiusNeighborsClassifier", () => {
  const X = tensor([
    [0, 0],
    [0.1, 0],
    [0, 0.1],
    [10, 10],
    [10.1, 10],
    [10, 10.1],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("fit and predict on training data", () => {
    const rnc = new RadiusNeighborsClassifier({ radius: 1 });
    rnc.fit(X, y);
    const pred = rnc.predict(X);
    expect(pred.size).toBe(6);
    for (let i = 0; i < 6; i++) {
      expect(Number(pred.data[pred.offset + i])).toBe(Number(y.data[y.offset + i]));
    }
  });

  it("score is 1.0 on well-separated data", () => {
    const rnc = new RadiusNeighborsClassifier({ radius: 1 });
    rnc.fit(X, y);
    expect(rnc.score(X, y)).toBe(1);
  });

  it("predictProba returns valid probabilities", () => {
    const rnc = new RadiusNeighborsClassifier({ radius: 1 });
    rnc.fit(X, y);
    const proba = rnc.predictProba(X);
    expect(proba.shape[0]).toBe(6);
    expect(proba.shape[1]).toBe(2);
    for (let i = 0; i < 6; i++) {
      let sum = 0;
      for (let c = 0; c < 2; c++) {
        const p = Number(proba.data[proba.offset + i * 2 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        sum += p;
      }
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("classes accessible after fit", () => {
    const rnc = new RadiusNeighborsClassifier({ radius: 1 });
    rnc.fit(X, y);
    expect(rnc.classes.size).toBe(2);
  });

  it("throws when not fitted", () => {
    const rnc = new RadiusNeighborsClassifier();
    expect(() => rnc.predict(X)).toThrow();
    expect(() => rnc.predictProba(X)).toThrow();
    expect(() => rnc.classes).toThrow();
  });

  it("throws for invalid radius", () => {
    expect(() => new RadiusNeighborsClassifier({ radius: 0 })).toThrow();
    expect(() => new RadiusNeighborsClassifier({ radius: -1 })).toThrow();
  });

  it("getParams returns options", () => {
    const rnc = new RadiusNeighborsClassifier({ radius: 5 });
    expect(rnc.getParams().radius).toBe(5);
  });
});

describe("RadiusNeighborsRegressor", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([2, 4, 6, 8, 10]);

  it("fit and predict", () => {
    const rnr = new RadiusNeighborsRegressor({ radius: 1.5 });
    rnr.fit(X, y);
    const pred = rnr.predict(X);
    expect(pred.size).toBe(5);
  });

  it("predicts mean of neighbors within radius", () => {
    const rnr = new RadiusNeighborsRegressor({ radius: 0.5 });
    rnr.fit(X, y);
    const pred = rnr.predict(tensor([[1]]));
    // Only [1] is within radius 0.5 of [1], so prediction = 2
    expect(Number(pred.data[pred.offset])).toBeCloseTo(2, 5);
  });

  it("score on training data", () => {
    const rnr = new RadiusNeighborsRegressor({ radius: 0.5 });
    rnr.fit(X, y);
    const r2 = rnr.score(X, y);
    expect(r2).toBe(1); // exact interpolation
  });

  it("throws when not fitted", () => {
    const rnr = new RadiusNeighborsRegressor();
    expect(() => rnr.predict(X)).toThrow();
  });

  it("throws for invalid radius", () => {
    expect(() => new RadiusNeighborsRegressor({ radius: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const rnr = new RadiusNeighborsRegressor({ radius: 3.5 });
    expect(rnr.getParams().radius).toBe(3.5);
  });
});
