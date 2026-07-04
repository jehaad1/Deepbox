import { describe, expect, it } from "vitest";
import { LabelSpreading } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("LabelSpreading", () => {
  const X = tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [5, 5],
    [6, 5],
  ]);
  const y = tensor([0, -1, -1, -1, 1, -1]);

  it("fits and predicts labels for semi-supervised data", () => {
    const ls = new LabelSpreading({ alpha: 0.2 });
    ls.fit(X, y);

    const pred = ls.predict(X);
    expect(pred.shape).toEqual([6]);

    // Labeled points should retain their labels
    expect(Number(pred.data[pred.offset])).toBe(0);
    // Point [5,5] should be class 1
    expect(Number(pred.data[pred.offset + 4])).toBe(1);
  });

  it("predictProba returns valid probabilities", () => {
    const ls = new LabelSpreading();
    ls.fit(X, y);

    const proba = ls.predictProba(X);
    expect(proba.shape).toEqual([6, 2]);

    // Each row should sum to ~1
    for (let i = 0; i < 6; i++) {
      let rowSum = 0;
      for (let c = 0; c < 2; c++) {
        const val = Number(proba.data[proba.offset + i * 2 + c]);
        expect(val).toBeGreaterThanOrEqual(0);
        rowSum += val;
      }
      expect(rowSum).toBeCloseTo(1, 1);
    }
  });

  it("score computes accuracy", () => {
    const ls = new LabelSpreading();
    ls.fit(X, y);
    const acc = ls.score(X, tensor([0, 0, 0, 0, 1, 1]));
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });

  it("exposes classes after fitting", () => {
    const ls = new LabelSpreading();
    ls.fit(X, y);
    const cls = ls.classes;
    expect(cls.size).toBe(2);
  });

  it("validates parameters", () => {
    expect(() => new LabelSpreading({ maxIter: 0 })).toThrow(/maxIter/);
    expect(() => new LabelSpreading({ gamma: -1 })).toThrow(/gamma/);
    expect(() => new LabelSpreading({ alpha: 2 })).toThrow(/alpha/);
    expect(() => new LabelSpreading({ alpha: -0.1 })).toThrow(/alpha/);
  });

  it("throws NotFittedError when not fitted", () => {
    const ls = new LabelSpreading();
    expect(() => ls.predict(X)).toThrow(/fitted/i);
    expect(() => ls.classes).toThrow(/fitted/i);
  });

  it("getParams returns constructor params", () => {
    const ls = new LabelSpreading({ alpha: 0.5, gamma: 10 });
    const params = ls.getParams();
    expect(params.alpha).toBe(0.5);
    expect(params.gamma).toBe(10);
  });
});
