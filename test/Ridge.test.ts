import { describe, expect, it } from "vitest";
import { Ridge } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("deepbox/ml - Ridge", () => {
  it("should throw if predict is called before fit", () => {
    const model = new Ridge();
    expect(() => model.predict(tensor([[1, 2]]))).toThrow("Ridge must be fitted");
  });

  it("should validate alpha", () => {
    const model = new Ridge({ alpha: -1 });
    const X = tensor([[1], [2]]);
    const y = tensor([1, 2]);
    expect(() => model.fit(X, y)).toThrow("alpha must be >= 0");
  });

  it("should fit and predict with intercept", () => {
    const model = new Ridge({ alpha: 1.0, fitIntercept: true });
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([1, 3, 5, 7]);

    model.fit(X, y);
    const pred = model.predict(X);

    expect(pred.shape).toEqual([4]);
    const r2 = model.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("should handle constant y in score", () => {
    const model = new Ridge({ alpha: 1.0 });
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([5, 5, 5, 5]);

    model.fit(X, y);
    const r2 = model.score(X, y);
    expect(r2).toBeCloseTo(1.0, 6);
  });

  it("should support normalization", () => {
    const model = new Ridge({ alpha: 1.0, normalize: true });
    const X = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
    ]);
    const y = tensor([2, 4, 6]);
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.shape).toEqual([3]);
  });

  it("setParams should reject unknown keys", () => {
    const model = new Ridge();
    expect(() => model.setParams({ unknown: 123 })).toThrow("Unknown parameter");
  });

  it("should throw on singular design matrix when alpha is zero", () => {
    const model = new Ridge({ alpha: 0, fitIntercept: false });
    const X = tensor([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    const y = tensor([1, 2, 3]);
    expect(() => model.fit(X, y)).toThrow(/singular|ill-conditioned/i);
  });
});
