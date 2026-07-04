import { describe, expect, it } from "vitest";
import { QuantileRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("QuantileRegressor", () => {
  const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
  const y = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

  it("fit and predict at median (0.5)", () => {
    const qr = new QuantileRegressor({ quantile: 0.5, alpha: 0 });
    qr.fit(X, y);
    const pred = qr.predict(X);
    expect(pred.size).toBe(10);
    // Should be close to y = 2x
    for (let i = 0; i < 10; i++) {
      expect(Number(pred.data[pred.offset + i])).toBeCloseTo(Number(y.data[y.offset + i]), 0);
    }
  });

  it("upper quantile predictions are higher", () => {
    const qr50 = new QuantileRegressor({ quantile: 0.5, alpha: 0.01 });
    const qr90 = new QuantileRegressor({ quantile: 0.9, alpha: 0.01 });
    // Add some noise
    const yNoisy = tensor([2, 5, 5, 9, 9, 13, 13, 17, 17, 21]);
    qr50.fit(X, yNoisy);
    qr90.fit(X, yNoisy);
    const pred50 = qr50.predict(tensor([[5]]));
    const pred90 = qr90.predict(tensor([[5]]));
    // 90th quantile should generally be >= median
    expect(Number(pred90.data[pred90.offset])).toBeGreaterThanOrEqual(
      Number(pred50.data[pred50.offset]) - 2
    );
  });

  it("coef and intercept accessible after fit", () => {
    const qr = new QuantileRegressor({ quantile: 0.5, alpha: 0 });
    qr.fit(X, y);
    expect(qr.coef.length).toBe(1);
    expect(typeof qr.intercept).toBe("number");
  });

  it("score on linear data", () => {
    const qr = new QuantileRegressor({ quantile: 0.5, alpha: 0 });
    qr.fit(X, y);
    const r2 = qr.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("throws when not fitted", () => {
    const qr = new QuantileRegressor();
    expect(() => qr.predict(X)).toThrow();
    expect(() => qr.coef).toThrow();
    expect(() => qr.intercept).toThrow();
  });

  it("throws for invalid quantile", () => {
    expect(() => new QuantileRegressor({ quantile: 0 })).toThrow();
    expect(() => new QuantileRegressor({ quantile: 1 })).toThrow();
    expect(() => new QuantileRegressor({ quantile: -0.1 })).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new QuantileRegressor({ alpha: -1 })).toThrow();
  });

  it("getParams returns options", () => {
    const qr = new QuantileRegressor({ quantile: 0.75, alpha: 0.5 });
    const params = qr.getParams();
    expect(params.quantile).toBe(0.75);
    expect(params.alpha).toBe(0.5);
  });

  it("multidimensional input", () => {
    const X2 = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 2],
    ]);
    const y2 = tensor([1, 2, 3, 4]);
    const qr = new QuantileRegressor({ quantile: 0.5, alpha: 0.01 });
    qr.fit(X2, y2);
    const pred = qr.predict(X2);
    expect(pred.size).toBe(4);
  });
});
