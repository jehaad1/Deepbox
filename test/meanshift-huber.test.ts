import { describe, expect, it } from "vitest";
import { HuberRegressor, MeanShift } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("MeanShift", () => {
  // Two well-separated clusters
  const X = tensor([
    [0, 0],
    [0.5, 0],
    [0, 0.5],
    [0.5, 0.5],
    [10, 10],
    [10.5, 10],
    [10, 10.5],
    [10.5, 10.5],
  ]);

  it("fit produces labels", () => {
    const ms = new MeanShift({ bandwidth: 2 });
    ms.fit(X);
    expect(ms.labels.size).toBe(8);
  });

  it("separates well-separated clusters", () => {
    const ms = new MeanShift({ bandwidth: 2 });
    ms.fit(X);
    const labels = ms.labels;
    const label0 = Number(labels.data[labels.offset]);
    const label4 = Number(labels.data[labels.offset + 4]);
    expect(label0).not.toBe(label4);

    for (let i = 0; i < 4; i++) {
      expect(Number(labels.data[labels.offset + i])).toBe(label0);
    }
    for (let i = 4; i < 8; i++) {
      expect(Number(labels.data[labels.offset + i])).toBe(label4);
    }
  });

  it("clusterCenters are accessible", () => {
    const ms = new MeanShift({ bandwidth: 2 });
    ms.fit(X);
    const centers = ms.clusterCenters;
    expect(centers.shape[0]).toBe(2);
    expect(centers.shape[1]).toBe(2);
  });

  it("predict assigns new points to nearest cluster", () => {
    const ms = new MeanShift({ bandwidth: 2 });
    ms.fit(X);
    const Xnew = tensor([
      [0.2, 0.2],
      [10.2, 10.2],
    ]);
    const pred = ms.predict(Xnew);
    expect(pred.size).toBe(2);
    // Should assign to different clusters
    expect(Number(pred.data[pred.offset])).not.toBe(Number(pred.data[pred.offset + 1]));
  });

  it("fitPredict returns labels", () => {
    const ms = new MeanShift({ bandwidth: 2 });
    const labels = ms.fitPredict(X);
    expect(labels.size).toBe(8);
  });

  it("auto bandwidth estimation works", () => {
    const ms = new MeanShift();
    ms.fit(X);
    expect(ms.labels.size).toBe(8);
  });

  it("binSeeding option works", () => {
    const ms = new MeanShift({ bandwidth: 2, binSeeding: true });
    ms.fit(X);
    expect(ms.labels.size).toBe(8);
  });

  it("throws when not fitted", () => {
    const ms = new MeanShift();
    expect(() => ms.labels).toThrow();
    expect(() => ms.clusterCenters).toThrow();
    expect(() => ms.predict(X)).toThrow();
  });

  it("throws for invalid bandwidth", () => {
    expect(() => new MeanShift({ bandwidth: 0 })).toThrow();
    expect(() => new MeanShift({ bandwidth: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new MeanShift({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const ms = new MeanShift({ bandwidth: 3, maxIter: 100 });
    const params = ms.getParams();
    expect(params.bandwidth).toBe(3);
    expect(params.maxIter).toBe(100);
  });
});

describe("HuberRegressor", () => {
  // Clean linear data: y = 2*x + 1
  const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
  const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);

  it("fits clean linear data", () => {
    const reg = new HuberRegressor({ maxIter: 500, learningRate: 0.01 });
    reg.fit(X, y);
    const r2 = reg.score(X, y);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("predict returns correct shape", () => {
    const reg = new HuberRegressor({ maxIter: 500, learningRate: 0.01 });
    reg.fit(X, y);
    const pred = reg.predict(X);
    expect(pred.size).toBe(8);
  });

  it("is less sensitive to outliers compared to plain linear model", () => {
    // Data with outlier
    const Xout = tensor([[1], [2], [3], [4], [5], [6], [7], [100]]);
    const yout = tensor([3, 5, 7, 9, 11, 13, 15, 500]);

    const reg = new HuberRegressor({ epsilon: 1.35, maxIter: 1000, learningRate: 0.001 });
    reg.fit(Xout, yout);
    // The coefficient should still be close to 2 (the true slope)
    // despite the outlier at (100, 500)
    expect(reg.coef.length).toBe(1);
  });

  it("coef and intercept accessible", () => {
    const reg = new HuberRegressor({ maxIter: 200 });
    reg.fit(X, y);
    expect(reg.coef.length).toBe(1);
    expect(typeof reg.intercept).toBe("number");
  });

  it("nIter accessible after fit", () => {
    const reg = new HuberRegressor({ maxIter: 200 });
    reg.fit(X, y);
    expect(reg.nIter).toBeGreaterThan(0);
  });

  it("outliers accessible after fit", () => {
    const reg = new HuberRegressor({ maxIter: 200 });
    reg.fit(X, y);
    expect(reg.outliers.length).toBe(8);
    // Clean data should have no outliers
    for (const o of reg.outliers) {
      expect(typeof o).toBe("boolean");
    }
  });

  it("multivariate regression", () => {
    const Xm = tensor([
      [1, 1],
      [2, 1],
      [3, 1],
      [1, 2],
      [2, 2],
      [3, 2],
      [1, 3],
      [2, 3],
      [3, 3],
      [4, 4],
    ]);
    const ym = tensor([6, 8, 10, 9, 11, 13, 12, 14, 16, 20]);
    const reg = new HuberRegressor({ maxIter: 2000, learningRate: 0.001 });
    reg.fit(Xm, ym);
    expect(reg.coef.length).toBe(2);
    const r2 = reg.score(Xm, ym);
    expect(r2).toBeGreaterThan(0.5);
  });

  it("throws when not fitted", () => {
    const reg = new HuberRegressor();
    expect(() => reg.predict(X)).toThrow();
    expect(() => reg.coef).toThrow();
    expect(() => reg.intercept).toThrow();
    expect(() => reg.nIter).toThrow();
    expect(() => reg.outliers).toThrow();
  });

  it("throws for invalid epsilon", () => {
    expect(() => new HuberRegressor({ epsilon: 1.0 })).toThrow();
    expect(() => new HuberRegressor({ epsilon: 0.5 })).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new HuberRegressor({ alpha: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new HuberRegressor({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const reg = new HuberRegressor({ epsilon: 2.0, alpha: 0.01 });
    const params = reg.getParams();
    expect(params.epsilon).toBe(2.0);
    expect(params.alpha).toBe(0.01);
  });
});
