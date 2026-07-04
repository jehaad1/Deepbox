import { describe, expect, it } from "vitest";
import { IsolationForest } from "../src/ml/anomaly/IsolationForest";
import { LocalOutlierFactor } from "../src/ml/anomaly/LocalOutlierFactor";
import { tensor } from "../src/ndarray";

describe("IsolationForest extended branches", () => {
  const X = tensor([
    [0, 0],
    [0.1, -0.1],
    [0.2, 0.1],
    [-0.1, 0.2],
    [0, 0.1],
    [0.1, 0],
    [-0.1, -0.1],
    [0.2, -0.2],
    [100, 100],
  ]);

  it("fit and predict with contamination", () => {
    const ifo = new IsolationForest({ nEstimators: 50, contamination: 0.25, randomState: 42 });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([9]);
    // Labels should be -1 or 1
    const arr = labels.toArray() as number[];
    for (const l of arr) {
      expect(l === -1 || l === 1).toBe(true);
    }
  });

  it("fit and predict with contamination=auto", () => {
    const ifo = new IsolationForest({ nEstimators: 20, contamination: "auto", randomState: 42 });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const ifo = new IsolationForest({ nEstimators: 20, randomState: 42 });
    const labels = ifo.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("scoreSamples returns anomaly scores", () => {
    const ifo = new IsolationForest({ nEstimators: 20, randomState: 42 });
    ifo.fit(X);
    const scores = ifo.scoreSamples(X);
    expect(scores.shape).toEqual([9]);
    // Scores should be negative (more negative = more anomalous)
    const arr = scores.toArray() as number[];
    for (const s of arr) {
      expect(s).toBeLessThanOrEqual(0);
    }
  });

  it("maxSamples as number", () => {
    const ifo = new IsolationForest({ nEstimators: 10, maxSamples: 5, randomState: 42 });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("throws when predicting before fit", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("throws when scoring before fit", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.scoreSamples(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nEstimators", () => {
    expect(() => new IsolationForest({ nEstimators: 0 })).toThrow();
    expect(() => new IsolationForest({ nEstimators: -1 })).toThrow();
    expect(() => new IsolationForest({ nEstimators: 1.5 })).toThrow();
  });

  it("getParams returns correct values", () => {
    const ifo = new IsolationForest({
      nEstimators: 50,
      maxSamples: 100,
      contamination: 0.1,
      maxFeatures: 0.5,
      randomState: 42,
    });
    const params = ifo.getParams();
    expect(params.nEstimators).toBe(50);
    expect(params.maxSamples).toBe(100);
    expect(params.contamination).toBe(0.1);
    expect(params.maxFeatures).toBe(0.5);
    expect(params.randomState).toBe(42);
  });

  it("setParams updates nEstimators", () => {
    const ifo = new IsolationForest();
    ifo.setParams({ nEstimators: 200 });
    expect(ifo.getParams().nEstimators).toBe(200);
  });

  it("setParams updates maxSamples", () => {
    const ifo = new IsolationForest();
    ifo.setParams({ maxSamples: "auto" });
    expect(ifo.getParams().maxSamples).toBe("auto");
    ifo.setParams({ maxSamples: 100 });
    expect(ifo.getParams().maxSamples).toBe(100);
  });

  it("setParams updates contamination", () => {
    const ifo = new IsolationForest();
    ifo.setParams({ contamination: 0.1 });
    expect(ifo.getParams().contamination).toBe(0.1);
    ifo.setParams({ contamination: "auto" });
    expect(ifo.getParams().contamination).toBe("auto");
  });

  it("setParams updates maxFeatures", () => {
    const ifo = new IsolationForest();
    ifo.setParams({ maxFeatures: 0.5 });
    expect(ifo.getParams().maxFeatures).toBe(0.5);
  });

  it("setParams updates randomState", () => {
    const ifo = new IsolationForest();
    ifo.setParams({ randomState: 42 });
    expect(ifo.getParams().randomState).toBe(42);
  });

  it("setParams rejects invalid nEstimators", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ nEstimators: 0 })).toThrow();
    expect(() => ifo.setParams({ nEstimators: 1.5 })).toThrow();
  });

  it("setParams rejects invalid maxSamples", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ maxSamples: 0 })).toThrow();
    expect(() => ifo.setParams({ maxSamples: -1 })).toThrow();
  });

  it("setParams rejects invalid contamination", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ contamination: 0 })).toThrow();
    expect(() => ifo.setParams({ contamination: 0.6 })).toThrow();
  });

  it("setParams rejects invalid maxFeatures", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ maxFeatures: 0 })).toThrow();
    expect(() => ifo.setParams({ maxFeatures: 1.5 })).toThrow();
  });

  it("setParams rejects invalid randomState", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ randomState: NaN })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });

  it("without randomState uses Math.random", () => {
    const ifo = new IsolationForest({ nEstimators: 5 });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([9]);
  });
});

describe("LocalOutlierFactor extended branches", () => {
  const X = tensor([
    [0, 0],
    [0.1, -0.1],
    [0.2, 0.1],
    [-0.1, 0.2],
    [0, 0.1],
    [0.1, 0],
    [-0.1, -0.1],
    [0.2, -0.2],
    [100, 100],
  ]);

  it("fit and predict with contamination", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.25 });
    lof.fit(X);
    const labels = lof.predict(X);
    expect(labels.shape).toEqual([9]);
    const arr = labels.toArray() as number[];
    for (const l of arr) {
      expect(l === -1 || l === 1).toBe(true);
    }
  });

  it("fit and predict with contamination=auto", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3, contamination: "auto" });
    lof.fit(X);
    const labels = lof.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    const labels = lof.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("scoreSamples on training data", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    lof.fit(X);
    const scores = lof.scoreSamples(X);
    expect(scores.shape).toEqual([9]);
  });

  it("predict on new data (different size)", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    lof.fit(X);
    const Xnew = tensor([
      [0, 0],
      [50, 50],
    ]);
    const labels = lof.predict(Xnew);
    expect(labels.shape).toEqual([2]);
  });

  it("scoreSamples on new data (different size)", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    lof.fit(X);
    const Xnew = tensor([
      [0, 0],
      [50, 50],
    ]);
    const scores = lof.scoreSamples(Xnew);
    expect(scores.shape).toEqual([2]);
  });

  it("negativeLofScores accessor", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    lof.fit(X);
    const scores = lof.negativeLofScores;
    expect(scores.shape).toEqual([9]);
  });

  it("negativeLofScores throws before fit", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.negativeLofScores).toThrow(/fitted/i);
  });

  it("throws when predicting before fit", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("throws when scoring before fit", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.scoreSamples(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid nNeighbors", () => {
    expect(() => new LocalOutlierFactor({ nNeighbors: 0 })).toThrow();
    expect(() => new LocalOutlierFactor({ nNeighbors: -1 })).toThrow();
    expect(() => new LocalOutlierFactor({ nNeighbors: 1.5 })).toThrow();
  });

  it("rejects too few samples for nNeighbors", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 100 });
    expect(() => lof.fit(tensor([[1, 2]]))).toThrow(/not enough/i);
  });

  it("getParams returns correct values", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 10, contamination: 0.1 });
    const params = lof.getParams();
    expect(params.nNeighbors).toBe(10);
    expect(params.contamination).toBe(0.1);
  });

  it("setParams updates nNeighbors", () => {
    const lof = new LocalOutlierFactor();
    lof.setParams({ nNeighbors: 10 });
    expect(lof.getParams().nNeighbors).toBe(10);
  });

  it("setParams updates contamination", () => {
    const lof = new LocalOutlierFactor();
    lof.setParams({ contamination: 0.1 });
    expect(lof.getParams().contamination).toBe(0.1);
    lof.setParams({ contamination: "auto" });
    expect(lof.getParams().contamination).toBe("auto");
  });

  it("setParams rejects invalid nNeighbors", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.setParams({ nNeighbors: 0 })).toThrow();
    expect(() => lof.setParams({ nNeighbors: 1.5 })).toThrow();
  });

  it("setParams rejects invalid contamination", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.setParams({ contamination: 0 })).toThrow();
    expect(() => lof.setParams({ contamination: 0.6 })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });
});
