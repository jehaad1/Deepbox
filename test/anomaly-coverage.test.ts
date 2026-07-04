import { describe, expect, it } from "vitest";
import { IsolationForest } from "../src/ml/anomaly/IsolationForest";
import { LocalOutlierFactor } from "../src/ml/anomaly/LocalOutlierFactor";
import { tensor } from "../src/ndarray";

describe("IsolationForest", () => {
  const X = tensor([
    [0, 0],
    [0.1, -0.1],
    [0.2, 0.1],
    [-0.1, 0.2],
    [0, 0.1],
    [0.15, 0],
    [0.05, -0.05],
    [100, 100],
  ]);

  it("fits and predicts outliers", () => {
    const ifo = new IsolationForest({
      nEstimators: 50,
      contamination: 0.15,
      randomState: 42,
    });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([8]);
    // The outlier [100,100] should be marked as -1
    const arr = labels.toArray() as number[];
    expect(arr[7]).toBe(-1);
  });

  it("fitPredict returns labels", () => {
    const ifo = new IsolationForest({ nEstimators: 20, randomState: 42 });
    const labels = ifo.fitPredict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("scoreSamples returns anomaly scores", () => {
    const ifo = new IsolationForest({ nEstimators: 20, randomState: 42 });
    ifo.fit(X);
    const scores = ifo.scoreSamples(X);
    expect(scores.shape).toEqual([8]);
    // Outlier should have more negative score
    const arr = scores.toArray() as number[];
    expect(arr[7]).toBeLessThan(arr[0] as number);
  });

  it("throws when predicting before fit", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => ifo.scoreSamples(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("validates feature count on predict", () => {
    const ifo = new IsolationForest({ nEstimators: 10, randomState: 1 });
    ifo.fit(X);
    expect(() => ifo.predict(tensor([[1, 2, 3]]))).toThrow(/features/i);
  });

  it("getParams returns all options", () => {
    const ifo = new IsolationForest({
      nEstimators: 50,
      maxSamples: 128,
      contamination: 0.1,
      maxFeatures: 0.5,
      randomState: 7,
    });
    const p = ifo.getParams();
    expect(p.nEstimators).toBe(50);
    expect(p.maxSamples).toBe(128);
    expect(p.contamination).toBe(0.1);
    expect(p.maxFeatures).toBe(0.5);
    expect(p.randomState).toBe(7);
  });

  it("works with auto maxSamples", () => {
    const ifo = new IsolationForest({
      nEstimators: 5,
      maxSamples: "auto",
      randomState: 1,
    });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("works with auto contamination", () => {
    const ifo = new IsolationForest({
      nEstimators: 5,
      contamination: "auto",
      randomState: 1,
    });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("works with numeric maxSamples", () => {
    const ifo = new IsolationForest({
      nEstimators: 5,
      maxSamples: 4,
      randomState: 1,
    });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("validates constructor params", () => {
    expect(() => new IsolationForest({ nEstimators: 0 })).toThrow(/nEstimators/);
    expect(() => new IsolationForest({ nEstimators: -1 })).toThrow(/nEstimators/);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const ifo = new IsolationForest();
      ifo.setParams({
        nEstimators: 50,
        maxSamples: 128,
        contamination: 0.2,
        maxFeatures: 0.8,
        randomState: 42,
      });
      const p = ifo.getParams();
      expect(p.nEstimators).toBe(50);
      expect(p.maxSamples).toBe(128);
      expect(p.contamination).toBe(0.2);
      expect(p.maxFeatures).toBe(0.8);
      expect(p.randomState).toBe(42);
    });

    it("accepts auto maxSamples", () => {
      const ifo = new IsolationForest();
      ifo.setParams({ maxSamples: "auto" });
      expect(ifo.getParams().maxSamples).toBe("auto");
    });

    it("accepts auto contamination", () => {
      const ifo = new IsolationForest();
      ifo.setParams({ contamination: "auto" });
      expect(ifo.getParams().contamination).toBe("auto");
    });

    it("accepts undefined randomState", () => {
      const ifo = new IsolationForest({ randomState: 42 });
      ifo.setParams({ randomState: undefined });
      expect(ifo.getParams().randomState).toBeUndefined();
    });

    it("rejects invalid nEstimators", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ nEstimators: 0 })).toThrow(/nEstimators/);
      expect(() => ifo.setParams({ nEstimators: 1.5 })).toThrow(/nEstimators/);
      expect(() => ifo.setParams({ nEstimators: "10" })).toThrow(/nEstimators/);
    });

    it("rejects invalid maxSamples", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ maxSamples: 0 })).toThrow(/maxSamples/);
      expect(() => ifo.setParams({ maxSamples: -1 })).toThrow(/maxSamples/);
      expect(() => ifo.setParams({ maxSamples: false })).toThrow(/maxSamples/);
    });

    it("rejects invalid contamination", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ contamination: 0 })).toThrow(/contamination/);
      expect(() => ifo.setParams({ contamination: 0.6 })).toThrow(/contamination/);
      expect(() => ifo.setParams({ contamination: -1 })).toThrow(/contamination/);
    });

    it("rejects invalid maxFeatures", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ maxFeatures: 0 })).toThrow(/maxFeatures/);
      expect(() => ifo.setParams({ maxFeatures: 1.1 })).toThrow(/maxFeatures/);
      expect(() => ifo.setParams({ maxFeatures: "all" })).toThrow(/maxFeatures/);
    });

    it("rejects invalid randomState", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ randomState: "42" })).toThrow(/randomState/);
      expect(() => ifo.setParams({ randomState: Infinity })).toThrow(/randomState/);
    });

    it("rejects unknown params", () => {
      const ifo = new IsolationForest();
      expect(() => ifo.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});

describe("LocalOutlierFactor", () => {
  const X = tensor([
    [0, 0],
    [0.1, -0.1],
    [0.2, 0.1],
    [-0.1, 0.2],
    [0, 0.1],
    [100, 100],
  ]);

  it("fits and predicts outliers", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.2 });
    lof.fit(X);
    const labels = lof.predict(X);
    expect(labels.shape).toEqual([6]);
  });

  it("fitPredict returns labels", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    const labels = lof.fitPredict(X);
    expect(labels.shape).toEqual([6]);
  });

  it("scoreSamples on training data", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    lof.fit(X);
    const scores = lof.scoreSamples(X);
    expect(scores.shape).toEqual([6]);
  });

  it("scoreSamples on new data", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    lof.fit(X);
    const Xnew = tensor([
      [0.05, 0.05],
      [50, 50],
    ]);
    const scores = lof.scoreSamples(Xnew);
    expect(scores.shape).toEqual([2]);
  });

  it("predict on new data (different size)", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 2, contamination: 0.2 });
    lof.fit(X);
    const Xnew = tensor([
      [0.05, 0.05],
      [50, 50],
    ]);
    const labels = lof.predict(Xnew);
    expect(labels.shape).toEqual([2]);
  });

  it("negativeLofScores getter", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    lof.fit(X);
    const scores = lof.negativeLofScores;
    expect(scores.shape).toEqual([6]);
  });

  it("throws when predicting before fit", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => lof.scoreSamples(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => lof.negativeLofScores).toThrow(/fitted/i);
  });

  it("validates constructor params", () => {
    expect(() => new LocalOutlierFactor({ nNeighbors: 0 })).toThrow(/nNeighbors/);
    expect(() => new LocalOutlierFactor({ nNeighbors: -1 })).toThrow(/nNeighbors/);
    expect(() => new LocalOutlierFactor({ nNeighbors: 1.5 })).toThrow(/nNeighbors/);
  });

  it("getParams returns options", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 5, contamination: 0.1 });
    const p = lof.getParams();
    expect(p.nNeighbors).toBe(5);
    expect(p.contamination).toBe(0.1);
  });

  it("works with auto contamination", () => {
    const lof = new LocalOutlierFactor({ contamination: "auto" });
    lof.fit(X);
    const labels = lof.predict(X);
    expect(labels.shape).toEqual([6]);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const lof = new LocalOutlierFactor();
      lof.setParams({ nNeighbors: 10, contamination: 0.3 });
      const p = lof.getParams();
      expect(p.nNeighbors).toBe(10);
      expect(p.contamination).toBe(0.3);
    });

    it("accepts auto contamination", () => {
      const lof = new LocalOutlierFactor();
      lof.setParams({ contamination: "auto" });
      expect(lof.getParams().contamination).toBe("auto");
    });

    it("rejects invalid nNeighbors", () => {
      const lof = new LocalOutlierFactor();
      expect(() => lof.setParams({ nNeighbors: 0 })).toThrow(/nNeighbors/);
      expect(() => lof.setParams({ nNeighbors: 1.5 })).toThrow(/nNeighbors/);
      expect(() => lof.setParams({ nNeighbors: "5" })).toThrow(/nNeighbors/);
    });

    it("rejects invalid contamination", () => {
      const lof = new LocalOutlierFactor();
      expect(() => lof.setParams({ contamination: 0 })).toThrow(/contamination/);
      expect(() => lof.setParams({ contamination: 0.6 })).toThrow(/contamination/);
    });

    it("rejects unknown params", () => {
      const lof = new LocalOutlierFactor();
      expect(() => lof.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});
