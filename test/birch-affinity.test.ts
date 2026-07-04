import { describe, expect, it } from "vitest";
import { AffinityPropagation, Birch } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("Birch", () => {
  const X = tensor([
    [0, 0],
    [0.1, 0],
    [0, 0.1],
    [0.1, 0.1],
    [10, 10],
    [10.1, 10],
    [10, 10.1],
    [10.1, 10.1],
  ]);

  it("fit produces labels", () => {
    const birch = new Birch({ nClusters: 2, threshold: 0.5 });
    birch.fit(X);
    expect(birch.labels.size).toBe(8);
  });

  it("separates well-separated clusters", () => {
    const birch = new Birch({ nClusters: 2, threshold: 0.5 });
    birch.fit(X);
    const labels = birch.labels;
    const label0 = Number(labels.data[labels.offset]);
    const label4 = Number(labels.data[labels.offset + 4]);
    expect(label0).not.toBe(label4);
  });

  it("clusterCenters are accessible", () => {
    const birch = new Birch({ nClusters: 2, threshold: 0.5 });
    birch.fit(X);
    const centers = birch.clusterCenters;
    expect(centers.shape[0]).toBe(2);
    expect(centers.shape[1]).toBe(2);
  });

  it("predict assigns new points", () => {
    const birch = new Birch({ nClusters: 2, threshold: 0.5 });
    birch.fit(X);
    const Xnew = tensor([
      [0.05, 0.05],
      [10.05, 10.05],
    ]);
    const pred = birch.predict(Xnew);
    expect(pred.size).toBe(2);
    expect(Number(pred.data[pred.offset])).not.toBe(Number(pred.data[pred.offset + 1]));
  });

  it("fitPredict returns labels", () => {
    const birch = new Birch({ nClusters: 2, threshold: 0.5 });
    const labels = birch.fitPredict(X);
    expect(labels.size).toBe(8);
  });

  it("throws when not fitted", () => {
    const birch = new Birch();
    expect(() => birch.labels).toThrow();
    expect(() => birch.clusterCenters).toThrow();
    expect(() => birch.predict(X)).toThrow();
  });

  it("throws for invalid nClusters", () => {
    expect(() => new Birch({ nClusters: 0 })).toThrow();
  });

  it("throws for invalid threshold", () => {
    expect(() => new Birch({ threshold: 0 })).toThrow();
    expect(() => new Birch({ threshold: -1 })).toThrow();
  });

  it("throws for invalid branchingFactor", () => {
    expect(() => new Birch({ branchingFactor: 1 })).toThrow();
  });

  it("getParams returns options", () => {
    const birch = new Birch({ nClusters: 4, threshold: 1.0 });
    const params = birch.getParams();
    expect(params.nClusters).toBe(4);
    expect(params.threshold).toBe(1.0);
  });

  it("setParams works and rejects unknown params", () => {
    const birch = new Birch();
    expect(birch.setParams({})).toBe(birch);
    expect(() => birch.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("AffinityPropagation", () => {
  const X = tensor([
    [0, 0],
    [0.1, 0],
    [0, 0.1],
    [0.1, 0.1],
    [10, 10],
    [10.1, 10],
    [10, 10.1],
    [10.1, 10.1],
  ]);

  it("fit produces labels", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    expect(ap.labels.size).toBe(8);
  });

  it("separates well-separated clusters", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    const labels = ap.labels;
    const label0 = Number(labels.data[labels.offset]);
    const label4 = Number(labels.data[labels.offset + 4]);
    expect(label0).not.toBe(label4);
  });

  it("clusterCenters are accessible", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    const centers = ap.clusterCenters;
    expect(centers.shape[0]).toBeGreaterThanOrEqual(1);
    expect(centers.shape[1]).toBe(2);
  });

  it("clusterCentersIndices are accessible", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    const indices = ap.clusterCentersIndices;
    expect(indices.length).toBeGreaterThanOrEqual(1);
  });

  it("predict assigns new points", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    const Xnew = tensor([
      [0.05, 0.05],
      [10.05, 10.05],
    ]);
    const pred = ap.predict(Xnew);
    expect(pred.size).toBe(2);
  });

  it("fitPredict returns labels", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    const labels = ap.fitPredict(X);
    expect(labels.size).toBe(8);
  });

  it("nIter is positive after fit", () => {
    const ap = new AffinityPropagation({ damping: 0.5, maxIter: 100 });
    ap.fit(X);
    expect(ap.nIter).toBeGreaterThan(0);
  });

  it("throws when not fitted", () => {
    const ap = new AffinityPropagation();
    expect(() => ap.labels).toThrow();
    expect(() => ap.clusterCenters).toThrow();
    expect(() => ap.clusterCentersIndices).toThrow();
    expect(() => ap.nIter).toThrow();
    expect(() => ap.predict(X)).toThrow();
  });

  it("throws for invalid damping", () => {
    expect(() => new AffinityPropagation({ damping: 0.3 })).toThrow();
    expect(() => new AffinityPropagation({ damping: 1.0 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new AffinityPropagation({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const ap = new AffinityPropagation({ damping: 0.7, maxIter: 300 });
    const params = ap.getParams();
    expect(params.damping).toBe(0.7);
    expect(params.maxIter).toBe(300);
  });

  it("setParams works and rejects unknown params", () => {
    const ap = new AffinityPropagation();
    expect(ap.setParams({})).toBe(ap);
    expect(() => ap.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });

  it("preference parameter works", () => {
    const ap = new AffinityPropagation({
      damping: 0.5,
      maxIter: 100,
      preference: -50,
    });
    ap.fit(X);
    expect(ap.labels.size).toBe(8);
  });
});
