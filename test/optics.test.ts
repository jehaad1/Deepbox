import { describe, expect, it } from "vitest";
import { OPTICS } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("OPTICS", () => {
  // Two well-separated clusters
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
    const optics = new OPTICS({ minSamples: 2, eps: 1 });
    optics.fit(X);
    expect(optics.labels.size).toBe(8);
  });

  it("separates well-separated clusters (dbscan method)", () => {
    const optics = new OPTICS({
      minSamples: 2,
      eps: 1,
      clusterMethod: "dbscan",
    });
    optics.fit(X);
    const labels = optics.labels;
    const label0 = Number(labels.data[labels.offset]);
    const label4 = Number(labels.data[labels.offset + 4]);
    // Should be in different clusters (or one is noise)
    expect(label0).not.toBe(label4);
  });

  it("fitPredict returns labels", () => {
    const optics = new OPTICS({ minSamples: 2, eps: 1 });
    const labels = optics.fitPredict(X);
    expect(labels.size).toBe(8);
  });

  it("reachability is accessible after fit", () => {
    const optics = new OPTICS({ minSamples: 2 });
    optics.fit(X);
    expect(optics.reachability.length).toBe(8);
  });

  it("ordering is accessible after fit", () => {
    const optics = new OPTICS({ minSamples: 2 });
    optics.fit(X);
    expect(optics.ordering.length).toBe(8);
    // ordering should be a permutation of 0..7
    const sorted = Array.from(optics.ordering).sort((a, b) => a - b);
    for (let i = 0; i < 8; i++) {
      expect(sorted[i]).toBe(i);
    }
  });

  it("coreDistances is accessible after fit", () => {
    const optics = new OPTICS({ minSamples: 2 });
    optics.fit(X);
    expect(optics.coreDistances.length).toBe(8);
  });

  it("xi cluster method works", () => {
    const optics = new OPTICS({ minSamples: 2, clusterMethod: "xi", xi: 0.05 });
    optics.fit(X);
    expect(optics.labels.size).toBe(8);
  });

  it("predict works after fit (transductive)", () => {
    const optics = new OPTICS({ minSamples: 2, eps: 1 });
    optics.fit(X);
    const labels = optics.predict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("clusterCenters accessible after fit", () => {
    const optics = new OPTICS({ minSamples: 2, eps: 1 });
    optics.fit(X);
    const centers = optics.clusterCenters;
    expect(centers).toBeDefined();
  });

  it("throws when not fitted", () => {
    const optics = new OPTICS();
    expect(() => optics.labels).toThrow();
    expect(() => optics.reachability).toThrow();
    expect(() => optics.ordering).toThrow();
    expect(() => optics.coreDistances).toThrow();
  });

  it("throws for invalid minSamples", () => {
    expect(() => new OPTICS({ minSamples: 0 })).toThrow();
    expect(() => new OPTICS({ minSamples: -1 })).toThrow();
  });

  it("throws for invalid maxEps", () => {
    expect(() => new OPTICS({ maxEps: 0 })).toThrow();
    expect(() => new OPTICS({ maxEps: -1 })).toThrow();
  });

  it("getParams returns options", () => {
    const optics = new OPTICS({
      minSamples: 3,
      maxEps: 5,
      clusterMethod: "xi",
    });
    const params = optics.getParams();
    expect(params.minSamples).toBe(3);
    expect(params.maxEps).toBe(5);
    expect(params.clusterMethod).toBe("xi");
  });

  it("setParams works and rejects unknown params", () => {
    const optics = new OPTICS();
    expect(optics.setParams({})).toBe(optics);
    expect(() => optics.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });

  it("handles single-cluster data", () => {
    const Xsingle = tensor([
      [0, 0],
      [0.1, 0.1],
      [0.2, 0],
      [0, 0.2],
      [0.1, 0.2],
    ]);
    const optics = new OPTICS({ minSamples: 2, eps: 1 });
    optics.fit(Xsingle);
    expect(optics.labels.size).toBe(5);
  });
});
