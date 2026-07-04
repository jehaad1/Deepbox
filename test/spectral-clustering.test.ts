import { describe, expect, it } from "vitest";
import { SpectralClustering } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("SpectralClustering", () => {
  // Two well-separated clusters
  const X = tensor([
    [0, 0],
    [0.1, 0.1],
    [0.2, 0],
    [0, 0.2],
    [10, 10],
    [10.1, 10.1],
    [10.2, 10],
    [10, 10.2],
  ]);

  it("fit produces correct number of labels", () => {
    const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
    sc.fit(X);
    expect(sc.labels.size).toBe(8);
  });

  it("separates well-separated clusters", () => {
    const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
    sc.fit(X);
    const labels = sc.labels;
    // First 4 points should share a label, last 4 should share another
    const label0 = Number(labels.data[labels.offset]);
    const label4 = Number(labels.data[labels.offset + 4]);
    expect(label0).not.toBe(label4);

    // Check consistency within clusters
    for (let i = 0; i < 4; i++) {
      expect(Number(labels.data[labels.offset + i])).toBe(label0);
    }
    for (let i = 4; i < 8; i++) {
      expect(Number(labels.data[labels.offset + i])).toBe(label4);
    }
  });

  it("fitPredict returns labels", () => {
    const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
    const labels = sc.fitPredict(X);
    expect(labels.size).toBe(8);
  });

  it("nearest_neighbors affinity works", () => {
    const sc = new SpectralClustering({
      nClusters: 2,
      affinity: "nearest_neighbors",
      nNeighbors: 3,
      randomState: 42,
    });
    sc.fit(X);
    expect(sc.labels.size).toBe(8);
  });

  it("labels contain valid cluster indices", () => {
    const sc = new SpectralClustering({ nClusters: 3, randomState: 42 });
    const X3 = tensor([
      [0, 0],
      [0.1, 0],
      [5, 5],
      [5.1, 5],
      [10, 0],
      [10.1, 0],
    ]);
    sc.fit(X3);
    const labels = sc.labels;
    const uniqueLabels = new Set<number>();
    for (let i = 0; i < labels.size; i++) {
      const l = Number(labels.data[labels.offset + i]);
      expect(l).toBeGreaterThanOrEqual(0);
      expect(l).toBeLessThan(3);
      uniqueLabels.add(l);
    }
  });

  it("throws when not fitted", () => {
    const sc = new SpectralClustering({ nClusters: 2 });
    expect(() => sc.labels).toThrow();
  });

  it("predict works after fit (transductive)", () => {
    const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
    sc.fit(X);
    const labels = sc.predict(X);
    expect(labels.shape).toEqual([8]);
  });

  it("clusterCenters accessible after fit", () => {
    const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
    sc.fit(X);
    const centers = sc.clusterCenters;
    expect(centers).toBeDefined();
  });

  it("throws for invalid nClusters", () => {
    expect(() => new SpectralClustering({ nClusters: 0 })).toThrow();
    expect(() => new SpectralClustering({ nClusters: -1 })).toThrow();
  });

  it("throws for invalid gamma", () => {
    expect(() => new SpectralClustering({ gamma: 0 })).toThrow();
    expect(() => new SpectralClustering({ gamma: -1 })).toThrow();
  });

  it("throws when nSamples < nClusters", () => {
    const sc = new SpectralClustering({ nClusters: 10 });
    expect(() => sc.fit(X)).toThrow();
  });

  it("getParams returns constructor options", () => {
    const sc = new SpectralClustering({
      nClusters: 3,
      gamma: 0.5,
      randomState: 42,
    });
    const params = sc.getParams();
    expect(params.nClusters).toBe(3);
    expect(params.gamma).toBe(0.5);
    expect(params.randomState).toBe(42);
    expect(params.affinity).toBe("rbf");
  });

  it("deterministic with randomState", () => {
    const sc1 = new SpectralClustering({ nClusters: 2, randomState: 42 });
    const sc2 = new SpectralClustering({ nClusters: 2, randomState: 42 });
    sc1.fit(X);
    sc2.fit(X);
    const l1 = sc1.labels;
    const l2 = sc2.labels;
    for (let i = 0; i < l1.size; i++) {
      expect(Number(l1.data[l1.offset + i])).toBe(Number(l2.data[l2.offset + i]));
    }
  });
});
