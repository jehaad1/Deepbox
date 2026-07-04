import { describe, expect, it } from "vitest";
import { AgglomerativeClustering } from "../src/ml/clustering/AgglomerativeClustering";
import { DBSCAN } from "../src/ml/clustering/DBSCAN";
import { OPTICS } from "../src/ml/clustering/OPTICS";
import { SpectralClustering } from "../src/ml/clustering/SpectralClustering";
import { tensor } from "../src/ndarray";

describe("Transductive clusterers predict() with nearest-neighbor assignment", () => {
  // Simple 2-cluster dataset
  const X = tensor([
    [0, 0],
    [0.1, 0.1],
    [0.2, 0],
    [5, 5],
    [5.1, 5.1],
    [5.2, 5],
  ]);

  describe("DBSCAN.predict()", () => {
    it("assigns new points to nearest core sample cluster", () => {
      const dbscan = new DBSCAN({ eps: 1.0, minSamples: 2 });
      dbscan.fit(X);

      const newPoints = tensor([
        [0.05, 0.05], // near cluster 0
        [5.05, 5.05], // near cluster 1
      ]);
      const pred = dbscan.predict(newPoints);
      expect(pred.shape).toEqual([2]);
      // Both should get valid cluster labels (non-noise)
      const p0 = Number(pred.data[0]);
      const p1 = Number(pred.data[1]);
      expect(p0).toBeGreaterThanOrEqual(0);
      expect(p1).toBeGreaterThanOrEqual(0);
      // They should be in different clusters
      expect(p0).not.toEqual(p1);
    });

    it("assigns noise (-1) to distant points", () => {
      const dbscan = new DBSCAN({ eps: 0.5, minSamples: 2 });
      dbscan.fit(X);

      const farPoint = tensor([[100, 100]]);
      const pred = dbscan.predict(farPoint);
      expect(Number(pred.data[0])).toBe(-1);
    });

    it("throws NotFittedError if not fitted", () => {
      const dbscan = new DBSCAN();
      expect(() => dbscan.predict(tensor([[0, 0]]))).toThrow();
    });
  });

  describe("AgglomerativeClustering.predict()", () => {
    it("assigns new points to nearest training sample cluster", () => {
      const agg = new AgglomerativeClustering({ nClusters: 2 });
      agg.fit(X);

      const newPoints = tensor([
        [0.05, 0.05],
        [5.05, 5.05],
      ]);
      const pred = agg.predict(newPoints);
      expect(pred.shape).toEqual([2]);
      const p0 = Number(pred.data[0]);
      const p1 = Number(pred.data[1]);
      expect(p0).not.toEqual(p1);
    });

    it("throws NotFittedError if not fitted", () => {
      const agg = new AgglomerativeClustering();
      expect(() => agg.predict(tensor([[0, 0]]))).toThrow();
    });

    it("clusterCenters returns computed centroids", () => {
      const agg = new AgglomerativeClustering({ nClusters: 2 });
      agg.fit(X);
      const centers = agg.clusterCenters;
      expect(centers.shape[0]).toBe(2);
      expect(centers.shape[1]).toBe(2);
    });
  });

  describe("SpectralClustering.predict()", () => {
    it("assigns new points to nearest training sample cluster", () => {
      const spec = new SpectralClustering({ nClusters: 2, randomState: 42 });
      spec.fit(X);

      const newPoints = tensor([
        [0.05, 0.05],
        [5.05, 5.05],
      ]);
      const pred = spec.predict(newPoints);
      expect(pred.shape).toEqual([2]);
      const p0 = Number(pred.data[0]);
      const p1 = Number(pred.data[1]);
      expect(p0).not.toEqual(p1);
    });

    it("throws NotFittedError if not fitted", () => {
      const spec = new SpectralClustering();
      expect(() => spec.predict(tensor([[0, 0]]))).toThrow();
    });

    it("clusterCenters returns computed centroids", () => {
      const spec = new SpectralClustering({ nClusters: 2, randomState: 42 });
      spec.fit(X);
      const centers = spec.clusterCenters;
      expect(centers.shape[0]).toBe(2);
      expect(centers.shape[1]).toBe(2);
    });
  });

  describe("OPTICS.predict()", () => {
    it("assigns new points to nearest training sample cluster", () => {
      const optics = new OPTICS({ minSamples: 2, maxEps: 3 });
      optics.fit(X);

      const newPoints = tensor([
        [0.05, 0.05],
        [5.05, 5.05],
      ]);
      const pred = optics.predict(newPoints);
      expect(pred.shape).toEqual([2]);
    });

    it("throws NotFittedError if not fitted", () => {
      const optics = new OPTICS();
      expect(() => optics.predict(tensor([[0, 0]]))).toThrow();
    });

    it("clusterCenters returns computed centroids", () => {
      const optics = new OPTICS({ minSamples: 2, maxEps: 3 });
      optics.fit(X);
      // May or may not have clusters depending on parameters
      // Just verify no crash
      const centers = optics.clusterCenters;
      expect(centers.ndim).toBe(2);
    });
  });
});
