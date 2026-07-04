import { describe, expect, it } from "vitest";
import {
  AffinityPropagation,
  AgglomerativeClustering,
  Birch,
  DBSCAN,
  GaussianMixture,
  KMeans,
  MeanShift,
  MiniBatchKMeans,
  OPTICS,
  SpectralClustering,
} from "../src/ml";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 2],
  [1.5, 1.8],
  [1.2, 2.1],
  [5, 8],
  [6, 7],
  [5.5, 8.2],
  [9, 11],
  [8, 10],
  [9.5, 10.5],
]);

// ---- GaussianMixture ----
describe("GaussianMixture branches", () => {
  it("fit and predict", () => {
    const gmm = new GaussianMixture({ nComponents: 3, randomState: 42 });
    gmm.fit(X);
    const labels = gmm.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    const labels = gmm.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("labels getter", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    gmm.fit(X);
    expect(gmm.labels.size).toBe(9);
  });

  it("clusterCenters getter", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    gmm.fit(X);
    const centers = gmm.clusterCenters;
    expect(centers.shape[0]).toBe(2);
  });

  it("labels throws before fit", () => {
    const gmm = new GaussianMixture({ nComponents: 2 });
    expect(() => gmm.labels).toThrow();
  });

  it("clusterCenters throws before fit", () => {
    const gmm = new GaussianMixture({ nComponents: 2 });
    expect(() => gmm.clusterCenters).toThrow();
  });

  it("getParams", () => {
    const gmm = new GaussianMixture({ nComponents: 3, maxIter: 50, tol: 0.01 });
    const p = gmm.getParams();
    expect(p.nComponents).toBe(3);
    expect(p.maxIter).toBe(50);
    expect(p.tol).toBe(0.01);
  });

  it("setParams valid", () => {
    const gmm = new GaussianMixture({ nComponents: 2 });
    gmm.setParams({ nComponents: 5 });
    expect(gmm.getParams().nComponents).toBe(5);
    gmm.setParams({ maxIter: 200 });
    expect(gmm.getParams().maxIter).toBe(200);
    gmm.setParams({ tol: 0.05 });
    expect(gmm.getParams().tol).toBe(0.05);
    gmm.setParams({ nInit: 3 });
    expect(gmm.getParams().nInit).toBe(3);
    gmm.setParams({ regCovar: 1e-5 });
    expect(gmm.getParams().regCovar).toBe(1e-5);
    gmm.setParams({ randomState: 123 });
    expect(gmm.getParams().randomState).toBe(123);
  });

  it("setParams invalid nComponents", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ nComponents: 0 })).toThrow();
    expect(() => gmm.setParams({ nComponents: -1 })).toThrow();
    expect(() => gmm.setParams({ nComponents: 1.5 })).toThrow();
  });

  it("setParams invalid maxIter", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ maxIter: 0 })).toThrow();
  });

  it("setParams invalid tol", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ tol: -1 })).toThrow();
  });

  it("setParams invalid nInit", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ nInit: 0 })).toThrow();
  });

  it("setParams invalid regCovar", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ regCovar: -1 })).toThrow();
  });

  it("setParams invalid randomState", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ randomState: "abc" as unknown })).toThrow();
  });

  it("setParams unknown key", () => {
    const gmm = new GaussianMixture();
    expect(() => gmm.setParams({ badKey: 1 })).toThrow();
  });

  it("constructor invalid nComponents", () => {
    expect(() => new GaussianMixture({ nComponents: 0 })).toThrow();
    expect(() => new GaussianMixture({ nComponents: -1 })).toThrow();
  });

  it("nInit > 1 runs multiple initializations", () => {
    const gmm = new GaussianMixture({
      nComponents: 2,
      nInit: 3,
      randomState: 42,
    });
    gmm.fit(X);
    expect(gmm.labels.size).toBe(9);
  });

  it("predict before fit throws", () => {
    const gmm = new GaussianMixture({ nComponents: 2 });
    expect(() => gmm.predict(X)).toThrow();
  });
});

// ---- KMeans ----
describe("KMeans branches", () => {
  it("fit with random init", () => {
    const km = new KMeans({ nClusters: 3, init: "random", randomState: 42 });
    km.fit(X);
    expect(km.labels.size).toBe(9);
  });

  it("fit with kmeans++ init", () => {
    const km = new KMeans({ nClusters: 3, init: "kmeans++", randomState: 42 });
    km.fit(X);
    expect(km.labels.size).toBe(9);
  });

  it("predict after fit", () => {
    const km = new KMeans({ nClusters: 3, randomState: 42 });
    km.fit(X);
    const labels = km.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const km = new KMeans({ nClusters: 2, randomState: 42 });
    const labels = km.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("clusterCenters", () => {
    const km = new KMeans({ nClusters: 3, randomState: 42 });
    km.fit(X);
    expect(km.clusterCenters.shape[0]).toBe(3);
  });

  it("setParams all valid", () => {
    const km = new KMeans({ nClusters: 2 });
    km.setParams({ nClusters: 5 });
    expect(km.getParams().nClusters).toBe(5);
    km.setParams({ maxIter: 200 });
    km.setParams({ tol: 0.05 });
    km.setParams({ init: "random" });
    km.setParams({ nInit: 3 });
    km.setParams({ randomState: 123 });
    km.setParams({ algorithm: "lloyd" });
    expect(km.getParams().algorithm).toBe("lloyd");
  });

  it("setParams invalid values", () => {
    const km = new KMeans({ nClusters: 2 });
    expect(() => km.setParams({ nClusters: 0 })).toThrow();
    expect(() => km.setParams({ maxIter: 0 })).toThrow();
    expect(() => km.setParams({ tol: -1 })).toThrow();
    expect(() => km.setParams({ init: "bad" })).toThrow();
    expect(() => km.setParams({ nInit: 0 })).toThrow();
    expect(() => km.setParams({ randomState: "abc" as unknown })).toThrow();
    expect(() => km.setParams({ algorithm: "bad" })).toThrow();
    expect(() => km.setParams({ unknown: 1 })).toThrow();
  });

  it("nInit > 1 runs multiple initializations", () => {
    const km = new KMeans({ nClusters: 3, nInit: 3, randomState: 42 });
    km.fit(X);
    expect(km.labels.size).toBe(9);
  });

  it("algorithm elkan", () => {
    const km = new KMeans({
      nClusters: 3,
      algorithm: "elkan",
      randomState: 42,
    });
    km.fit(X);
    expect(km.labels.size).toBe(9);
  });
});

// ---- MiniBatchKMeans ----
describe("MiniBatchKMeans branches", () => {
  it("fit and predict", () => {
    const mbk = new MiniBatchKMeans({
      nClusters: 3,
      batchSize: 5,
      randomState: 42,
    });
    mbk.fit(X);
    const labels = mbk.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const mbk = new MiniBatchKMeans({ nClusters: 2, randomState: 42 });
    const labels = mbk.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams/setParams", () => {
    const mbk = new MiniBatchKMeans({ nClusters: 2 });
    expect(mbk.getParams().nClusters).toBe(2);
    mbk.setParams({ nClusters: 5 });
    expect(mbk.getParams().nClusters).toBe(5);
  });
});

// ---- MeanShift ----
describe("MeanShift branches", () => {
  it("fit", () => {
    const ms = new MeanShift();
    ms.fit(X);
    expect(ms.labels.size).toBe(9);
  });

  it("fitPredict", () => {
    const ms = new MeanShift();
    const labels = ms.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict after fit", () => {
    const ms = new MeanShift();
    ms.fit(X);
    const labels = ms.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("clusterCenters", () => {
    const ms = new MeanShift();
    ms.fit(X);
    expect(ms.clusterCenters.shape[0]).toBeGreaterThan(0);
  });

  it("getParams/setParams", () => {
    const ms = new MeanShift({ bandwidth: 2.0 });
    expect(ms.getParams().bandwidth).toBe(2.0);
    ms.setParams({ bandwidth: 3.0 });
    expect(ms.getParams().bandwidth).toBe(3.0);
  });

  it("setParams with auto bandwidth", () => {
    const ms = new MeanShift();
    ms.setParams({ bandwidth: "auto" });
    expect(ms.getParams().bandwidth).toBe("auto");
  });
});

// ---- AffinityPropagation ----
describe("AffinityPropagation branches", () => {
  it("fit", () => {
    const ap = new AffinityPropagation({ damping: 0.7 });
    ap.fit(X);
    expect(ap.labels.size).toBe(9);
  });

  it("fitPredict", () => {
    const ap = new AffinityPropagation({ damping: 0.7 });
    const labels = ap.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict after fit", () => {
    const ap = new AffinityPropagation({ damping: 0.7 });
    ap.fit(X);
    const labels = ap.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams/setParams", () => {
    const ap = new AffinityPropagation({ damping: 0.7 });
    expect(ap.getParams().damping).toBe(0.7);
    ap.setParams({ damping: 0.9 });
    expect(ap.getParams().damping).toBe(0.9);
  });
});

// ---- AgglomerativeClustering ----
describe("AgglomerativeClustering branches", () => {
  it("different linkage methods", () => {
    for (const linkage of ["single", "complete", "average", "ward"] as const) {
      const ac = new AgglomerativeClustering({ nClusters: 3, linkage });
      ac.fit(X);
      expect(ac.labels.size).toBe(9);
    }
  });

  it("getParams/setParams", () => {
    const ac = new AgglomerativeClustering({ nClusters: 3 });
    expect(ac.getParams().nClusters).toBe(3);
    ac.setParams({ nClusters: 5 });
    expect(ac.getParams().nClusters).toBe(5);
  });

  it("clusterCenters", () => {
    const ac = new AgglomerativeClustering({ nClusters: 3 });
    ac.fit(X);
    const centers = ac.clusterCenters;
    expect(centers.shape[0]).toBe(3);
  });
});

// ---- DBSCAN ----
describe("DBSCAN branches", () => {
  it("fit with different minSamples", () => {
    const db = new DBSCAN({ eps: 2, minSamples: 2 });
    db.fit(X);
    expect(db.labels.size).toBe(9);
  });

  it("predict after fit", () => {
    const db = new DBSCAN({ eps: 2, minSamples: 2 });
    db.fit(X);
    const labels = db.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams/setParams", () => {
    const db = new DBSCAN({ eps: 1.0 });
    expect(db.getParams().eps).toBe(1.0);
    db.setParams({ eps: 2.0 });
    expect(db.getParams().eps).toBe(2.0);
  });
});

// ---- OPTICS ----
describe("OPTICS branches", () => {
  it("predict after fit", () => {
    const op = new OPTICS({ minSamples: 2, maxEps: 10 });
    op.fit(X);
    const labels = op.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("clusterCenters", () => {
    const op = new OPTICS({ minSamples: 2 });
    op.fit(X);
    const centers = op.clusterCenters;
    expect(centers).toBeDefined();
  });

  it("getParams/setParams", () => {
    const op = new OPTICS({ minSamples: 3 });
    expect(op.getParams().minSamples).toBe(3);
    op.setParams({ minSamples: 5 });
    expect(op.getParams().minSamples).toBe(5);
  });

  it("xi cluster method", () => {
    const op = new OPTICS({ minSamples: 2, clusterMethod: "xi", xi: 0.1 });
    op.fit(X);
    expect(op.labels.size).toBe(9);
  });
});

// ---- Birch ----
describe("Birch branches", () => {
  it("fit and predict", () => {
    const birch = new Birch({ nClusters: 3, threshold: 1.5 });
    birch.fit(X);
    const labels = birch.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("fitPredict", () => {
    const birch = new Birch({ nClusters: 2 });
    const labels = birch.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams/setParams", () => {
    const birch = new Birch({ nClusters: 3 });
    expect(birch.getParams().nClusters).toBe(3);
    birch.setParams({ nClusters: 5 });
    expect(birch.getParams().nClusters).toBe(5);
  });

  it("clusterCenters", () => {
    const birch = new Birch({ nClusters: 3 });
    birch.fit(X);
    expect(birch.clusterCenters.shape[0]).toBe(3);
  });
});

// ---- SpectralClustering ----
describe("SpectralClustering branches", () => {
  it("predict after fit", () => {
    const sc = new SpectralClustering({ nClusters: 3, randomState: 42 });
    sc.fit(X);
    const labels = sc.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("clusterCenters", () => {
    const sc = new SpectralClustering({ nClusters: 3, randomState: 42 });
    sc.fit(X);
    const centers = sc.clusterCenters;
    expect(centers.shape[0]).toBe(3);
  });

  it("getParams/setParams", () => {
    const sc = new SpectralClustering({ nClusters: 3 });
    expect(sc.getParams().nClusters).toBe(3);
    sc.setParams({ nClusters: 5 });
    expect(sc.getParams().nClusters).toBe(5);
  });
});
