import { describe, expect, it } from "vitest";
import { AgglomerativeClustering } from "../src/ml/clustering/AgglomerativeClustering";
import { Birch } from "../src/ml/clustering/Birch";
import { GaussianMixture } from "../src/ml/clustering/GaussianMixture";
import { MiniBatchKMeans } from "../src/ml/clustering/MiniBatchKMeans";
import { OPTICS } from "../src/ml/clustering/OPTICS";
import { SpectralClustering } from "../src/ml/clustering/SpectralClustering";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 0],
  [1.1, 0.1],
  [0.9, -0.1],
  [5, 5],
  [5.1, 5.1],
  [4.9, 4.9],
  [10, 0],
  [10.1, 0.1],
  [9.9, -0.1],
]);

// ────── SpectralClustering ──────
describe("SpectralClustering", () => {
  it("fitPredict", () => {
    const sc = new SpectralClustering({ nClusters: 3, randomState: 42 });
    const labels = sc.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict after fit works (transductive)", () => {
    const sc = new SpectralClustering({ nClusters: 3, randomState: 42 });
    sc.fit(X);
    const labels = sc.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams / setParams", () => {
    const sc = new SpectralClustering({ nClusters: 3 });
    expect(sc.getParams().nClusters).toBe(3);
    sc.setParams({ nClusters: 5 });
    expect(sc.getParams().nClusters).toBe(5);
  });
});

// ────── GaussianMixture ──────
describe("GaussianMixture", () => {
  it("fits and predicts", () => {
    const gm = new GaussianMixture({
      nComponents: 3,
      randomState: 42,
      maxIter: 10,
    });
    gm.fit(X);
    const labels = gm.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict_proba", () => {
    const gm = new GaussianMixture({
      nComponents: 3,
      randomState: 42,
      maxIter: 10,
    });
    gm.fit(X);
    const proba = gm.predictProba(X);
    expect(proba.shape[0]).toBe(9);
    expect(proba.shape[1]).toBe(3);
  });

  it("throws for not fitted", () => {
    const gm = new GaussianMixture({ nComponents: 3 });
    expect(() => gm.predict(X)).toThrow(/fitted/i);
  });

  it("getParams / setParams", () => {
    const gm = new GaussianMixture({ nComponents: 3 });
    expect(gm.getParams().nComponents).toBe(3);
  });
});

// ────── MiniBatchKMeans ──────
describe("MiniBatchKMeans", () => {
  it("fits and predicts", () => {
    const km = new MiniBatchKMeans({
      nClusters: 3,
      randomState: 42,
      batchSize: 3,
    });
    km.fit(X);
    const labels = km.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("throws for not fitted", () => {
    const km = new MiniBatchKMeans({ nClusters: 3 });
    expect(() => km.predict(X)).toThrow(/fitted/i);
  });

  it("getParams / setParams", () => {
    const km = new MiniBatchKMeans({ nClusters: 3 });
    expect(km.getParams().nClusters).toBe(3);
  });
});

// ────── Birch ──────
describe("Birch", () => {
  it("fits and predicts", () => {
    const b = new Birch({ nClusters: 3 });
    b.fit(X);
    const labels = b.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("throws for not fitted", () => {
    const b = new Birch({ nClusters: 3 });
    expect(() => b.predict(X)).toThrow(/fitted/i);
  });

  it("getParams / setParams", () => {
    const b = new Birch({ nClusters: 3 });
    expect(b.getParams().nClusters).toBe(3);
  });

  it("custom threshold and branching_factor", () => {
    const b = new Birch({ nClusters: 3, threshold: 0.8, branchingFactor: 10 });
    b.fit(X);
    expect(b.predict(X).shape).toEqual([9]);
  });
});

// ────── AgglomerativeClustering ──────
describe("AgglomerativeClustering", () => {
  it("fitPredict", () => {
    const ac = new AgglomerativeClustering({ nClusters: 3 });
    const labels = ac.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("linkage=complete", () => {
    const ac = new AgglomerativeClustering({
      nClusters: 3,
      linkage: "complete",
    });
    const labels = ac.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("linkage=average", () => {
    const ac = new AgglomerativeClustering({
      nClusters: 3,
      linkage: "average",
    });
    const labels = ac.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict after fit works (transductive)", () => {
    const ac = new AgglomerativeClustering({ nClusters: 3 });
    ac.fit(X);
    const labels = ac.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams / setParams", () => {
    const ac = new AgglomerativeClustering({ nClusters: 3 });
    expect(ac.getParams().nClusters).toBe(3);
  });
});

// ────── OPTICS ──────
describe("OPTICS", () => {
  it("fitPredict", () => {
    const op = new OPTICS({ minSamples: 2 });
    const labels = op.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("with maxEps", () => {
    const op = new OPTICS({ minSamples: 2, maxEps: 3 });
    const labels = op.fitPredict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("predict after fit works (transductive)", () => {
    const op = new OPTICS({ minSamples: 2 });
    op.fit(X);
    const labels = op.predict(X);
    expect(labels.shape).toEqual([9]);
  });

  it("getParams / setParams", () => {
    const op = new OPTICS({ minSamples: 2 });
    expect(op.getParams().minSamples).toBe(2);
  });
});
