import { describe, expect, it } from "vitest";
import { Isomap, MDS, SpectralEmbedding } from "../src/ml";
import { tensor } from "../src/ndarray";

const X = tensor([
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [5, 5, 5],
  [6, 5, 5],
]);

describe("Isomap", () => {
  it("reduces dimensionality via geodesic distances", () => {
    const iso = new Isomap({ nComponents: 2, nNeighbors: 3 });
    const emb = iso.fitTransform(X);
    expect(emb.shape).toEqual([6, 2]);
  });

  it("exposes embedding after fit", () => {
    const iso = new Isomap({ nComponents: 2, nNeighbors: 3 });
    iso.fit(X);
    expect(iso.embedding.shape).toEqual([6, 2]);
  });

  it("validates parameters", () => {
    expect(() => new Isomap({ nComponents: 0 })).toThrow(/nComponents/);
    expect(() => new Isomap({ nNeighbors: 0 })).toThrow(/nNeighbors/);
  });

  it("throws if nNeighbors >= n_samples", () => {
    const iso = new Isomap({ nNeighbors: 10 });
    expect(() => iso.fit(X)).toThrow(/nNeighbors/);
  });

  it("throws NotFittedError before fitting", () => {
    const iso = new Isomap();
    expect(() => iso.embedding).toThrow(/fitted/i);
  });

  it("getParams returns constructor params", () => {
    const iso = new Isomap({ nComponents: 3, nNeighbors: 4 });
    expect(iso.getParams()).toEqual({ nComponents: 3, nNeighbors: 4 });
  });
});

describe("MDS", () => {
  it("reduces dimensionality preserving pairwise distances", () => {
    const mds = new MDS({ nComponents: 2 });
    const emb = mds.fitTransform(X);
    expect(emb.shape).toEqual([6, 2]);
  });

  it("exposes embedding after fit", () => {
    const mds = new MDS({ nComponents: 2 });
    mds.fit(X);
    expect(mds.embedding.shape).toEqual([6, 2]);
  });

  it("validates parameters", () => {
    expect(() => new MDS({ nComponents: 0 })).toThrow(/nComponents/);
  });

  it("throws NotFittedError before fitting", () => {
    const mds = new MDS();
    expect(() => mds.embedding).toThrow(/fitted/i);
  });
});

describe("SpectralEmbedding", () => {
  it("reduces dimensionality via graph Laplacian", () => {
    const se = new SpectralEmbedding({ nComponents: 2, gamma: 1 });
    const emb = se.fitTransform(X);
    expect(emb.shape).toEqual([6, 2]);
  });

  it("exposes embedding after fit", () => {
    const se = new SpectralEmbedding({ nComponents: 2 });
    se.fit(X);
    expect(se.embedding.shape).toEqual([6, 2]);
  });

  it("validates parameters", () => {
    expect(() => new SpectralEmbedding({ nComponents: 0 })).toThrow(/nComponents/);
    expect(() => new SpectralEmbedding({ gamma: -1 })).toThrow(/gamma/);
  });

  it("throws NotFittedError before fitting", () => {
    const se = new SpectralEmbedding();
    expect(() => se.embedding).toThrow(/fitted/i);
  });

  it("getParams returns constructor params", () => {
    const se = new SpectralEmbedding({ nComponents: 3, gamma: 2 });
    expect(se.getParams()).toEqual({ nComponents: 3, gamma: 2 });
  });
});
