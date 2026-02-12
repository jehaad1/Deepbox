import { describe, expect, it } from "vitest";
import { DBSCAN } from "../src/ml/clustering/DBSCAN";
import { TSNE } from "../src/ml/manifold/TSNE";
import { tensor } from "../src/ndarray";
import { toNumArr } from "./_helpers";

describe("deepbox/ml - Manifold and Clustering", () => {
  describe("TSNE", () => {
    it("fits and transforms a small dataset", () => {
      const X = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const tsne = new TSNE({
        nComponents: 2,
        nIter: 50,
        randomState: 42,
        perplexity: 2,
      });
      const embedding = tsne.fitTransform(X);
      expect(embedding.shape).toEqual([4, 2]);
      const params = tsne.getParams();
      expect(params.nComponents).toBe(2);
      const embeddingResult = tsne.embeddingResult;
      expect(embeddingResult.shape).toEqual([4, 2]);
    });

    it("supports approximate mode for larger datasets", () => {
      const data: number[][] = [];
      for (let i = 0; i < 60; i++) {
        data.push([i, i % 7, (i * 3) % 11, (i * i) % 13]);
      }
      const X = tensor(data);
      const tsne = new TSNE({
        nComponents: 2,
        nIter: 20,
        randomState: 42,
        perplexity: 5,
        method: "approximate",
        approximateNeighbors: 20,
        negativeSamples: 15,
      });
      const embedding = tsne.fitTransform(X);
      expect(embedding.shape).toEqual([60, 2]);
    });

    it("guards exact mode for large datasets", () => {
      const data: number[][] = [];
      for (let i = 0; i < 8; i++) {
        data.push([i, i + 1, i + 2]);
      }
      const X = tensor(data);
      const tsne = new TSNE({
        nComponents: 2,
        perplexity: 2,
        maxExactSamples: 5,
      });
      expect(() => tsne.fitTransform(X)).toThrow(/Exact t-SNE/);
    });

    it("throws when too few samples", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const tsne = new TSNE({
        nComponents: 2,
        nIter: 10,
        randomState: 1,
        perplexity: 2,
      });
      expect(() => tsne.fitTransform(X)).toThrow();
    });
  });

  describe("DBSCAN", () => {
    it("clusters points and marks noise", () => {
      const X = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [5, 5],
        [5, 6],
        [6, 5],
        [10, 10],
      ]);
      const dbscan = new DBSCAN({ eps: 1.6, minSamples: 2 });
      const labels = dbscan.fitPredict(X);
      const labelArray = toNumArr(labels.toArray());
      expect(labelArray).toContain(-1);
      const core = dbscan.coreIndices;
      expect(core.length).toBeGreaterThan(0);
    });

    it("supports manhattan distance", () => {
      const X = tensor([
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11],
      ]);
      const dbscan = new DBSCAN({
        eps: 1.1,
        minSamples: 2,
        metric: "manhattan",
      });
      dbscan.fit(X);
      const labels = toNumArr(dbscan.labels.toArray());
      expect(labels.filter((v) => v === -1).length).toBe(0);
    });

    it("validates eps and minSamples", () => {
      expect(() => new DBSCAN({ eps: 0 })).toThrow();
      expect(() => new DBSCAN({ minSamples: 0 })).toThrow();
    });
  });
});
