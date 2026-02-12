import { describe, expect, it } from "vitest";
import { Ridge } from "../src/ml/linear/Ridge";
import { TSNE } from "../src/ml/manifold/TSNE";
import { RandomForestClassifier } from "../src/ml/tree/RandomForest";
import { tensor } from "../src/ndarray";
import { dot, matmul } from "../src/ndarray/linalg/basic";

describe("Advanced Coverage Fixes", () => {
  describe("TSNE", () => {
    it("should validate constructor parameters", () => {
      expect(() => new TSNE({ nComponents: -1 })).toThrow(/nComponents/);
      expect(() => new TSNE({ perplexity: -5 })).toThrow(/perplexity/);
      expect(() => new TSNE({ learningRate: 0 })).toThrow(/learningRate/);
      expect(() => new TSNE({ nIter: 0 })).toThrow(/nIter/);
      expect(() => new TSNE({ earlyExaggeration: -1 })).toThrow(/earlyExaggeration/);
      expect(() => new TSNE({ minGradNorm: -1 })).toThrow(/minGradNorm/);
      expect(() => new TSNE({ maxExactSamples: 0 })).toThrow(/maxExactSamples/);
      // @ts-expect-error Testing invalid method
      expect(() => new TSNE({ method: "invalid" })).toThrow(/method/);
    });

    it("should support approximate method", () => {
      const X = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);
      const tsne = new TSNE({
        method: "approximate",
        nIter: 10,
        perplexity: 1,
      });
      const embedding = tsne.fitTransform(X);
      expect(embedding.shape).toEqual([4, 2]);
    });
  });

  describe("RandomForestClassifier", () => {
    it("should validate constructor parameters", () => {
      expect(() => new RandomForestClassifier({ nEstimators: 0 })).toThrow(/nEstimators/);
      expect(() => new RandomForestClassifier({ maxDepth: 0 })).toThrow(/maxDepth/);
      expect(() => new RandomForestClassifier({ minSamplesSplit: 1 })).toThrow(/minSamplesSplit/);
      expect(() => new RandomForestClassifier({ minSamplesLeaf: 0 })).toThrow(/minSamplesLeaf/);
      expect(() => new RandomForestClassifier({ maxFeatures: 0 })).toThrow(/maxFeatures/);
    });

    it("should support custom maxFeatures", () => {
      const clf = new RandomForestClassifier({
        maxFeatures: 1,
        nEstimators: 2,
      });
      const X = tensor([
        [0, 0],
        [1, 1],
      ]);
      const y = tensor([0, 1], { dtype: "int32" });
      clf.fit(X, y);
      expect(clf.predict(X).shape).toEqual([2]);
    });
  });

  describe("Linear Algebra Basic", () => {
    it("matmul should validate dimensions", () => {
      const A = tensor([[1, 2]]);
      const B = tensor([1, 2]); // 1D
      expect(() => matmul(A, B)).toThrow(/matmul requires 2D tensors/);
    });

    it("matmul should validate inner dimensions", () => {
      const A = tensor([[1, 2]]);
      const B = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      expect(() => matmul(A, B)).toThrow(/mismatch/);
    });

    it("dot should validate dimensions", () => {
      const A = tensor([[1, 2]]);
      const B = tensor([1, 2]);
      expect(() => dot(A, B)).toThrow(/dot requires 1D tensors/);
    });

    it("dot should validate size mismatch", () => {
      const A = tensor([1, 2]);
      const B = tensor([1, 2, 3]);
      expect(() => dot(A, B)).toThrow(/mismatch/);
    });
  });

  describe("Ridge Solvers", () => {
    it("should support lsqr solver", () => {
      const model = new Ridge({ solver: "lsqr" });
      const X = tensor([[1], [2]]);
      const y = tensor([1, 2]);
      expect(() => model.fit(X, y)).not.toThrow();
    });

    it("should support svd solver", () => {
      const model = new Ridge({ solver: "svd" });
      const X = tensor([[1], [2], [3]]);
      const y = tensor([1, 2, 3]);
      model.fit(X, y);
      expect(model.score(X, y)).toBeGreaterThan(0.85);
    });
  });
});
