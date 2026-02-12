import { describe, expect, it } from "vitest";
import {
  adjustedMutualInfoScore,
  adjustedRandScore,
  calinskiHarabaszScore,
  completenessScore,
  daviesBouldinScore,
  fowlkesMallowsScore,
  homogeneityScore,
  normalizedMutualInfoScore,
  silhouetteSamples,
  silhouetteScore,
  vMeasureScore,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("Clustering Metrics - Comprehensive Tests", () => {
  // Standard test data
  const X = tensor([
    [1, 2],
    [1, 4],
    [1, 0],
    [4, 2],
    [4, 4],
    [4, 0],
  ]);
  const labels = tensor([0, 0, 0, 1, 1, 1]);
  const labelsTrue = tensor([0, 0, 0, 1, 1, 1]);
  const labelsPred = tensor([0, 0, 1, 1, 2, 2]);

  describe("silhouetteScore", () => {
    it("should calculate silhouette score", () => {
      const score = silhouetteScore(X, labels);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return high score for well-separated clusters", () => {
      const wellSeparatedX = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
        [11, 10],
      ]);
      const wellSeparatedLabels = tensor([0, 0, 0, 1, 1, 1]);
      const score = silhouetteScore(wellSeparatedX, wellSeparatedLabels);
      expect(score).toBeGreaterThan(0.5);
    });

    it("should return low score for poorly separated clusters", () => {
      const poorlyX = tensor([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0.5, 0.5],
        [0.5, 0.6],
      ]);
      const poorlyLabels = tensor([0, 0, 0, 1, 1, 1]);
      const score = silhouetteScore(poorlyX, poorlyLabels);
      expect(score).toBeLessThan(0.5);
    });

    it("should reject single-cluster input", () => {
      const singleCluster = tensor([0, 0, 0, 0]);
      const subsetX = tensor([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
      ]);
      expect(() => silhouetteScore(subsetX, singleCluster)).toThrow(
        /2 <= n_clusters <= n_samples - 1/
      );
    });

    it("should handle perfect clustering", () => {
      const perfectX = tensor([
        [0, 0],
        [0, 0],
        [10, 10],
        [10, 10],
      ]);
      const perfectLabels = tensor([0, 0, 1, 1]);
      const score = silhouetteScore(perfectX, perfectLabels);
      expect(score).toBeGreaterThan(0);
    });

    it("should handle 3D data", () => {
      const X3D = tensor([
        [1, 2, 3],
        [1, 4, 3],
        [1, 0, 3],
        [4, 2, 6],
        [4, 4, 6],
        [4, 0, 6],
      ]);
      const score = silhouetteScore(X3D, labels);
      expect(typeof score).toBe("number");
    });
  });

  describe("silhouetteSamples", () => {
    it("should calculate silhouette for each sample", () => {
      const samples = silhouetteSamples(X, labels);
      expect(samples.size).toBe(X.shape[0]);
    });

    it("should return values in range [-1, 1]", () => {
      const samples = silhouetteSamples(X, labels);
      for (let i = 0; i < samples.size; i++) {
        const val = Number(samples.data[samples.offset + i]);
        expect(val).toBeGreaterThanOrEqual(-1);
        expect(val).toBeLessThanOrEqual(1);
      }
    });

    it("should have average equal to silhouette score", () => {
      const samples = silhouetteSamples(X, labels);
      const score = silhouetteScore(X, labels);

      let sum = 0;
      for (let i = 0; i < samples.size; i++) {
        sum += Number(samples.data[samples.offset + i]);
      }
      const avg = sum / samples.size;

      expect(avg).toBeCloseTo(score);
    });

    it("should reject single-cluster input", () => {
      const singleCluster = tensor([0, 0, 0, 0]);
      const subsetX = tensor([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
      ]);
      expect(() => silhouetteSamples(subsetX, singleCluster)).toThrow(
        /2 <= n_clusters <= n_samples - 1/
      );
    });
  });

  describe("daviesBouldinScore", () => {
    it("should calculate Davies-Bouldin score", () => {
      const score = daviesBouldinScore(X, labels);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it("should return lower score for better clustering", () => {
      const wellSeparatedX = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
        [11, 10],
      ]);
      const wellSeparatedLabels = tensor([0, 0, 0, 1, 1, 1]);

      const poorlyX = tensor([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0.5, 0.5],
        [0.5, 0.6],
      ]);
      const poorlyLabels = tensor([0, 0, 0, 1, 1, 1]);

      const goodScore = daviesBouldinScore(wellSeparatedX, wellSeparatedLabels);
      const poorScore = daviesBouldinScore(poorlyX, poorlyLabels);

      expect(goodScore).toBeLessThan(poorScore);
    });

    it("should handle single cluster", () => {
      const singleCluster = tensor([0, 0, 0, 0]);
      const subsetX = tensor([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
      ]);
      const score = daviesBouldinScore(subsetX, singleCluster);
      expect(typeof score).toBe("number");
    });

    it("should handle multiple clusters", () => {
      const multiLabels = tensor([0, 0, 1, 1, 2, 2]);
      const score = daviesBouldinScore(X, multiLabels);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
    });
  });

  describe("calinskiHarabaszScore", () => {
    it("should calculate Calinski-Harabasz score", () => {
      const score = calinskiHarabaszScore(X, labels);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it("should return higher score for better clustering", () => {
      const wellSeparatedX = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
        [11, 10],
      ]);
      const wellSeparatedLabels = tensor([0, 0, 0, 1, 1, 1]);

      const poorlyX = tensor([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0.5, 0.5],
        [0.5, 0.6],
      ]);
      const poorlyLabels = tensor([0, 0, 0, 1, 1, 1]);

      const goodScore = calinskiHarabaszScore(wellSeparatedX, wellSeparatedLabels);
      const poorScore = calinskiHarabaszScore(poorlyX, poorlyLabels);

      expect(goodScore).toBeGreaterThan(poorScore);
    });

    it("should handle single cluster", () => {
      const singleCluster = tensor([0, 0, 0, 0]);
      const subsetX = tensor([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
      ]);
      const score = calinskiHarabaszScore(subsetX, singleCluster);
      expect(score).toBe(0);
    });

    it("should handle multiple clusters", () => {
      const multiLabels = tensor([0, 0, 1, 1, 2, 2]);
      const score = calinskiHarabaszScore(X, multiLabels);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it("should handle 3D data", () => {
      const X3D = tensor([
        [1, 2, 3],
        [1, 4, 3],
        [1, 0, 3],
        [4, 2, 6],
        [4, 4, 6],
        [4, 0, 6],
      ]);
      const score = calinskiHarabaszScore(X3D, labels);
      expect(typeof score).toBe("number");
    });
  });

  describe("adjustedRandScore", () => {
    it("should calculate adjusted Rand index", () => {
      const score = adjustedRandScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for identical clusterings", () => {
      const score = adjustedRandScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should return value close to 0 for random clusterings", () => {
      const random1 = tensor([0, 1, 0, 1, 0, 1]);
      const random2 = tensor([0, 0, 1, 1, 0, 1]);
      const score = adjustedRandScore(random1, random2);
      expect(Math.abs(score)).toBeLessThan(0.5);
    });

    it("should handle single element", () => {
      const single = tensor([0]);
      const score = adjustedRandScore(single, single);
      expect(score).toBe(1);
    });

    it("should handle all same cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = adjustedRandScore(same, same);
      expect(score).toBe(1);
    });

    it("should handle completely different clusterings", () => {
      const labels1 = tensor([0, 0, 0, 1, 1, 1]);
      const labels2 = tensor([0, 1, 2, 3, 4, 5]);
      const score = adjustedRandScore(labels1, labels2);
      expect(score).toBeLessThan(0.5);
    });

    it("should be symmetric", () => {
      const score1 = adjustedRandScore(labelsTrue, labelsPred);
      const score2 = adjustedRandScore(labelsPred, labelsTrue);
      expect(score1).toBeCloseTo(score2);
    });
  });

  describe("adjustedMutualInfoScore", () => {
    it("should calculate adjusted mutual information", () => {
      const score = adjustedMutualInfoScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
    });

    it("should return high score for similar clusterings", () => {
      const score = adjustedMutualInfoScore(labelsTrue, labelsTrue);
      expect(score).toBeGreaterThan(0.5);
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = adjustedMutualInfoScore(same, same);
      expect(typeof score).toBe("number");
    });

    it("should be symmetric", () => {
      const score1 = adjustedMutualInfoScore(labelsTrue, labelsPred);
      const score2 = adjustedMutualInfoScore(labelsPred, labelsTrue);
      expect(score1).toBeCloseTo(score2);
    });
  });

  describe("normalizedMutualInfoScore", () => {
    it("should calculate normalized mutual information", () => {
      const score = normalizedMutualInfoScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for identical clusterings", () => {
      const score = normalizedMutualInfoScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should return low score for random clusterings", () => {
      const random1 = tensor([0, 1, 0, 1, 0, 1]);
      const random2 = tensor([0, 0, 1, 1, 0, 1]);
      const score = normalizedMutualInfoScore(random1, random2);
      expect(score).toBeLessThan(0.8);
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = normalizedMutualInfoScore(same, same);
      expect(typeof score).toBe("number");
    });

    it("should be symmetric", () => {
      const score1 = normalizedMutualInfoScore(labelsTrue, labelsPred);
      const score2 = normalizedMutualInfoScore(labelsPred, labelsTrue);
      expect(score1).toBeCloseTo(score2);
    });

    it("should handle multiple clusters", () => {
      const labels1 = tensor([0, 0, 1, 1, 2, 2]);
      const labels2 = tensor([0, 1, 1, 2, 2, 0]);
      const score = normalizedMutualInfoScore(labels1, labels2);
      expect(typeof score).toBe("number");
    });
  });

  describe("fowlkesMallowsScore", () => {
    it("should calculate Fowlkes-Mallows score", () => {
      const score = fowlkesMallowsScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for identical clusterings", () => {
      const score = fowlkesMallowsScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should return low score for different clusterings", () => {
      const labels1 = tensor([0, 0, 0, 1, 1, 1]);
      const labels2 = tensor([0, 0, 1, 1, 2, 2]);
      const score = fowlkesMallowsScore(labels1, labels2);
      expect(score).toBeLessThan(0.8);
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = fowlkesMallowsScore(same, same);
      expect(typeof score).toBe("number");
    });

    it("should be symmetric", () => {
      const score1 = fowlkesMallowsScore(labelsTrue, labelsPred);
      const score2 = fowlkesMallowsScore(labelsPred, labelsTrue);
      expect(score1).toBeCloseTo(score2);
    });

    it("should handle small datasets", () => {
      const small1 = tensor([0, 1]);
      const small2 = tensor([0, 1]);
      const score = fowlkesMallowsScore(small1, small2);
      expect(typeof score).toBe("number");
    });
  });

  describe("homogeneityScore", () => {
    it("should calculate homogeneity score", () => {
      const score = homogeneityScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for perfect homogeneity", () => {
      const score = homogeneityScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should return high score when clusters contain single class", () => {
      const trueLabels = tensor([0, 0, 0, 1, 1, 1]);
      const predLabels = tensor([0, 0, 1, 2, 2, 2]);
      const score = homogeneityScore(trueLabels, predLabels);
      expect(score).toBeGreaterThan(0.5);
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = homogeneityScore(same, same);
      expect(score).toBeCloseTo(1.0);
    });

    it("should handle multiple clusters", () => {
      const labels1 = tensor([0, 0, 1, 1, 2, 2]);
      const labels2 = tensor([0, 1, 1, 2, 2, 0]);
      const score = homogeneityScore(labels1, labels2);
      expect(typeof score).toBe("number");
    });
  });

  describe("completenessScore", () => {
    it("should calculate completeness score", () => {
      const score = completenessScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for perfect completeness", () => {
      const score = completenessScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should return high score when all class members in same cluster", () => {
      const trueLabels = tensor([0, 0, 0, 1, 1, 1]);
      const predLabels = tensor([0, 0, 0, 0, 1, 1]);
      const score = completenessScore(trueLabels, predLabels);
      expect(score).toBeGreaterThanOrEqual(0.5);
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = completenessScore(same, same);
      expect(score).toBeCloseTo(1.0);
    });

    it("should handle multiple clusters", () => {
      const labels1 = tensor([0, 0, 1, 1, 2, 2]);
      const labels2 = tensor([0, 1, 1, 2, 2, 0]);
      const score = completenessScore(labels1, labels2);
      expect(typeof score).toBe("number");
    });
  });

  describe("vMeasureScore", () => {
    it("should calculate V-measure score", () => {
      const score = vMeasureScore(labelsTrue, labelsPred);
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("should return 1 for identical clusterings", () => {
      const score = vMeasureScore(labelsTrue, labelsTrue);
      expect(score).toBeCloseTo(1.0);
    });

    it("should be harmonic mean of homogeneity and completeness", () => {
      const h = homogeneityScore(labelsTrue, labelsPred);
      const c = completenessScore(labelsTrue, labelsPred);
      const v = vMeasureScore(labelsTrue, labelsPred);

      if (h + c > 0) {
        const expected = (2 * h * c) / (h + c);
        expect(v).toBeCloseTo(expected);
      }
    });

    it("should handle beta parameter", () => {
      const v1 = vMeasureScore(labelsTrue, labelsPred, 1.0);
      const v2 = vMeasureScore(labelsTrue, labelsPred, 2.0);

      expect(typeof v1).toBe("number");
      expect(typeof v2).toBe("number");
    });

    it("should handle single cluster", () => {
      const same = tensor([0, 0, 0, 0]);
      const score = vMeasureScore(same, same);
      expect(typeof score).toBe("number");
    });

    it("should be symmetric when beta=1", () => {
      const score1 = vMeasureScore(labelsTrue, labelsPred, 1.0);
      const score2 = vMeasureScore(labelsPred, labelsTrue, 1.0);
      expect(score1).toBeCloseTo(score2);
    });
  });

  describe("Edge Cases and Error Handling", () => {
    it("should handle empty inputs for clustering metrics", () => {
      const emptyX = tensor([]).reshape([0, 2]);
      const emptyLabels = tensor([]);

      expect(() => silhouetteScore(emptyX, emptyLabels)).toThrow(/at least 2 samples/i);
      expect(daviesBouldinScore(emptyX, emptyLabels)).toBe(0);
      expect(calinskiHarabaszScore(emptyX, emptyLabels)).toBe(0);
    });

    it("should handle size mismatch", () => {
      const short = tensor([0, 1]);
      const long = tensor([0, 1, 2]);

      expect(() => adjustedRandScore(short, long)).toThrow();
    });

    it("should reject mismatched labels length for clustering scores", () => {
      const X = tensor([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);
      const labels = tensor([0, 1]);
      expect(() => silhouetteScore(X, labels)).toThrow();
      expect(() => daviesBouldinScore(X, labels)).toThrow();
      expect(() => calinskiHarabaszScore(X, labels)).toThrow();
    });

    it("should reject string tensors for clustering metrics", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const labels = tensor(["a", "b"]);
      expect(() => silhouetteScore(X, labels)).toThrow(/string/);
      expect(() => daviesBouldinScore(X, labels)).toThrow(/string/);
      expect(() => calinskiHarabaszScore(X, labels)).toThrow(/string/);
    });

    it("should handle single element", () => {
      const single = tensor([0]);
      const singleX = tensor([[1, 2]]);

      expect(adjustedRandScore(single, single)).toBe(1);
      expect(() => silhouetteScore(singleX, single)).toThrow(/at least 2 samples/i);
    });

    it("should handle all same labels", () => {
      const same = tensor([0, 0, 0, 0, 0, 0]);

      expect(adjustedRandScore(same, same)).toBe(1);
      expect(homogeneityScore(same, same)).toBeCloseTo(1.0);
      expect(completenessScore(same, same)).toBeCloseTo(1.0);
    });

    it("should handle all different labels", () => {
      const different = tensor([0, 1, 2, 3, 4, 5]);
      const differentX = tensor([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
      ]);

      expect(() => silhouetteScore(differentX, different)).toThrow(
        /2 <= n_clusters <= n_samples - 1/
      );
      expect(typeof daviesBouldinScore(differentX, different)).toBe("number");
    });

    it("should handle large number of clusters", () => {
      const size = 100;
      const manyLabels = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 10)
      );
      const manyX = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => [i % 10, Math.floor(i / 10)])
      );

      expect(typeof silhouetteScore(manyX, manyLabels)).toBe("number");
      expect(typeof daviesBouldinScore(manyX, manyLabels)).toBe("number");
      expect(typeof calinskiHarabaszScore(manyX, manyLabels)).toBe("number");
    });

    it("should handle high-dimensional data", () => {
      const highDimX = tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 4, 7],
        [10, 11, 12, 13, 14],
        [10, 11, 12, 13, 15],
        [10, 11, 12, 13, 16],
      ]);
      const highDimLabels = tensor([0, 0, 0, 1, 1, 1]);

      expect(typeof silhouetteScore(highDimX, highDimLabels)).toBe("number");
      expect(typeof daviesBouldinScore(highDimX, highDimLabels)).toBe("number");
      expect(typeof calinskiHarabaszScore(highDimX, highDimLabels)).toBe("number");
    });

    it("rejects unsupported silhouette distance metrics", () => {
      // @ts-expect-error - deliberately passing unsupported metric for runtime validation
      expect(() => silhouetteScore(X, labels, "manhattan")).toThrow(
        /Unsupported metric.*Must be 'euclidean' or 'precomputed'/
      );
      // @ts-expect-error - deliberately passing unsupported metric for runtime validation
      expect(() => silhouetteSamples(X, labels, "cosine")).toThrow(
        /Unsupported metric.*Must be 'euclidean' or 'precomputed'/
      );
    });
  });

  describe("Performance Tests", () => {
    it("should handle moderately large datasets", () => {
      const size = 1000;
      const largeX = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => [(i % 10) + Math.random(), Math.floor(i / 10) + Math.random()])
      );
      const largeLabels = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 5)
      );

      const start = Date.now();
      silhouetteScore(largeX, largeLabels);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(10000); // Should complete in under 10 seconds
    });

    it("should handle comparison metrics efficiently", () => {
      const size = 5000;
      const labels1 = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 10)
      );
      const labels2 = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => (i + 1) % 10)
      );

      const start = Date.now();
      adjustedRandScore(labels1, labels2);
      normalizedMutualInfoScore(labels1, labels2);
      fowlkesMallowsScore(labels1, labels2);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(5000);
    }, 20000);
  });

  describe("Consistency Tests", () => {
    it("should have V-measure as harmonic mean of homogeneity and completeness", () => {
      const h = homogeneityScore(labelsTrue, labelsPred);
      const c = completenessScore(labelsTrue, labelsPred);
      const v = vMeasureScore(labelsTrue, labelsPred, 1.0);

      if (h + c > 0) {
        const expected = (2 * h * c) / (h + c);
        expect(v).toBeCloseTo(expected);
      } else {
        expect(v).toBe(0);
      }
    });

    it("should have consistent scores for perfect clustering", () => {
      const perfect = labelsTrue;

      expect(adjustedRandScore(perfect, perfect)).toBeCloseTo(1.0);
      expect(homogeneityScore(perfect, perfect)).toBeCloseTo(1.0);
      expect(completenessScore(perfect, perfect)).toBeCloseTo(1.0);
      expect(vMeasureScore(perfect, perfect)).toBeCloseTo(1.0);
    });

    it("should have silhouette score consistent with samples", () => {
      const score = silhouetteScore(X, labels);
      const samples = silhouetteSamples(X, labels);

      let sum = 0;
      for (let i = 0; i < samples.size; i++) {
        sum += Number(samples.data[samples.offset + i]);
      }
      const avgSamples = sum / samples.size;

      expect(score).toBeCloseTo(avgSamples);
    });
  });

  describe("Boundary Conditions", () => {
    it("should handle two clusters", () => {
      const twoClusterX = tensor([
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11],
      ]);
      const twoClusterLabels = tensor([0, 0, 1, 1]);

      expect(typeof silhouetteScore(twoClusterX, twoClusterLabels)).toBe("number");
      expect(typeof daviesBouldinScore(twoClusterX, twoClusterLabels)).toBe("number");
      expect(typeof calinskiHarabaszScore(twoClusterX, twoClusterLabels)).toBe("number");
    });

    it("should handle many small clusters", () => {
      const manySmallX = tensor([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
      ]);
      const manySmallLabels = tensor([0, 1, 2, 3, 4, 5]);

      expect(() => silhouetteScore(manySmallX, manySmallLabels)).toThrow(
        /2 <= n_clusters <= n_samples - 1/
      );
      expect(typeof daviesBouldinScore(manySmallX, manySmallLabels)).toBe("number");
    });

    it("should handle imbalanced clusters", () => {
      const imbalancedX = tensor([
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [10, 10],
      ]);
      const imbalancedLabels = tensor([0, 0, 0, 0, 0, 1]);

      expect(typeof silhouetteScore(imbalancedX, imbalancedLabels)).toBe("number");
      expect(typeof daviesBouldinScore(imbalancedX, imbalancedLabels)).toBe("number");
    });
  });

  describe("P0 Edge Cases and Error Handling", () => {
    describe("daviesBouldinScore with identical centroids", () => {
      it("should handle identical centroids gracefully", () => {
        const identicalX = tensor([
          [1, 1],
          [1, 1],
          [2, 2],
          [2, 2],
        ]);
        const identicalLabels = tensor([0, 0, 1, 1]);

        const score = daviesBouldinScore(identicalX, identicalLabels);
        expect(Number.isFinite(score)).toBe(true);
        expect(score).toBeGreaterThanOrEqual(0);
      });
    });

    describe("normalizedMutualInfoScore with zero entropy", () => {
      it("should return 1.0 when both have zero entropy (single class)", () => {
        const singleClass = tensor([0, 0, 0, 0]);

        const score = normalizedMutualInfoScore(singleClass, singleClass);
        expect(score).toBe(1.0);
      });

      it("should return 0.0 when one has zero entropy and other doesn't", () => {
        const singleClass = tensor([0, 0, 0, 0]);
        const multiClass = tensor([0, 1, 0, 1]);

        const score = normalizedMutualInfoScore(singleClass, multiClass);
        expect(score).toBe(0.0);
      });
    });

    describe("fowlkesMallowsScore with division by zero", () => {
      it("should return 0.0 when TP + FP = 0", () => {
        // All pairs have different true labels but same predicted labels
        const labelsTrue = tensor([0, 1, 2]);
        const labelsPred = tensor([0, 0, 0]);

        const score = fowlkesMallowsScore(labelsTrue, labelsPred);
        expect(score).toBe(0.0);
      });

      it("should return 0.0 when TP + FN = 0", () => {
        // All pairs have same true labels but different predicted labels
        const labelsTrue = tensor([0, 0, 0]);
        const labelsPred = tensor([0, 1, 2]);

        const score = fowlkesMallowsScore(labelsTrue, labelsPred);
        expect(score).toBe(0.0);
      });
    });

    describe("adjustedMutualInfoScore vs normalizedMutualInfoScore", () => {
      it("should match sklearn AMI and differ from NMI for non-trivial partitions", () => {
        const labelsTrue = tensor([0, 0, 1, 1, 2, 2]);
        const labelsPred = tensor([0, 1, 1, 2, 2, 0]);

        const ami = adjustedMutualInfoScore(labelsTrue, labelsPred);
        const nmi = normalizedMutualInfoScore(labelsTrue, labelsPred);

        // Reference values from sklearn.metrics
        expect(ami).toBeCloseTo(-0.25, 12);
        expect(nmi).toBeCloseTo(0.3690702464285424, 12);
        expect(ami).not.toBeCloseTo(nmi, 12);
      });
    });

    describe("error type validation", () => {
      it("should throw ShapeError for labels size mismatch", () => {
        const labelsTrue = tensor([0, 1, 1]);
        const labelsPred = tensor([0, 1]);

        expect(() => adjustedRandScore(labelsTrue, labelsPred)).toThrow("size");
      });

      it("should throw DTypeError for string labels", () => {
        const labelsTrue = tensor(["a", "b", "c"]);
        const labelsPred = tensor(["a", "b", "c"]);

        expect(() => adjustedRandScore(labelsTrue, labelsPred)).toThrow(
          "string labels not supported"
        );
      });

      it("should throw ShapeError for X and labels shape mismatch", () => {
        const X = tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const labels = tensor([0, 1]); // Wrong size

        expect(() => silhouetteScore(X, labels)).toThrow("labels length must match");
      });
    });
  });
});
