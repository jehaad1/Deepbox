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

describe("Clustering Metrics", () => {
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

  it("should calculate silhouette score in [-1, 1]", () => {
    const score = silhouetteScore(X, labels);
    expect(score).toBeGreaterThanOrEqual(-1);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate silhouette samples with correct size", () => {
    const samples = silhouetteSamples(X, labels);
    expect(samples.size).toBe(6);
    for (let i = 0; i < samples.size; i++) {
      const val = Number(samples.data[samples.offset + i]);
      expect(val).toBeGreaterThanOrEqual(-1);
      expect(val).toBeLessThanOrEqual(1);
    }
  });

  it("should calculate Davies-Bouldin score >= 0", () => {
    const score = daviesBouldinScore(X, labels);
    expect(score).toBeGreaterThanOrEqual(0);
  });

  it("should calculate Calinski-Harabasz score > 0", () => {
    const score = calinskiHarabaszScore(X, labels);
    expect(score).toBeGreaterThan(0);
  });

  it("should calculate adjusted Rand score in [-1, 1]", () => {
    const score = adjustedRandScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(-1);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate adjusted mutual information score in [-1, 1]", () => {
    const score = adjustedMutualInfoScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(-1);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate normalized mutual information score in [0, 1]", () => {
    const score = normalizedMutualInfoScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate Fowlkes-Mallows score in [0, 1]", () => {
    const score = fowlkesMallowsScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate homogeneity score in [0, 1]", () => {
    const score = homogeneityScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate completeness score in [0, 1]", () => {
    const score = completenessScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should calculate V-measure score in [0, 1]", () => {
    const score = vMeasureScore(labelsTrue, labelsPred);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});
