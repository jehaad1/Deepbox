import { describe, expect, it } from "vitest";
import { KMeans } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("KMeans", () => {
  it("should cluster simple 2D data", () => {
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ]);

    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    kmeans.fit(X);

    const labels = kmeans.predict(X);
    expect(labels.shape).toEqual([6]);

    // Check that we have 2 clusters
    const uniqueLabels = new Set<number>();
    for (let i = 0; i < labels.size; i++) {
      uniqueLabels.add(Number(labels.data[labels.offset + i]));
    }
    expect(uniqueLabels.size).toBe(2);
  });

  it("should have cluster centers", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [8, 9],
      [9, 10],
    ]);

    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    kmeans.fit(X);

    const centers = kmeans.clusterCenters;
    expect(centers.shape[0]).toBe(2);
    expect(centers.shape[1]).toBe(2);
  });

  it("should calculate inertia", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [8, 9],
      [9, 10],
    ]);

    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    kmeans.fit(X);

    const inertia = kmeans.inertia;
    expect(typeof inertia).toBe("number");
    expect(inertia).toBeGreaterThanOrEqual(0);
  });

  it("should throw error if not fitted", () => {
    const kmeans = new KMeans({ nClusters: 2 });
    const X = tensor([[1, 2]]);

    expect(() => kmeans.predict(X)).toThrow();
  });

  it("should validate feature count on predict", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [8, 9],
      [9, 10],
    ]);
    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    kmeans.fit(X);

    expect(() => kmeans.predict(tensor([[1, 2, 3]]))).toThrow(/features/);
  });

  it("should support fitPredict", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [8, 9],
      [9, 10],
    ]);

    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    const labels = kmeans.fitPredict(X);

    expect(labels.shape).toEqual([4]);
  });
});
