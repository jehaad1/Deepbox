import { describe, expect, it } from "vitest";
import { KDTree } from "../src/ml";

describe("KDTree", () => {
  // 5 points in 2D
  const data = new Float64Array([0, 0, 1, 0, 0, 1, 1, 1, 5, 5]);

  it("query finds nearest neighbor", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([0.1, 0.1]);
    const { distances, indices } = tree.query(q, 1);
    expect(indices.length).toBe(1);
    expect(indices[0]).toBe(0); // closest to [0,0]
    expect(distances[0]).toBeLessThan(0.2);
  });

  it("query finds k nearest neighbors in order", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([0.5, 0.5]);
    const { distances, indices } = tree.query(q, 3);
    expect(indices.length).toBe(3);
    // All 4 corners are equidistant, but [0,0],[1,0],[0,1],[1,1] are closest
    // The 3 nearest should be among the first 4 points (not [5,5])
    for (const idx of indices) {
      expect(idx).toBeLessThan(4);
    }
    // Distances should be sorted ascending
    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1]!);
    }
  });

  it("query returns all points when k >= n", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([0, 0]);
    const { indices } = tree.query(q, 10);
    expect(indices.length).toBe(5);
  });

  it("queryRadius finds points within radius", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([0, 0]);
    const { distances, indices } = tree.queryRadius(q, 1.5);
    // [0,0] dist=0, [1,0] dist=1, [0,1] dist=1, [1,1] dist=sqrt(2)≈1.41
    expect(indices.length).toBe(4);
    for (const d of distances) {
      expect(d).toBeLessThanOrEqual(1.5);
    }
  });

  it("queryRadius returns empty for tiny radius in sparse area", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([100, 100]);
    const { indices } = tree.queryRadius(q, 0.01);
    expect(indices.length).toBe(0);
  });

  it("queryBatch processes multiple queries", () => {
    const tree = new KDTree(data, 5, 2);
    const queries = new Float64Array([0, 0, 5, 5]);
    const { distances, indices } = tree.queryBatch(queries, 2, 1);
    expect(indices[0]).toBe(0); // nearest to [0,0]
    expect(indices[1]).toBe(4); // nearest to [5,5]
    expect(distances[0]).toBeCloseTo(0);
    expect(distances[1]).toBeCloseTo(0);
  });

  it("works with higher dimensions", () => {
    const data3d = new Float64Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const tree = new KDTree(data3d, 3, 3);
    const q = new Float64Array([0.9, 0.9, 0.9]);
    const { indices } = tree.query(q, 1);
    expect(indices[0]).toBe(1); // closest to [1,1,1]
  });

  it("handles single point", () => {
    const singleData = new Float64Array([3, 4]);
    const tree = new KDTree(singleData, 1, 2);
    const q = new Float64Array([0, 0]);
    const { distances, indices } = tree.query(q, 1);
    expect(indices[0]).toBe(0);
    expect(distances[0]).toBeCloseTo(5); // sqrt(9+16)
  });

  it("handles duplicate points", () => {
    const dupData = new Float64Array([1, 1, 1, 1, 1, 1]);
    const tree = new KDTree(dupData, 3, 2);
    const q = new Float64Array([1, 1]);
    const { distances } = tree.query(q, 3);
    for (const d of distances) {
      expect(d).toBeCloseTo(0);
    }
  });

  it("query results are correct distances", () => {
    const tree = new KDTree(data, 5, 2);
    const q = new Float64Array([3, 4]);
    const { distances, indices } = tree.query(q, 1);
    // Nearest should be [1,1] at dist=sqrt(4+9)=sqrt(13) or [5,5] at dist=sqrt(4+1)=sqrt(5)
    expect(indices[0]).toBe(4); // [5,5] is closer
    expect(distances[0]).toBeCloseTo(Math.sqrt(4 + 1), 5);
  });
});
