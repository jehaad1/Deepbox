import { describe, expect, it } from "vitest";
import { BallTree } from "../src/ml";

describe("BallTree", () => {
  const data = new Float64Array([0, 0, 1, 0, 0, 1, 1, 1, 5, 5, 6, 5]);
  const nSamples = 6;
  const nDims = 2;

  it("finds k nearest neighbors correctly", () => {
    const tree = new BallTree(data, nSamples, nDims);
    const { distances, indices } = tree.query(new Float64Array([0.1, 0.1]), 2);

    expect(indices.length).toBe(2);
    expect(indices[0]).toBe(0);
    expect(distances[0]).toBeCloseTo(Math.sqrt(0.02), 4);
    expect(distances[0]).toBeLessThanOrEqual(distances[1]!);
  });

  it("finds all points within radius", () => {
    const tree = new BallTree(data, nSamples, nDims);
    const { distances, indices } = tree.queryRadius(new Float64Array([0, 0]), 1.05);

    expect(indices).toContain(0);
    expect(indices).toContain(1);
    expect(indices).toContain(2);
    expect(indices).not.toContain(4);
    expect(indices).not.toContain(5);

    for (const d of distances) {
      expect(d).toBeLessThanOrEqual(1.05);
    }
  });

  it("handles batch queries", () => {
    const tree = new BallTree(data, nSamples, nDims);
    const queryPoints = new Float64Array([0, 0, 5, 5]);
    const { distances, indices } = tree.queryBatch(queryPoints, 2, 1);

    expect(indices.length).toBe(2);
    expect(indices[0]).toBe(0);
    expect(indices[1]).toBe(4);
    expect(distances[0]).toBeCloseTo(0, 6);
    expect(distances[1]).toBeCloseTo(0, 6);
  });

  it("returns sorted distances", () => {
    const tree = new BallTree(data, nSamples, nDims);
    const { distances } = tree.query(new Float64Array([2.5, 2.5]), 4);

    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1]!);
    }
  });

  it("works with custom leaf size", () => {
    const tree = new BallTree(data, nSamples, nDims, 2);
    const { indices } = tree.query(new Float64Array([0, 0]), 1);
    expect(indices[0]).toBe(0);
  });
});
