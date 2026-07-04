import { describe, expect, it } from "vitest";
import { NearestNeighbors } from "../src/ml";
import { tensor } from "../src/ndarray";
import { toNum2D } from "./_helpers";

describe("NearestNeighbors", () => {
  const X = tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [3, 3],
  ]);

  it("fits unsupervised data and returns k-nearest neighbors for query points", () => {
    const nn = new NearestNeighbors({ nNeighbors: 2 });
    nn.fit(X);

    const { distances, indices } = nn.kneighbors(tensor([[0.1, 0.1]]));
    expect(distances.shape).toEqual([1, 2]);
    expect(indices.shape).toEqual([1, 2]);

    const indexRow = toNum2D(indices.toArray())[0];
    expect(indexRow).toEqual([0, 1]);

    const distanceRow = toNum2D(distances.toArray())[0];
    expect(distanceRow[0]).toBeLessThanOrEqual(distanceRow[1]);
    expect(distanceRow[0]).toBeCloseTo(Math.sqrt(0.02), 6);
  });

  it("excludes each sample itself when querying training data by default", () => {
    const nn = new NearestNeighbors({ nNeighbors: 1 });
    nn.fit(X);

    const { indices } = nn.kneighbors();
    const rows = toNum2D(indices.toArray());
    expect(rows[0]).toEqual([1]);
    expect(rows[1]).toEqual([0]);
    expect(rows[2]).toEqual([0]);
    expect(rows[3]).toEqual([1]);
  });

  it("returns neighbors within radius for explicit queries and training data", () => {
    const nn = new NearestNeighbors({ radius: 1.05 });
    nn.fit(X);

    const explicit = nn.radiusNeighbors(tensor([[0, 0]]));
    expect(explicit.indices[0]).toEqual([0, 1, 2]);
    expect(explicit.distances[0][0]).toBeCloseTo(0, 6);

    const trainQuery = nn.radiusNeighbors();
    expect(trainQuery.indices[0]).toEqual([1, 2]);
    expect(trainQuery.indices[3]).toEqual([]);
  });

  it("supports parameter updates and alternate metrics", () => {
    const nn = new NearestNeighbors({ nNeighbors: 1, radius: 0.5 });
    nn.setParams({ nNeighbors: 2, radius: 2, metric: "manhattan" });
    expect(nn.getParams()).toEqual({
      nNeighbors: 2,
      radius: 2,
      metric: "manhattan",
    });

    nn.fit(X);
    const { indices } = nn.kneighbors(tensor([[0.2, 0.2]]));
    expect(toNum2D(indices.toArray())[0]).toEqual([0, 1]);
  });

  it("validates parameters and usage", () => {
    expect(() => new NearestNeighbors({ nNeighbors: 0 })).toThrow(/>= 1/);
    expect(() => new NearestNeighbors({ radius: 0 })).toThrow(/radius/i);

    const nn = new NearestNeighbors({ nNeighbors: 2 });
    expect(() => nn.kneighbors(X)).toThrow(/fitted/i);
    expect(() => nn.radiusNeighbors(X)).toThrow(/fitted/i);

    nn.fit(X);
    expect(() => nn.kneighbors(tensor([[1, 2, 3]]))).toThrow(/features/i);
    expect(() => nn.setParams({ metric: "chebyshev" })).toThrow(/metric/i);
    expect(() => nn.setParams({ unknown: 1 })).toThrow(/Unknown parameter/i);
  });

  it("rejects impossible neighbor counts for self-query on tiny datasets", () => {
    const nn = new NearestNeighbors({ nNeighbors: 1 });
    nn.fit(tensor([[42, 24]]));
    expect(() => nn.kneighbors()).toThrow(/nNeighbors/i);
  });
});
