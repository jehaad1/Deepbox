import { describe, expect, it } from "vitest";
import { filterDataset, loadIris, mapDataset, randomSplit, Subset } from "../src/datasets";

describe("Subset", () => {
  it("creates a subset with given indices", () => {
    const iris = loadIris();
    const sub = new Subset(iris, [0, 1, 2, 3, 4]);
    expect(sub.data.shape[0]).toBe(5);
    expect(sub.target.shape[0]).toBe(5);
    expect(sub.indices).toEqual([0, 1, 2, 3, 4]);
  });

  it("preserves feature names and description", () => {
    const iris = loadIris();
    const sub = new Subset(iris, [0]);
    expect(sub.featureNames).toEqual(iris.featureNames);
    expect(sub.description).toBe(iris.description);
  });

  it("preserves data dimensions", () => {
    const iris = loadIris();
    const sub = new Subset(iris, [10, 20, 30]);
    expect(sub.data.shape).toEqual([3, 4]);
  });

  it("throws for out-of-bounds index", () => {
    const iris = loadIris();
    expect(() => new Subset(iris, [999])).toThrow();
  });

  it("throws for negative index", () => {
    const iris = loadIris();
    expect(() => new Subset(iris, [-1])).toThrow();
  });

  it("empty indices produces empty subset", () => {
    const iris = loadIris();
    const sub = new Subset(iris, []);
    expect(sub.data.shape[0]).toBe(0);
    expect(sub.target.shape[0]).toBe(0);
  });
});

describe("randomSplit", () => {
  it("splits dataset into two parts", () => {
    const iris = loadIris();
    const [train, test] = randomSplit(iris, [120, 30], 42);
    expect(train.data.shape[0]).toBe(120);
    expect(test.data.shape[0]).toBe(30);
  });

  it("splits are deterministic with same seed", () => {
    const iris = loadIris();
    const [a1, b1] = randomSplit(iris, [100, 50], 123);
    const [a2, b2] = randomSplit(iris, [100, 50], 123);
    expect(a1.indices).toEqual(a2.indices);
    expect(b1.indices).toEqual(b2.indices);
  });

  it("splits into three parts", () => {
    const iris = loadIris();
    const [train, val, test] = randomSplit(iris, [100, 30, 20], 42);
    expect(train.data.shape[0]).toBe(100);
    expect(val.data.shape[0]).toBe(30);
    expect(test.data.shape[0]).toBe(20);
  });

  it("throws if lengths don't sum to dataset size", () => {
    const iris = loadIris();
    expect(() => randomSplit(iris, [100, 100], 42)).toThrow();
  });

  it("throws for negative lengths", () => {
    const iris = loadIris();
    expect(() => randomSplit(iris, [160, -10], 42)).toThrow();
  });

  it("non-overlapping indices", () => {
    const iris = loadIris();
    const [a, b] = randomSplit(iris, [75, 75], 42);
    const aSet = new Set(a.indices);
    const bSet = new Set(b.indices);
    expect(aSet.size).toBe(75);
    expect(bSet.size).toBe(75);
    for (const idx of a.indices) {
      expect(bSet.has(idx)).toBe(false);
    }
  });
});

describe("mapDataset", () => {
  it("doubles feature values", () => {
    const iris = loadIris();
    const mapped = mapDataset(iris, (data, target) => ({
      data: data.map((v) => v * 2),
      target,
    }));
    expect(mapped.data.shape).toEqual(iris.data.shape);
    expect(mapped.target.shape).toEqual(iris.target.shape);

    const origVal = iris.data.at(0, 0) as number;
    const mappedVal = mapped.data.at(0, 0) as number;
    expect(mappedVal).toBeCloseTo(origVal * 2, 10);
  });

  it("can change target values", () => {
    const iris = loadIris();
    const mapped = mapDataset(iris, (data, target) => ({
      data,
      target: target + 10,
    }));
    const origT = iris.target.at(0) as number;
    const mappedT = mapped.target.at(0) as number;
    expect(mappedT).toBeCloseTo(origT + 10, 10);
  });

  it("preserves metadata", () => {
    const iris = loadIris();
    const mapped = mapDataset(iris, (data, target) => ({ data, target }));
    expect(mapped.featureNames).toEqual(iris.featureNames);
    expect(mapped.description).toBe(iris.description);
  });
});

describe("filterDataset", () => {
  it("filters by target class", () => {
    const iris = loadIris();
    const filtered = filterDataset(iris, (_data, target) => target === 0);
    expect(filtered.data.shape[0]).toBe(50);
    expect(filtered.target.shape[0]).toBe(50);
    // All targets should be 0
    for (let i = 0; i < 50; i++) {
      expect(filtered.target.at(i)).toBe(0);
    }
  });

  it("filters by feature value", () => {
    const iris = loadIris();
    const filtered = filterDataset(iris, (data) => (data[0] ?? 0) > 6.0);
    expect(filtered.data.shape[0]).toBeGreaterThan(0);
    expect(filtered.data.shape[0]).toBeLessThan(150);
    // All first features should be > 6.0
    for (let i = 0; i < (filtered.data.shape[0] ?? 0); i++) {
      expect(filtered.data.at(i, 0) as number).toBeGreaterThan(6.0);
    }
  });

  it("empty result when nothing matches", () => {
    const iris = loadIris();
    const filtered = filterDataset(iris, () => false);
    expect(filtered.data.shape[0]).toBe(0);
    expect(filtered.target.shape[0]).toBe(0);
  });

  it("keeps all when predicate always true", () => {
    const iris = loadIris();
    const filtered = filterDataset(iris, () => true);
    expect(filtered.data.shape).toEqual(iris.data.shape);
  });

  it("preserves metadata", () => {
    const iris = loadIris();
    const filtered = filterDataset(iris, () => true);
    expect(filtered.featureNames).toEqual(iris.featureNames);
    expect(filtered.description).toBe(iris.description);
  });
});
