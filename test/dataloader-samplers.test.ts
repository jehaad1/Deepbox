import { describe, expect, it } from "vitest";
import {
  DataLoader,
  SequentialSampler,
  SubsetRandomSampler,
  WeightedRandomSampler,
} from "../src/datasets";
import { tensor } from "../src/ndarray";

describe("SequentialSampler", () => {
  it("produces indices 0..n-1 in order", () => {
    const sampler = new SequentialSampler(5);
    expect([...sampler]).toEqual([0, 1, 2, 3, 4]);
    expect(sampler.length).toBe(5);
  });

  it("produces empty iteration for length 0", () => {
    const sampler = new SequentialSampler(0);
    expect([...sampler]).toEqual([]);
    expect(sampler.length).toBe(0);
  });

  it("throws on negative length", () => {
    expect(() => new SequentialSampler(-1)).toThrow();
  });
});

describe("SubsetRandomSampler", () => {
  it("produces a permutation of the given indices", () => {
    const indices = [0, 2, 4, 6, 8];
    const sampler = new SubsetRandomSampler(indices, { seed: 42 });
    const result = [...sampler];
    expect(result.length).toBe(5);
    expect(result.sort((a, b) => a - b)).toEqual([0, 2, 4, 6, 8]);
  });

  it("length matches indices length", () => {
    const sampler = new SubsetRandomSampler([1, 3, 5]);
    expect(sampler.length).toBe(3);
  });

  it("is deterministic with seed", () => {
    const indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const s1 = new SubsetRandomSampler(indices, { seed: 123 });
    const s2 = new SubsetRandomSampler(indices, { seed: 123 });
    expect([...s1]).toEqual([...s2]);
  });

  it("throws on invalid indices", () => {
    expect(() => new SubsetRandomSampler([-1])).toThrow();
    expect(() => new SubsetRandomSampler([1.5])).toThrow();
  });
});

describe("WeightedRandomSampler", () => {
  it("produces numSamples indices (with replacement)", () => {
    const sampler = new WeightedRandomSampler([1, 1, 1, 1], {
      numSamples: 10,
      replacement: true,
      seed: 42,
    });
    const result = [...sampler];
    expect(result.length).toBe(10);
    for (const idx of result) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(4);
    }
    expect(sampler.length).toBe(10);
  });

  it("produces numSamples indices (without replacement)", () => {
    const sampler = new WeightedRandomSampler([1, 2, 3, 4], {
      numSamples: 3,
      replacement: false,
      seed: 42,
    });
    const result = [...sampler];
    expect(result.length).toBe(3);
    // All unique
    expect(new Set(result).size).toBe(3);
    for (const idx of result) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(4);
    }
  });

  it("heavily weighted index appears more often (statistical)", () => {
    // weight[0] = 100, rest = 1 → index 0 should dominate
    const sampler = new WeightedRandomSampler([100, 1, 1, 1], {
      numSamples: 100,
      replacement: true,
      seed: 42,
    });
    const result = [...sampler];
    const count0 = result.filter((i) => i === 0).length;
    expect(count0).toBeGreaterThan(80); // should be ~97%
  });

  it("is deterministic with seed", () => {
    const s1 = new WeightedRandomSampler([1, 2, 3], {
      numSamples: 20,
      seed: 99,
    });
    const s2 = new WeightedRandomSampler([1, 2, 3], {
      numSamples: 20,
      seed: 99,
    });
    expect([...s1]).toEqual([...s2]);
  });

  it("defaults to replacement=true and numSamples=weights.length", () => {
    const sampler = new WeightedRandomSampler([1, 1, 1], { seed: 42 });
    const result = [...sampler];
    expect(result.length).toBe(3);
    expect(sampler.length).toBe(3);
  });

  it("throws on empty weights", () => {
    expect(() => new WeightedRandomSampler([])).toThrow();
  });

  it("throws on negative weights", () => {
    expect(() => new WeightedRandomSampler([-1, 1])).toThrow();
  });

  it("throws on all-zero weights", () => {
    expect(() => new WeightedRandomSampler([0, 0, 0])).toThrow();
  });

  it("throws if numSamples > length without replacement", () => {
    expect(
      () =>
        new WeightedRandomSampler([1, 1], {
          numSamples: 5,
          replacement: false,
        })
    ).toThrow();
  });
});

describe("DataLoader with sampler", () => {
  const X = tensor([
    [10, 11],
    [20, 21],
    [30, 31],
    [40, 41],
    [50, 51],
  ]);
  const y = tensor([0, 1, 2, 3, 4]);

  it("uses SubsetRandomSampler to select a subset", () => {
    const sampler = new SubsetRandomSampler([0, 2, 4], { seed: 42 });
    const loader = new DataLoader(X, y, { batchSize: 3, sampler });
    const batches = [...loader];
    expect(batches.length).toBe(1);
    const [xBatch, yBatch] = batches[0]!;
    expect(xBatch.shape).toEqual([3, 2]);
    expect(yBatch.shape).toEqual([3]);
  });

  it("uses WeightedRandomSampler for oversampling", () => {
    const sampler = new WeightedRandomSampler([1, 1, 1, 1, 1], {
      numSamples: 8,
      replacement: true,
      seed: 42,
    });
    const loader = new DataLoader(X, y, { batchSize: 4, sampler });
    const batches = [...loader];
    expect(batches.length).toBe(2);
    expect(batches[0]![0].shape).toEqual([4, 2]);
    expect(batches[1]![0].shape).toEqual([4, 2]);
  });

  it("uses SequentialSampler (same as default)", () => {
    const sampler = new SequentialSampler(5);
    const loader = new DataLoader(X, y, { batchSize: 2, sampler });
    const batches = [...loader];
    expect(batches.length).toBe(3); // 2+2+1
    expect(batches[0]![0].shape).toEqual([2, 2]);
    expect(batches[2]![0].shape).toEqual([1, 2]);
  });

  it("throws when sampler and shuffle are both set", () => {
    const sampler = new SequentialSampler(5);
    expect(() => new DataLoader(X, y, { sampler, shuffle: true })).toThrow(
      "Cannot use both sampler and shuffle"
    );
  });

  it("works with dropLast and sampler", () => {
    const sampler = new SubsetRandomSampler([0, 1, 2, 3, 4], { seed: 42 });
    const loader = new DataLoader(X, y, {
      batchSize: 2,
      sampler,
      dropLast: true,
    });
    const batches = [...loader];
    expect(batches.length).toBe(2); // 5 / 2 = 2 (drop last 1)
  });

  it("works without y (X-only loader)", () => {
    const sampler = new SubsetRandomSampler([0, 2, 4], { seed: 42 });
    const loader = new DataLoader(X, undefined, { batchSize: 3, sampler });
    const batches = [...loader];
    expect(batches.length).toBe(1);
    expect(batches[0]![0].shape).toEqual([3, 2]);
    expect(batches[0]!.length).toBe(1); // only X, no y
  });
});
