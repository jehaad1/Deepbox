import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  beta,
  binomial,
  choice,
  exponential,
  gamma,
  getSeed,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  randn,
  setSeed,
  shuffle,
  uniform,
} from "../src/random";
import { numRawData } from "./_helpers";

describe("Random Seed Management", () => {
  it("should set and get random seed", () => {
    setSeed(42);
    expect(getSeed()).toBe(42);
  });

  it("should make rand deterministic when seeded", () => {
    setSeed(123);
    const a = rand([5], { dtype: "float64" });
    setSeed(123);
    const b = rand([5], { dtype: "float64" });
    expect(numRawData(a.data)).toEqual(numRawData(b.data));
  });

  it("should make randn deterministic when seeded", () => {
    setSeed(321);
    const a = randn([5], { dtype: "float64" });
    setSeed(321);
    const b = randn([5], { dtype: "float64" });
    expect(numRawData(a.data)).toEqual(numRawData(b.data));
  });
});

describe("Basic Random Distributions", () => {
  it("should generate random uniform values", () => {
    const x = rand([10]);
    expect(x.shape).toEqual([10]);
  });

  it("should generate random normal values", () => {
    const x = randn([10]);
    expect(x.shape).toEqual([10]);
  });

  it("should generate random integers", () => {
    const x = randint(0, 10, [5]);
    expect(x.shape).toEqual([5]);
  });

  it("should generate uniform distribution", () => {
    const x = uniform(-1, 1, [100]);
    expect(x.shape).toEqual([100]);
  });

  it("should generate normal distribution", () => {
    const x = normal(0, 2, [100]);
    expect(x.shape).toEqual([100]);
  });
});

describe("Advanced Distributions", () => {
  it("should generate binomial distribution", () => {
    const x = binomial(10, 0.5, [100]);
    expect(x.shape).toEqual([100]);
  });

  it("should generate Poisson distribution", () => {
    const x = poisson(5, [100]);
    expect(x.shape).toEqual([100]);
  });

  it("should generate exponential distribution", () => {
    const x = exponential(2, [100]);
    expect(x.shape).toEqual([100]);
  });

  it("should generate gamma distribution", () => {
    const x = gamma(2, 2, [100]);
    expect(x.shape).toEqual([100]);
  });

  it("should generate beta distribution", () => {
    const x = beta(2, 5, [100]);
    expect(x.shape).toEqual([100]);
  });
});

describe("Sampling Operations", () => {
  it("should sample with choice", () => {
    const sample = choice(10, 5);
    expect(sample.shape[0]).toBe(5);
  });

  it("should sample from a tensor and preserve dtype", () => {
    const x = tensor([1, 2, 3, 4], { dtype: "int32" });
    setSeed(7);
    const y = choice(x, 3, true);
    expect(y.dtype).toBe("int32");
    expect(y.shape).toEqual([3]);
  });

  it("should be deterministic for choice(tensor) when seeded", () => {
    const x = tensor([10, 20, 30, 40, 50], { dtype: "float64" });
    setSeed(999);
    const a = choice(x, 4, true);
    setSeed(999);
    const b = choice(x, 4, true);
    expect(numRawData(a.data)).toEqual(numRawData(b.data));
  });

  it("should shuffle array in-place", () => {
    const x = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
    setSeed(2024);
    shuffle(x);
    // Same multiset of values.
    expect(numRawData(x.data).sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });

  it("should be deterministic for shuffle when seeded", () => {
    const a = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
    const b = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
    setSeed(2025);
    shuffle(a);
    setSeed(2025);
    shuffle(b);
    expect(numRawData(a.data)).toEqual(numRawData(b.data));
  });

  it("should create random permutation", () => {
    const perm = permutation(10);
    expect(perm.shape[0]).toBe(10);
  });

  it("should make permutation deterministic when seeded", () => {
    setSeed(77);
    const a = permutation(10);
    setSeed(77);
    const b = permutation(10);
    expect(numRawData(a.data)).toEqual(numRawData(b.data));
  });
});

describe("Parameter Validation", () => {
  it("randint should throw when high <= low", () => {
    expect(() => randint(5, 5, [1])).toThrow();
  });

  it("uniform should throw when high < low", () => {
    expect(() => uniform(1, -1, [1])).toThrow();
  });

  it("normal should throw when std < 0", () => {
    expect(() => normal(0, -1, [1])).toThrow();
  });

  it("binomial should throw when p is out of range", () => {
    expect(() => binomial(10, -0.1, [1])).toThrow();
    expect(() => binomial(10, 1.1, [1])).toThrow();
  });

  it("poisson should throw when lambda < 0", () => {
    expect(() => poisson(-1, [1])).toThrow();
  });

  it("exponential should throw when scale <= 0", () => {
    expect(() => exponential(0, [1])).toThrow();
  });

  it("gamma should throw when shape_param <= 0", () => {
    expect(() => gamma(0, 1, [1])).toThrow();
  });

  it("beta should throw when alpha/beta <= 0", () => {
    expect(() => beta(0, 1, [1])).toThrow();
    expect(() => beta(1, 0, [1])).toThrow();
  });

  it("choice(number) should throw on invalid population", () => {
    expect(() => choice(-1, 1)).toThrow();
  });
});
