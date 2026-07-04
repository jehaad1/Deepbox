import { describe, expect, it } from "vitest";
import { bernoulli, geometric, lognormal, setSeed } from "../src/random";

describe("bernoulli", () => {
  it("produces 0s and 1s", () => {
    setSeed(42);
    const t = bernoulli(0.5, [100]);
    const arr = Array.from(t.data as Int32Array);
    expect(arr.every((v) => v === 0 || v === 1)).toBe(true);
  });

  it("p=0 produces all 0s", () => {
    const t = bernoulli(0, [10]);
    const arr = Array.from(t.data as Int32Array);
    expect(arr.every((v) => v === 0)).toBe(true);
  });

  it("p=1 produces all 1s", () => {
    const t = bernoulli(1, [10]);
    const arr = Array.from(t.data as Int32Array);
    expect(arr.every((v) => v === 1)).toBe(true);
  });

  it("has correct shape", () => {
    const t = bernoulli(0.5, [3, 4]);
    expect(t.shape).toEqual([3, 4]);
  });

  it("throws on invalid p", () => {
    expect(() => bernoulli(-0.1, [1])).toThrow();
    expect(() => bernoulli(1.1, [1])).toThrow();
  });

  it("is deterministic with seed", () => {
    setSeed(123);
    const a = bernoulli(0.5, [20]);
    setSeed(123);
    const b = bernoulli(0.5, [20]);
    expect(Array.from(a.data as Int32Array)).toEqual(Array.from(b.data as Int32Array));
  });
});

describe("geometric", () => {
  it("produces positive integers", () => {
    setSeed(42);
    const t = geometric(0.5, [100]);
    const arr = Array.from(t.data as Int32Array);
    expect(arr.every((v) => v >= 1)).toBe(true);
  });

  it("p=1 always returns 1", () => {
    const t = geometric(1, [10]);
    const arr = Array.from(t.data as Int32Array);
    expect(arr.every((v) => v === 1)).toBe(true);
  });

  it("has correct shape", () => {
    const t = geometric(0.3, [2, 5]);
    expect(t.shape).toEqual([2, 5]);
  });

  it("throws on invalid p", () => {
    expect(() => geometric(0, [1])).toThrow();
    expect(() => geometric(-0.1, [1])).toThrow();
    expect(() => geometric(1.1, [1])).toThrow();
  });

  it("is deterministic with seed", () => {
    setSeed(99);
    const a = geometric(0.3, [20]);
    setSeed(99);
    const b = geometric(0.3, [20]);
    expect(Array.from(a.data as Int32Array)).toEqual(Array.from(b.data as Int32Array));
  });
});

describe("lognormal", () => {
  it("produces positive values", () => {
    setSeed(42);
    const t = lognormal(0, 1, [100]);
    const arr = Array.from(t.data as Float32Array);
    expect(arr.every((v) => v > 0)).toBe(true);
  });

  it("has correct shape", () => {
    const t = lognormal(0, 1, [3, 4]);
    expect(t.shape).toEqual([3, 4]);
  });

  it("std=0 produces exp(mean)", () => {
    const t = lognormal(2, 0, [5]);
    const arr = Array.from(t.data as Float32Array);
    const expected = Math.fround(Math.exp(2));
    expect(arr.every((v) => Math.abs(v - expected) < 0.001)).toBe(true);
  });

  it("throws on invalid params", () => {
    expect(() => lognormal(0, -1, [1])).toThrow();
    expect(() => lognormal(Number.NaN, 1, [1])).toThrow();
  });

  it("is deterministic with seed", () => {
    setSeed(77);
    const a = lognormal(0, 1, [20]);
    setSeed(77);
    const b = lognormal(0, 1, [20]);
    expect(Array.from(a.data as Float32Array)).toEqual(Array.from(b.data as Float32Array));
  });
});
