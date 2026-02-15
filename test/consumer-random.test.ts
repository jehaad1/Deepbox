import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  beta,
  binomial,
  choice,
  clearSeed,
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

describe("consumer API: random", () => {
  it("setSeed / getSeed / clearSeed", () => {
    setSeed(42);
    expect(getSeed()).toBe(42);
    clearSeed();
    expect(getSeed()).toBeUndefined();
  });

  it("rand, randn, randint", () => {
    expect(rand([3, 3]).shape).toEqual([3, 3]);
    expect(randn([2, 4]).shape).toEqual([2, 4]);
    expect(randint(0, 10, [5]).size).toBe(5);
  });

  it("distributions: uniform, normal, binomial, poisson, exponential, gamma, beta", () => {
    expect(uniform(-1, 1, [10]).size).toBe(10);
    expect(normal(0, 1, [10]).size).toBe(10);
    expect(binomial(10, 0.5, [20]).size).toBe(20);
    expect(poisson(5, [15]).size).toBe(15);
    expect(exponential(2, [10]).size).toBe(10);
    expect(gamma(2, 1, [10]).size).toBe(10);
    expect(beta(2, 5, [10]).size).toBe(10);
  });

  it("reproducibility via seed", () => {
    setSeed(123);
    const a = rand([5]);
    setSeed(123);
    const b = rand([5]);
    expect(a.at(0)).toBe(b.at(0));
    clearSeed();
  });

  it("choice, permutation, shuffle", () => {
    const data = tensor([10, 20, 30, 40, 50]);
    expect(choice(data, 3).size).toBe(3);
    expect(permutation(5).size).toBe(5);
    expect(permutation(tensor([10, 20, 30])).size).toBe(3);
    const toShuffle = tensor([1, 2, 3, 4, 5]);
    shuffle(toShuffle);
    expect(toShuffle.size).toBe(5);
  });
});
