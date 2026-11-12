import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

// pandas 3.1 on the same data: DataFrame.corr("spearman") is exactly +-1 for every monotone
// pair below, and DataFrame.corr("pearson") is exactly 1 for integer-valued linear pairs.
const lcg = (seed: number) => {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
};

describe("DataFrame.corr reports exactly +-1 for perfect correlation (DOC-7)", () => {
  it("spearman is exactly 1 or -1 for monotone pairs, with and without gaps", () => {
    const rng = lcg(12345);
    for (let t = 0; t < 300; t++) {
      const n = 3 + Math.floor(rng() * 20);
      const x = Array.from({ length: n }, () => Math.round(((rng() * 1000) / 7) * 1000) / 1000);
      const sign = t % 2 === 0 ? 1 : -1;
      const y = x.map((v) => sign * (0.5 + t) * v + 3);
      const gappy = y.map((v, i) => (i % 4 === 3 ? null : v));
      const full = new DataFrame({ x, y }).corr("spearman").get("y").toArray()[0];
      const partial = new DataFrame({ x, y: gappy }).corr("spearman").get("y").toArray()[0];
      if (new Set(x).size > 2) {
        expect(full).toBe(sign);
        expect(partial).toBe(sign);
      }
    }
  });

  it("pearson is exactly 1 or -1 for integer-valued linear pairs", () => {
    const rng = lcg(777);
    for (let t = 0; t < 300; t++) {
      const n = 3 + Math.floor(rng() * 12);
      const x = Array.from({ length: n }, () => Math.floor(rng() * 50));
      const a = 1 + Math.floor(rng() * 4);
      const sign = t % 2 === 0 ? 1 : -1;
      const y = x.map((v) => sign * a * v + 7);
      if (new Set(x).size < 2) continue;
      const r = new DataFrame({ x, y }).corr().get("y").toArray()[0] as number;
      expect(Math.abs(r)).toBeLessThanOrEqual(1);
      expect(r).toBeCloseTo(sign, 14);
    }
    const exact = new DataFrame({ a: [1, 2, 3], b: [2, 4, 6], c: [3, 2, 1] }).corr();
    expect(exact.get("b").toArray()).toEqual([1, 1, -1]);
    expect(exact.get("c").toArray()).toEqual([-1, -1, 1]);
  });

  it("pearson never leaves [-1, 1] for random floats", () => {
    const rng = lcg(4242);
    for (let t = 0; t < 300; t++) {
      const n = 3 + Math.floor(rng() * 20);
      const x = Array.from({ length: n }, () => rng() * 1e3);
      const y = x.map((v) => 2.5 * v - 1);
      const r = new DataFrame({ x, y }).corr().get("y").toArray()[0] as number;
      expect(Math.abs(r)).toBeLessThanOrEqual(1);
      expect(Math.abs(r - 1)).toBeLessThan(1e-14);
    }
  });
});
