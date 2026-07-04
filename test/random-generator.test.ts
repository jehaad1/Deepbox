import { describe, expect, it } from "vitest";
import { Generator } from "../src/random";

describe("Generator class", () => {
  describe("constructor", () => {
    it("creates with valid seed", () => {
      const rng = new Generator(42);
      expect(rng.seed).toBe(42);
    });

    it("throws for non-finite seed", () => {
      expect(() => new Generator(NaN)).toThrow();
      expect(() => new Generator(Infinity)).toThrow();
      expect(() => new Generator(-Infinity)).toThrow();
    });
  });

  describe("determinism", () => {
    it("same seed produces same sequence", () => {
      const rng1 = new Generator(123);
      const rng2 = new Generator(123);

      for (let i = 0; i < 100; i++) {
        expect(rng1.random()).toBe(rng2.random());
      }
    });

    it("different seeds produce different sequences", () => {
      const rng1 = new Generator(1);
      const rng2 = new Generator(2);

      let allSame = true;
      for (let i = 0; i < 10; i++) {
        if (rng1.random() !== rng2.random()) {
          allSame = false;
          break;
        }
      }
      expect(allSame).toBe(false);
    });
  });

  describe("random()", () => {
    it("returns values in [0, 1)", () => {
      const rng = new Generator(42);
      for (let i = 0; i < 1000; i++) {
        const v = rng.random();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });
  });

  describe("randomArray()", () => {
    it("returns correct size", () => {
      const rng = new Generator(42);
      const arr = rng.randomArray(100);
      expect(arr.length).toBe(100);
      expect(arr).toBeInstanceOf(Float64Array);
    });

    it("all values in [0, 1)", () => {
      const rng = new Generator(42);
      const arr = rng.randomArray(500);
      for (let i = 0; i < arr.length; i++) {
        expect(arr[i]).toBeGreaterThanOrEqual(0);
        expect(arr[i]).toBeLessThan(1);
      }
    });
  });

  describe("normal()", () => {
    it("produces values with approximately correct mean and std", () => {
      const rng = new Generator(42);
      const n = 10000;
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = rng.normal(5, 2);
        sum += v;
        sumSq += v * v;
      }
      const mean = sum / n;
      const variance = sumSq / n - mean * mean;
      expect(mean).toBeCloseTo(5, 0);
      expect(Math.sqrt(variance)).toBeCloseTo(2, 0);
    });

    it("throws for negative std", () => {
      const rng = new Generator(42);
      expect(() => rng.normal(0, -1)).toThrow();
    });
  });

  describe("normalArray()", () => {
    it("returns correct size", () => {
      const rng = new Generator(42);
      const arr = rng.normalArray(0, 1, 50);
      expect(arr.length).toBe(50);
    });
  });

  describe("uniform()", () => {
    it("returns values in [low, high)", () => {
      const rng = new Generator(42);
      for (let i = 0; i < 500; i++) {
        const v = rng.uniform(2, 5);
        expect(v).toBeGreaterThanOrEqual(2);
        expect(v).toBeLessThan(5);
      }
    });

    it("throws when low >= high", () => {
      const rng = new Generator(42);
      expect(() => rng.uniform(5, 5)).toThrow();
      expect(() => rng.uniform(6, 5)).toThrow();
    });
  });

  describe("uniformArray()", () => {
    it("returns correct size and range", () => {
      const rng = new Generator(42);
      const arr = rng.uniformArray(-1, 1, 100);
      expect(arr.length).toBe(100);
      for (let i = 0; i < arr.length; i++) {
        expect(arr[i]).toBeGreaterThanOrEqual(-1);
        expect(arr[i]).toBeLessThan(1);
      }
    });
  });

  describe("randint()", () => {
    it("returns integers in [low, high)", () => {
      const rng = new Generator(42);
      for (let i = 0; i < 500; i++) {
        const v = rng.randint(0, 10);
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(10);
      }
    });

    it("throws for non-integer bounds", () => {
      const rng = new Generator(42);
      expect(() => rng.randint(0.5, 5)).toThrow();
    });

    it("throws when low >= high", () => {
      const rng = new Generator(42);
      expect(() => rng.randint(5, 5)).toThrow();
    });
  });

  describe("randintArray()", () => {
    it("returns correct size and range", () => {
      const rng = new Generator(42);
      const arr = rng.randintArray(0, 100, 50);
      expect(arr.length).toBe(50);
      expect(arr).toBeInstanceOf(Int32Array);
      for (let i = 0; i < arr.length; i++) {
        expect(arr[i]).toBeGreaterThanOrEqual(0);
        expect(arr[i]).toBeLessThan(100);
      }
    });
  });

  describe("exponential()", () => {
    it("returns positive values", () => {
      const rng = new Generator(42);
      for (let i = 0; i < 100; i++) {
        expect(rng.exponential(1)).toBeGreaterThan(0);
      }
    });

    it("throws for non-positive scale", () => {
      const rng = new Generator(42);
      expect(() => rng.exponential(0)).toThrow();
      expect(() => rng.exponential(-1)).toThrow();
    });
  });

  describe("bernoulli()", () => {
    it("returns 0 or 1", () => {
      const rng = new Generator(42);
      for (let i = 0; i < 100; i++) {
        const v = rng.bernoulli(0.5);
        expect(v === 0 || v === 1).toBe(true);
      }
    });

    it("respects probability", () => {
      const rng = new Generator(42);
      let ones = 0;
      const n = 10000;
      for (let i = 0; i < n; i++) {
        ones += rng.bernoulli(0.7);
      }
      expect(ones / n).toBeCloseTo(0.7, 1);
    });

    it("throws for invalid p", () => {
      const rng = new Generator(42);
      expect(() => rng.bernoulli(-0.1)).toThrow();
      expect(() => rng.bernoulli(1.1)).toThrow();
    });
  });

  describe("choice()", () => {
    it("selects from weighted distribution", () => {
      const rng = new Generator(42);
      const counts = [0, 0, 0];
      const weights = [1, 2, 7]; // heavily biased toward index 2
      for (let i = 0; i < 1000; i++) {
        counts[rng.choice(weights)]!++;
      }
      // Index 2 should be most frequent
      expect(counts[2]).toBeGreaterThan(counts[0]!);
      expect(counts[2]).toBeGreaterThan(counts[1]!);
    });

    it("throws for empty weights", () => {
      const rng = new Generator(42);
      expect(() => rng.choice([])).toThrow();
    });
  });

  describe("shuffle()", () => {
    it("shuffles array in place", () => {
      const rng = new Generator(42);
      const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const original = [...arr];
      rng.shuffle(arr);

      // Same elements
      expect(arr.sort((a, b) => a - b)).toEqual(original);
    });

    it("is deterministic", () => {
      const rng1 = new Generator(42);
      const rng2 = new Generator(42);
      const a1 = [1, 2, 3, 4, 5];
      const a2 = [1, 2, 3, 4, 5];
      rng1.shuffle(a1);
      rng2.shuffle(a2);
      expect(a1).toEqual(a2);
    });
  });

  describe("permutation()", () => {
    it("returns a permutation of [0, n)", () => {
      const rng = new Generator(42);
      const perm = rng.permutation(10);
      expect(perm.length).toBe(10);
      expect(perm).toBeInstanceOf(Int32Array);

      // All values 0..9 present
      const sorted = Array.from(perm).sort((a, b) => a - b);
      for (let i = 0; i < 10; i++) {
        expect(sorted[i]).toBe(i);
      }
    });

    it("throws for negative n", () => {
      const rng = new Generator(42);
      expect(() => rng.permutation(-1)).toThrow();
    });

    it("is deterministic", () => {
      const rng1 = new Generator(42);
      const rng2 = new Generator(42);
      const p1 = rng1.permutation(20);
      const p2 = rng2.permutation(20);
      expect(Array.from(p1)).toEqual(Array.from(p2));
    });
  });
});
