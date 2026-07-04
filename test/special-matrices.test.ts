import { describe, expect, it } from "vitest";
import {
  circulant,
  companion,
  hadamard,
  hankel,
  hilbert,
  toeplitz,
  vandermonde,
} from "../src/linalg";

describe("hilbert", () => {
  it("creates 3x3 Hilbert matrix", () => {
    const H = hilbert(3);
    expect(H.shape).toEqual([3, 3]);
    expect(H.at(0, 0)).toBeCloseTo(1);
    expect(H.at(0, 1)).toBeCloseTo(1 / 2);
    expect(H.at(0, 2)).toBeCloseTo(1 / 3);
    expect(H.at(1, 0)).toBeCloseTo(1 / 2);
    expect(H.at(1, 1)).toBeCloseTo(1 / 3);
    expect(H.at(2, 2)).toBeCloseTo(1 / 5);
  });

  it("creates 1x1 Hilbert matrix", () => {
    const H = hilbert(1);
    expect(H.shape).toEqual([1, 1]);
    expect(H.at(0, 0)).toBe(1);
  });

  it("throws on invalid n", () => {
    expect(() => hilbert(0)).toThrow();
    expect(() => hilbert(-1)).toThrow();
    expect(() => hilbert(1.5)).toThrow();
  });
});

describe("toeplitz", () => {
  it("creates symmetric Toeplitz from column only", () => {
    const T = toeplitz([1, 2, 3]);
    expect(T.shape).toEqual([3, 3]);
    // Should be symmetric
    expect(T.at(0, 0)).toBe(1);
    expect(T.at(0, 1)).toBe(2);
    expect(T.at(0, 2)).toBe(3);
    expect(T.at(1, 0)).toBe(2);
    expect(T.at(1, 1)).toBe(1);
    expect(T.at(2, 0)).toBe(3);
  });

  it("creates non-symmetric Toeplitz with column and row", () => {
    const T = toeplitz([1, 2, 3], [1, 4, 5]);
    expect(T.shape).toEqual([3, 3]);
    expect(T.at(0, 0)).toBe(1);
    expect(T.at(0, 1)).toBe(4);
    expect(T.at(0, 2)).toBe(5);
    expect(T.at(1, 0)).toBe(2);
    expect(T.at(1, 1)).toBe(1);
    expect(T.at(1, 2)).toBe(4);
  });

  it("throws on empty column", () => {
    expect(() => toeplitz([])).toThrow();
  });
});

describe("vandermonde", () => {
  it("creates default (decreasing) Vandermonde", () => {
    const V = vandermonde([1, 2, 3]);
    expect(V.shape).toEqual([3, 3]);
    // Row for x=2: [4, 2, 1] (x^2, x^1, x^0)
    expect(V.at(1, 0)).toBe(4);
    expect(V.at(1, 1)).toBe(2);
    expect(V.at(1, 2)).toBe(1);
  });

  it("creates increasing Vandermonde", () => {
    const V = vandermonde([1, 2, 3], undefined, true);
    expect(V.shape).toEqual([3, 3]);
    // Row for x=2: [1, 2, 4] (x^0, x^1, x^2)
    expect(V.at(1, 0)).toBe(1);
    expect(V.at(1, 1)).toBe(2);
    expect(V.at(1, 2)).toBe(4);
  });

  it("allows custom number of columns", () => {
    const V = vandermonde([1, 2], 4);
    expect(V.shape).toEqual([2, 4]);
  });

  it("throws on empty input", () => {
    expect(() => vandermonde([])).toThrow();
  });
});

describe("hadamard", () => {
  it("creates H_1", () => {
    const H = hadamard(1);
    expect(H.shape).toEqual([1, 1]);
    expect(H.at(0, 0)).toBe(1);
  });

  it("creates H_2", () => {
    const H = hadamard(2);
    expect(H.shape).toEqual([2, 2]);
    expect(H.at(0, 0)).toBe(1);
    expect(H.at(0, 1)).toBe(1);
    expect(H.at(1, 0)).toBe(1);
    expect(H.at(1, 1)).toBe(-1);
  });

  it("creates H_4", () => {
    const H = hadamard(4);
    expect(H.shape).toEqual([4, 4]);
    // All entries should be +1 or -1
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const v = H.at(i, j);
        expect(v === 1 || v === -1).toBe(true);
      }
    }
  });

  it("throws on non-power-of-2", () => {
    expect(() => hadamard(3)).toThrow();
    expect(() => hadamard(5)).toThrow();
    expect(() => hadamard(0)).toThrow();
  });
});

describe("companion", () => {
  it("creates companion matrix for x^2 + 2x + 3", () => {
    const C = companion([1, 2, 3]);
    expect(C.shape).toEqual([2, 2]);
    // First row: [-2, -3]
    expect(C.at(0, 0)).toBe(-2);
    expect(C.at(0, 1)).toBe(-3);
    // Second row: [1, 0]
    expect(C.at(1, 0)).toBe(1);
    expect(C.at(1, 1)).toBe(0);
  });

  it("scales by leading coefficient", () => {
    const C = companion([2, 4, 6]);
    expect(C.at(0, 0)).toBe(-2);
    expect(C.at(0, 1)).toBe(-3);
  });

  it("throws on too few coefficients", () => {
    expect(() => companion([1])).toThrow();
    expect(() => companion([])).toThrow();
  });

  it("throws on zero leading coefficient", () => {
    expect(() => companion([0, 1, 2])).toThrow();
  });
});

describe("circulant", () => {
  it("creates circulant matrix", () => {
    const C = circulant([1, 2, 3]);
    expect(C.shape).toEqual([3, 3]);
    // Row 0: [1, 3, 2]  (c[0], c[-1], c[-2]) = (c[0], c[2], c[1])
    expect(C.at(0, 0)).toBe(1);
    expect(C.at(0, 1)).toBe(3);
    expect(C.at(0, 2)).toBe(2);
    // Row 1: [2, 1, 3]
    expect(C.at(1, 0)).toBe(2);
    expect(C.at(1, 1)).toBe(1);
    expect(C.at(1, 2)).toBe(3);
  });

  it("creates 1x1 circulant", () => {
    const C = circulant([5]);
    expect(C.shape).toEqual([1, 1]);
    expect(C.at(0, 0)).toBe(5);
  });

  it("throws on empty input", () => {
    expect(() => circulant([])).toThrow();
  });
});

describe("hankel", () => {
  it("creates Hankel matrix from column only (zeros padding)", () => {
    const H = hankel([1, 2, 3]);
    expect(H.shape).toEqual([3, 3]);
    expect(H.at(0, 0)).toBe(1);
    expect(H.at(0, 1)).toBe(2);
    expect(H.at(0, 2)).toBe(3);
    expect(H.at(1, 0)).toBe(2);
    expect(H.at(1, 1)).toBe(3);
    expect(H.at(1, 2)).toBe(0);
    expect(H.at(2, 0)).toBe(3);
    expect(H.at(2, 1)).toBe(0);
    expect(H.at(2, 2)).toBe(0);
  });

  it("creates Hankel matrix with last row", () => {
    const H = hankel([1, 2, 3], [3, 4, 5]);
    expect(H.shape).toEqual([3, 3]);
    expect(H.at(0, 0)).toBe(1);
    expect(H.at(0, 1)).toBe(2);
    expect(H.at(0, 2)).toBe(3);
    expect(H.at(1, 0)).toBe(2);
    expect(H.at(1, 1)).toBe(3);
    expect(H.at(1, 2)).toBe(4);
    expect(H.at(2, 0)).toBe(3);
    expect(H.at(2, 1)).toBe(4);
    expect(H.at(2, 2)).toBe(5);
  });

  it("throws on empty column", () => {
    expect(() => hankel([])).toThrow();
  });
});
