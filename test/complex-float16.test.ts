/**
 * Complex number and half-precision array tests.
 *
 * Reference values follow NumPy complex semantics (principal branches,
 * z**0 == 1) and IEEE 754 binary16 / bfloat16 round-to-nearest-even.
 */

import { describe, expect, it } from "vitest";
import {
  BFloat16Array,
  Complex,
  Complex64Array,
  Complex128Array,
  Float16Array as DbFloat16Array,
} from "../src/ndarray";
import {
  bfloat16BitsToFloat64,
  float16BitsToFloat64,
  float64ToBFloat16Bits,
  float64ToFloat16Bits,
} from "../src/ndarray/tensor/float16";

describe("Complex scalar", () => {
  it("abs/phase/conj/neg", () => {
    const z = new Complex(3, -4);
    expect(z.abs()).toBe(5);
    expect(new Complex(0, 1).phase()).toBeCloseTo(Math.PI / 2, 12);
    expect(z.conj().im).toBe(4);
    expect(z.neg().re).toBe(-3);
    // hypot avoids overflow: |1e200 + 1e200j| is finite-scaled
    expect(new Complex(1e200, 1e200).abs()).toBeCloseTo(Math.SQRT2 * 1e200, 185);
  });

  it("arithmetic", () => {
    const a = new Complex(1, 2);
    const b = new Complex(3, -1);
    expect(a.add(b).equals(new Complex(4, 1))).toBe(true);
    expect(a.sub(b).equals(new Complex(-2, 3))).toBe(true);
    // (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 5 + 5i
    expect(a.mul(b).equals(new Complex(5, 5))).toBe(true);
    // division: (5+5i)/(3-i) = (1+2i)
    expect(new Complex(5, 5).div(b).equals(a)).toBe(true);
  });

  it("division uses Smith's algorithm (no under/overflow)", () => {
    const tiny = new Complex(1e-300, 1e-300);
    const q = tiny.div(tiny);
    expect(q.re).toBeCloseTo(1, 12);
    expect(q.im).toBeCloseTo(0, 12);
    const huge = new Complex(1e300, 1e300);
    const q2 = huge.div(huge);
    expect(q2.re).toBeCloseTo(1, 12);

    const byZero = new Complex(1, 1).div(new Complex(0, 0));
    expect(Number.isFinite(byZero.re)).toBe(false);
  });

  it("exp/log/sqrt principal branches", () => {
    // Euler: e^{iπ} = -1
    const e = new Complex(0, Math.PI).exp();
    expect(e.re).toBeCloseTo(-1, 12);
    expect(e.im).toBeCloseTo(0, 12);

    const l = new Complex(-1, 0).log();
    expect(l.re).toBeCloseTo(0, 12);
    expect(l.im).toBeCloseTo(Math.PI, 12);

    const s = new Complex(-1, 0).sqrt();
    expect(s.re).toBeCloseTo(0, 12);
    expect(s.im).toBeCloseTo(1, 12);

    const s2 = new Complex(0, 2).sqrt(); // sqrt(2i) = 1 + i
    expect(s2.re).toBeCloseTo(1, 12);
    expect(s2.im).toBeCloseTo(1, 12);
  });

  it("pow follows NumPy conventions", () => {
    const z = new Complex(2, 0);
    expect(z.pow(10).re).toBeCloseTo(1024, 8);
    // z ** 0 == 1 for all z, including 0
    expect(new Complex(0, 0).pow(0).equals(Complex.ONE)).toBe(true);
    expect(new Complex(0, 0).pow(3).equals(Complex.ZERO)).toBe(true);
    expect(Number.isNaN(new Complex(0, 0).pow(-1).re)).toBe(true);
    // i^2 = -1
    const i2 = Complex.I.pow(2);
    expect(i2.re).toBeCloseTo(-1, 12);
    expect(i2.im).toBeCloseTo(0, 12);
    // complex exponent: i^i = e^{-π/2}
    const ii = Complex.I.pow(Complex.I);
    expect(ii.re).toBeCloseTo(Math.exp(-Math.PI / 2), 12);
    expect(ii.im).toBeCloseTo(0, 12);
  });

  it("toString / fromPolar / constants", () => {
    expect(new Complex(2, 0).toString()).toBe("2");
    expect(new Complex(0, 3).toString()).toBe("3j");
    expect(new Complex(1, -2).toString()).toBe("(1-2j)");
    expect(new Complex(1, 2).toString()).toBe("(1+2j)");
    const p = Complex.fromPolar(2, Math.PI / 2);
    expect(p.re).toBeCloseTo(0, 12);
    expect(p.im).toBeCloseTo(2, 12);
    expect(Complex.ONE.re).toBe(1);
    expect(Complex.I.im).toBe(1);
  });
});

describe("Complex64Array / Complex128Array", () => {
  it("stores interleaved values with float32 rounding", () => {
    const arr = Complex64Array.fromComplexArray([new Complex(0.1, 0.2), new Complex(3, 4)]);
    expect(arr.length).toBe(2);
    expect(arr.getReal(0)).toBe(Math.fround(0.1));
    expect(arr.getImag(0)).toBe(Math.fround(0.2));
    expect(arr.getComplex(1).equals(new Complex(3, 4))).toBe(true);
    // Proxy index access returns the real part
    expect(arr[1]).toBe(3);
  });

  it("setComplex/setRI/fill/slice/set/fromInterleaved", () => {
    const arr = new Complex64Array(3);
    arr.setComplex(0, new Complex(1, -1));
    arr.setRI(1, 2, -2);
    arr.fill(new Complex(9, 9), 2, 3);
    expect(arr.getComplex(0).equals(new Complex(1, -1))).toBe(true);
    expect(arr.getComplex(1).equals(new Complex(2, -2))).toBe(true);
    expect(arr.getComplex(2).equals(new Complex(9, 9))).toBe(true);

    const sl = arr.slice(1, 3);
    expect(sl.length).toBe(2);
    expect(sl.getComplex(0).equals(new Complex(2, -2))).toBe(true);

    const fromInter = Complex64Array.fromInterleaved([1, 2, 3, 4]);
    expect(fromInter.length).toBe(2);
    expect(fromInter.getComplex(1).equals(new Complex(3, 4))).toBe(true);

    const target = new Complex64Array(2);
    target.set([5, 6, 7, 8]);
    expect(target.getComplex(1).equals(new Complex(7, 8))).toBe(true);

    expect(arr.toComplexArray()).toHaveLength(3);
    expect(String(arr)).toContain("Complex64Array");
  });

  it("Complex128Array keeps float64 precision", () => {
    const arr = Complex128Array.fromComplexArray([new Complex(0.1, 0.2)]);
    expect(arr.getReal(0)).toBe(0.1);
    expect(arr.getImag(0)).toBe(0.2);
    expect(arr.BYTES_PER_ELEMENT).toBe(16);
    const sl = arr.slice(0, 1);
    expect(sl.getComplex(0).equals(new Complex(0.1, 0.2), 1e-15)).toBe(true);
  });
});

describe("Float16Array (binary16)", () => {
  it("bit conversions match IEEE 754 binary16", () => {
    expect(float64ToFloat16Bits(1)).toBe(0x3c00);
    expect(float64ToFloat16Bits(-2)).toBe(0xc000);
    expect(float64ToFloat16Bits(65504)).toBe(0x7bff); // max finite
    expect(float64ToFloat16Bits(Number.POSITIVE_INFINITY)).toBe(0x7c00);
    expect(float16BitsToFloat64(0x3c00)).toBe(1);
    expect(float16BitsToFloat64(0x7c00)).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(float16BitsToFloat64(0x7e00))).toBe(true);
    // smallest subnormal: 2^-24
    expect(float16BitsToFloat64(0x0001)).toBe(2 ** -24);
  });

  it("rounds to nearest even", () => {
    const arr = new DbFloat16Array(4);
    arr[0] = 2049; // tie between 2048 and 2050 -> even 2048
    arr[1] = 2051; // tie between 2050 and 2052 -> even 2052
    arr[2] = 0.1; // nearest binary16 to 0.1
    arr[3] = 65520; // above max finite midpoint -> Infinity
    expect(arr[0]).toBe(2048);
    expect(arr[1]).toBe(2052);
    expect(arr[2]).toBeCloseTo(0.0999755859375, 12);
    expect(arr[3]).toBe(Number.POSITIVE_INFINITY);
  });

  it("supports from/of/at/slice/subarray/fill/copyWithin/set/toArray", () => {
    const arr = DbFloat16Array.from([1, 2, 3, 4]);
    expect(arr.length).toBe(4);
    expect(arr.at(-1)).toBe(4);
    expect(arr.at(10)).toBeUndefined();

    const sl = arr.slice(1, 3);
    expect(sl.toArray()).toEqual([2, 3]);
    // slice copies; subarray shares
    sl[0] = 9;
    expect(arr[1]).toBe(2);
    const sub = arr.subarray(2);
    sub[0] = 7;
    expect(arr[2]).toBe(7);

    arr.fill(0.5, 0, 2);
    expect(arr.toArray().slice(0, 2)).toEqual([0.5, 0.5]);

    arr.set([1.5, 2.5], 2);
    expect(arr[2]).toBe(1.5);
    expect(arr[3]).toBe(2.5);

    arr.copyWithin(0, 2, 4);
    expect(arr[0]).toBe(1.5);
    expect(arr[1]).toBe(2.5);

    expect(DbFloat16Array.of(1, 2).toArray()).toEqual([1, 2]);
    expect(String(arr)).toContain("Float16Array");
  });
});

describe("BFloat16Array", () => {
  it("bit conversions match bfloat16 (truncated float32 with rounding)", () => {
    expect(float64ToBFloat16Bits(1)).toBe(0x3f80);
    expect(bfloat16BitsToFloat64(0x3f80)).toBe(1);
    expect(bfloat16BitsToFloat64(0x4049)).toBeCloseTo(3.140625, 12); // π in bf16
    // bf16 keeps the float32 exponent range: 1e38 stays finite
    const big = bfloat16BitsToFloat64(float64ToBFloat16Bits(1e38));
    expect(Number.isFinite(big)).toBe(true);
    expect(big).toBeCloseTo(1e38, -36);
    expect(Number.isNaN(bfloat16BitsToFloat64(float64ToBFloat16Bits(Number.NaN)))).toBe(true);
  });

  it("stores with 8-bit mantissa precision", () => {
    const arr = BFloat16Array.from([0.1, 256, -1.5]);
    expect(arr[0]).toBeCloseTo(0.10009765625, 12);
    expect(arr[1]).toBe(256);
    expect(arr[2]).toBe(-1.5);

    const sl = arr.slice(0, 2);
    expect(sl.length).toBe(2);
    arr.fill(2, 0, 1);
    expect(arr[0]).toBe(2);
    expect(arr.at(-1)).toBe(-1.5);
    expect(BFloat16Array.of(4).toArray()).toEqual([4]);
    expect(String(arr)).toContain("BFloat16Array");
  });
});
