import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Backend, DeviceBuffer, KernelBackend } from "../../src/core";
import {
  DataValidationError,
  DeviceError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  registerBackend,
  resetConfig,
  ShapeError,
  setConfig,
} from "../../src/core";
import { Complex, Complex64Array, Complex128Array } from "../../src/ndarray/tensor/complex";
import {
  arange,
  empty,
  eye,
  full,
  geomspace,
  linspace,
  logspace,
  ones,
  randn,
  tensor,
  zeros,
} from "../../src/ndarray/tensor/creation";
import {
  BFloat16Array,
  Float16Array,
  roundToBFloat16,
  roundToFloat16,
} from "../../src/ndarray/tensor/float16";
import { transpose } from "../../src/ndarray/tensor/shape";
import { Tensor } from "../../src/ndarray/tensor/Tensor";

describe("c37 Complex arithmetic accuracy (references: NumPy 2.4)", () => {
  it("sqrt is exact on the axes and respects signed zero", () => {
    // np.sqrt(-4+0j) = 2j, np.sqrt(-4-0j) = -2j
    const up = new Complex(-4, 0).sqrt();
    expect(up.re).toBe(0);
    expect(up.im).toBe(2);
    const down = new Complex(-4, -0).sqrt();
    expect(down.re).toBe(0);
    expect(down.im).toBe(-2);
    // np.sqrt(3+4j) = 2+1j, np.sqrt(-3+4j) = 1+2j, np.sqrt(-3-4j) = 1-2j
    expect(new Complex(3, 4).sqrt().equals(new Complex(2, 1), 1e-15)).toBe(true);
    expect(new Complex(-3, 4).sqrt().equals(new Complex(1, 2), 1e-15)).toBe(true);
    expect(new Complex(-3, -4).sqrt().equals(new Complex(1, -2), 1e-15)).toBe(true);
    // np.sqrt(-3.5+1.25j)
    const z = new Complex(-3.5, 1.25).sqrt();
    expect(z.re).toBeCloseTo(0.32902673662608817, 14);
    expect(z.im).toBeCloseTo(1.8995416798309042, 14);
  });

  it("sqrt handles zero, infinities and NaN like C99/NumPy", () => {
    expect(new Complex(0, 0).sqrt().equals(Complex.ZERO)).toBe(true);
    const a = new Complex(Number.POSITIVE_INFINITY, 1).sqrt();
    expect(a.re).toBe(Number.POSITIVE_INFINITY);
    expect(a.im).toBe(0);
    const b = new Complex(Number.NEGATIVE_INFINITY, 1).sqrt();
    expect(b.re).toBe(0);
    expect(b.im).toBe(Number.POSITIVE_INFINITY);
    const c = new Complex(1, Number.POSITIVE_INFINITY).sqrt();
    expect(c.re).toBe(Number.POSITIVE_INFINITY);
    expect(c.im).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(new Complex(1, Number.NaN).sqrt().re)).toBe(true);
  });

  it("sqrt does not overflow for huge magnitudes", () => {
    const z = new Complex(-1e308, 1e308).sqrt();
    expect(Number.isFinite(z.re)).toBe(true);
    expect(Number.isFinite(z.im)).toBe(true);
    expect(z.mul(z).re / -1e308).toBeCloseTo(1, 10);
  });

  it("sqrt stays accurate for subnormal and near-overflow inputs", () => {
    // cmath.sqrt(5e-324j) = 1.5717277847026288e-162 + 1.5717277847026288e-162j
    const tiny = new Complex(0, 5e-324).sqrt();
    expect(tiny.re / 1.5717277847026288e-162).toBeCloseTo(1, 12);
    expect(tiny.im / 1.5717277847026288e-162).toBeCloseTo(1, 12);
    // cmath.sqrt(1.7e308+1.7e308j): the modulus 2.4e308 overflows a plain hypot
    const huge = new Complex(1.7e308, 1.7e308).sqrt();
    expect(huge.re / 1.4325088230154573e154).toBeCloseTo(1, 12);
    expect(huge.im / 5.933645827121221e153).toBeCloseTo(1, 12);
    // cmath.sqrt(inf+nanj) = inf+nanj
    const inf = new Complex(Number.POSITIVE_INFINITY, Number.NaN).sqrt();
    expect(inf.re).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(inf.im)).toBe(true);
  });

  it("exp keeps a zero imaginary part exactly real", () => {
    // np.exp(inf+0j) = inf+0j
    const e = new Complex(Number.POSITIVE_INFINITY, 0).exp();
    expect(e.re).toBe(Number.POSITIVE_INFINITY);
    expect(e.im).toBe(0);
    // np.exp(1+2j)
    const z = new Complex(1, 2).exp();
    expect(z.re).toBeCloseTo(-1.1312043837568135, 14);
    expect(z.im).toBeCloseTo(2.4717266720048188, 14);
  });

  it("log keeps the real part accurate near the unit circle", () => {
    // np.log(1+1e-10j).real = 5.0000000000000005e-21 (log(|z|) would give 0)
    const near = new Complex(1, 1e-10).log();
    expect(near.re / 5.0000000000000005e-21).toBeCloseTo(1, 12);
    expect(near.im).toBeCloseTo(1e-10, 22);
    // np.log(0.9999999999+1e-5j).real = -5.000000827153708e-11
    const z = new Complex(0.9999999999, 1e-5).log();
    expect(z.re / -5.000000827153708e-11).toBeCloseTo(1, 12);
    // np.log(-2+3j)
    const w = new Complex(-2, 3).log();
    expect(w.re).toBeCloseTo(1.2824746787307684, 14);
    expect(w.im).toBeCloseTo(2.158798930342464, 14);
    // np.log(0) = -inf+0j
    expect(new Complex(0, 0).log().re).toBe(Number.NEGATIVE_INFINITY);
  });

  it("pow uses exact repeated multiplication for small integer exponents", () => {
    // np.power(1+2j, 2) = -3+4j exactly; (1+2j)**3 = -11-2j
    const sq = new Complex(1, 2).pow(2);
    expect(sq.re).toBe(-3);
    expect(sq.im).toBe(4);
    const cube = new Complex(1, 2).pow(3);
    expect(cube.re).toBe(-11);
    expect(cube.im).toBe(-2);
    // np.power(1+2j, -2) = -0.12-0.16j
    expect(new Complex(1, 2).pow(-2).equals(new Complex(-0.12, -0.16), 1e-15)).toBe(true);
    // np.power(1.5-0.5j, 7) = -15.5625-19.1875j (all dyadic, exact)
    const p7 = new Complex(1.5, -0.5).pow(7);
    expect(p7.re).toBe(-15.5625);
    expect(p7.im).toBe(-19.1875);
    // exponents of magnitude >= 100 take the log/exp path
    const p100 = new Complex(1.5, -0.5).pow(100);
    expect(p100.re / 5.722679833107513e19).toBeCloseTo(1, 10);
    // np.power(1+2j, 0.5)
    const half = new Complex(1, 2).pow(0.5);
    expect(half.re).toBeCloseTo(1.272019649514069, 14);
    expect(half.im).toBeCloseTo(0.7861513777574233, 14);
  });

  it("equals accepts tol = 0 and matching infinities", () => {
    const a = new Complex(1.5, -2);
    expect(a.equals(new Complex(1.5, -2), 0)).toBe(true);
    expect(a.equals(new Complex(1.5, -2.0000001), 0)).toBe(false);
    const inf = new Complex(Number.POSITIVE_INFINITY, 0);
    expect(inf.equals(new Complex(Number.POSITIVE_INFINITY, 0))).toBe(true);
    expect(new Complex(Number.NaN, 0).equals(new Complex(Number.NaN, 0))).toBe(false);
  });

  it("toString prints a sign before a NaN imaginary part", () => {
    expect(new Complex(1, Number.NaN).toString()).toBe("(1+NaNj)");
    expect(new Complex(1, -2).toString()).toBe("(1-2j)");
  });
});

describe("c37 Complex64Array / Complex128Array validation", () => {
  it("rejects invalid lengths", () => {
    expect(() => new Complex64Array(-1)).toThrow(InvalidParameterError);
    expect(() => new Complex64Array(1.5)).toThrow(InvalidParameterError);
    expect(() => new Complex128Array(Number.NaN)).toThrow(InvalidParameterError);
    expect(new Complex128Array(0).length).toBe(0);
  });

  it("throws on out-of-range element access instead of returning garbage", () => {
    const a = new Complex128Array(2);
    expect(() => a.getReal(2)).toThrow(IndexError);
    expect(() => a.getImag(-1)).toThrow(IndexError);
    expect(() => a.getComplex(1.5)).toThrow(IndexError);
    expect(() => a.setRI(5, 1, 1)).toThrow(IndexError);
    expect(() => a.setComplex(2, new Complex(1, 1))).toThrow(IndexError);
    const b = new Complex64Array(1);
    expect(() => b.getReal(1)).toThrow(IndexError);
    b.setRI(0, 1, 2);
    expect(b.getImag(0)).toBe(2);
  });

  it("fromInterleaved rejects an odd number of values", () => {
    expect(() => Complex64Array.fromInterleaved([1, 2, 3])).toThrow(InvalidParameterError);
    expect(() => Complex128Array.fromInterleaved([1])).toThrow(InvalidParameterError);
    const ok = Complex128Array.fromInterleaved(new Float64Array([1, 2, 3, 4]));
    expect(ok.getComplex(1).equals(new Complex(3, 4))).toBe(true);
  });

  it("fill follows TypedArray bounds (negative and clamped)", () => {
    const a = new Complex128Array(4);
    a.fill(new Complex(1, 1), -2);
    expect(a.getReal(1)).toBe(0);
    expect(a.getReal(2)).toBe(1);
    expect(a.getImag(3)).toBe(1);
    a.fill(new Complex(2, 2), 0, 100);
    expect(a.toComplexArray().every((z) => z.re === 2 && z.im === 2)).toBe(true);
  });

  it("set copies another complex array element-wise and checks bounds", () => {
    const src = Complex64Array.fromComplexArray([new Complex(1, 2), new Complex(3, 4)]);
    const dst = new Complex64Array(3);
    dst.set(src, 1);
    expect(dst.getComplex(0).equals(Complex.ZERO)).toBe(true);
    expect(dst.getComplex(1).equals(new Complex(1, 2))).toBe(true);
    expect(dst.getComplex(2).equals(new Complex(3, 4))).toBe(true);
    expect(() => dst.set(src, 2)).toThrow(IndexError);
    expect(() => dst.set([1, 2, 3])).toThrow(InvalidParameterError);
    const wide = new Complex128Array(2);
    wide.set(src);
    expect(wide.getComplex(1).equals(new Complex(3, 4))).toBe(true);
  });

  it("slice with fractional or NaN bounds stays valid", () => {
    const a = Complex128Array.fromComplexArray([
      new Complex(1, 0),
      new Complex(2, 0),
      new Complex(3, 0),
    ]);
    expect(a.slice(1).length).toBe(2);
    expect(a.slice(-2, -1).getReal(0)).toBe(2);
    expect(a.slice(2, 1).length).toBe(0);
  });
});

describe("c37 Float16Array / BFloat16Array fixes", () => {
  it("subarray reports the shared buffer window", () => {
    const arr = Float16Array.from([1, 2, 3, 4]);
    const sub = arr.subarray(1, 3);
    expect(sub.length).toBe(2);
    expect(sub.byteOffset).toBe(2);
    expect(sub.byteLength).toBe(4);
    expect(sub.buffer).toBe(arr.buffer);
    sub[0] = 9;
    expect(arr[1]).toBe(9);
    const bsub = BFloat16Array.from([1, 2, 3]).subarray(-2);
    expect(bsub.length).toBe(2);
    expect(bsub.byteOffset).toBe(2);
  });

  it("slice truncates fractional bounds like TypedArray.slice", () => {
    const arr = Float16Array.from([1, 2, 3, 4]);
    expect(arr.slice(0.5, 2.5).toArray()).toEqual([1, 2]);
    expect(arr.slice(Number.NaN).toArray()).toEqual([1, 2, 3, 4]);
    expect(arr.slice(-1.5).toArray()).toEqual([4]);
    expect(BFloat16Array.from([1, 2, 3]).slice(1.9, 3).toArray()).toEqual([2, 3]);
  });

  it("set throws when the values do not fit (like TypedArray.set)", () => {
    const arr = new Float16Array(3);
    expect(() => arr.set([1, 2, 3, 4])).toThrow(IndexError);
    expect(() => arr.set([1, 2], 2)).toThrow(IndexError);
    expect(() => arr.set([1], -1)).toThrow(IndexError);
    arr.set([1.5, 2.5], 1);
    expect(arr.toArray()).toEqual([0, 1.5, 2.5]);
    const b = new BFloat16Array(2);
    expect(() => b.set([1, 2, 3])).toThrow(IndexError);
  });

  it("validates the numeric length and coerces holes to NaN", () => {
    expect(() => new Float16Array(-1)).toThrow(InvalidParameterError);
    expect(() => new BFloat16Array(2.5)).toThrow(InvalidParameterError);
    // TypedArray semantics: undefined becomes NaN, not -Infinity
    const holes = new Float16Array([1, undefined as unknown as number]);
    expect(holes[0]).toBe(1);
    expect(Number.isNaN(holes[1])).toBe(true);
    expect(Float16Array.BYTES_PER_ELEMENT).toBe(2);
    expect(BFloat16Array.BYTES_PER_ELEMENT).toBe(2);
  });

  it("at truncates fractional indices", () => {
    const arr = Float16Array.from([1, 2, 3]);
    expect(arr.at(1.9)).toBe(2);
    expect(arr.at(-1.5)).toBe(3);
    expect(arr.at(Number.NaN)).toBe(1);
  });

  it("round helpers match NumPy float16 and torch bfloat16", () => {
    // np.float16(0.1) = 0.0999755859375; np.float16(65519.999) = 65504; np.float16(1e-8) = 0
    expect(roundToFloat16(0.1)).toBe(0.0999755859375);
    expect(roundToFloat16(65519.999)).toBe(65504);
    expect(roundToFloat16(70000)).toBe(Number.POSITIVE_INFINITY);
    expect(roundToFloat16(1e-8)).toBe(0);
    // torch.tensor([0.1, 1/3], dtype=torch.bfloat16)
    expect(roundToBFloat16(0.1)).toBe(0.10009765625);
    expect(roundToBFloat16(1 / 3)).toBe(0.333984375);
  });
});

describe("c37 half-precision tensor creation and astype", () => {
  it("tensor() rounds values onto the float16 / bfloat16 grid", () => {
    expect(tensor([0.1, 70000, 1e-8], { dtype: "float16" }).toArray()).toEqual([
      0.0999755859375,
      Number.POSITIVE_INFINITY,
      0,
    ]);
    expect(tensor([0.1, 1 / 3], { dtype: "bfloat16" }).toArray()).toEqual([
      0.10009765625, 0.333984375,
    ]);
  });

  it("tensor() accepts a Float32Array for half dtypes without mutating it", () => {
    const raw = new Float32Array([0.1, 1 / 3]);
    const before = Array.from(raw);
    const t = tensor(raw, { dtype: "float16" });
    expect(t.dtype).toBe("float16");
    expect(t.toArray()).toEqual([0.0999755859375, 0.333251953125]);
    expect(Array.from(raw)).toEqual(before);
    expect(() => tensor(new Float64Array([1]), { dtype: "float16" })).toThrow(DTypeError);
  });

  it("astype rounds to half precision", () => {
    // np.array([0.1, 1/3]).astype(np.float16)
    const t = tensor([0.1, 1 / 3], { dtype: "float64" });
    expect(t.astype("float16").toArray()).toEqual([0.0999755859375, 0.333251953125]);
    expect(t.astype("bfloat16").toArray()).toEqual([0.10009765625, 0.333984375]);
  });

  it("astype rounds float64 to float16 once, without a float32 detour", () => {
    // np.array([65519.999]).astype(np.float16) = 65504 (float32 would round to 65520 first)
    expect(tensor([65519.999], { dtype: "float64" }).astype("float16").toArray()).toEqual([65504]);
    expect(tensor([65520], { dtype: "float64" }).astype("float16").toArray()).toEqual([
      Number.POSITIVE_INFINITY,
    ]);
    expect(tensor(["1.5", "2.5"]).astype("float16").toArray()).toEqual([1.5, 2.5]);
    expect(tensor([3, 4], { dtype: "int64" }).astype("bfloat16").toArray()).toEqual([3, 4]);
  });

  it("linspace, arange, full and randn honor half precision", () => {
    expect(linspace(0, 1, 3, true, { dtype: "float16" }).toArray()).toEqual([0, 0.5, 1]);
    expect(arange(0, 0.3, 0.1, { dtype: "float16" }).toArray()).toEqual([
      0, 0.0999755859375, 0.199951171875,
    ]);
    expect(full([2], 0.1, { dtype: "float16" }).toArray()).toEqual([
      0.0999755859375, 0.0999755859375,
    ]);
    const r = randn([64], { dtype: "float16" }).toArray() as number[];
    expect(r.every((v) => roundToFloat16(v) === v)).toBe(true);
  });
});

describe("c37 astype conversions", () => {
  it("prints the shortest round-trip text for float32 / float16", () => {
    // np.array([0.1, 1/3], dtype=np.float32).astype(str) = ['0.1', '0.33333334']
    expect(
      tensor([0.1, 1 / 3])
        .astype("string")
        .toArray()
    ).toEqual(["0.1", "0.33333334"]);
    // np.array([0.1], dtype=np.float16).astype(str) = ['0.1']
    expect(tensor([0.1], { dtype: "float16" }).astype("string").toArray()).toEqual(["0.1"]);
    expect(tensor([2, -3.5], { dtype: "float32" }).astype("string").toArray()).toEqual([
      "2",
      "-3.5",
    ]);
  });

  it("parses integer strings to int64 exactly and rejects bad values", () => {
    expect(tensor(["9007199254740993", " -7 "]).astype("int64").toArray()).toEqual([
      9007199254740993n,
      -7n,
    ]);
    expect(tensor(["3.9"]).astype("int64").toArray()).toEqual([3n]);
    expect(() => tensor(["abc"]).astype("int64")).toThrow(DTypeError);
    expect(() => tensor(["9223372036854775808"]).astype("int64")).toThrow(DTypeError);
  });

  it("throws instead of wrapping values outside the int64 range", () => {
    expect(() => tensor([1e30], { dtype: "float64" }).astype("int64")).toThrow(DTypeError);
    expect(() => tensor([1e30], { dtype: "int64" })).toThrow(DTypeError);
    expect(
      tensor([-(2 ** 63)], { dtype: "float64" })
        .astype("int64")
        .toArray()
    ).toEqual([-9223372036854775808n]);
  });

  it("rejects complex targets with a clear error", () => {
    expect(() => tensor([1, 2]).astype("complex64")).toThrow(/Complex64Array/);
    expect(() => zeros([2], { dtype: "complex128" })).toThrow(DTypeError);
    expect(() => tensor([1], { dtype: "complex64" })).toThrow(DTypeError);
  });

  it("converts strided views in logical order", () => {
    const v = transpose(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float64" }
      )
    );
    expect(v.astype("int32").toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("c37 strided copies", () => {
  it("reshape of a permuted 3-D view matches NumPy's logical order", () => {
    // np.arange(24).reshape(2,3,4).transpose(2,0,1).reshape(-1)
    const a = tensor(Array.from({ length: 24 }, (_, i) => i)).reshape([2, 3, 4]);
    const p = transpose(a, [2, 0, 1]);
    expect(p.reshape([-1]).toArray()).toEqual([
      0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23,
    ]);
    expect(p.astype("int32").reshape([-1]).toArray()).toEqual(p.reshape([-1]).toArray());
    const s = tensor(Array.from({ length: 6 }, (_, i) => `s${i}`)).reshape([2, 3]);
    expect(transpose(s).reshape([-1]).toArray()).toEqual(["s0", "s3", "s1", "s4", "s2", "s5"]);
    const big = tensor([1, 2, 3, 4, 5, 6], { dtype: "int64" }).reshape([2, 3]);
    expect(transpose(big).reshape([6]).toArray()).toEqual([1n, 4n, 2n, 5n, 3n, 6n]);
  });

  it("astype reads from the tensor's offset", () => {
    const data = new Float64Array([9, 9, 1.5, 2.5, -3.5]);
    const t = Tensor.fromTypedArray({
      data,
      shape: [2],
      dtype: "float64",
      device: "cpu",
      offset: 2,
    });
    expect(t.astype("int32").toArray()).toEqual([1, 2]);
    expect(t.astype("float32").toArray()).toEqual([1.5, 2.5]);
    expect(t.astype("float16").toArray()).toEqual([1.5, 2.5]);
    expect(t.astype("bool").toArray()).toEqual([1, 1]);
    expect(t.astype("int64").toArray()).toEqual([1n, 2n]);
    expect(t.astype("string").toArray()).toEqual(["1.5", "2.5"]);
  });

  it("0-D and empty tensors survive reshape and astype", () => {
    expect(tensor(3.5).astype("int32").item()).toBe(3);
    expect(tensor(3.5).reshape([1, 1]).toArray()).toEqual([[3.5]]);
    expect(zeros([0, 3]).astype("float64").shape).toEqual([0, 3]);
    expect(transpose(zeros([0, 3])).reshape([0]).shape).toEqual([0]);
  });
});

describe("c37 Tensor.reshape / slice / constructor validation", () => {
  it("reshape infers one -1 dimension like NumPy", () => {
    const t = tensor([1, 2, 3, 4, 5, 6]);
    expect(t.reshape([2, -1]).shape).toEqual([2, 3]);
    expect(t.reshape([-1]).shape).toEqual([6]);
    expect(t.reshape([-1, 1]).shape).toEqual([6, 1]);
    expect(() => t.reshape([-1, -1])).toThrow(ShapeError);
    expect(() => t.reshape([4, -1])).toThrow(ShapeError);
    expect(() => t.reshape([-2, 3])).toThrow(DataValidationError);
  });

  it("reshape of a non-contiguous view copies in logical order", () => {
    const m = transpose(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(m.reshape([-1]).toArray()).toEqual([1, 4, 2, 5, 3, 6]);
    expect(m.flatten().toArray()).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("slice rejects non-integer indices and bounds", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => t.slice(1.5)).toThrow(InvalidParameterError);
    expect(() => t.slice(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => t.slice({ start: 0.5 })).toThrow(InvalidParameterError);
    expect(() => t.slice({ end: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => t.slice(5)).toThrow(IndexError);
  });

  it("slice accepts infinite bounds as 'to the edge'", () => {
    const t = tensor([1, 2, 3, 4], { dtype: "float64" });
    expect(
      t.slice({ start: Number.NEGATIVE_INFINITY, end: Number.POSITIVE_INFINITY }).toArray()
    ).toEqual([1, 2, 3, 4]);
    expect(t.slice({ start: Number.POSITIVE_INFINITY, step: -1 }).toArray()).toEqual([4, 3, 2, 1]);
  });

  it("slice supports 0-D tensors and empty results", () => {
    expect(tensor(7).slice().item()).toBe(7);
    expect(tensor([1, 2, 3]).slice({ start: 2, end: 1 }).shape).toEqual([0]);
  });

  it("fromTypedArray requires an integer offset", () => {
    const data = new Float32Array(4);
    expect(() =>
      Tensor.fromTypedArray({ data, shape: [2], dtype: "float32", device: "cpu", offset: 1.5 })
    ).toThrow(InvalidParameterError);
    expect(() =>
      Tensor.fromTypedArray({
        data,
        shape: [2],
        dtype: "float32",
        device: "cpu",
        offset: Number.NaN,
      })
    ).toThrow(InvalidParameterError);
    const ok = Tensor.fromTypedArray({
      data,
      shape: [2],
      dtype: "float32",
      device: "cpu",
      offset: 2,
    });
    expect(ok.size).toBe(2);
  });
});

describe("c37 creation functions (references: NumPy 2.4)", () => {
  it("linspace floors integer dtypes and ends exactly on stop", () => {
    // np.linspace(-5, 5, 4, dtype=int) = [-5, -2, 1, 5]
    expect(linspace(-5, 5, 4, true, { dtype: "int32" }).toArray()).toEqual([-5, -2, 1, 5]);
    expect(linspace(-5, 5, 4, true, { dtype: "int64" }).toArray()).toEqual([-5n, -2n, 1n, 5n]);
    // np.linspace(0.1, 0.7, 7)[-1] == 0.7
    const x = linspace(0.1, 0.7, 7, true, { dtype: "float64" }).toArray() as number[];
    expect(x[6]).toBe(0.7);
    expect(x[0]).toBe(0.1);
  });

  it("linspace matches NumPy for num = 1 and endpoint = false", () => {
    expect(linspace(3, 9, 1).toArray()).toEqual([3]);
    expect(linspace(0, 1, 1, false).toArray()).toEqual([0]);
    const open = linspace(0, 1, 7, false, { dtype: "float64" }).toArray() as number[];
    const ref = [
      0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714,
      0.7142857142857142, 0.8571428571428571,
    ];
    open.forEach((v, i) => {
      expect(v).toBeCloseTo(ref[i] as number, 15);
    });
    expect(linspace(0, 1, 0).shape).toEqual([0]);
  });

  it("linspace validates num", () => {
    expect(() => linspace(0, 1, 2.5)).toThrow(InvalidParameterError);
    expect(() => linspace(0, 1, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => linspace(0, 1, -1)).toThrow(InvalidParameterError);
    expect(() => logspace(0, 1, 1.5)).toThrow(InvalidParameterError);
    expect(() => geomspace(1, 10, Number.NaN)).toThrow(InvalidParameterError);
  });

  it("logspace truncates integer dtypes like NumPy", () => {
    // np.logspace(0, 2, 5, dtype=int) = [1, 3, 10, 31, 100]
    expect(logspace(0, 2, 5, 10, true, { dtype: "int64" }).toArray()).toEqual([
      1n,
      3n,
      10n,
      31n,
      100n,
    ]);
    expect(logspace(0, 3, 4, 10, true, { dtype: "int32" }).toArray()).toEqual([1, 10, 100, 1000]);
    // np.logspace(-1, 1, 5, base=2.0)
    const v = logspace(-1, 1, 5, 2, true, { dtype: "float64" }).toArray() as number[];
    const ref = [0.5, Math.SQRT1_2, 1.0, Math.SQRT2, 2.0];
    v.forEach((x, i) => {
      expect(x).toBeCloseTo(ref[i] as number, 15);
    });
  });

  it("geomspace hits both endpoints exactly", () => {
    // np.geomspace(1, 1000, 4) = [1, 10, 100, 1000]; ints too
    const g = geomspace(1, 1000, 4, true, { dtype: "float64" }).toArray() as number[];
    expect(g[0]).toBe(1);
    expect(g[3]).toBe(1000);
    expect(geomspace(1, 1000, 4, true, { dtype: "int32" }).toArray()).toEqual([1, 10, 100, 1000]);
    // np.geomspace(2, 200, 5)
    const h = geomspace(2, 200, 5, true, { dtype: "float64" }).toArray() as number[];
    const ref = [2.0, 6.32455532033676, 20.000000000000004, 63.24555320336759, 200.0];
    h.forEach((x, i) => {
      expect(x).toBeCloseTo(ref[i] as number, 12);
    });
    expect(h[4]).toBe(200);
    // np.geomspace(-1, -1000, 4) = [-1, -10, -100, -1000]
    expect(geomspace(-1, -1000, 4, true, { dtype: "float64" }).toArray()).toEqual([
      -1, -10, -100, -1000,
    ]);
    // endpoint = false excludes stop but still starts exactly on start
    const open = geomspace(3, 300, 4, false, { dtype: "float64" }).toArray() as number[];
    expect(open[0]).toBe(3);
    expect(open[3]).toBeLessThan(300);
  });

  it("geomspace rejects non-finite endpoints", () => {
    expect(() => geomspace(1, Number.POSITIVE_INFINITY, 3)).toThrow(InvalidParameterError);
    expect(() => geomspace(Number.NaN, 1, 3)).toThrow(InvalidParameterError);
    expect(() => geomspace(0, 1, 3)).toThrow(InvalidParameterError);
    expect(() => geomspace(-1, 1, 3)).toThrow(InvalidParameterError);
  });

  it("arange rejects non-finite arguments with a clear error", () => {
    expect(() => arange(0, Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(() => arange(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => arange(0, 5, Number.NaN)).toThrow(InvalidParameterError);
    // np.arange(0, 1, 0.1).size == 10, np.arange(1, 0, -0.25) = [1, .75, .5, .25]
    expect(arange(0, 1, 0.1, { dtype: "float64" }).size).toBe(10);
    expect(arange(1, 0, -0.25).toArray()).toEqual([1, 0.75, 0.5, 0.25]);
    expect(arange(3, 3).shape).toEqual([0]);
  });

  it("arange to int64 checks the range instead of wrapping", () => {
    expect(arange(3, undefined, 1, { dtype: "int64" }).toArray()).toEqual([0n, 1n, 2n]);
    expect(() => arange(2 ** 63, 2 ** 63 + 4096, 2048, { dtype: "int64" })).toThrow(DTypeError);
  });

  it("full converts the value like tensor() does", () => {
    // bool tensors only ever hold 0/1
    expect(full([3], 5, { dtype: "bool" }).toArray()).toEqual([1, 1, 1]);
    expect(full([2], Number.NaN, { dtype: "bool" }).toArray()).toEqual([1, 1]);
    expect(full([2], 0, { dtype: "bool" }).toArray()).toEqual([0, 0]);
    expect(full([2], true).toArray()).toEqual([1, 1]);
    expect(full([2], 7, { dtype: "int64" }).toArray()).toEqual([7n, 7n]);
    expect(() => full([2], 1.5, { dtype: "int64" })).toThrow(DTypeError);
    expect(() => full([1], "x", { dtype: "int32" })).toThrow(DTypeError);
    expect(() => full([1], 1, { dtype: "string" })).toThrow(DTypeError);
    expect(full([2], "x", { dtype: "string" }).toArray()).toEqual(["x", "x"]);
  });

  it("tensor() maps NaN to true for bool, like astype and NumPy", () => {
    // np.array([np.nan, 0.0, 2.0]).astype(bool) = [True, False, True]
    expect(tensor([Number.NaN, 0, 2], { dtype: "bool" }).toArray()).toEqual([1, 0, 1]);
  });

  it("empty() string tensors hold empty strings, not holes", () => {
    const e = empty([2, 2], { dtype: "string" });
    expect(e.toArray()).toEqual([
      ["", ""],
      ["", ""],
    ]);
    expect(zeros([2], { dtype: "string" }).toArray()).toEqual(["", ""]);
  });

  it("eye builds off-diagonals and rejects bad arguments", () => {
    // np.eye(3, 4, 1), np.eye(3, 3, -1), np.eye(2, 3, 5)
    expect(eye(3, 4, 1).toArray()).toEqual([
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);
    expect(eye(3, 3, -1).toArray()).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    expect(eye(2, 3, 5).toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(eye(2, 2, 0, { dtype: "int64" }).toArray()).toEqual([
      [1n, 0n],
      [0n, 1n],
    ]);
    expect(() => eye(3, 3, 0.5)).toThrow(InvalidParameterError);
    expect(() => eye(2, 2, 0, { dtype: "string" })).toThrow(DTypeError);
    expect(eye(0).shape).toEqual([0, 0]);
  });

  it("randn refuses bool and stays reproducible with a seed", async () => {
    expect(() => randn([2], { dtype: "bool" })).toThrow(DTypeError);
    const { setSeed } = await import("../../src/core");
    setSeed(7);
    const a = randn([5], { dtype: "float64" }).toArray();
    setSeed(7);
    const b = randn([5], { dtype: "float64" }).toArray();
    expect(a).toEqual(b);
  });

  it("ones() keeps existing behavior for numeric and string dtypes", () => {
    expect(ones([2], { dtype: "int64" }).toArray()).toEqual([1n, 1n]);
    expect(ones([1], { dtype: "string" }).toArray()).toEqual(["1"]);
  });
});

describe("c37 tensor() dtype inference follows the configured default", () => {
  afterAll(() => {
    resetConfig();
  });

  it("uses defaultDtype for numeric input", () => {
    expect(tensor([1, 2]).dtype).toBe("float32");
    setConfig({ defaultDtype: "float64" });
    expect(tensor([1.1, 2]).dtype).toBe("float64");
    expect(tensor([]).dtype).toBe("float64");
    expect(tensor([true]).dtype).toBe("bool");
    expect(tensor(["a"]).dtype).toBe("string");
    expect(linspace(0, 1, 1).dtype).toBe("float64");
    expect(tensor([1], { dtype: "int32" }).dtype).toBe("int32");
    resetConfig();
    expect(tensor([1.1]).dtype).toBe("float32");
  });
});

describe("c37 device tensors (minimal in-process kernel backend)", () => {
  type Buf = DeviceBuffer & { data: Float32Array };
  let freed = 0;

  const stub = (): never => {
    throw new DeviceError("stub kernel backend: op not implemented");
  };

  const fake = {
    info: () => ({
      device: "webgpu",
      name: "c37 fake",
      available: true,
      capabilities: [],
    }),
    supports: () => false,
    init: async () => {},
    dispose: () => {},
    upload(data: Float32Array): Buf {
      const copy = new Float32Array(data);
      return { device: "webgpu", byteLength: copy.byteLength, size: copy.length, data: copy };
    },
    async download(buffer: DeviceBuffer): Promise<Float32Array> {
      return new Float32Array((buffer as Buf).data);
    },
    free(): void {
      freed++;
    },
    fill: stub,
    binary: stub,
    unary: stub,
    matmul: stub,
    reduce: stub,
    reduceAxis: stub,
    matmulBatched: stub,
    ternary: stub,
    im2col: stub,
    col2im: stub,
    pool2d: stub,
    pool2dBackward: stub,
  } as unknown as KernelBackend & Backend;

  beforeAll(() => {
    registerBackend("webgpu", fake);
  });

  afterAll(() => {
    resetConfig();
  });

  it("an empty slice of a strided device view keeps a valid offset", () => {
    const t = tensor([1, 2, 3, 4, 5, 6], { device: "webgpu" });
    // shape [2] with stride 5 reaches offset 5 of 6; slicing from index 2 of an
    // axis with stride 5 would otherwise land at offset 10 (> buffer size).
    const strided = t.slice({ start: 0, step: 5 });
    expect(strided.strides).toEqual([5]);
    const empty2 = strided.slice({ start: 2, end: 2 });
    expect(empty2.shape).toEqual([0]);
    expect(empty2.offset).toBeLessThanOrEqual(6);
  });

  it("using a disposed tensor throws DeviceError from every entry point", async () => {
    const t = tensor([1, 2, 3, 4], { device: "webgpu" });
    const view = t.reshape([2, 2]);
    const before = freed;
    t.dispose();
    t.dispose();
    expect(freed).toBe(before);
    expect(() => t.reshape([4])).toThrow(DeviceError);
    expect(() => t.slice({ start: 0, end: 1 })).toThrow(DeviceError);
    expect(() => t.view([4])).toThrow(DeviceError);
    expect(() => t.astype("float64")).toThrow(DeviceError);
    expect(() => t.deviceBuffer).toThrow(DeviceError);
    await expect(t.to("cpu")).rejects.toThrow(DeviceError);
    // the view still owns the allocation
    expect((await view.cpu()).toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    view.dispose();
    expect(freed).toBe(before + 1);
  });

  it("astype on a live device tensor asks for a CPU transfer", () => {
    const t = tensor([1, 2], { device: "webgpu" });
    expect(() => t.astype("float64")).toThrow(/cpu\(\)/);
  });
});
