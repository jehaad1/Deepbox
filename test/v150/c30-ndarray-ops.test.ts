/**
 * Regression tests for the 1.5.0 audit of src/ndarray/ops
 * (_internal, activation, arithmetic, broadcast, comparison).
 *
 * Reference values were computed with NumPy 2.4 / PyTorch 2.12 / mpmath.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Backend, DeviceBuffer, KernelLayout } from "../../src/core";
import {
  DataValidationError,
  DeviceError,
  DTypeError,
  InvalidParameterError,
  registerBackend,
  resetConfig,
  ShapeError,
  unregisterBackend,
} from "../../src/core";
import { tensor, transpose } from "../../src/ndarray";
import { bigintToNumberSafe, readNumbers } from "../../src/ndarray/ops/_internal";
import {
  elu,
  gelu,
  hardtanh,
  leakyRelu,
  logSoftmax,
  mish,
  relu,
  sigmoid,
  softmax,
  softplus,
  swish,
  tanhshrink,
} from "../../src/ndarray/ops/activation";
import {
  abs,
  add,
  addScalar,
  clip,
  div,
  floorDiv,
  maximum,
  minimum,
  mod,
  mul,
  mulScalar,
  neg,
  pow,
  reciprocal,
  sign,
  sub,
} from "../../src/ndarray/ops/arithmetic";
import { canBroadcast, getBroadcastShape } from "../../src/ndarray/ops/broadcast";
import {
  allclose,
  arrayEqual,
  equal,
  greater,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
} from "../../src/ndarray/ops/comparison";
import { Tensor } from "../../src/ndarray/tensor/Tensor";

const i32 = (data: number[] | number[][]): Tensor => tensor(data, { dtype: "int32" });
const i64 = (values: bigint[]): Tensor =>
  Tensor.fromTypedArray({
    data: new BigInt64Array(values),
    shape: [values.length],
    dtype: "int64",
    device: "cpu",
  });
const f64 = (data: number[] | number[][]): Tensor => tensor(data, { dtype: "float64" });
const f32 = (data: number[] | number[][]): Tensor => tensor(data, { dtype: "float32" });

/** Read an int64 tensor back as bigint[]. */
const bigValues = (t: Tensor): bigint[] => Array.from(t.data as BigInt64Array);

describe("c30 broadcast helpers", () => {
  it("getBroadcastShape follows NumPy rules", () => {
    expect(getBroadcastShape([3, 1], [1, 4])).toEqual([3, 4]);
    expect(getBroadcastShape([5], [2, 3, 5])).toEqual([2, 3, 5]);
    expect(getBroadcastShape([], [2, 2])).toEqual([2, 2]);
    expect(getBroadcastShape([0, 3], [1, 3])).toEqual([0, 3]);
    expect(getBroadcastShape([0], [0])).toEqual([0]);
    expect(() => getBroadcastShape([0, 3], [2, 3])).toThrow(ShapeError);
    expect(() => getBroadcastShape([2, 3], [4, 3])).toThrow(ShapeError);
  });

  it("canBroadcast agrees with getBroadcastShape", () => {
    expect(canBroadcast([3, 1], [1, 4])).toBe(true);
    expect(canBroadcast([0], [2])).toBe(false);
    expect(canBroadcast([0], [1])).toBe(true);
    expect(canBroadcast([2, 3], [3, 2])).toBe(false);
  });
});

describe("c30 internal helpers", () => {
  it("bigintToNumberSafe reports the offending value", () => {
    expect(bigintToNumberSafe(2n ** 53n - 1n)).toBe(Number.MAX_SAFE_INTEGER);
    expect(() => bigintToNumberSafe(2n ** 60n)).toThrow(DataValidationError);
    expect(() => bigintToNumberSafe(-(2n ** 60n))).toThrow(/1152921504606846976/);
  });

  it("readNumbers gathers strided views and converts int64", () => {
    const t = transpose(
      f64([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(Array.from(readNumbers(t, "test"))).toEqual([1, 4, 2, 5, 3, 6]);
    expect(Array.from(readNumbers(i64([1n, -2n]), "test"))).toEqual([1, -2]);
    expect(() => readNumbers(i64([2n ** 62n]), "test")).toThrow(DataValidationError);
    expect(Array.from(readNumbers(i64([2n ** 62n]), "test", false))).toEqual([2 ** 62]);
    expect(() => readNumbers(tensor(["a"]), "test")).toThrow(DTypeError);
  });
});

describe("c30 int32 multiplication wraps exactly", () => {
  const a = i32([
    [2147483647, -2147483648],
    [65536, 123456789],
  ]);

  it("contiguous same-shape path", () => {
    const x = i32([2147483647, -2147483648, 65536, 123456789]);
    const y = i32([2147483647, -1, 65536, 987654321]);
    // NumPy: int32 product wraps modulo 2^32
    expect(mul(x, y).toArray()).toEqual([1, -2147483648, 0, -67153019]);
  });

  it("broadcast path", () => {
    expect(mul(a, i32([2147483647, -1])).toArray()).toEqual([
      [1, -2147483648],
      [-65536, -123456789],
    ]);
  });

  it("strided same-shape path", () => {
    expect(mul(transpose(a), a).toArray()).toEqual([
      [1, 0],
      [0, -1757895751],
    ]);
  });

  it("mulScalar / addScalar keep int32 and wrap", () => {
    const m = mulScalar(a, 2147483647);
    expect(m.dtype).toBe("int32");
    expect(m.toArray()).toEqual([
      [1, -2147483648],
      [-65536, 2024026859],
    ]);
    expect(mulScalar(a, 100000).toArray()).toEqual([
      [-100000, 0],
      [-2036334592, 1942891296],
    ]);
    expect(addScalar(a, 2147483647).toArray()).toEqual([
      [-2, -1],
      [-2147418113, -2024026860],
    ]);
  });
});

describe("c30 integer power is exact and wraps", () => {
  it("int32", () => {
    const base = i32([7, 3, -3, 2, 10]);
    const exp = i32([20, 30, 31, 31, 12]);
    expect(pow(base, exp).toArray()).toEqual([
      -1199696159, -1010140999, -1264544299, -2147483648, -727379968,
    ]);
  });

  it("uint8", () => {
    const base = tensor([3, 2, 5], { dtype: "uint8" });
    const exp = tensor([7, 9, 4], { dtype: "uint8" });
    expect(pow(base, exp).toArray()).toEqual([139, 0, 113]);
  });

  it("int64 wraps modulo 2^64 without building huge BigInts", () => {
    const r = pow(i64([3n, -2n, 7n]), i64([50n, 63n, 30n]));
    expect(bigValues(r)).toEqual([
      6048575297968530377n,
      -9223372036854775808n,
      1576789505350337489n,
    ]);
    // A huge exponent returns promptly (square-and-multiply with wrap).
    expect(bigValues(pow(i64([1n]), i64([10n ** 12n])))).toEqual([1n]);
  });

  it("negative integer exponents promote to float32", () => {
    const r = pow(i32([2, 4]), i32([-1, 2]));
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([0.5, 16]);
    const r64 = pow(i64([2n]), i64([-2n]));
    expect(r64.dtype).toBe("float32");
    expect(r64.toArray()).toEqual([0.25]);
  });

  it("float64 cube matches NumPy (no one-ulp error from v*v*v)", () => {
    const x = f64([0.1, 1.1, -2.3, 3.7]);
    const r = pow(x, tensor(3, { dtype: "float64" }));
    expect(Array.from(r.data as Float64Array)).toEqual([
      0.0010000000000000002, 1.3310000000000004, -12.166999999999998, 50.653000000000006,
    ]);
  });

  it("float32 cube", () => {
    const r = pow(f32([0.1, 1.1, -2.3]), tensor(3, { dtype: "float32" }));
    expect(Array.from(r.data as Float32Array)).toEqual([
      0.0010000000474974513, 1.3310000896453857, -12.166998863220215,
    ]);
  });
});

describe("c30 floorDiv and mod", () => {
  const a = [-7, 7, -7, 7, 1, 5.5, -5.5, 1e17, 0.3];
  const b = [2, -2, -2, 2, 0.1, 0.5, 0.5, 3, 0.1];

  it("float floorDiv matches NumPy (1 // 0.1 is 9)", () => {
    expect(floorDiv(f64(a), f64(b)).toArray()).toEqual([
      -4, -4, 3, 3, 9, 11, -11, 3.3333333333333332e16, 2,
    ]);
  });

  it("float mod uses the exact remainder", () => {
    const r = mod(f64(a), f64(b)).toArray() as number[];
    expect(r.slice(0, 4)).toEqual([1, -1, -1, 1]);
    expect(r[4]).toBe(0.09999999999999995);
    expect(r[5]).toBe(0);
    expect(r[6]).toBe(0);
    // a - floor(a / b) * b would give 0 here
    expect(r[7]).toBe(1);
    expect(r[8]).toBe(0.09999999999999998);
  });

  it("quotients near 2^52 follow NumPy's remainder-based result", () => {
    // The rounded quotient is x.5 here, but NumPy (and Python) return the
    // fmod-corrected value, one below floor(a / b).
    const x = f64([-4398410523.621062, 2230571266.082552, 4577175622.58921]);
    const y = f64([0.0000011341374277606595, -6.597472441412466e-7, 0.0000012669360951157922]);
    expect(floorDiv(x, y).toArray()).toEqual([
      -3878198898969124, -3380948212956848, 3612791237249322,
    ]);
    expect(mod(x, y).toArray()).toEqual([
      7.284730967937483e-7, -4.897831581910365e-7, 6.156403476279123e-7,
    ]);
  });

  it("float32 floorDiv rounds its intermediates like NumPy's float32 kernel", () => {
    // np.floor_divide(float32(2162.3251953125), float32(-0.00038725545164197683))
    const r = floorDiv(f32([2162.3251953125, 1, 7]), f32([-0.00038725545164197683, 0.1, 2]));
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([-5583719, 9, 3]);
  });

  it("infinite operands and signed zeros", () => {
    const x = f64([5, -5, 5, Infinity]);
    const y = f64([Infinity, Infinity, -Infinity, 3]);
    expect(floorDiv(x, y).toArray()).toEqual([0, -1, -1, Number.NaN]);
    expect(mod(x, y).toArray()).toEqual([5, Infinity, -Infinity, Number.NaN]);
    const z = mod(f64([-0, 0]), f64([3, -3])).data as Float64Array;
    expect(Object.is(z[0], 0)).toBe(true);
    expect(Object.is(z[1], -0)).toBe(true);
    expect(Object.is((floorDiv(f64([-0]), f64([3])).data as Float64Array)[0], -0)).toBe(true);
  });

  it("float division by zero follows IEEE", () => {
    expect(floorDiv(f64([1, -1, 0]), f64([0, 0, 0])).toArray()).toEqual([
      Infinity,
      -Infinity,
      Number.NaN,
    ]);
    expect(mod(f64([5]), f64([0])).toArray()).toEqual([Number.NaN]);
  });

  it("int32 floorDiv/mod: sign of divisor, zero divisor gives 0", () => {
    const x = i32([-7, 7, -7, 7, 5]);
    const y = i32([2, -2, -2, 2, 0]);
    expect(floorDiv(x, y).toArray()).toEqual([-4, -4, 3, 3, 0]);
    expect(mod(x, y).toArray()).toEqual([1, -1, -1, 1, 0]);
  });

  it("int64 floorDiv/mod do not throw on zero", () => {
    expect(bigValues(floorDiv(i64([7n, -7n, 5n]), i64([2n, 2n, 0n])))).toEqual([3n, -4n, 0n]);
    expect(bigValues(mod(i64([7n, -7n, 5n]), i64([-2n, 2n, 0n])))).toEqual([-1n, 1n, 0n]);
  });

  it("broadcasts and handles strided operands", () => {
    const m = f64([
      [7, 8],
      [9, 10],
    ]);
    expect(floorDiv(m, f64([3, 4])).toArray()).toEqual([
      [2, 2],
      [3, 2],
    ]);
    expect(mod(transpose(m), f64([[3], [4]])).toArray()).toEqual([
      [1, 0],
      [0, 2],
    ]);
    expect(floorDiv(m, tensor(2, { dtype: "float64" })).toArray()).toEqual([
      [3, 4],
      [4, 5],
    ]);
  });
});

describe("c30 add / sub", () => {
  it("add and sub handle strided, broadcast and scalar operands", () => {
    const m = f64([
      [1, 2],
      [3, 4],
    ]);
    expect(add(transpose(m), m).toArray()).toEqual([
      [2, 5],
      [5, 8],
    ]);
    expect(sub(m, f64([1, 1])).toArray()).toEqual([
      [0, 1],
      [2, 3],
    ]);
    expect(sub(tensor(10, { dtype: "float64" }), m).toArray()).toEqual([
      [9, 8],
      [7, 6],
    ]);
    expect(add(i32([2147483647]), i32([1])).toArray()).toEqual([-2147483648]);
    expect(bigValues(add(i64([1n]), i64([2n])))).toEqual([3n]);
    expect(bigValues(sub(i64([1n]), i64([2n])))).toEqual([-1n]);
  });

  it("bool add is logical OR and bool sub is rejected", () => {
    const a = tensor([true, false, true]);
    const b = tensor([true, false, false]);
    expect(add(a, b).toArray()).toEqual([1, 0, 1]);
    expect(() => sub(a, b)).toThrow(DTypeError);
  });

  it("mixed dtypes promote and mismatched shapes throw typed errors", () => {
    const mixed = add(f64([1]), f32([1]));
    expect(mixed.dtype).toBe("float64");
    expect(mixed.toArray()).toEqual([2]);
    expect(() => add(tensor(["a"]), f64([1]))).toThrow(DTypeError);
    expect(() => add(f64([1, 2]), f64([1, 2, 3]))).toThrow(ShapeError);
  });
});

describe("c30 dtype handling in arithmetic", () => {
  it("div keeps float16/bfloat16 dtypes", () => {
    const a = tensor([1, 2, 3], { dtype: "float16" });
    const b = tensor([2, 2, 2], { dtype: "float16" });
    const r = div(a, b);
    expect(r.dtype).toBe("float16");
    expect(r.toArray()).toEqual([0.5, 1, 1.5]);
    expect(div(tensor([1], { dtype: "bfloat16" }), tensor([4], { dtype: "bfloat16" })).dtype).toBe(
      "bfloat16"
    );
    // integers give float32
    expect(div(i32([1, 2]), i32([2, 4])).dtype).toBe("float32");
    expect(div(i64([1n, 2n]), i64([2n, 4n])).toArray()).toEqual([0.5, 0.5]);
  });

  it("int32 div broadcasts and promotes to float32", () => {
    const r = div(i32([[7, 8, -9]]), i32([[2], [4]]));
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([
      [3.5, 4, -4.5],
      [1.75, 2, -2.25],
    ]);
    expect(div(i32([7, 8, -9]), i32([2, 4, 2])).toArray()).toEqual([3.5, 2, -4.5]);
  });

  it("reciprocal keeps half dtypes and handles int64", () => {
    expect(reciprocal(tensor([2, 4], { dtype: "float16" })).dtype).toBe("float16");
    expect(reciprocal(i64([2n, 0n, -4n])).toArray()).toEqual([0.5, Infinity, -0.25]);
    expect(reciprocal(i32([2, 0, -4])).toArray()).toEqual([0.5, Infinity, -0.25]);
  });

  it("neg rejects bool instead of wrapping to 255", () => {
    const b = tensor([true, false]);
    expect(b.dtype).toBe("bool");
    expect(() => neg(b)).toThrow(DTypeError);
    expect(neg(tensor([1, 0, 200], { dtype: "uint8" })).toArray()).toEqual([255, 0, 56]);
    expect(neg(i32([1, -2])).toArray()).toEqual([-1, 2]);
  });

  it("sign maps negative zero to zero and keeps NaN", () => {
    const r = sign(f64([-0, Number.NaN, -3, 2])).data as Float64Array;
    expect(Object.is(r[0], 0)).toBe(true);
    expect(r[1]).toBeNaN();
    expect(r[2]).toBe(-1);
    expect(r[3]).toBe(1);
  });

  it("maximum/minimum propagate NaN and support int64 and broadcasting", () => {
    expect(maximum(f64([1, Number.NaN, 3]), f64([2, 2, Number.NaN])).toArray()).toEqual([
      2,
      Number.NaN,
      Number.NaN,
    ]);
    expect(bigValues(maximum(i64([1n, 5n]), i64([3n, 2n])))).toEqual([3n, 5n]);
    expect(bigValues(minimum(i64([1n, 5n]), i64([3n, 2n])))).toEqual([1n, 2n]);
    expect(
      maximum(
        f64([
          [1, 5],
          [3, 2],
        ]),
        f64([2, 2])
      ).toArray()
    ).toEqual([
      [2, 5],
      [3, 2],
    ]);
    expect(
      minimum(
        transpose(
          f64([
            [1, 5],
            [3, 2],
          ])
        ),
        tensor(2, { dtype: "float64" })
      ).toArray()
    ).toEqual([
      [1, 2],
      [2, 2],
    ]);
  });

  it("abs and strided unary ops read views correctly", () => {
    const t = transpose(
      f64([
        [-1, 2, -3],
        [4, -5, 6],
      ])
    );
    expect(abs(t).toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expect(neg(t).toArray()).toEqual([
      [1, -4],
      [-2, 5],
      [3, -6],
    ]);
  });
});

describe("c30 scalar helpers", () => {
  it("float32 scalar is rounded to float32 first (NumPy / PyTorch parity)", () => {
    const x = f32([0.1, 0.7, 1.3, 3.25, 100.5]);
    const r = mulScalar(x, 0.1);
    expect(r.dtype).toBe("float32");
    const bits = Array.from(new Uint32Array((r.data as Float32Array).buffer.slice(0)));
    // np.float32 array * 0.1
    expect(bits).toEqual([1008981771, 1032805417, 1040522936, 1051092583, 1092668621]);
    const added = addScalar(x, 0.1);
    const addBits = Array.from(new Uint32Array((added.data as Float32Array).buffer.slice(0)));
    expect(addBits).toEqual([1045220557, 1061997773, 1068708659, 1079404134, 1120482099]);
  });

  it("fractional scalars promote integer tensors to float32", () => {
    const a = addScalar(i32([1, 2, 3]), 0.5);
    expect(a.dtype).toBe("float32");
    expect(a.toArray()).toEqual([1.5, 2.5, 3.5]);
    const m = mulScalar(i32([1, 2, 3]), 0.5);
    expect(m.dtype).toBe("float32");
    expect(m.toArray()).toEqual([0.5, 1, 1.5]);
    expect(mulScalar(i64([1n, 2n]), 0.5).toArray()).toEqual([0.5, 1]);
    // NaN / Infinity no longer reach BigInt() and throw a raw RangeError
    expect(addScalar(i64([1n]), Number.POSITIVE_INFINITY).toArray()).toEqual([Infinity]);
    expect(addScalar(i32([1]), Number.NaN).toArray()).toEqual([Number.NaN]);
  });

  it("int64 and uint8 scalar arithmetic wraps like NumPy", () => {
    const r = mulScalar(i64([9007199254740993n, 2n]), 3);
    expect(bigValues(r)).toEqual([27021597764222979n, 6n]);
    const u = tensor([200, 100], { dtype: "uint8" });
    expect(mulScalar(u, 3).toArray()).toEqual([88, 44]);
    expect(addScalar(u, 100).toArray()).toEqual([44, 200]);
  });

  it("bool tensors become int32 instead of storing 2 in a bool", () => {
    const r = addScalar(tensor([true, false]), 1);
    expect(r.dtype).toBe("int32");
    expect(r.toArray()).toEqual([2, 1]);
  });

  it("validates the scalar and the dtype", () => {
    expect(() => addScalar(tensor(["a"]), 1)).toThrow(DTypeError);
    expect(() => mulScalar(f64([1]), "2" as unknown as number)).toThrow(InvalidParameterError);
  });

  it("works on strided views", () => {
    const t = transpose(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(addScalar(t, 10).toArray()).toEqual([
      [11, 13],
      [12, 14],
    ]);
  });
});

describe("c30 clip", () => {
  it("fractional bounds promote integer tensors to float32", () => {
    const r = clip(i32([1, 2, 3, 4, 5]), 1.5, 3.5);
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([1.5, 2, 3, 3.5, 3.5]);
  });

  it("integer bounds keep the dtype", () => {
    const r = clip(i32([1, 2, 3, 4, 5]), 2, 4);
    expect(r.dtype).toBe("int32");
    expect(r.toArray()).toEqual([2, 2, 3, 4, 4]);
  });

  it("out-of-range bounds are not wrapped into the integer dtype", () => {
    const r = clip(tensor([1, 2, 3], { dtype: "uint8" }), -1, 2);
    expect(r.dtype).toBe("int32");
    expect(r.toArray()).toEqual([1, 2, 2]);
    // an integer bound beyond int32 widens an int32 tensor to int64 (exact, no float rounding)
    const wide = clip(i32([1, 2]), undefined, 3e9);
    expect(wide.dtype).toBe("int64");
    expect(bigValues(wide)).toEqual([1n, 2n]);
  });

  it("infinite bounds are ignored and do not break int64", () => {
    const x = i64([-5n, 0n, 5n, 10n]);
    expect(bigValues(clip(x, -2, 3))).toEqual([-2n, 0n, 3n, 3n]);
    expect(bigValues(clip(x, Number.NEGATIVE_INFINITY, 3))).toEqual([-5n, 0n, 3n, 3n]);
    expect(bigValues(clip(x, -2, Number.POSITIVE_INFINITY))).toEqual([-2n, 0n, 5n, 10n]);
  });

  it("NaN bound turns everything into NaN, like NumPy", () => {
    expect(clip(f64([1, 2]), Number.NaN, 5).toArray()).toEqual([Number.NaN, Number.NaN]);
    const r = clip(i32([1, 2]), Number.NaN, 5);
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([Number.NaN, Number.NaN]);
  });

  it("float tensors keep dtype and propagate NaN", () => {
    const r = clip(f32([-5, Number.NaN, 5]), -1, 1);
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([-1, Number.NaN, 1]);
  });

  it("rejects min > max", () => {
    expect(() => clip(f64([1]), 3, 2)).toThrow(InvalidParameterError);
  });

  it("hardtanh validates its bounds and keeps dtype", () => {
    expect(() => hardtanh(f64([1]), 2, 1)).toThrow(/hardtanh/);
    expect(hardtanh(f32([-2, 0.5, 3])).toArray()).toEqual([-1, 0.5, 1]);
  });
});

describe("c30 comparison", () => {
  it("isclose / allclose support int64 tensors", () => {
    expect(isclose(i64([1n, 2n]), i64([1n, 3n])).toArray()).toEqual([1, 0]);
    expect(allclose(i64([1n, 2n]), i64([1n, 2n]))).toBe(true);
    expect(allclose(i64([1n, 2n]), i32([1, 3]))).toBe(false);
  });

  it("equalNan option (NumPy isclose / allclose / array_equal)", () => {
    const a = f64([1, Number.NaN, Infinity, 1.0001]);
    const b = f64([1, Number.NaN, Infinity, 1.0]);
    expect(isclose(a, b, 1e-3, 0, true).toArray()).toEqual([1, 1, 1, 1]);
    expect(isclose(a, b, 1e-3, 0).toArray()).toEqual([1, 0, 1, 1]);
    expect(allclose(a, b, 1e-3, 0, true)).toBe(true);
    expect(allclose(a, b, 1e-3, 0)).toBe(false);
    expect(arrayEqual(f64([1, Number.NaN]), f64([1, Number.NaN]))).toBe(false);
    expect(arrayEqual(f64([1, Number.NaN]), f64([1, Number.NaN]), true)).toBe(true);
  });

  it("rejects negative or NaN tolerances", () => {
    expect(() => isclose(f64([1]), f64([1]), -1)).toThrow(InvalidParameterError);
    expect(() => isclose(f64([1]), f64([1]), 1e-5, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => allclose(f64([1]), f64([1]), 1e-5, -1)).toThrow(/atol/);
  });

  it("allclose broadcasts strided operands and stops on mismatch", () => {
    const m = transpose(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(
      allclose(
        m,
        f64([
          [1, 3],
          [2, 4],
        ])
      )
    ).toBe(true);
    expect(allclose(m, f64([1, 3]))).toBe(false);
    expect(allclose(f64([1, 1, 1]), tensor(1, { dtype: "float64" }))).toBe(true);
    expect(allclose(f64([1, 2]), f64([1, 2, 3]))).toBe(false);
  });

  it("comparisons on strided and broadcast operands", () => {
    const m = transpose(
      f64([
        [1, 5],
        [3, 2],
      ])
    );
    expect(
      greater(
        m,
        f64([
          [1, 3],
          [1, 3],
        ])
      ).toArray()
    ).toEqual([
      [0, 0],
      [1, 0],
    ]);
    expect(less(m, f64([2, 4])).toArray()).toEqual([
      [1, 1],
      [0, 1],
    ]);
    expect(equal(m, tensor(3, { dtype: "float64" })).toArray()).toEqual([
      [0, 1],
      [0, 0],
    ]);
  });

  it("int64 compares exactly against doubles", () => {
    const big = i64([2n ** 62n + 1n]);
    expect(equal(big, tensor(2 ** 62, { dtype: "float64" })).toArray()).toEqual([0]);
    expect(greater(big, tensor(2 ** 62, { dtype: "float64" })).toArray()).toEqual([1]);
  });

  it("isnan / isinf / isfinite on integer, bool and strided tensors", () => {
    expect(isnan(i32([1, 2])).toArray()).toEqual([0, 0]);
    expect(isinf(i64([1n])).toArray()).toEqual([0]);
    expect(isfinite(tensor([true, false])).toArray()).toEqual([1, 1]);
    const t = transpose(
      f64([
        [1, Number.NaN],
        [Infinity, 4],
      ])
    );
    expect(isnan(t).toArray()).toEqual([
      [0, 0],
      [1, 0],
    ]);
    expect(isinf(t).toArray()).toEqual([
      [0, 1],
      [0, 0],
    ]);
    expect(isfinite(t).toArray()).toEqual([
      [1, 0],
      [0, 1],
    ]);
    expect(() => isnan(tensor(["a"]))).toThrow(DTypeError);
  });
});

describe("c30 activations: numerical accuracy", () => {
  it("gelu stays accurate for negative inputs (no 1 + tanh cancellation)", () => {
    const x = [-8, -5, -1, -1e-3, 0, 1e-3, 2, 5];
    const expected = [
      -3.1077829375011112e-21, -2.291796196629506e-7, -0.1588080093917233, -0.000499601057786418, 0,
      0.000500398942213582, 1.954597694087775, 4.999999770820381,
    ];
    const r = gelu(f64(x)).toArray() as number[];
    for (let i = 0; i < x.length; i++) {
      const e = expected[i] as number;
      expect(Math.abs((r[i] as number) - e)).toBeLessThanOrEqual(
        1e-14 * Math.max(Math.abs(e), 1e-300)
      );
    }
  });

  it("gelu / mish limits and special values", () => {
    expect(gelu(f64([-Infinity, Infinity])).toArray()).toEqual([0, Infinity]);
    expect(mish(f64([-Infinity])).toArray()).toEqual([0]);
    expect(swish(f64([-Infinity])).toArray()).toEqual([0]);
    expect(gelu(f64([Number.NaN])).toArray()).toEqual([Number.NaN]);
    expect(gelu(f64([-1e200])).toArray()).toEqual([-0]);
  });

  it("elu uses expm1 near zero", () => {
    const r = elu(f64([-1e-10, -1e-5, -2, 3]), 2).toArray() as number[];
    expect(r[0]).toBeCloseTo(-1.9999999999000001e-10, 24);
    expect(Math.abs((r[0] as number) / -1.9999999999000001e-10 - 1)).toBeLessThan(1e-15);
    expect(Math.abs((r[1] as number) / -1.9999900000333333e-5 - 1)).toBeLessThan(1e-15);
    expect(r[2]).toBeCloseTo(-1.7293294335267746, 14);
    expect(r[3]).toBe(3);
  });

  it("tanhshrink keeps relative accuracy for small |x|", () => {
    const x = [1e-8, 1e-4, 0.01, 0.3, 0.49, 0.51, 2, -0.2];
    // mpmath: x - tanh(x) at 60 digits
    const expected = [
      3.3333333333333335e-25, 3.3333333200000004e-13, 3.333200005396607e-7, 0.008687387548409094,
      0.03578356731774093, 0.04005480106696238, 1.035972419924183, -0.0026246797750959995,
    ];
    const r = tanhshrink(f64(x)).toArray() as number[];
    for (let i = 0; i < x.length; i++) {
      expect(Math.abs((r[i] as number) / (expected[i] as number) - 1)).toBeLessThan(2e-14);
    }
    expect(tanhshrink(f64([Infinity, -Infinity])).toArray()).toEqual([Infinity, -Infinity]);
    expect(tanhshrink(f64([Number.NaN])).toArray()).toEqual([Number.NaN]);
  });

  it("sigmoid keeps the denormal tail and symmetric values", () => {
    const r = sigmoid(f64([-740, -40, 0, 10, Infinity, -Infinity])).toArray() as number[];
    expect(r[0]).toBeGreaterThan(0);
    expect(r[0]).toBeLessThan(1e-320);
    expect(r[1]).toBeCloseTo(4.248354255291589e-18, 30);
    expect(r[2]).toBe(0.5);
    expect(r[3]).toBeCloseTo(0.9999546021312976, 15);
    expect(r[4]).toBe(1);
    expect(r[5]).toBe(0);
    expect(sigmoid(f64([Number.NaN])).toArray()).toEqual([Number.NaN]);
  });

  it("softplus is stable at both extremes", () => {
    const r = softplus(f64([-800, -1, 0, 1, 800])).toArray() as number[];
    expect(r[0]).toBe(0);
    expect(r[1]).toBeCloseTo(0.31326168751822286, 15);
    expect(r[2]).toBeCloseTo(Math.LN2, 15);
    expect(r[3]).toBeCloseTo(1.3132616875182228, 15);
    expect(r[4]).toBe(800);
  });

  it("mish matches PyTorch", () => {
    const r = mish(f64([-1, 0, 1, 30])).toArray() as number[];
    expect(r[0]).toBeCloseTo(-0.30340146137410895, 14);
    expect(r[1]).toBe(0);
    expect(r[2]).toBeCloseTo(0.8650983882673103, 14);
    expect(r[3]).toBeCloseTo(30, 12);
  });
});

describe("c30 softmax / logSoftmax", () => {
  it("rows, large values and non-last axes", () => {
    const r = softmax(
      f64([
        [1, 2, 3],
        [1000, 1000, 1000],
      ])
    );
    const rows = r.toArray() as number[][];
    const refRow = [0.09003057317038045, 0.2447284710547976, 0.6652409557748218];
    refRow.forEach((v, k) => {
      expect(rows[0]?.[k]).toBeCloseTo(v, 15);
    });
    expect(rows[1]).toEqual([1 / 3, 1 / 3, 1 / 3]);
    const ax0 = softmax(
      f64([
        [1, 2],
        [3, 5],
      ]),
      0
    ).toArray() as number[][];
    expect(ax0[0]?.[0]).toBeCloseTo(0.11920292202211755, 15);
    expect(ax0[0]?.[1]).toBeCloseTo(0.04742587317756679, 15);
    expect(ax0[1]?.[0]).toBeCloseTo(0.8807970779778823, 15);
    expect(ax0[1]?.[1]).toBeCloseTo(0.9525741268224334, 15);
  });

  it("reads strided views in logical order", () => {
    const t = transpose(
      f64([
        [1, 3],
        [2, 5],
      ])
    );
    const r = softmax(t, 0).toArray() as number[][];
    // transpose(...) is [[1,2],[3,5]]; softmax along axis 0 as in the test above
    expect(r[0]?.[0]).toBeCloseTo(0.11920292202211755, 15);
    expect(r[1]?.[1]).toBeCloseTo(0.9525741268224334, 15);
  });

  it("0-d tensors softmax to 1 (PyTorch accepts dim 0 and -1)", () => {
    expect(softmax(tensor(5, { dtype: "float64" })).toArray()).toBe(1);
    expect(logSoftmax(tensor(5, { dtype: "float64" }), 0).toArray()).toBe(0);
    expect(() => softmax(tensor(5, { dtype: "float64" }), 1)).toThrow(InvalidParameterError);
  });

  it("logSoftmax keeps the tail that log(softmax) would lose", () => {
    const r = logSoftmax(
      f64([
        [1, 2, 3],
        [1000, 0, -1000],
      ])
    ).toArray() as number[][];
    expect(r[0]?.[0]).toBeCloseTo(-2.4076059644443806, 14);
    expect(r[0]?.[2]).toBeCloseTo(-0.4076059644443804, 14);
    expect(r[1]).toEqual([0, -1000, -2000]);
  });

  it("int tensors are accepted and NaN poisons only its slice", () => {
    const r = softmax(
      i32([
        [1, 1],
        [0, 0],
      ]),
      1
    ).toArray();
    expect(r).toEqual([
      [0.5, 0.5],
      [0.5, 0.5],
    ]);
    const n = softmax(
      f64([
        [1, Number.NaN],
        [1, 1],
      ]),
      1
    ).toArray() as number[][];
    expect(n[0]?.every(Number.isNaN)).toBe(true);
    expect(n[1]).toEqual([0.5, 0.5]);
  });

  it("empty tensors and string tensors", () => {
    const empty = Tensor.fromTypedArray({
      data: new Float64Array(0),
      shape: [0, 3],
      dtype: "float64",
      device: "cpu",
    });
    expect(softmax(empty).shape).toEqual([0, 3]);
    expect(() => softmax(tensor(["a"]))).toThrow(DTypeError);
  });
});

describe("c30 activation dtypes and parameters", () => {
  it("sigmoid/relu/leakyRelu keep half precision dtypes", () => {
    const h = tensor([-1, 0, 1], { dtype: "float16" });
    expect(sigmoid(h).dtype).toBe("float16");
    expect(relu(h).dtype).toBe("float16");
    expect(leakyRelu(h).dtype).toBe("float16");
    expect(relu(tensor([-1, 1], { dtype: "bfloat16" })).dtype).toBe("bfloat16");
    expect(relu(f32([-1, 1])).dtype).toBe("float32");
    // relu is a clip: integer input keeps its dtype
    expect(relu(i32([-1, 1])).dtype).toBe("int32");
  });

  it("int64 relu stays exact and other int64 activations reject unsafe values", () => {
    expect(bigValues(relu(i64([-3n, 4n])))).toEqual([0n, 4n]);
    expect(bigValues(relu(i64([2n ** 62n])))).toEqual([2n ** 62n]);
    expect(() => sigmoid(i64([2n ** 62n]))).toThrow(DataValidationError);
  });

  it("leakyRelu/elu validate alpha", () => {
    expect(() => leakyRelu(f64([1]), Number.NaN)).toThrow(InvalidParameterError);
    expect(() => elu(f64([1]), Infinity)).toThrow(/alpha/);
    expect(leakyRelu(f64([-2, 3]), 0.5).toArray()).toEqual([-1, 3]);
  });

  it("string tensors are rejected with a typed error", () => {
    const s = tensor(["a"]);
    for (const fn of [sigmoid, relu, gelu, swish, mish, softplus, tanhshrink, elu, leakyRelu]) {
      expect(() => fn(s)).toThrow(DTypeError);
    }
  });
});

// ---------------------------------------------------------------------------
// Device dispatch for maximum / minimum / reciprocal / sign / clip / softplus
// ---------------------------------------------------------------------------

type FakeBuffer = DeviceBuffer & { data: Float32Array };

function gather(buffer: DeviceBuffer, layout: KernelLayout): Float32Array {
  const buf = buffer as FakeBuffer;
  const size = layout.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(size);
  const ndim = layout.shape.length;
  for (let i = 0; i < size; i++) {
    let rem = i;
    let idx = layout.offset;
    for (let k = ndim - 1; k >= 0; k--) {
      const dim = layout.shape[k] ?? 1;
      idx += (rem % dim) * (layout.strides[k] ?? 0);
      rem = Math.floor(rem / dim);
    }
    out[i] = buf.data[idx] ?? 0;
  }
  return out;
}

const wrap = (data: Float32Array): FakeBuffer => ({
  device: "webgpu",
  byteLength: data.length * 4,
  size: data.length,
  dtype: "float32",
  data,
});

const unsupported = (): never => {
  throw new DeviceError("fake backend: unsupported");
};

const calls: string[] = [];

const fakeBackend = {
  info: () => ({ device: "webgpu", name: "c30 fake", available: true, capabilities: [] }),
  supports: () => false,
  init: async () => {},
  dispose: () => {},
  upload: (data: Float32Array) => wrap(data.slice()),
  download: async (b: DeviceBuffer) => (b as FakeBuffer).data.slice(),
  free: () => {},
  fill: (value: number, size: number) => wrap(new Float32Array(Math.max(size, 1)).fill(value)),
  binary: (
    op: string,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ) => {
    calls.push(op);
    const av = gather(a, { ...aLayout, shape: outShape });
    const bv = gather(b, { ...bLayout, shape: outShape });
    const fn = op === "maximum" ? Math.max : op === "minimum" ? Math.min : unsupported;
    return wrap(av.map((x, i) => fn(x, bv[i] ?? 0)));
  },
  unary: (op: string, x: DeviceBuffer, layout: KernelLayout) => {
    calls.push(op);
    const fn =
      op === "reciprocal"
        ? (v: number) => 1 / v
        : op === "sign"
          ? Math.sign
          : op === "copy"
            ? (v: number) => v
            : op === "softplus"
              ? (v: number) => Math.log1p(Math.exp(v))
              : unsupported;
    return wrap(gather(x, layout).map((v) => fn(v)));
  },
  matmul: unsupported,
  reduce: unsupported,
  reduceAxis: unsupported,
  matmulBatched: unsupported,
  ternary: unsupported,
  im2col: unsupported,
  col2im: unsupported,
  pool2d: unsupported,
  pool2dBackward: unsupported,
};

describe("c30 device dispatch", () => {
  beforeAll(() => {
    registerBackend("webgpu", fakeBackend as unknown as Backend);
  });
  afterAll(() => {
    unregisterBackend("webgpu");
    resetConfig();
  });

  const onDevice = (data: number[]): Tensor => tensor(data, { device: "webgpu" });

  it("maximum / minimum run on the device", async () => {
    calls.length = 0;
    const r = maximum(onDevice([1, 5, 3]), onDevice([4, 2, 6]));
    expect(r.isDeviceTensor).toBe(true);
    expect((await r.cpu()).toArray()).toEqual([4, 5, 6]);
    const m = minimum(onDevice([1, 5, 3]), tensor(2, { dtype: "float32" }));
    expect((await m.cpu()).toArray()).toEqual([1, 2, 2]);
    expect(calls).toContain("maximum");
    expect(calls).toContain("minimum");
  });

  it("reciprocal / sign / softplus run on the device", async () => {
    calls.length = 0;
    expect((await reciprocal(onDevice([2, 4])).cpu()).toArray()).toEqual([0.5, 0.25]);
    expect((await sign(onDevice([-3, 0, 2])).cpu()).toArray()).toEqual([-1, 0, 1]);
    const sp = (await softplus(onDevice([0])).cpu()).toArray() as number;
    expect((sp as unknown as number[])[0]).toBeCloseTo(Math.LN2, 6);
    expect(calls).toEqual(expect.arrayContaining(["reciprocal", "sign", "softplus"]));
  });

  it("clip and hardtanh compose maximum/minimum on the device", async () => {
    const x = onDevice([-5, 0.5, 5]);
    expect((await clip(x, -1, 1).cpu()).toArray()).toEqual([-1, 0.5, 1]);
    expect((await clip(x, 0).cpu()).toArray()).toEqual([0, 0.5, 5]);
    expect((await clip(x, undefined, 1).cpu()).toArray()).toEqual([-5, 0.5, 1]);
    expect((await hardtanh(x).cpu()).toArray()).toEqual([-1, 0.5, 1]);
    const same = clip(x);
    expect(same).not.toBe(x);
    expect((await same.cpu()).toArray()).toEqual([-5, 0.5, 5]);
  });
});
