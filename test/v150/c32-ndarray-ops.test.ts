import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  argsort,
  columnStack,
  concatenate,
  digitize,
  exp2,
  gradient,
  hstack,
  interp,
  repeat,
  round,
  rsqrt,
  sort,
  split,
  square,
  stack,
  tensor,
  tile,
  trapz,
  vstack,
} from "../../src/ndarray";
import { radixArgsortF64, radixSortF64 } from "../../src/ndarray/ops/radix";
import { dropoutMask } from "../../src/ndarray/ops/random";
import { Tensor } from "../../src/ndarray/tensor";
import { transpose } from "../../src/ndarray/tensor/shape";
import { setSeed } from "../../src/random";

const f64 = { dtype: "float64" } as const;

function i64(values: bigint[]) {
  return Tensor.fromTypedArray({
    data: BigInt64Array.from(values),
    shape: [values.length],
    dtype: "int64",
    device: "cpu",
  });
}

/** 3x2 strided view of [[0, 1, 2], [3, 4, 5]]. */
function transposedView() {
  return transpose(
    tensor(
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
      f64
    )
  );
}

describe("v1.5.0 ndarray/ops/manipulation", () => {
  it("concatenate reads strided views through their strides", () => {
    // numpy: np.concatenate([np.arange(6.).reshape(2,3).T, [[10,11],[12,13],[14,15]]], axis=1)
    const b = tensor(
      [
        [10, 11],
        [12, 13],
        [14, 15],
      ],
      f64
    );
    const out = concatenate([transposedView(), b], 1);
    expect(out.shape).toEqual([3, 4]);
    expect(out.toArray()).toEqual([
      [0, 3, 10, 11],
      [1, 4, 12, 13],
      [2, 5, 14, 15],
    ]);
  });

  it("concatenate reads sliced views with an offset", () => {
    const base = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      f64
    );
    const tail = base.slice({ start: 1, end: 3 }, { start: 1, end: 3 }); // [[5, 6], [8, 9]]
    const out = concatenate([tail, tail], 0);
    expect(out.toArray()).toEqual([
      [5, 6],
      [8, 9],
      [5, 6],
      [8, 9],
    ]);
  });

  it("concatenate of one tensor returns an independent copy and validates the axis", () => {
    const a = tensor([1, 2, 3], f64);
    const c = concatenate([a]);
    expect(c.toArray()).toEqual([1, 2, 3]);
    expect(c.data).not.toBe(a.data);
    // A single 0-d tensor has no axis to concatenate along (numpy raises too).
    expect(() => concatenate([tensor(5)])).toThrow(InvalidParameterError);
    expect(() => concatenate([a], 3)).toThrow(InvalidParameterError);
  });

  it("concatenate handles empty pieces and int64/string dtypes", () => {
    const empty = tensor([], f64).reshape([0, 2]);
    const a = tensor([[1, 2]], f64);
    expect(concatenate([empty, a, empty], 0).toArray()).toEqual([[1, 2]]);
    const big = i64([2n ** 60n, 1n]);
    expect(concatenate([big, big]).toArray()).toEqual([2n ** 60n, 1n, 2n ** 60n, 1n]);
    expect(concatenate([tensor(["a", "b"]), tensor(["c"])]).toArray()).toEqual(["a", "b", "c"]);
  });

  it("stack places the new axis correctly for strided inputs", () => {
    // numpy: np.stack([a, a + 100], axis=-1) with a = np.arange(6.).reshape(2,3).T
    const a = transposedView();
    const shifted = tensor(
      [
        [100, 103],
        [101, 104],
        [102, 105],
      ],
      f64
    );
    const out = stack([a, shifted], -1);
    expect(out.shape).toEqual([3, 2, 2]);
    expect(out.toArray()).toEqual([
      [
        [0, 100],
        [3, 103],
      ],
      [
        [1, 101],
        [4, 104],
      ],
      [
        [2, 102],
        [5, 105],
      ],
    ]);
    expect(stack([tensor(["x", "y"]), tensor(["z", "w"])], 1).toArray()).toEqual([
      ["x", "z"],
      ["y", "w"],
    ]);
  });

  it("split copies sections out of a strided view", () => {
    // numpy: np.split(a, [1, 1], axis=0) and np.split(a, 2, axis=1) with a = arange(6.).reshape(2,3).T
    const a = transposedView();
    const parts = split(a, [1, 1], 0);
    expect(parts.map((p) => p.shape)).toEqual([
      [1, 2],
      [0, 2],
      [2, 2],
    ]);
    expect(parts[0]?.toArray()).toEqual([[0, 3]]);
    expect(parts[2]?.toArray()).toEqual([
      [1, 4],
      [2, 5],
    ]);
    const cols = split(a, 2, 1);
    expect(cols[0]?.toArray()).toEqual([[0], [1], [2]]);
    expect(cols[1]?.toArray()).toEqual([[3], [4], [5]]);
  });

  it("tile matches numpy for strided input, extra dims and zero reps", () => {
    // numpy: np.tile(np.array([[1,2],[3,4]]).T, (2,1,2))
    const m = transpose(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        f64
      )
    );
    const out = tile(m, [2, 1, 2]);
    expect(out.shape).toEqual([2, 2, 4]);
    expect(out.toArray()).toEqual([
      [
        [1, 3, 1, 3],
        [2, 4, 2, 4],
      ],
      [
        [1, 3, 1, 3],
        [2, 4, 2, 4],
      ],
    ]);
    expect(tile(tensor([1, 2], f64), [0]).shape).toEqual([0]);
    expect(tile(tensor([[1, 2]], f64), [0, 2]).shape).toEqual([0, 4]);
    expect(tile(tensor(["a", "b"]), [3]).toArray()).toEqual(["a", "b", "a", "b", "a", "b"]);
  });

  it("repeat accepts one count per element like numpy", () => {
    // numpy: np.repeat([[1,2,3],[4,5,6]].T, [1,0,2], axis=0)
    const m = transpose(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        f64
      )
    );
    expect(repeat(m, [1, 0, 2], 0).toArray()).toEqual([
      [1, 4],
      [3, 6],
      [3, 6],
    ]);
    // flattened strided input: np.repeat([[1,2],[3,4]].T, [1,2,0,1]) -> [1,3,3,4]
    const sq = transpose(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        f64
      )
    );
    expect(repeat(sq, [1, 2, 0, 1]).toArray()).toEqual([1, 3, 3, 4]);
    // a one-element list is broadcast
    expect(
      repeat(
        tensor(
          [
            [1, 2],
            [3, 4],
          ],
          f64
        ),
        [2],
        1
      ).toArray()
    ).toEqual([
      [1, 1, 2, 2],
      [3, 3, 4, 4],
    ]);
    expect(repeat(tensor([1, 2, 3], f64), 2).toArray()).toEqual([1, 1, 2, 2, 3, 3]);
    expect(repeat(i64([2n, 3n]), 3).toArray()).toEqual([2n, 2n, 2n, 3n, 3n, 3n]);
  });

  it("repeat rejects bad counts and mismatched lengths", () => {
    const t = tensor([1, 2, 3], f64);
    expect(() => repeat(t, [1, 2])).toThrow(InvalidParameterError);
    expect(() => repeat(t, [1, -1, 1])).toThrow(InvalidParameterError);
    expect(() => repeat(t, 1.5)).toThrow(InvalidParameterError);
  });
});

describe("v1.5.0 ndarray/ops/math", () => {
  it("round keeps the sign of zero and rounds halves to even", () => {
    const out = round(tensor([-0.4, -0.5, 0.5, 1.5, 2.5, -1.5], f64)).toArray() as number[];
    // numpy: np.round([-0.4, -0.5, 0.5, 1.5, 2.5, -1.5]) -> [-0, -0, 0, 2, 2, -2]
    expect(Object.is(out[0], -0)).toBe(true);
    expect(Object.is(out[1], -0)).toBe(true);
    expect(out.slice(2)).toEqual([0, 2, 2, -2]);
  });

  it("round supports decimals like numpy", () => {
    // numpy: np.round(2.675, 2) = 2.68, np.round(1234.5678, -2) = 1200.0,
    //        np.round(5.5, 1) = 5.5, np.round(-1.25, 1) = -1.2
    expect(round(tensor([2.675], f64), 2).toArray()).toEqual([2.68]);
    expect(round(tensor([1234.5678], f64), -2).toArray()).toEqual([1200]);
    expect(round(tensor([5.5, -1.25], f64), 1).toArray()).toEqual([5.5, -1.2]);
    expect(round(tensor([Number.POSITIVE_INFINITY, Number.NaN, 0.5], f64), 2).toArray()).toEqual([
      Number.POSITIVE_INFINITY,
      Number.NaN,
      0.5,
    ]);
    expect(() => round(tensor([1.5], f64), 0.5)).toThrow(InvalidParameterError);
  });

  it("square, rsqrt and exp2 read strided views correctly", () => {
    const v = transposedView(); // [[0, 3], [1, 4], [2, 5]]
    expect(square(v).toArray()).toEqual([
      [0, 9],
      [1, 16],
      [4, 25],
    ]);
    expect((rsqrt(v).toArray() as number[][])[1]).toEqual([1, 0.5]);
    expect(exp2(v).toArray()).toEqual([
      [1, 8],
      [2, 16],
      [4, 32],
    ]);
  });
});

describe("v1.5.0 ndarray/ops/numerical", () => {
  it("interp returns fp at the end points, left/right only outside", () => {
    // numpy: np.interp([0, 1], [0, 1], [5, 6], left=-1, right=-2) -> [5, 6]
    const xp = tensor([0, 1], f64);
    const fp = tensor([5, 6], f64);
    expect(interp(tensor([0, 1, -0.1, 1.1], f64), xp, fp, -1, -2).toArray()).toEqual([
      5, 6, -1, -2,
    ]);
  });

  it("interp handles repeated xp, NaN input and infinite fp like numpy", () => {
    // numpy: np.interp([0, .5, 1, 1.5, 2, 3], [0, 1, 1, 2], [0, 1, 2, 3]) -> [0, .5, 2, 2.5, 3, 3]
    const out = interp(
      tensor([0, 0.5, 1, 1.5, 2, 3], f64),
      tensor([0, 1, 1, 2], f64),
      tensor([0, 1, 2, 3], f64)
    );
    expect(out.toArray()).toEqual([0, 0.5, 2, 2.5, 3, 3]);
    // numpy: np.interp([2, 5], [1, 3, 5], [0, inf, 5], left=-1, right=-2) -> [inf, 5]
    expect(
      interp(
        tensor([2, 5], f64),
        tensor([1, 3, 5], f64),
        tensor([0, Number.POSITIVE_INFINITY, 5], f64),
        -1,
        -2
      ).toArray()
    ).toEqual([Number.POSITIVE_INFINITY, 5]);
    // inf - inf in the slope must not poison a flat segment: np.interp(.5, [0, 1], [inf, inf]) = inf
    expect(
      interp(
        tensor([0.5], f64),
        tensor([0, 1], f64),
        tensor([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY], f64)
      ).toArray()
    ).toEqual([Number.POSITIVE_INFINITY]);
    const nan = interp(tensor([Number.NaN], f64), tensor([1], f64), tensor([5], f64)).toArray();
    expect(nan).toEqual([Number.NaN]);
    expect(() =>
      interp(tensor([1], f64), tensor([0, Number.NaN], f64), tensor([1, 2], f64))
    ).toThrow(InvalidParameterError);
  });

  it("trapz sums with compensation", () => {
    // Exact value of the trapezoid sum is 5.000000000000004e16 (computed with fractions);
    // plain left-to-right summation gives 5e16 because every +2 is lost against 1e17.
    const y = tensor([1e17, ...new Array<number>(40).fill(1)], f64);
    expect(trapz(y)).toBe(5.000000000000004e16);
    // validated even when there are fewer than 2 samples
    expect(() => trapz(tensor([1], f64), tensor([1, 2], f64))).toThrow(ShapeError);
  });

  it("gradient matches numpy on a non-uniform grid and rejects repeated coordinates", () => {
    // numpy: np.gradient([1, 2, 4, 7, 11], [0, 1, 3, 4, 10])
    const g = gradient(tensor([1, 2, 4, 7, 11], f64), tensor([0, 1, 3, 4, 10], f64));
    const ref = [1, 1, 2.3333333333333326, 2.6666666666666674, 0.6666666666666666];
    (g.toArray() as number[]).forEach((v, i) => {
      expect(v).toBeCloseTo(ref[i] as number, 12);
    });
    expect(() => gradient(tensor([1, 2, 3], f64), tensor([0, 1, 1], f64))).toThrow(
      InvalidParameterError
    );
    expect(() => gradient(tensor([1, 2, 3], f64), tensor([0, 1, 0], f64))).toThrow(
      InvalidParameterError
    );
    expect(() => gradient(tensor([1, 2, 3], f64), Number.NaN)).toThrow(InvalidParameterError);
  });

  it("digitize supports decreasing bins, NaN and empty bins like numpy", () => {
    // numpy: np.digitize([3, 2, 1, 0, .5], [3, 2, 1]) -> [0, 1, 2, 3, 3]; right=True -> [1, 2, 3, 3, 3]
    const x = tensor([3, 2, 1, 0, 0.5], f64);
    const dec = tensor([3, 2, 1], f64);
    expect(digitize(x, dec).toArray()).toEqual([0, 1, 2, 3, 3]);
    expect(digitize(x, dec, true).toArray()).toEqual([1, 2, 3, 3, 3]);
    // numpy: np.digitize([nan, 1.5], [1, 2, 3]) -> [3, 1]; np.digitize([nan, 1.5, 3, 5], [3, 2, 1]) -> [0, 2, 0, 0]
    expect(digitize(tensor([Number.NaN, 1.5], f64), tensor([1, 2, 3], f64)).toArray()).toEqual([
      3, 1,
    ]);
    expect(digitize(tensor([Number.NaN, 1.5, 3, 5], f64), dec).toArray()).toEqual([0, 2, 0, 0]);
    expect(digitize(tensor([1], f64), tensor([], f64)).toArray()).toEqual([0]);
    expect(() => digitize(tensor([1], f64), tensor([1, Number.NaN], f64))).toThrow(
      InvalidParameterError
    );
  });

  it("vstack keeps the dtype and accepts mixed 1D/2D and 0D inputs", () => {
    const rows = vstack([tensor([1, 2], { dtype: "int32" }), tensor([3, 4], { dtype: "int32" })]);
    expect(rows.dtype).toBe("int32");
    expect(rows.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    // numpy: np.vstack([[1,2,3], [[4,5,6],[7,8,9]]])
    const mixed = vstack([
      tensor([1, 2, 3], f64),
      tensor(
        [
          [4, 5, 6],
          [7, 8, 9],
        ],
        f64
      ),
    ]);
    expect(mixed.toArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(vstack([tensor(1), tensor(2)]).shape).toEqual([2, 1]);
    const big = 2n ** 62n;
    expect(vstack([i64([big]), i64([1n])]).toArray()).toEqual([[big], [1n]]);
    expect(vstack([tensor(["a"]), tensor(["b"])]).toArray()).toEqual([["a"], ["b"]]);
  });

  it("hstack and columnStack keep dtypes and promote mixed numeric dtypes like binary ops", () => {
    const i32 = tensor([1], { dtype: "int32" });
    const f32 = tensor([2.5], { dtype: "float32" });
    // PyTorch-style promotion: (int32, float32) -> float32; (int32, int64) -> int64
    expect(hstack([i32, f32]).dtype).toBe("float32");
    expect(hstack([i32, tensor([2], { dtype: "int64" })]).dtype).toBe("int64");
    expect(
      hstack([tensor([1], { dtype: "uint8" }), tensor([2.5], { dtype: "float16" })]).dtype
    ).toBe("float16");
    expect(hstack([tensor(1), tensor([2, 3])]).toArray()).toEqual([1, 2, 3]);
    expect(() => hstack([tensor(["a"]), i32])).toThrow();

    const cols = columnStack([
      tensor([1, 2], { dtype: "int32" }),
      tensor([3, 4], { dtype: "int32" }),
    ]);
    expect(cols.dtype).toBe("int32");
    expect(cols.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });
});

describe("v1.5.0 ndarray/ops/radix", () => {
  it("radixSortF64 keeps signed zeros and orders -0 first like TypedArray.sort", () => {
    const values = new Float64Array([0, -0, 3, -0, 0, Number.NaN, -1]);
    const expected = Float64Array.from(values).sort();
    radixSortF64(values);
    expect(Array.from(values).map((v) => Object.is(v, -0))).toEqual(
      Array.from(expected).map((v) => Object.is(v, -0))
    );
    expect(Array.from(values.slice(0, 6))).toEqual(Array.from(expected.slice(0, 6)));
    expect(Number.isNaN(values[6])).toBe(true);
  });

  it("sort of a large lane preserves the sign of zero", () => {
    const n = 9000;
    const raw = Array.from({ length: n }, (_, i) => ((i * 7) % 101) - 50 + 0.5);
    raw[10] = 0;
    raw[11] = -0;
    const sorted = sort(tensor(raw, f64)).toArray() as number[];
    const zeros = sorted.filter((v) => v === 0);
    expect(zeros.length).toBe(2);
    expect(Object.is(zeros[0], -0)).toBe(true);
    expect(Object.is(zeros[1], 0)).toBe(true);
  });

  it("radixArgsortF64 is stable, treats signed zeros as equal and sorts NaN last", () => {
    const values = new Float64Array([2, 0, -0, Number.NaN, 2, -1, Number.NaN, 0]);
    const out = new Int32Array(values.length);
    radixArgsortF64(values, out);
    expect(Array.from(out)).toEqual([5, 1, 2, 7, 0, 4, 3, 6]);
  });

  it("radix paths agree on data whose low mantissa digits are all zero (skipped passes)", () => {
    const n = 9000;
    const raw = Array.from({ length: n }, (_, i) => Math.fround(Math.sin(i) * 100));
    const t = tensor(raw, f64);
    const sorted = sort(t).toArray() as number[];
    expect(sorted).toEqual([...raw].sort((a, b) => a - b));
    const order = argsort(t).toArray() as number[];
    expect(order.map((i) => raw[i])).toEqual(sorted);
  });
});

describe("v1.5.0 ndarray/ops/random", () => {
  it("dropoutMask is reproducible under a seed and respects p", () => {
    setSeed(11);
    const a = dropoutMask([40000], 0.25, 1 / 0.75, "float64", "cpu");
    setSeed(11);
    const b = dropoutMask([40000], 0.25, 1 / 0.75, "float64", "cpu");
    expect(Array.from(a.data as Float64Array)).toEqual(Array.from(b.data as Float64Array));
    const kept = (a.data as Float64Array).reduce((s, v) => s + (v === 0 ? 0 : 1), 0) / 40000;
    expect(kept).toBeGreaterThan(0.73);
    expect(kept).toBeLessThan(0.77);
  });

  it("dropoutMask with p = 0 keeps every element", () => {
    const m = dropoutMask([1000], 0, 1, "float32", "cpu");
    expect((m.data as Float32Array).every((v) => v === 1)).toBe(true);
  });

  it("dropoutMask normalises bool masks and truncates int64 scales like astype", () => {
    setSeed(3);
    const bool = dropoutMask([200], 0.5, 2, "bool", "cpu");
    expect(Array.from(bool.data as Uint8Array).every((v) => v === 0 || v === 1)).toBe(true);
    setSeed(3);
    const big = dropoutMask([200], 0.5, 1.9, "int64", "cpu");
    const values = new Set(Array.from(big.data as BigInt64Array));
    expect([...values].every((v) => v === 0n || v === 1n)).toBe(true);
  });

  it("dropoutMask validates scale and shape", () => {
    expect(() => dropoutMask([3], 0.5, Number.NaN, "float32", "cpu")).toThrow(
      InvalidParameterError
    );
    expect(() => dropoutMask([3], 0.5, Number.POSITIVE_INFINITY, "int64", "cpu")).toThrow(
      InvalidParameterError
    );
    expect(() => dropoutMask([-1], 0.5, 2, "float32", "cpu")).toThrow(DataValidationError);
    expect(() => dropoutMask([3], 1, 2, "float32", "cpu")).toThrow(InvalidParameterError);
  });
});
