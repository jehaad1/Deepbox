/**
 * Wave 3 tests for the ndarray-shape group: takeAlongAxis, putAlongAxis, nonzero,
 * argwhere, countNonzero, unique options, batched cross and int32 searchsorted.
 *
 * Reference values were computed with NumPy 2.4.
 */
import { describe, expect, it } from "vitest";
import { DTypeError, IndexError, InvalidParameterError, ShapeError } from "../../src/core";
import { Tensor, tensor, transpose } from "../../src/ndarray";
import {
  argwhere,
  countNonzero,
  cross,
  nonzero,
  putAlongAxis,
  searchsorted,
  takeAlongAxis,
  unique,
} from "../../src/ndarray/ops/utils";

const f64 = (data: unknown): Tensor => tensor(data as never, { dtype: "float64" });
const i32 = (data: unknown): Tensor => tensor(data as never, { dtype: "int32" });

function int64(values: readonly bigint[], shape: number[] = [values.length]): Tensor {
  return Tensor.fromTypedArray({
    data: new BigInt64Array(values),
    shape,
    dtype: "int64",
    device: "cpu",
  });
}

describe("takeAlongAxis", () => {
  const a = f64([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  it("matches np.take_along_axis", () => {
    expect(
      takeAlongAxis(
        a,
        i32([
          [2, 0],
          [1, 1],
        ]),
        1
      ).toArray()
    ).toEqual([
      [3, 1],
      [5, 5],
    ]);
    expect(takeAlongAxis(a, i32([[0, 1, 0]]), 0).toArray()).toEqual([[1, 5, 3]]);
  });

  it("defaults to the last axis and accepts negative axes and negative indices", () => {
    expect(
      takeAlongAxis(
        a,
        i32([
          [-1, 0],
          [1, -3],
        ])
      ).toArray()
    ).toEqual([
      [3, 1],
      [5, 4],
    ]);
    expect(takeAlongAxis(a, i32([[-1], [0]]), -1).toArray()).toEqual([[3], [4]]);
  });

  it("broadcasts every axis except the indexed one", () => {
    // np.take_along_axis(a, [[0, 2]], axis=1) broadcasts the single index row over both rows
    expect(takeAlongAxis(a, i32([[0, 2]]), 1).toArray()).toEqual([
      [1, 3],
      [4, 6],
    ]);
    // a size-1 array axis stretches to the index length
    expect(
      takeAlongAxis(
        f64([[7], [8]]),
        i32([
          [0, 0, 0],
          [0, 0, 0],
        ]),
        1
      ).toArray()
    ).toEqual([
      [7, 7, 7],
      [8, 8, 8],
    ]);
    expect(() => takeAlongAxis(a, i32([[0], [0], [0]]), 1)).toThrow(ShapeError);
  });

  it("flattens the array when axis is null", () => {
    // np.take_along_axis(a.ravel(), [5, 0, -1], axis=0)
    expect(takeAlongAxis(a, i32([5, 0, -1]), null).toArray()).toEqual([6, 1, 6]);
    expect(() => takeAlongAxis(a, i32([[0, 1]]), null)).toThrow(ShapeError);
  });

  it("is the inverse companion of an argsort", () => {
    const x = f64([
      [10, 30, 20],
      [60, 40, 50],
    ]);
    const order = i32([
      [0, 2, 1],
      [1, 2, 0],
    ]);
    expect(takeAlongAxis(x, order, 1).toArray()).toEqual([
      [10, 20, 30],
      [40, 50, 60],
    ]);
  });

  it("reads strided views, keeps the dtype and handles int64", () => {
    const t = transpose(a); // [[1, 4], [2, 5], [3, 6]] as a view
    expect(takeAlongAxis(t, i32([[1], [0], [1]]), 1).toArray()).toEqual([[4], [2], [6]]);
    const big = int64([9007199254740993n, 5n, 7n]);
    const out = takeAlongAxis(big, i32([0, 2]), 0);
    expect(out.dtype).toBe("int64");
    expect(out.toArray()).toEqual([9007199254740993n, 7n]);
    expect(takeAlongAxis(tensor([1, 0], { dtype: "bool" }), i32([1, 1, 0]), 0).dtype).toBe("bool");
    // non-contiguous indices
    const idx = transpose(
      i32([
        [0, 1],
        [1, 0],
        [0, 0],
      ])
    ); // shape [2, 3]
    expect(takeAlongAxis(a, idx, 1).toArray()).toEqual([
      [1, 2, 1],
      [5, 4, 4],
    ]);
  });

  it("accepts integer-valued float indices and handles empty indices", () => {
    expect(takeAlongAxis(f64([5, 6, 7]), f64([2, 0]), 0).toArray()).toEqual([7, 5]);
    const e = takeAlongAxis(a, i32([[], []]), 1);
    expect(e.shape).toEqual([2, 0]);
  });

  it("validates its inputs", () => {
    expect(() => takeAlongAxis(a, i32([[3]]), 1)).toThrow(IndexError);
    expect(() => takeAlongAxis(a, i32([[-4]]), 1)).toThrow(IndexError);
    expect(() => takeAlongAxis(a, f64([[0.5]]), 1)).toThrow(InvalidParameterError);
    expect(() => takeAlongAxis(a, tensor([[1]], { dtype: "bool" }), 1)).toThrow(DTypeError);
    expect(() => takeAlongAxis(a, i32([0, 1]), 1)).toThrow(ShapeError);
    expect(() => takeAlongAxis(a, i32([[0]]), 2)).toThrow(InvalidParameterError);
    expect(() => takeAlongAxis(tensor(["a", "b"]), i32([0]), 0)).toThrow(DTypeError);
  });
});

describe("putAlongAxis", () => {
  const a = f64([
    [10, 30, 20],
    [60, 40, 50],
  ]);

  it("matches np.put_along_axis and leaves the input unchanged", () => {
    // np.put_along_axis(a, [[0], [2]], 99, axis=1)
    const r = putAlongAxis(a, i32([[0], [2]]), 99, 1);
    expect(r.toArray()).toEqual([
      [99, 30, 20],
      [60, 40, 99],
    ]);
    expect(a.toArray()).toEqual([
      [10, 30, 20],
      [60, 40, 50],
    ]);
    // tensor values broadcast to the index shape
    expect(
      putAlongAxis(
        a,
        i32([
          [0, 1],
          [2, 2],
        ]),
        f64([
          [1, 2],
          [3, 4],
        ]),
        1
      ).toArray()
    ).toEqual([
      [1, 2, 20],
      [60, 40, 4],
    ]);
    expect(putAlongAxis(a, i32([[1, 0, 1]]), f64([7, 8, 9]), 0).toArray()).toEqual([
      [10, 8, 20],
      [7, 40, 9],
    ]);
  });

  it("lets the last write win for repeated indices", () => {
    expect(putAlongAxis(f64([0, 0, 0]), i32([1, 1]), f64([5, 6]), 0).toArray()).toEqual([0, 6, 0]);
  });

  it("supports axis null, negative axes and scalar bigint values", () => {
    expect(putAlongAxis(a, i32([0, 5]), f64([1, 2]), null).toArray()).toEqual([
      [1, 30, 20],
      [60, 40, 2],
    ]);
    expect(putAlongAxis(a, i32([[1], [0]]), 0, -1).toArray()).toEqual([
      [10, 0, 20],
      [0, 40, 50],
    ]);
    const big = int64([1n, 2n, 3n]);
    const r = putAlongAxis(big, i32([1]), 9007199254740993n, 0);
    expect(r.toArray()).toEqual([1n, 9007199254740993n, 3n]);
    expect(r.dtype).toBe("int64");
  });

  it("converts values to the array dtype and works on strided views", () => {
    const boolT = tensor([0, 0, 0], { dtype: "bool" });
    expect(putAlongAxis(boolT, i32([2]), 7, 0).toArray()).toEqual([0, 0, 1]);
    expect(putAlongAxis(i32([1, 2, 3]), i32([0]), 4.9, 0).toArray()).toEqual([4, 2, 3]);
    const t = transpose(a); // view [[10, 60], [30, 40], [20, 50]]
    expect(putAlongAxis(t, i32([[1], [1], [0]]), 0, 1).toArray()).toEqual([
      [10, 0],
      [30, 0],
      [0, 50],
    ]);
    const half = putAlongAxis(tensor([0, 0], { dtype: "float16" }), i32([0]), 0.1, 0);
    expect(half.dtype).toBe("float16");
    const h = (half.toArray() as number[])[0] as number;
    expect(Math.abs(h - 0.1)).toBeLessThan(1e-3);
    expect(h).not.toBe(0.1);
  });

  it("validates its inputs", () => {
    expect(() => putAlongAxis(a, i32([[3]]), 1, 1)).toThrow(IndexError);
    expect(() => putAlongAxis(a, i32([[0, 1]]), f64([1, 2, 3]), 1)).toThrow(ShapeError);
    expect(() => putAlongAxis(a, i32([[0]]), f64([[[1]]]), 1)).toThrow(ShapeError);
    expect(() => putAlongAxis(a, i32([0]), 1, 1)).toThrow(ShapeError);
    expect(() => putAlongAxis(a, tensor([[1]], { dtype: "bool" }), 1, 1)).toThrow(DTypeError);
    expect(() => putAlongAxis(a, i32([[0]]), tensor(["x"]), 1)).toThrow(DTypeError);
    expect(putAlongAxis(a, i32([[], []]), 1, 1).toArray()).toEqual(a.toArray());
  });
});

describe("nonzero, argwhere and countNonzero", () => {
  const a = f64([
    [0, 2, Number.NaN],
    [3, 0, 0],
  ]);

  it("nonzero returns one int32 tensor per dimension", () => {
    // np.nonzero -> (array([0, 0, 1]), array([1, 2, 0]))
    const [rows, cols] = nonzero(a);
    expect(rows?.dtype).toBe("int32");
    expect(rows?.toArray()).toEqual([0, 0, 1]);
    expect(cols?.toArray()).toEqual([1, 2, 0]);
    expect(nonzero(f64([0, 1, 0, 4]))[0]?.toArray()).toEqual([1, 3]);
  });

  it("nonzero handles empty results, views, int64, bool and 0-d input", () => {
    const none = nonzero(f64([[0, 0]]));
    expect(none.length).toBe(2);
    expect(none[0]?.shape).toEqual([0]);
    expect(nonzero(f64([]))[0]?.shape).toEqual([0]);
    // np.nonzero(a.T) -> (array([0, 1, 2]), array([1, 0, 0]))
    const [r, c] = nonzero(transpose(a));
    expect(r?.toArray()).toEqual([0, 1, 2]);
    expect(c?.toArray()).toEqual([1, 0, 0]);
    expect(nonzero(int64([0n, 5n, 0n]))[0]?.toArray()).toEqual([1]);
    expect(nonzero(tensor([1, 0, 1], { dtype: "bool" }))[0]?.toArray()).toEqual([0, 2]);
    expect(() => nonzero(tensor(5))).toThrow(ShapeError);
    expect(() => nonzero(tensor(["a"]))).toThrow(DTypeError);
  });

  it("argwhere returns an int32 [count, ndim] matrix", () => {
    // np.argwhere -> [[0, 1], [0, 2], [1, 0]]
    const w = argwhere(a);
    expect(w.dtype).toBe("int32");
    expect(w.shape).toEqual([3, 2]);
    expect(w.toArray()).toEqual([
      [0, 1],
      [0, 2],
      [1, 0],
    ]);
    // np.argwhere on a (2, 3, 4) array of arange(24) % 3 has shape (16, 3)
    const b = f64(Array.from({ length: 24 }, (_, i) => i % 3)).reshape([2, 3, 4]);
    expect(argwhere(b).shape).toEqual([16, 3]);
    expect(argwhere(f64([[0, 0]])).shape).toEqual([0, 2]);
    expect(argwhere(f64([]).reshape([0, 3])).shape).toEqual([0, 2]);
    // np.argwhere(np.array(5)).shape == (1, 0); np.array(0) -> (0, 0)
    expect(argwhere(tensor(5)).shape).toEqual([1, 0]);
    expect(argwhere(tensor(0)).shape).toEqual([0, 0]);
    expect(argwhere(transpose(a)).toArray()).toEqual([
      [0, 1],
      [1, 0],
      [2, 0],
    ]);
  });

  it("countNonzero counts all, along axes and with keepdims", () => {
    // np.count_nonzero: all -> 3, axis=0 -> [1, 1, 1], axis=1 keepdims -> [[2], [1]]
    const all = countNonzero(a);
    expect(all.dtype).toBe("int32");
    expect(all.shape).toEqual([]);
    expect(all.toArray()).toBe(3);
    expect(countNonzero(a, 0).toArray()).toEqual([1, 1, 1]);
    expect(countNonzero(a, -1, true).toArray()).toEqual([[2], [1]]);
    expect(countNonzero(a, [0, 1]).toArray()).toBe(3);
    const b = f64(Array.from({ length: 24 }, (_, i) => i % 3)).reshape([2, 3, 4]);
    expect(countNonzero(b, [0, 2]).toArray()).toEqual([4, 6, 6]);
    expect(countNonzero(b, -1).toArray()).toEqual([
      [2, 3, 3],
      [2, 3, 3],
    ]);
    expect(countNonzero(b, [0, 2], true).shape).toEqual([1, 3, 1]);
  });

  it("countNonzero handles empty, 0-d, views and invalid axes", () => {
    expect(countNonzero(f64([])).toArray()).toBe(0);
    expect(countNonzero(tensor(3)).toArray()).toBe(1);
    expect(countNonzero(tensor(0)).toArray()).toBe(0);
    expect(countNonzero(f64([]).reshape([0, 3]), 0).toArray()).toEqual([0, 0, 0]);
    expect(countNonzero(transpose(a), 1).toArray()).toEqual([1, 1, 1]);
    expect(countNonzero(int64([0n, 2n, 3n])).toArray()).toBe(2);
    expect(() => countNonzero(a, 2)).toThrow(InvalidParameterError);
    expect(() => countNonzero(a, [0, 0])).toThrow(InvalidParameterError);
    expect(() => countNonzero(tensor(["a"]))).toThrow(DTypeError);
  });
});

describe("unique options", () => {
  const nan = Number.NaN;

  it("keeps the original call form and return type", () => {
    const r = unique(f64([3, 1, 2, 1, 3]));
    expect(Object.keys(r)).toEqual(["values"]);
    expect(r.values.toArray()).toEqual([1, 2, 3]);
    const c = unique(f64([3, 1, 2, 1, 3]), true);
    expect(c.counts?.dtype).toBe("float64");
    expect(c.counts?.toArray()).toEqual([2, 1, 2]);
    expect(Object.keys(unique(f64([1]), false))).toEqual(["values"]);
  });

  it("returns only the requested fields in options form", () => {
    const x = f64([3, 1, 2, 1, 3]);
    expect(Object.keys(unique(x, {}))).toEqual(["values"]);
    // np.unique(x, return_index=True, return_inverse=True, return_counts=True)
    const r = unique(x, { returnIndex: true, returnInverse: true, returnCounts: true });
    expect(r.values.toArray()).toEqual([1, 2, 3]);
    expect(r.indices.toArray()).toEqual([1, 2, 0]);
    expect(r.inverse.toArray()).toEqual([2, 0, 1, 0, 2]);
    expect(r.counts.toArray()).toEqual([2, 1, 2]);
    expect(r.indices.dtype).toBe("int32");
    expect(r.inverse.dtype).toBe("int32");
    expect(r.counts.dtype).toBe("int32");
    const only = unique(x, { returnCounts: true });
    expect(Object.keys(only)).toEqual(["values", "counts"]);
    expect(only.counts.dtype).toBe("int32");
    expect(Object.keys(unique(x, { returnInverse: true }))).toEqual(["values", "inverse"]);
  });

  it("collapses NaN and merges signed zeros like NumPy", () => {
    // np.unique([3, 1, 2, 1, 3, nan, nan, 0., -0.], ...)
    //   -> [0, 1, 2, 3, nan], index [7, 1, 2, 0, 5], inverse [3, 1, 2, 1, 3, 4, 4, 0, 0], counts [2, 2, 1, 2, 2]
    const r = unique(f64([3, 1, 2, 1, 3, nan, nan, 0, -0]), {
      returnIndex: true,
      returnInverse: true,
      returnCounts: true,
    });
    const v = r.values.toArray() as number[];
    expect(v.slice(0, 4)).toEqual([0, 1, 2, 3]);
    expect(Object.is(v[0], 0)).toBe(true);
    expect(v[4]).toBeNaN();
    expect(r.indices.toArray()).toEqual([7, 1, 2, 0, 5]);
    expect(r.inverse.toArray()).toEqual([3, 1, 2, 1, 3, 4, 4, 0, 0]);
    expect(r.counts.toArray()).toEqual([2, 2, 1, 2, 2]);
    // np.unique([-0., 0., 1.], return_index=True) keeps the first zero, which is -0
    const z = unique(f64([-0, 0, 1]), { returnIndex: true });
    expect(Object.is((z.values.toArray() as number[])[0], -0)).toBe(true);
    expect(z.indices.toArray()).toEqual([0, 2]);
  });

  it("gives the inverse the shape of the input when no axis is given", () => {
    // np.unique([[3, 1], [1, 3], [2, 2]], return_index/inverse/counts)
    const m = f64([
      [3, 1],
      [1, 3],
      [2, 2],
    ]);
    const r = unique(m, { returnIndex: true, returnInverse: true, returnCounts: true });
    expect(r.values.toArray()).toEqual([1, 2, 3]);
    expect(r.indices.toArray()).toEqual([1, 4, 0]);
    expect(r.inverse.toArray()).toEqual([
      [2, 0],
      [0, 2],
      [1, 1],
    ]);
    expect(r.counts.toArray()).toEqual([2, 2, 2]);
    // a strided view is read in logical order
    // np.unique(m.T, return_index=True) -> index [1, 2, 0]
    expect(unique(transpose(m), { returnIndex: true }).indices.toArray()).toEqual([1, 2, 0]);
  });

  it("finds unique sub-tensors along an axis", () => {
    const m = f64([
      [3, 1],
      [1, 3],
      [2, 2],
      [1, 3],
    ]);
    // np.unique(m, axis=0, ...) -> [[1, 3], [2, 2], [3, 1]], index [1, 2, 0], inverse [2, 0, 1, 0], counts [2, 1, 1]
    const r0 = unique(m, { axis: 0, returnIndex: true, returnInverse: true, returnCounts: true });
    expect(r0.values.toArray()).toEqual([
      [1, 3],
      [2, 2],
      [3, 1],
    ]);
    expect(r0.indices.toArray()).toEqual([1, 2, 0]);
    expect(r0.inverse.toArray()).toEqual([2, 0, 1, 0]);
    expect(r0.counts.toArray()).toEqual([2, 1, 1]);
    // np.unique(a, axis=1) for a = [[3, 1], [1, 3], [2, 2]] -> [[1, 3], [3, 1], [2, 2]], index [1, 0]
    const a = f64([
      [3, 1],
      [1, 3],
      [2, 2],
    ]);
    const r1 = unique(a, { axis: 1, returnIndex: true, returnInverse: true });
    expect(r1.values.toArray()).toEqual([
      [1, 3],
      [3, 1],
      [2, 2],
    ]);
    expect(r1.indices.toArray()).toEqual([1, 0]);
    expect(r1.inverse.toArray()).toEqual([1, 0]);
    // negative axis and a 3-d input
    const cube = f64([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [1, 2],
        [3, 4],
      ],
    ]);
    const c = unique(cube, { axis: -3, returnCounts: true });
    expect(c.values.shape).toEqual([1, 2, 2]);
    expect(c.counts.toArray()).toEqual([2]);
  });

  it("handles axis edge cases like NumPy", () => {
    // np.unique(np.empty((0, 3)), axis=0) -> shape (0, 3)
    const e = unique(f64([]).reshape([0, 3]), {
      axis: 0,
      returnIndex: true,
      returnInverse: true,
      returnCounts: true,
    });
    expect(e.values.shape).toEqual([0, 3]);
    expect(e.indices.shape).toEqual([0]);
    expect(e.inverse.shape).toEqual([0]);
    // np.unique(np.empty((3, 0)), axis=0, ...) -> values (1, 0), index [0], inverse [0, 0, 0], counts [3]
    const z = unique(f64([]).reshape([3, 0]), {
      axis: 0,
      returnIndex: true,
      returnInverse: true,
      returnCounts: true,
    });
    expect(z.values.shape).toEqual([1, 0]);
    expect(z.indices.toArray()).toEqual([0]);
    expect(z.inverse.toArray()).toEqual([0, 0, 0]);
    expect(z.counts.toArray()).toEqual([3]);
    // rows holding NaN compare equal to each other and sort last
    const n = unique(
      f64([
        [1, Number.NaN],
        [0, 1],
        [1, Number.NaN],
      ]),
      { axis: 0, returnCounts: true }
    );
    expect(n.values.shape).toEqual([2, 2]);
    expect(n.counts.toArray()).toEqual([1, 2]);
    expect(() => unique(tensor(5), { axis: 0 })).toThrow(InvalidParameterError);
    expect(() => unique(f64([[1]]), { axis: 2 })).toThrow(InvalidParameterError);
    expect(() => unique(tensor(["a"]), { returnIndex: true })).toThrow(DTypeError);
  });

  it("is exact for int64, keeps dtypes and handles empty and 0-d input", () => {
    const a = 9007199254740992n;
    const b = 9007199254740993n;
    const r = unique(int64([b, a, b]), {
      returnIndex: true,
      returnInverse: true,
      returnCounts: true,
    });
    expect(r.values.dtype).toBe("int64");
    expect(r.values.toArray()).toEqual([a, b]);
    expect(r.indices.toArray()).toEqual([1, 0]);
    expect(r.inverse.toArray()).toEqual([1, 0, 1]);
    expect(r.counts.toArray()).toEqual([1, 2]);
    const rows = unique(int64([b, a, b, a], [2, 2]), { axis: 0, returnCounts: true });
    expect(rows.values.toArray()).toEqual([[b, a]]);
    expect(rows.counts.toArray()).toEqual([2]);
    expect(unique(i32([2, 1, 2]), { returnIndex: true }).values.dtype).toBe("int32");
    expect(unique(tensor([1, 1, 0], { dtype: "bool" }), { returnInverse: true }).values.dtype).toBe(
      "bool"
    );
    const empty = unique(f64([]), { returnIndex: true, returnInverse: true, returnCounts: true });
    expect(empty.values.shape).toEqual([0]);
    expect(empty.indices.shape).toEqual([0]);
    expect(empty.inverse.shape).toEqual([0]);
    expect(empty.counts.shape).toEqual([0]);
    const scalar = unique(tensor(5), { returnInverse: true, returnIndex: true });
    expect(scalar.values.toArray()).toEqual([5]);
    expect(scalar.inverse.shape).toEqual([]);
    expect(scalar.inverse.toArray()).toBe(0);
  });

  it("agrees with NumPy on a large input (radix path)", () => {
    // Deterministic data: 20000 values in [0, 50)
    const n = 20000;
    const data = new Float64Array(n);
    let s = 12345;
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) % 2147483648;
      data[i] = Math.floor(s / 4096) % 50;
    }
    const t = Tensor.fromTypedArray({ data, shape: [n], dtype: "float64", device: "cpu" });
    const r = unique(t, { returnIndex: true, returnInverse: true, returnCounts: true });
    expect(r.values.size).toBe(50);
    const vals = r.values.data as Float64Array;
    const inv = r.inverse.data as Int32Array;
    const idx = r.indices.data as Int32Array;
    const counts = r.counts.data as Int32Array;
    let total = 0;
    for (let k = 0; k < 50; k++) {
      expect(vals[k]).toBe(k);
      expect(data[idx[k] as number]).toBe(k);
      // first occurrence: nothing earlier has this value
      for (let j = 0; j < (idx[k] as number); j++) expect(data[j]).not.toBe(k);
      total += counts[k] as number;
    }
    expect(total).toBe(n);
    for (let i = 0; i < n; i += 97) expect(vals[inv[i] as number]).toBe(data[i]);
  });
});

describe("searchsorted dtype", () => {
  it("returns int32 indices with the same positions as before", () => {
    const r = searchsorted(f64([1, 3, 5, 7]), f64([2, 4, 6, 7, 0, 9]));
    expect(r.dtype).toBe("int32");
    expect(r.data).toBeInstanceOf(Int32Array);
    // np.searchsorted([1, 3, 5, 7], [2, 4, 6, 7, 0, 9]) -> [1, 2, 3, 3, 0, 4]
    expect(r.toArray()).toEqual([1, 2, 3, 3, 0, 4]);
    expect(searchsorted(f64([1, 3, 5, 7]), f64([3, 5]), "right").toArray()).toEqual([2, 3]);
    expect(searchsorted(f64([1, 2, 3]), f64([[0, 4]])).shape).toEqual([1, 2]);
    expect(searchsorted(f64([]), f64([1])).dtype).toBe("int32");
    // the result is a valid index tensor
    const sorted = f64([10, 20, 30]);
    const pos = searchsorted(sorted, f64([15, 25]));
    expect(takeAlongAxis(sorted, pos, 0).toArray()).toEqual([20, 30]);
  });
});

describe("cross", () => {
  it("computes the single-vector product and keeps the float dtype", () => {
    expect(cross(f64([1, 2, 3]), f64([4, 5, 6])).toArray()).toEqual([-3, 6, -3]);
    expect(cross(tensor([1, 0, 0]), tensor([0, 1, 0])).dtype).toBe("float32");
    expect(cross(f64([1, 0, 0]), f64([0, 1, 0])).dtype).toBe("float64");
    expect(cross(tensor([1, 2, 3], { dtype: "float32" }), f64([4, 5, 6])).dtype).toBe("float64");
    expect(
      cross(tensor([1, 2, 3], { dtype: "float16" }), tensor([4, 5, 6], { dtype: "float16" })).dtype
    ).toBe("float16");
    expect(
      cross(tensor([1, 2, 3], { dtype: "float16" }), tensor([4, 5, 6], { dtype: "bfloat16" })).dtype
    ).toBe("float32");
  });

  it("supports batched inputs with broadcasting", () => {
    // np.cross([[1, 2, 3], [4, 5, 6]] float32, [7, 8, 9]) -> [[-6, 12, -6], [-3, 6, -3]]
    const r = cross(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float32" }
      ),
      f64([7, 8, 9]).astype("float32")
    );
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([
      [-6, 12, -6],
      [-3, 6, -3],
    ]);
    // [2, 1, 3] x [2, 3] -> [2, 2, 3]
    const a = f64([[[1, 0, 0]], [[0, 1, 0]]]);
    const b = f64([
      [0, 1, 0],
      [0, 0, 1],
    ]);
    const c = cross(a, b);
    expect(c.shape).toEqual([2, 2, 3]);
    expect(c.toArray()).toEqual([
      [
        [0, 0, 1],
        [0, -1, 0],
      ],
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
    ]);
    expect(() =>
      cross(
        f64([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        f64([
          [1, 2, 3],
          [1, 2, 3],
          [1, 2, 3],
        ])
      )
    ).toThrow(ShapeError);
  });

  it("supports the axis option and negative axes", () => {
    const a = f64([
      [1, 0],
      [0, 1],
      [0, 0],
    ]); // two vectors stored along axis 0
    const b = f64([
      [0, 0],
      [0, 0],
      [1, 1],
    ]);
    // columns: [1,0,0] x [0,0,1] = [0,-1,0]; [0,1,0] x [0,0,1] = [1,0,0]
    const r = cross(a, b, { axis: 0 });
    expect(r.shape).toEqual([3, 2]);
    expect(r.toArray()).toEqual([
      [0, 1],
      [-1, 0],
      [0, 0],
    ]);
    expect(cross(a, b, 0).toArray()).toEqual(r.toArray());
    // np.cross(a, b, axis=-2) for (2, 3, 4) arrays equals axis=1
    const x = f64(Array.from({ length: 24 }, (_, i) => (i * 7) % 11)).reshape([2, 3, 4]);
    const y = f64(Array.from({ length: 24 }, (_, i) => (i * 5) % 13)).reshape([2, 3, 4]);
    expect(cross(x, y, { axis: -2 }).toArray()).toEqual(cross(x, y, { axis: 1 }).toArray());
    // np.cross(x, y, axis=1)[0, :, 0] = [-1, 0, 0]
    const col = (cross(x, y, { axis: 1 }).toArray() as number[][][])[0]?.map((row) => row[0]);
    expect(col).toEqual([-1, 0, 0]);
  });

  it("reads strided views and keeps integer dtypes exact", () => {
    const m = f64([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    // transpose(m) is [[1, 2, 3], [4, 5, 6]] as a view
    expect(cross(transpose(m), f64([7, 8, 9])).toArray()).toEqual([
      [-6, 12, -6],
      [-3, 6, -3],
    ]);
    // np.cross of int32 vectors stays int32
    const ci = cross(i32([1, 2, 3]), i32([4, 5, 6]));
    expect(ci.dtype).toBe("int32");
    expect(ci.toArray()).toEqual([-3, 6, -3]);
    // uint8 stays uint8 and wraps like NumPy: np.cross([1, 2, 3], [4, 5, 6]) as uint8
    const cu = cross(tensor([1, 2, 3], { dtype: "uint8" }), tensor([4, 5, 6], { dtype: "uint8" }));
    expect(cu.dtype).toBe("uint8");
    expect(cu.toArray()).toEqual([253, 6, 253]);
    expect(cross(tensor([1, 0, 0], { dtype: "uint8" }), i32([0, 1, 0])).dtype).toBe("int32");
    expect(
      cross(tensor([1, 0, 0], { dtype: "bool" }), tensor([0, 1, 0], { dtype: "bool" })).dtype
    ).toBe("int32");
    const big = 4611686018427387904n; // 2^62
    const cb = cross(int64([big, 0n, 0n]), int64([0n, 1n, 0n]));
    expect(cb.dtype).toBe("int64");
    expect(cb.toArray()).toEqual([0n, 0n, big]);
    expect(cross(int64([1n, 2n, 3n]), f64([4, 5, 6])).dtype).toBe("float64");
  });

  it("handles empty batches and validates its inputs", () => {
    const empty = cross(f64([]).reshape([0, 3]), f64([1, 2, 3]));
    expect(empty.shape).toEqual([0, 3]);
    expect(() => cross(f64([1, 2]), f64([3, 4]))).toThrow(ShapeError);
    expect(() => cross(f64([[1, 2]]), f64([4, 5, 6]))).toThrow(ShapeError);
    expect(() => cross(f64([1, 2, 3]), f64([4, 5]))).toThrow(ShapeError);
    expect(() => cross(tensor(1), f64([1, 2, 3]))).toThrow(ShapeError);
    expect(() => cross(f64([1, 2, 3]), f64([1, 2, 3]), { axis: 1 })).toThrow(InvalidParameterError);
    expect(() => cross(tensor(["a", "b", "c"]), f64([1, 2, 3]))).toThrow(DTypeError);
  });
});
