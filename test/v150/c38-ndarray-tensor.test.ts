import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core";
import { gather, reshape, slice, Tensor, tensor, transpose } from "../../src/ndarray";
import { expandDims, squeeze, unsqueeze } from "../../src/ndarray/tensor/shape_ops";
import { normalizeIndex, normalizeRange } from "../../src/ndarray/tensor/slice_helpers";
import { isContiguous } from "../../src/ndarray/tensor/strides";

describe("c38 strides.isContiguous", () => {
  it("accepts row-major layouts and rejects gaps or permutations", () => {
    expect(isContiguous([2, 3], [3, 1])).toBe(true);
    expect(isContiguous([2, 3], [1, 2])).toBe(false);
    expect(isContiguous([2, 3], [6, 2])).toBe(false);
    expect(isContiguous([2, 3], [3])).toBe(false);
    expect(isContiguous([], [])).toBe(true);
  });

  it("keeps the strict size-1 behavior callers rely on", () => {
    expect(isContiguous([1, 3], [3, 1])).toBe(true);
    expect(isContiguous([3, 1], [1, 3])).toBe(false);
  });
});

describe("c38 slice_helpers", () => {
  it("rejects NaN and fractional indices instead of returning garbage", () => {
    expect(() => normalizeIndex(Number.NaN, 3)).toThrow(InvalidParameterError);
    expect(() => normalizeIndex(1.5, 3)).toThrow(InvalidParameterError);
    expect(() => normalizeIndex(3, 3)).toThrow(IndexError);
    expect(normalizeIndex(-1, 3)).toBe(2);
    const t = tensor([10, 20, 30]);
    expect(() => slice(t, 1.5)).toThrow(/integer/);
    expect(() => slice(t, Number.NaN)).toThrow(/integer/);
  });

  it("rejects NaN and fractional slice bounds", () => {
    expect(() => normalizeRange({ start: 0.5 }, 4)).toThrow(InvalidParameterError);
    expect(() => normalizeRange({ end: Number.NaN }, 4)).toThrow(InvalidParameterError);
    expect(() => normalizeRange({ start: 1, end: 2.5, step: -1 }, 4)).toThrow(/end/);
    expect(() => normalizeRange({ step: 1.5 }, 4)).toThrow(InvalidParameterError);
  });

  it("clamps infinite bounds like NumPy clamps out-of-range ones", () => {
    const t = tensor([0, 1, 2, 3, 4]);
    expect(slice(t, { start: 1, end: Number.POSITIVE_INFINITY }).toArray()).toEqual([1, 2, 3, 4]);
    expect(slice(t, { start: Number.NEGATIVE_INFINITY, end: 2 }).toArray()).toEqual([0, 1]);
    // x[::-1][1:4] -> [3, 2, 1]; x[inf:-inf:-1] -> [4, 3, 2, 1, 0]
    expect(
      slice(t, {
        start: Number.POSITIVE_INFINITY,
        end: Number.NEGATIVE_INFINITY,
        step: -1,
      }).toArray()
    ).toEqual([4, 3, 2, 1, 0]);
  });
});

describe("c38 reshape / transpose / unsqueeze validation", () => {
  it("names the offending dimension for invalid reshape sizes", () => {
    const t = tensor([1, 2, 3, 4]);
    // -3 does not divide 4, so the old code reported a failed inference instead.
    expect(() => reshape(t, [-3, -1])).toThrow(DataValidationError);
    expect(() => reshape(t, [-3, -1])).toThrow(/shape\[0\]/);
    expect(() => reshape(t, [1.5, -1])).toThrow(DataValidationError);
    expect(() => reshape(t, [-2, 2])).toThrow(DataValidationError);
    expect(() => reshape(t, [Number.NaN, 2])).toThrow(DataValidationError);
    expect(() => reshape(t, [2, -1, -1])).toThrow(/Only one dimension/);
    expect(() => reshape(t, [3, -1])).toThrow(ShapeError);
    expect(reshape(t, [2, -1]).shape).toEqual([2, 2]);
  });

  it("copies when reshaping a non-contiguous tensor", () => {
    const m = transpose(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(reshape(m, [6]).toArray()).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("rejects fractional transpose axes with a typed error", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => transpose(m, [0.5, 1])).toThrow(InvalidParameterError);
    expect(() => transpose(m, [0, 0])).toThrow(/duplicate/);
    expect(transpose(m, [-1, -2]).toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("rejects fractional and NaN unsqueeze axes", () => {
    const t = tensor([1, 2, 3]);
    expect(() => unsqueeze(t, 0.5)).toThrow(InvalidParameterError);
    expect(() => expandDims(t, Number.NaN)).toThrow(InvalidParameterError);
    expect(unsqueeze(t, -1).shape).toEqual([3, 1]);
    expect(unsqueeze(t, 1).shape).toEqual([3, 1]);
    expect(() => unsqueeze(t, 2)).toThrow(InvalidParameterError);
  });

  it("squeeze accepts a readonly axis list", () => {
    const t = tensor([[[1], [2], [3]]]);
    const axes: readonly number[] = [0, 2];
    expect(squeeze(t, axes).shape).toEqual([3]);
    expect(() => squeeze(t, 1)).toThrow(ShapeError);
  });
});

describe("c38 gather", () => {
  const a = Array.from({ length: 24 }, (_, i) => i);
  const t3 = tensor(a).reshape([2, 3, 4]);

  it("matches numpy.take on a middle axis", () => {
    const r = gather(t3, tensor([2, 0, 2], { dtype: "int32" }), 1);
    expect(r.shape).toEqual([2, 3, 4]);
    expect(r.toArray()).toEqual([
      [
        [8, 9, 10, 11],
        [0, 1, 2, 3],
        [8, 9, 10, 11],
      ],
      [
        [20, 21, 22, 23],
        [12, 13, 14, 15],
        [20, 21, 22, 23],
      ],
    ]);
  });

  it("matches numpy.take on a negative axis", () => {
    const r = gather(t3, tensor([3, 0]), -1);
    expect(r.toArray()).toEqual([
      [
        [3, 0],
        [7, 4],
        [11, 8],
      ],
      [
        [15, 12],
        [19, 16],
        [23, 20],
      ],
    ]);
  });

  it("reads non-contiguous (transposed) input and strided indices", () => {
    const tt = transpose(t3, [2, 0, 1]); // shape [4, 2, 3]
    const r = gather(tt, tensor([3, 1]), 0);
    expect(r.toArray()).toEqual([
      [
        [3, 7, 11],
        [15, 19, 23],
      ],
      [
        [1, 5, 9],
        [13, 17, 21],
      ],
    ]);
    const m = transpose(
      tensor([
        [0, 1, 2],
        [3, 4, 5],
      ])
    ); // [[0,3],[1,4],[2,5]]
    expect(gather(m, tensor([1, 1, 0]), 0).toArray()).toEqual([
      [1, 4],
      [1, 4],
      [0, 3],
    ]);
    const idx = slice(tensor([9, 1, 9, 0, 9]), { start: 1, end: 5, step: 2 }); // contiguous copy [1, 0]
    expect(gather(m, idx, 1).toArray()).toEqual([
      [3, 0],
      [4, 1],
      [5, 2],
    ]);
  });

  it("works for int64, string and 1D tensors and honors int64 indices", () => {
    const i64 = (v: bigint[]) =>
      Tensor.fromTypedArray({
        data: new BigInt64Array(v),
        shape: [v.length],
        dtype: "int64",
        device: "cpu",
      });
    const big = i64([10n, 20n, 30n]);
    const r = gather(big, i64([2n, 0n]), 0);
    expect(r.dtype).toBe("int64");
    expect(r.toArray()).toEqual([30n, 10n]);

    const s = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [2, 2] });
    expect(gather(s, tensor([1, 0, 1]), 1).toArray()).toEqual([
      ["b", "a", "b"],
      ["d", "c", "d"],
    ]);
  });

  it("returns empty results for empty indices and keeps the other dims", () => {
    const r = gather(t3, tensor([] as number[], { dtype: "int32" }), 1);
    expect(r.shape).toEqual([2, 0, 4]);
    expect(r.size).toBe(0);
  });

  it("validates indices even when the output is empty", () => {
    const empty = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    expect(() => gather(empty, tensor([5]), 1)).toThrow(IndexError);
    expect(() => gather(empty, tensor([0]), 0)).toThrow(IndexError);
    expect(gather(empty, tensor([2, 1]), 1).shape).toEqual([0, 2]);
  });

  it("throws typed errors for bad indices", () => {
    const t = tensor([1, 2, 3]);
    expect(() => gather(t, tensor([3]), 0)).toThrow(IndexError);
    expect(() => gather(t, tensor([-1]), 0)).toThrow(IndexError);
    expect(() => gather(t, tensor([0.5]), 0)).toThrow(InvalidParameterError);
    expect(() => gather(t, tensor([Number.NaN]), 0)).toThrow(InvalidParameterError);
    expect(() => gather(t, tensor([[0]]), 0)).toThrow(InvalidParameterError);
    expect(() => gather(t, Tensor.fromStringArray({ data: ["a"], shape: [1] }), 0)).toThrow(
      DTypeError
    );
    expect(() => gather(t, tensor([0]), 1)).toThrow(InvalidParameterError);
  });
});
