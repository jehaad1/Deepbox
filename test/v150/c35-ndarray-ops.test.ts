/**
 * Regression tests for src/ndarray/ops/utils.ts (v1.5.0 audit).
 *
 * Reference values were computed with NumPy 2.4 unless a comment says otherwise.
 */
import { describe, expect, it } from "vitest";
import { DTypeError, IndexError, InvalidParameterError, ShapeError } from "../../src/core";
import { Tensor, tensor, transpose } from "../../src/ndarray";
import {
  atleast1d,
  atleast2d,
  bincount,
  booleanIndex,
  broadcastTo,
  clone,
  cross,
  diag,
  diagonal,
  emptyLike,
  fancyIndex,
  flip,
  fullLike,
  histogram,
  indexSelect,
  isin,
  meshgrid,
  moveaxis,
  nanmax,
  nanmean,
  nanmin,
  nanstd,
  nansum,
  onesLike,
  pad,
  roll,
  rot90,
  scatter,
  searchsorted,
  swapaxes,
  tril,
  triu,
  unique,
  where,
  zerosLike,
} from "../../src/ndarray/ops/utils";

const f64 = (data: unknown): Tensor => tensor(data as never, { dtype: "float64" });

function int64(values: readonly bigint[], shape: number[] = [values.length]): Tensor {
  return Tensor.fromTypedArray({
    data: new BigInt64Array(values),
    shape,
    dtype: "int64",
    device: "cpu",
  });
}

/** Row-major copy of the logical contents, as plain numbers. */
function values(t: Tensor): number[] {
  const out: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) for (const e of x) walk(e);
    else out.push(Number(x));
  };
  walk(t.toArray());
  return out;
}

/** A 1-D tensor of `vals` whose elements sit at every second slot of the buffer (stride 2). */
function stridedVector(vals: number[]): Tensor {
  const data = new Float64Array(vals.length * 2).fill(99);
  vals.forEach((v, i) => {
    data[i * 2] = v;
  });
  return Tensor.fromTypedArray({
    data,
    shape: [vals.length],
    dtype: "float64",
    device: "cpu",
    strides: [2],
  });
}

/** A 2x3 matrix [[0,1,2],[3,4,5]] and its transposed (strided) view. */
const m23 = (): Tensor =>
  f64([
    [0, 1, 2],
    [3, 4, 5],
  ]);

describe("where", () => {
  it("broadcasts a condition against strided inputs", () => {
    const x = transpose(m23()); // [[0,3],[1,4],[2,5]]
    const y = f64([
      [10, 20],
      [30, 40],
      [50, 60],
    ]);
    const out = where(f64([1, 0]).astype("bool"), x, y);
    expect(out.toArray()).toEqual([
      [0, 20],
      [1, 40],
      [2, 60],
    ]);
  });

  it("selects int64 values without precision loss", () => {
    const big = 9007199254740993n;
    const out = where(tensor([1, 0], { dtype: "bool" }), int64([big, 1n]), int64([2n, big]));
    expect(out.toArray()).toEqual([big, big]);
  });

  it("treats NaN in the condition as true", () => {
    const out = where(f64([Number.NaN, 0]), f64([1, 2]), f64([3, 4]));
    expect(values(out)).toEqual([1, 4]);
  });
});

describe("*_like", () => {
  it("full_like stores 1 for any non-zero value in a bool tensor", () => {
    // np.full_like(np.array([True, False]), 7) -> [True, True]
    const out = fullLike(tensor([1, 0], { dtype: "bool" }), 7);
    expect(out.dtype).toBe("bool");
    expect(values(out)).toEqual([1, 1]);
    expect(values(fullLike(tensor([1, 0], { dtype: "bool" }), Number.NaN))).toEqual([1, 1]);
    expect(values(fullLike(tensor([1, 0], { dtype: "bool" }), 0))).toEqual([0, 0]);
  });

  it("full_like truncates toward zero for integer dtypes", () => {
    // np.full_like(np.array([1, 2], dtype=np.int32), 1.9) -> [1, 1]
    expect(values(fullLike(tensor([1, 2], { dtype: "int32" }), 1.9))).toEqual([1, 1]);
    expect(values(fullLike(tensor([1, 2], { dtype: "int32" }), -1.9))).toEqual([-1, -1]);
  });

  it("full_like rejects NaN and Infinity for integer dtypes", () => {
    expect(() => fullLike(tensor([1], { dtype: "int32" }), Number.NaN)).toThrow(
      InvalidParameterError
    );
    expect(() => fullLike(tensor([1], { dtype: "uint8" }), Number.POSITIVE_INFINITY)).toThrow(
      InvalidParameterError
    );
    expect(() => fullLike(int64([1n]), Number.NaN)).toThrow(InvalidParameterError);
  });

  it("full_like on int64 accepts fractional numbers and bigint values", () => {
    expect(fullLike(int64([1n, 2n]), 2.7).toArray()).toEqual([2n, 2n]);
    expect(fullLike(int64([1n]), 9007199254740993n).toArray()).toEqual([9007199254740993n]);
  });

  it("honours the dtype option", () => {
    const base = tensor([1.5, 2.5]);
    expect(zerosLike(base, { dtype: "int32" }).dtype).toBe("int32");
    expect(onesLike(base, { dtype: "int64" }).toArray()).toEqual([1n, 1n]);
    expect(emptyLike(base, { dtype: "uint8" }).dtype).toBe("uint8");
    expect(fullLike(base, 3, { dtype: "int32" }).toArray()).toEqual([3, 3]);
  });

  it("ignores a stray numeric second argument (Array.prototype.map usage)", () => {
    const out = [tensor([1, 2]), tensor([3])].map(zerosLike as (t: Tensor) => Tensor);
    expect(out.map((t) => t.shape)).toEqual([[2], [1]]);
  });
});

describe("clone", () => {
  it("materializes strided views in logical order", () => {
    const out = clone(transpose(m23()));
    expect(out.toArray()).toEqual([
      [0, 3],
      [1, 4],
      [2, 5],
    ]);
    expect(out.strides).toEqual([2, 1]);
    expect(out.offset).toBe(0);
  });

  it("copies string tensors, including views", () => {
    const s = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [2, 2] });
    expect(clone(swapaxes(s, 0, 1)).toArray()).toEqual([
      ["a", "c"],
      ["b", "d"],
    ]);
  });
});

describe("diag / diagonal", () => {
  it("rejects a non-integer offset", () => {
    expect(() => diag(tensor([1, 2]), 0.5)).toThrow(InvalidParameterError);
    expect(() => diagonal(m23(), Number.NaN)).toThrow(InvalidParameterError);
  });

  it("builds off-diagonals like NumPy", () => {
    // np.diag([1, 2], -1) -> [[0,0,0],[1,0,0],[0,2,0]]
    expect(diag(tensor([1, 2]), -1).toArray()).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 2, 0],
    ]);
  });

  it("extracts diagonals of strided views", () => {
    const a = f64([
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
    ]);
    // np.diagonal(a.T, 1) -> [3, 7]
    expect(values(diagonal(transpose(a), 1))).toEqual([3, 7]);
    expect(values(diagonal(a, 5))).toEqual([]);
    expect(values(diagonal(a, -5))).toEqual([]);
  });
});

describe("triu / tril", () => {
  it("works on batches of matrices", () => {
    const b = f64(
      Array.from({ length: 2 }, (_, i) =>
        Array.from({ length: 3 }, (_, j) => Array.from({ length: 4 }, (_, k) => i * 12 + j * 4 + k))
      )
    );
    // np.tril(b, 1)
    expect(tril(b, 1).toArray()).toEqual([
      [
        [0, 1, 0, 0],
        [4, 5, 6, 0],
        [8, 9, 10, 11],
      ],
      [
        [12, 13, 0, 0],
        [16, 17, 18, 0],
        [20, 21, 22, 23],
      ],
    ]);
    // np.triu(b, 1)
    expect(triu(b, 1).toArray()).toEqual([
      [
        [0, 1, 2, 3],
        [0, 0, 6, 7],
        [0, 0, 0, 11],
      ],
      [
        [0, 13, 14, 15],
        [0, 0, 18, 19],
        [0, 0, 0, 23],
      ],
    ]);
  });

  it("handles strided views", () => {
    // np.triu(a.T), np.tril(a.T, -1) for a = [[0,1,2],[3,4,5]]
    expect(triu(transpose(m23())).toArray()).toEqual([
      [0, 3],
      [0, 4],
      [0, 0],
    ]);
    expect(tril(transpose(m23()), -1).toArray()).toEqual([
      [0, 0],
      [1, 0],
      [2, 5],
    ]);
  });

  it("validates rank and offset", () => {
    expect(() => triu(tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => tril(m23(), 1.5)).toThrow(InvalidParameterError);
  });
});

describe("flip", () => {
  it("rejects repeated axes", () => {
    // np.flip(a, (0, 0)) raises ValueError
    expect(() => flip(m23(), [0, 0])).toThrow(InvalidParameterError);
    expect(() => flip(m23(), [0, -2])).toThrow(InvalidParameterError);
  });

  it("flips strided views correctly", () => {
    const t = transpose(m23()); // [[0,3],[1,4],[2,5]]
    expect(flip(t, 0).toArray()).toEqual([
      [2, 5],
      [1, 4],
      [0, 3],
    ]);
    expect(flip(t).toArray()).toEqual([
      [5, 2],
      [4, 1],
      [3, 0],
    ]);
  });

  it("accepts a single axis and an empty axis list", () => {
    expect(flip(m23(), 1).toArray()).toEqual([
      [2, 1, 0],
      [5, 4, 3],
    ]);
    expect(flip(m23(), []).toArray()).toEqual(m23().toArray());
  });

  it("flips scalars and empty tensors", () => {
    expect(flip(tensor(5)).toArray()).toBe(5);
    expect(flip(tensor([])).shape).toEqual([0]);
  });
});

describe("roll", () => {
  it("rejects non-integer shifts", () => {
    expect(() => roll(tensor([1, 2, 3]), 1.5)).toThrow(InvalidParameterError);
    expect(() => roll(tensor([1, 2, 3]), Number.NaN, 0)).toThrow(InvalidParameterError);
  });

  it("supports a shift per axis", () => {
    // np.roll(a, (1, 1), axis=(0, 1))
    expect(roll(m23(), [1, 1], [0, 1]).toArray()).toEqual([
      [5, 3, 4],
      [2, 0, 1],
    ]);
    // np.roll(a, 1, axis=(0, 1)): one shift is applied to every axis
    expect(roll(m23(), 1, [0, 1]).toArray()).toEqual([
      [5, 3, 4],
      [2, 0, 1],
    ]);
    // shifts along a repeated axis add up
    expect(roll(m23(), [1, 2], [1, 1]).toArray()).toEqual(roll(m23(), 3, 1).toArray());
    expect(() => roll(m23(), [1, 2, 3], [0, 1])).toThrow(InvalidParameterError);
    expect(() => roll(m23(), [1, 2])).toThrow(InvalidParameterError);
  });

  it("rolls strided views", () => {
    const t = transpose(m23()); // [[0,3],[1,4],[2,5]]
    // np.roll(a.T, 1, axis=0)
    expect(roll(t, 1, 0).toArray()).toEqual([
      [2, 5],
      [0, 3],
      [1, 4],
    ]);
    // np.roll(a.T, 2) rolls the flattened array
    expect(roll(t, 2).toArray()).toEqual([
      [2, 5],
      [0, 3],
      [1, 4],
    ]);
  });

  it("handles negative and oversized shifts and int64", () => {
    expect(values(roll(tensor([1, 2, 3, 4, 5]), -7))).toEqual([3, 4, 5, 1, 2]);
    expect(roll(int64([1n, 2n, 3n]), 1).toArray()).toEqual([3n, 1n, 2n]);
    expect(roll(tensor([]), 3).shape).toEqual([0]);
  });
});

describe("pad", () => {
  const v = (): Tensor => tensor([1, 2, 3]);

  it("matches NumPy for every mode with pads longer than the input", () => {
    expect(values(pad(v(), [[5, 4]], "reflect"))).toEqual([2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3]);
    expect(values(pad(v(), [[5, 4]], "symmetric"))).toEqual([2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1]);
    expect(values(pad(v(), [[5, 4]], "edge"))).toEqual([1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3]);
    expect(values(pad(v(), [[5, 4]], "wrap"))).toEqual([2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]);
    expect(values(pad(v(), [[5, 4]], "replicate"))).toEqual(values(pad(v(), [[5, 4]], "edge")));
    expect(values(pad(v(), [[5, 4]], "circular"))).toEqual(values(pad(v(), [[5, 4]], "wrap")));
  });

  it("matches NumPy for 2-D inputs", () => {
    const a = f64([
      [0, 1, 2],
      [3, 4, 5],
    ]);
    const w = [
      [1, 2],
      [2, 1],
    ] as const;
    expect(pad(a, w, "reflect").toArray()).toEqual([
      [5, 4, 3, 4, 5, 4],
      [2, 1, 0, 1, 2, 1],
      [5, 4, 3, 4, 5, 4],
      [2, 1, 0, 1, 2, 1],
      [5, 4, 3, 4, 5, 4],
    ]);
    expect(pad(a, w, "symmetric").toArray()).toEqual([
      [1, 0, 0, 1, 2, 2],
      [1, 0, 0, 1, 2, 2],
      [4, 3, 3, 4, 5, 5],
      [4, 3, 3, 4, 5, 5],
      [1, 0, 0, 1, 2, 2],
    ]);
    expect(pad(a, w, "wrap").toArray()).toEqual([
      [4, 5, 3, 4, 5, 3],
      [1, 2, 0, 1, 2, 0],
      [4, 5, 3, 4, 5, 3],
      [1, 2, 0, 1, 2, 0],
      [4, 5, 3, 4, 5, 3],
    ]);
    expect(pad(a, w, "edge").toArray()).toEqual([
      [0, 0, 0, 1, 2, 2],
      [0, 0, 0, 1, 2, 2],
      [3, 3, 3, 4, 5, 5],
      [3, 3, 3, 4, 5, 5],
      [3, 3, 3, 4, 5, 5],
    ]);
  });

  it("reflects a length-1 axis as a constant", () => {
    expect(values(pad(tensor([7]), [[3, 2]], "reflect"))).toEqual([7, 7, 7, 7, 7, 7]);
  });

  it("rejects unknown modes instead of reading out of range", () => {
    expect(() => pad(v(), [[1, 1]], "bogus" as never)).toThrow(InvalidParameterError);
  });

  it("rejects padding an empty axis with a non-constant mode", () => {
    expect(() => pad(tensor([]), [[1, 1]], "reflect")).toThrow(InvalidParameterError);
    expect(values(pad(tensor([]), [[1, 1]], "constant", 4))).toEqual([4, 4]);
    expect(pad(tensor([]), [[0, 0]], "reflect").shape).toEqual([0]);
  });

  it("validates pad widths", () => {
    expect(() => pad(v(), [[1.5, 1]])).toThrow(InvalidParameterError);
    expect(() => pad(v(), [[-1, 1]])).toThrow(InvalidParameterError);
    expect(() => pad(v(), [[Number.NaN, 1]])).toThrow(InvalidParameterError);
    expect(() =>
      pad(v(), [
        [1, 1],
        [1, 1],
      ])
    ).toThrow(/padWidth length 2/);
  });

  it("accepts a number, a single pair, or one pair for every axis", () => {
    const a = f64([
      [1, 2],
      [3, 4],
    ]);
    // np.pad(a, 1, constant_values=9)
    expect(pad(a, 1, "constant", 9).toArray()).toEqual([
      [9, 9, 9, 9],
      [9, 1, 2, 9],
      [9, 3, 4, 9],
      [9, 9, 9, 9],
    ]);
    expect(pad(a, [0, 1]).toArray()).toEqual([
      [1, 2, 0],
      [3, 4, 0],
      [0, 0, 0],
    ]);
    expect(pad(a, [[1, 0]]).shape).toEqual([3, 3]);
  });

  it("stores constants in the tensor dtype", () => {
    expect(values(pad(tensor([1, 0], { dtype: "bool" }), [[1, 1]], "constant", 5))).toEqual([
      1, 1, 0, 1,
    ]);
    expect(pad(int64([1n, 2n]), [[1, 1]], "constant", 7).toArray()).toEqual([7n, 1n, 2n, 7n]);
    expect(pad(int64([1n]), [[1, 0]], "constant", 7.9).toArray()).toEqual([7n, 1n]);
    expect(() => pad(tensor([1], { dtype: "int32" }), [[1, 0]], "constant", Number.NaN)).toThrow(
      InvalidParameterError
    );
  });

  it("pads strided views", () => {
    const t = transpose(m23());
    expect(
      pad(
        t,
        [
          [1, 0],
          [0, 1],
        ],
        "edge"
      ).toArray()
    ).toEqual([
      [0, 3, 3],
      [0, 3, 3],
      [1, 4, 4],
      [2, 5, 5],
    ]);
  });
});

describe("moveaxis / swapaxes", () => {
  it("rejects repeated axes", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => moveaxis(t, [0, 0], [0, 1])).toThrow(InvalidParameterError);
    expect(() => moveaxis(t, [0, 1], [1, 1])).toThrow(InvalidParameterError);
    expect(() => moveaxis(t, [0, -2], [0, 1])).toThrow(InvalidParameterError);
  });

  it("matches NumPy", () => {
    const cube = f64([
      [
        [0, 1],
        [2, 3],
      ],
      [
        [4, 5],
        [6, 7],
      ],
    ]);
    // np.moveaxis(c, 0, -1)
    expect(moveaxis(cube, 0, -1).toArray()).toEqual([
      [
        [0, 4],
        [1, 5],
      ],
      [
        [2, 6],
        [3, 7],
      ],
    ]);
    const t = Tensor.zeros([2, 3, 4], { dtype: "float32", device: "cpu" });
    // np.moveaxis(x, [0, 1], [-1, -2]).shape == (4, 3, 2)
    expect(moveaxis(t, [0, 1], [-1, -2]).shape).toEqual([4, 3, 2]);
  });

  it("returns views that share the buffer", () => {
    const t = m23();
    expect(swapaxes(t, 0, 1).data).toBe(t.data);
    expect(moveaxis(t, 0, 1).data).toBe(t.data);
  });

  it("supports string tensors", () => {
    const s = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [2, 2] });
    expect(swapaxes(s, 0, 1).toArray()).toEqual([
      ["a", "c"],
      ["b", "d"],
    ]);
  });
});

describe("broadcast_to", () => {
  it("returns a zero-copy view that shares the input buffer", () => {
    const a = tensor([1, 2, 3]);
    const b = broadcastTo(a, [2, 3]);
    expect(b.data).toBe(a.data);
    expect(b.strides).toEqual([0, 1]);
    expect(b.toArray()).toEqual([
      [1, 2, 3],
      [1, 2, 3],
    ]);
  });

  it("validates the target shape", () => {
    const a = tensor([1, 2, 3]);
    expect(() => broadcastTo(a, [-1, 3])).toThrow(InvalidParameterError);
    expect(() => broadcastTo(a, [2.5, 3])).toThrow(InvalidParameterError);
    expect(() => broadcastTo(a, [2, 4])).toThrow(/axis 1 has size 4/);
    expect(() => broadcastTo(a, [])).toThrow(ShapeError);
  });

  it("broadcasts strided views and string tensors", () => {
    const t = transpose(m23());
    expect(broadcastTo(t, [2, 3, 2]).toArray()).toEqual([
      [
        [0, 3],
        [1, 4],
        [2, 5],
      ],
      [
        [0, 3],
        [1, 4],
        [2, 5],
      ],
    ]);
    const s = Tensor.fromStringArray({ data: ["a", "b"], shape: [2] });
    expect(broadcastTo(s, [2, 2]).toArray()).toEqual([
      ["a", "b"],
      ["a", "b"],
    ]);
  });
});

describe("scatter", () => {
  it("throws when an index is out of bounds instead of writing into another row", () => {
    const t = f64([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(() => scatter(t, 1, f64([[5]]), f64([[9]]))).toThrow(IndexError);
    expect(() => scatter(t, 1, f64([[3]]), f64([[9]]))).toThrow(IndexError);
    expect(() => scatter(t, 1, f64([[-4]]), f64([[9]]))).toThrow(IndexError);
  });

  it("rejects non-integer indices", () => {
    expect(() => scatter(f64([0, 0]), 0, f64([0.5]), f64([1]))).toThrow(InvalidParameterError);
    expect(() => scatter(f64([0, 0]), 0, f64([Number.NaN]), f64([1]))).toThrow(
      InvalidParameterError
    );
  });

  it("reads src by index coordinates when src is larger than index", () => {
    // t[idx[0,j]][j] = src[0,j] for j in 0..1
    const out = scatter(
      f64([
        [0, 0, 0],
        [0, 0, 0],
      ]),
      0,
      f64([[1, 0]]),
      f64([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(out.toArray()).toEqual([
      [0, 2, 0],
      [1, 0, 0],
    ]);
  });

  it("scatters along an axis and leaves the input untouched", () => {
    const t = f64([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    const out = scatter(
      t,
      1,
      f64([
        [0, 1, 2],
        [2, 0, 1],
      ]),
      f64([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(out.toArray()).toEqual([
      [1, 2, 3],
      [5, 6, 4],
    ]);
    expect(t.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("accepts negative indices and converts src to the dtype of t", () => {
    expect(values(scatter(f64([0, 0, 0]), 0, f64([-1]), f64([9])))).toEqual([0, 0, 9]);
    const out = scatter(tensor([0, 0, 0], { dtype: "int32" }), 0, f64([1]), f64([9.7]));
    expect(out.dtype).toBe("int32");
    expect(values(out)).toEqual([0, 9, 0]);
    // int64 destination with a number source (previously dropped silently)
    expect(scatter(int64([0n, 0n]), 0, f64([1]), f64([5])).toArray()).toEqual([0n, 5n]);
    // bool destination normalizes to 0/1
    expect(values(scatter(tensor([0, 0], { dtype: "bool" }), 0, f64([0]), f64([7])))).toEqual([
      1, 0,
    ]);
  });

  it("validates the shapes", () => {
    const t = f64([
      [0, 0],
      [0, 0],
    ]);
    expect(() => scatter(t, 0, f64([0]), f64([1]))).toThrow(ShapeError);
    expect(() => scatter(t, 0, f64([[0, 0, 0]]), f64([[1, 2]]))).toThrow(ShapeError);
    expect(() => scatter(t, 1, f64([[0], [0], [0]]), f64([[1], [2], [3]]))).toThrow(ShapeError);
  });

  it("scatters strided index and source views", () => {
    const t = f64([
      [0, 0],
      [0, 0],
    ]);
    const idx = transpose(
      f64([
        [0, 1],
        [1, 0],
      ])
    ); // [[0,1],[1,0]]
    const src = transpose(
      f64([
        [1, 2],
        [3, 4],
      ])
    ); // [[1,3],[2,4]]
    expect(scatter(t, 1, idx, src).toArray()).toEqual([
      [1, 3],
      [4, 2],
    ]);
  });
});

describe("nan reductions", () => {
  const x = (): Tensor =>
    f64([
      [1, Number.NaN, 3],
      [4, 5, Number.NaN],
    ]);

  it("match NumPy over axes, tuples of axes and with keepdims", () => {
    expect(values(nansum(x(), 1, true))).toEqual([4, 9]);
    expect(nansum(x(), 1, true).shape).toEqual([2, 1]);
    expect(values(nanmean(x(), 0))).toEqual([2.5, 5, 3]);
    const sd = values(nanstd(x(), 0));
    expect(sd[0]).toBeCloseTo(1.5, 12);
    expect(sd[1]).toBe(0);
    expect(sd[2]).toBe(0);
    expect(values(nanmin(x(), 0))).toEqual([1, 5, 3]);
    expect(nanmax(x(), [0, 1]).toArray()).toBe(5);
    expect(nansum(x(), [0, 1], true).shape).toEqual([1, 1]);
    expect(nansum(x(), []).toArray()).toEqual([
      [1, 0, 3],
      [4, 5, 0],
    ]);
  });

  it("supports ddof", () => {
    // np.nanstd([1, nan, 3, 5], ddof=1) == 2.0
    expect(nanstd(f64([1, Number.NaN, 3, 5]), undefined, false, 1).toArray()).toBeCloseTo(2, 12);
    expect(nanstd(f64([1, Number.NaN]), undefined, false, 1).toArray()).toBeNaN();
    expect(() => nanstd(f64([1, 2]), undefined, false, -1)).toThrow(InvalidParameterError);
  });

  it("sums with compensation", () => {
    // math.fsum([1e16, 1, -1e16]) == 1.0 (a naive left-to-right sum gives 0)
    expect(nansum(f64([1e16, 1, -1e16])).toArray()).toBe(1);
    // 0.1 added a million times: the exact double nearest the true sum is 100000.00000000001
    const n = 1_000_000;
    const tenth = new Float64Array(n).fill(0.1);
    const s = nansum(
      Tensor.fromTypedArray({ data: tenth, shape: [n], dtype: "float64", device: "cpu" })
    );
    expect(Math.abs((s.toArray() as number) - 100000)).toBeLessThan(1e-9);
  });

  it("propagates infinities like NumPy", () => {
    expect(nansum(f64([Number.POSITIVE_INFINITY, 1, Number.NaN])).toArray()).toBe(
      Number.POSITIVE_INFINITY
    );
    expect(nansum(f64([Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])).toArray()).toBeNaN();
    expect(nanmin(f64([Number.POSITIVE_INFINITY, Number.NaN])).toArray()).toBe(
      Number.POSITIVE_INFINITY
    );
    expect(nanmean(f64([Number.POSITIVE_INFINITY, 1])).toArray()).toBe(Number.POSITIVE_INFINITY);
  });

  it("returns NaN (or 0 for sums) when no value is left", () => {
    const allNaN = f64([Number.NaN, Number.NaN]);
    expect(nansum(allNaN).toArray()).toBe(0);
    expect(nanmean(allNaN).toArray()).toBeNaN();
    expect(nanmin(allNaN).toArray()).toBeNaN();
    expect(nanmax(allNaN).toArray()).toBeNaN();
    expect(nanstd(allNaN).toArray()).toBeNaN();
    expect(nansum(tensor([])).toArray()).toBe(0);
    expect(
      nanmean(
        f64([
          [Number.NaN, Number.NaN],
          [3, 4],
        ]),
        1
      ).toArray()
    ).toEqual([Number.NaN, 3.5]);
  });

  it("rejects repeated and out-of-range axes", () => {
    expect(() => nansum(x(), [0, 0])).toThrow(InvalidParameterError);
    expect(() => nansum(x(), [0, -2])).toThrow(InvalidParameterError);
    expect(() => nanmean(x(), 2)).toThrow(InvalidParameterError);
  });

  it("reduces strided views and int64 tensors", () => {
    const t = transpose(x()); // shape [3, 2]
    expect(values(nansum(t, 1))).toEqual([5, 5, 3]);
    expect(nansum(int64([1n, 2n, 3n])).toArray()).toBe(6);
  });

  it("reduces a middle axis of a 3-D tensor", () => {
    const t = f64([
      [
        [1, 2],
        [3, Number.NaN],
      ],
      [
        [5, 6],
        [Number.NaN, Number.NaN],
      ],
    ]);
    expect(nanmin(t, 1).toArray()).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });
});

describe("unique", () => {
  it("collapses NaNs and signed zeros like NumPy", () => {
    // np.unique([3, 1, 2, 1, 3, nan, nan, 0., -0.], return_counts=True)
    const { values: u, counts } = unique(f64([3, 1, 2, 1, 3, Number.NaN, Number.NaN, 0, -0]), true);
    const got = values(u);
    expect(got.slice(0, 4).map((v) => v + 0)).toEqual([0, 1, 2, 3]);
    expect(got[4]).toBeNaN();
    expect(values(counts as Tensor)).toEqual([2, 2, 1, 2, 2]);
  });

  it("compares int64 values exactly beyond 2^53", () => {
    const a = 9007199254740992n;
    const b = 9007199254740993n;
    const { values: u, counts } = unique(int64([b, a, b]), true);
    expect(u.toArray()).toEqual([a, b]);
    expect(values(counts as Tensor)).toEqual([1, 2]);
  });

  it("handles empty input, strided views and keeps the dtype", () => {
    const empty = unique(tensor([]), true);
    expect(empty.values.shape).toEqual([0]);
    expect((empty.counts as Tensor).shape).toEqual([0]);
    expect(values(unique(transpose(m23())).values)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(unique(tensor([2, 1, 2], { dtype: "int32" })).values.dtype).toBe("int32");
    expect(unique(tensor([2, 1], { dtype: "bool" })).values.dtype).toBe("bool");
  });
});

describe("searchsorted", () => {
  it("sorts NaN after every number", () => {
    // np.searchsorted([1, 2, 3], [nan, 0, 5]) -> [3, 0, 3]
    expect(values(searchsorted(f64([1, 2, 3]), f64([Number.NaN, 0, 5])))).toEqual([3, 0, 3]);
    // np.searchsorted([1, 2, nan], [nan], 'left') -> [2], 'right' -> [3]
    expect(values(searchsorted(f64([1, 2, Number.NaN]), f64([Number.NaN]), "left"))).toEqual([2]);
    expect(values(searchsorted(f64([1, 2, Number.NaN]), f64([Number.NaN]), "right"))).toEqual([3]);
  });

  it("distinguishes left and right on ties", () => {
    expect(values(searchsorted(f64([1, 2, 2, 3]), f64([2]), "left"))).toEqual([1]);
    expect(values(searchsorted(f64([1, 2, 2, 3]), f64([2]), "right"))).toEqual([3]);
  });

  it("rejects an invalid side and keeps the shape of values", () => {
    expect(() => searchsorted(f64([1, 2]), f64([1]), "middle" as never)).toThrow(
      InvalidParameterError
    );
    expect(
      searchsorted(
        f64([1, 2, 3]),
        f64([
          [1, 2],
          [3, 4],
        ])
      ).toArray()
    ).toEqual([
      [0, 1],
      [2, 3],
    ]);
    expect(searchsorted(f64([]), f64([1])).toArray()).toEqual([0]);
  });

  it("handles strided inputs and int64", () => {
    const sorted = transpose(f64([[1, 2, 3]])); // shape [3, 1] -> not 1-D
    expect(() => searchsorted(sorted, f64([1]))).toThrow(ShapeError);
    expect(values(searchsorted(int64([1n, 5n, 9n]), int64([5n, 6n])))).toEqual([1, 2]);
  });
});

describe("histogram", () => {
  it("widens an explicit range with equal ends like NumPy", () => {
    // np.histogram([1, 2, 2, 3, 10], bins=3, range=(5, 5))
    const { counts, binEdges } = histogram(f64([1, 2, 2, 3, 10]), 3, [5, 5]);
    expect(values(counts)).toEqual([0, 0, 0]);
    const e = values(binEdges);
    const expected = [4.5, 4.833333333333333, 5.166666666666667, 5.5];
    for (let i = 0; i < expected.length; i++) expect(e[i]).toBeCloseTo(expected[i] as number, 12);
  });

  it("matches NumPy for a fixed range", () => {
    // np.histogram([1, 2, 2, 3, 10], bins=3, range=(0, 12)) -> [4, 0, 1], [0, 4, 8, 12]
    const { counts, binEdges } = histogram(f64([1, 2, 2, 3, 10]), 3, [0, 12]);
    expect(values(counts)).toEqual([4, 0, 1]);
    expect(values(binEdges)).toEqual([0, 4, 8, 12]);
  });

  it("puts values on a bin edge in the same bin as NumPy", () => {
    // np.histogram([0.3, 0.6, 0.9], bins=3) -> [1, 1, 1]
    expect(values(histogram(f64([0.3, 0.6, 0.9]), 3).counts)).toEqual([1, 1, 1]);
    // np.histogram([1, 2, 3, 4, 5], bins=5) -> [1, 1, 1, 1, 1]; the maximum is in the last bin
    expect(values(histogram(f64([1, 2, 3, 4, 5]), 5).counts)).toEqual([1, 1, 1, 1, 1]);
  });

  it("makes the last bin edge equal to the range end exactly", () => {
    const { binEdges } = histogram(f64([0.1, 0.7]), 7, [0.1, 0.7]);
    expect(values(binEdges)[7]).toBe(0.7);
  });

  it("rejects non-finite data ranges and inverted ranges", () => {
    expect(() => histogram(f64([1, 2, Number.POSITIVE_INFINITY]))).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1, 2]), 2, [5, 1])).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1, 2]), 2, [0, Number.NaN])).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1, 2]), 0)).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1, 2]), 1.5)).toThrow(InvalidParameterError);
  });

  it("ignores NaN samples and handles empty data", () => {
    expect(values(histogram(f64([1, Number.NaN, 2]), 2).counts)).toEqual([1, 1]);
    const empty = histogram(tensor([]), 4);
    expect(values(empty.counts)).toEqual([0, 0, 0, 0]);
    expect(values(empty.binEdges)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("supports explicit bin edges", () => {
    // np.histogram([1, 2, 3, 4], bins=[0, 2, 5]) -> [1, 3]
    expect(values(histogram(f64([1, 2, 3, 4]), [0, 2, 5]).counts)).toEqual([1, 3]);
    // np.histogram([1, 2, 3, 4], bins=[0, 2, 2, 5]) -> [1, 0, 3]
    expect(values(histogram(f64([1, 2, 3, 4]), [0, 2, 2, 5]).counts)).toEqual([1, 0, 3]);
    expect(() => histogram(f64([1]), [0])).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1]), [2, 1])).toThrow(InvalidParameterError);
    expect(() => histogram(f64([1]), [0, Number.NaN])).toThrow(InvalidParameterError);
  });

  it("supports weights and density", () => {
    // np.histogram([1, 2, 3, 4], bins=[0, 2, 5], weights=[1, 2, 3, 4], density=True) -> [0.05, 0.3]
    const out = histogram(f64([1, 2, 3, 4]), [0, 2, 5], undefined, {
      weights: f64([1, 2, 3, 4]),
      density: true,
    });
    const d = values(out.counts);
    expect(d[0]).toBeCloseTo(0.05, 12);
    expect(d[1]).toBeCloseTo(0.3, 12);
    // np.histogram([1, 2, 3, 4], bins=2, weights=[1, 2, 3, 4]) -> [3, 7], edges [1, 2.5, 4]
    const w = histogram(f64([1, 2, 3, 4]), 2, undefined, { weights: f64([1, 2, 3, 4]) });
    expect(values(w.counts)).toEqual([3, 7]);
    expect(values(w.binEdges)).toEqual([1, 2.5, 4]);
    expect(() => histogram(f64([1, 2]), 2, undefined, { weights: f64([1]) })).toThrow(ShapeError);
  });

  it("reads strided inputs in logical order", () => {
    const t = transpose(m23());
    expect(values(histogram(t, 3, [0, 6]).counts)).toEqual([2, 2, 2]);
  });
});

describe("bincount", () => {
  it("validates minlength", () => {
    expect(() => bincount(f64([1, 2]), -3)).toThrow(InvalidParameterError);
    expect(() => bincount(f64([1, 2]), 1.5)).toThrow(InvalidParameterError);
  });

  it("supports weights", () => {
    // np.bincount([0, 1, 1, 3], weights=[0.5, 1, 2, 3], minlength=6)
    expect(values(bincount(f64([0, 1, 1, 3]), 6, f64([0.5, 1, 2, 3])))).toEqual([
      0.5, 3, 0, 3, 0, 0,
    ]);
    expect(() => bincount(f64([0, 1]), 0, f64([1]))).toThrow(ShapeError);
  });

  it("rejects negative, fractional and huge values with typed errors", () => {
    expect(() => bincount(f64([-1, 0]))).toThrow(InvalidParameterError);
    expect(() => bincount(f64([0.5]))).toThrow(InvalidParameterError);
    expect(() => bincount(f64([2 ** 40]))).toThrow(InvalidParameterError);
  });

  it("returns an empty result for empty input and pads to minlength", () => {
    expect(bincount(tensor([])).shape).toEqual([0]);
    expect(values(bincount(tensor([]), 3))).toEqual([0, 0, 0]);
    expect(values(bincount(f64([0, 1]), 4))).toEqual([1, 1, 0, 0]);
  });
});

describe("booleanIndex", () => {
  it("reads strided tensors and masks in logical order", () => {
    const t = transpose(
      f64([
        [1, 2],
        [3, 4],
      ])
    ); // [[1,3],[2,4]]
    const mask = transpose(
      tensor(
        [
          [1, 0],
          [1, 1],
        ],
        { dtype: "bool" }
      )
    ); // [[1,1],[0,1]]
    expect(values(booleanIndex(t, mask))).toEqual([1, 3, 4]);
  });

  it("selects int64 values exactly and treats NaN as true", () => {
    const big = 9007199254740993n;
    expect(booleanIndex(int64([big, 1n]), tensor([1, 0], { dtype: "bool" })).toArray()).toEqual([
      big,
    ]);
    expect(values(booleanIndex(f64([5, 6]), f64([Number.NaN, 0])))).toEqual([5]);
  });

  it("returns an empty tensor for an all-false mask", () => {
    expect(booleanIndex(f64([1, 2]), f64([0, 0])).shape).toEqual([0]);
  });
});

describe("fancyIndex", () => {
  it("throws for out-of-bounds indices", () => {
    expect(() => fancyIndex(f64([1, 2, 3]), f64([7]))).toThrow(IndexError);
    expect(() => fancyIndex(f64([1, 2, 3]), f64([-4]))).toThrow(IndexError);
    expect(() => fancyIndex(f64([]), f64([0]))).toThrow(IndexError);
  });

  it("rejects fractional and boolean indices", () => {
    expect(() => fancyIndex(f64([1, 2, 3]), f64([0.5]))).toThrow(InvalidParameterError);
    expect(() => fancyIndex(f64([1, 2, 3]), f64([Number.NaN]))).toThrow(InvalidParameterError);
    expect(() => fancyIndex(f64([1, 2, 3]), tensor([1, 0], { dtype: "bool" }))).toThrow(DTypeError);
  });

  it("wraps negative indices", () => {
    // np.take([1, 2, 3], [-1, 0]) -> [3, 1]
    expect(values(fancyIndex(f64([1, 2, 3]), f64([-1, 0])))).toEqual([3, 1]);
  });

  it("returns the shape of the indices for N-D index tensors (np.take)", () => {
    // np.take(arange(6).reshape(2, 3), [[0, 1], [2, 0]], axis=1)
    const out = fancyIndex(
      m23(),
      f64([
        [0, 1],
        [2, 0],
      ]),
      1
    );
    expect(out.shape).toEqual([2, 2, 2]);
    expect(out.toArray()).toEqual([
      [
        [0, 1],
        [2, 0],
      ],
      [
        [3, 4],
        [5, 3],
      ],
    ]);
  });

  it("keeps the axis for a 0-d index tensor", () => {
    expect(fancyIndex(m23(), tensor(1), 0).shape).toEqual([1, 3]);
  });

  it("indexes strided views and keeps dtype", () => {
    expect(fancyIndex(transpose(m23()), f64([2, 0]), 0).toArray()).toEqual([
      [2, 5],
      [0, 3],
    ]);
    expect(fancyIndex(int64([5n, 6n, 7n]), f64([2])).toArray()).toEqual([7n]);
    expect(fancyIndex(tensor([1, 2], { dtype: "int32" }), f64([1])).dtype).toBe("int32");
  });
});

describe("index_select", () => {
  it("preserves the dtype of the input", () => {
    const out = indexSelect(
      tensor([1, 2, 3], { dtype: "int32" }),
      0,
      tensor([0, 2], { dtype: "int32" })
    );
    expect(out.dtype).toBe("int32");
    expect(values(out)).toEqual([1, 3]);
    expect(indexSelect(tensor([1, 2, 3]), 0, tensor([1], { dtype: "int32" })).dtype).toBe(
      "float32"
    );
  });

  it("keeps int64 values exact", () => {
    const big = 9007199254740993n;
    expect(indexSelect(int64([1n, big]), 0, tensor([1], { dtype: "int32" })).toArray()).toEqual([
      big,
    ]);
  });

  it("throws IndexError for out-of-range or negative indices", () => {
    expect(() => indexSelect(f64([1, 2, 3]), 0, f64([3]))).toThrow(IndexError);
    expect(() => indexSelect(f64([1, 2, 3]), 0, f64([-1]))).toThrow(IndexError);
    expect(() => indexSelect(f64([1, 2, 3]), 0, f64([0.5]))).toThrow(InvalidParameterError);
    expect(() => indexSelect(f64([1, 2, 3]), 3, f64([0]))).toThrow(InvalidParameterError);
  });

  it("selects from strided views and long contiguous rows", () => {
    expect(indexSelect(transpose(m23()), 1, f64([1, 0, 1])).toArray()).toEqual([
      [3, 0, 3],
      [4, 1, 4],
      [5, 2, 5],
    ]);
    const wide = f64(
      Array.from({ length: 3 }, (_, i) => Array.from({ length: 20 }, (_, j) => i * 20 + j))
    );
    const out = indexSelect(wide, 0, f64([2, 0]));
    expect(values(out).slice(0, 3)).toEqual([40, 41, 42]);
    expect(values(out).slice(20, 23)).toEqual([0, 1, 2]);
    expect(out.shape).toEqual([2, 20]);
  });

  it("accepts string axis aliases and rejects non-1-D indices", () => {
    expect(indexSelect(m23(), "columns", f64([2])).toArray()).toEqual([[2], [5]]);
    expect(() => indexSelect(f64([1, 2]), 0, f64([[0]]))).toThrow(ShapeError);
  });
});

describe("isin", () => {
  it("never matches NaN", () => {
    // np.isin([nan, 1, 2], [nan, 1]) -> [False, True, False]
    expect(values(isin(f64([Number.NaN, 1, 2]), [Number.NaN, 1]))).toEqual([0, 1, 0]);
    expect(values(isin(f64([Number.NaN, 1]), f64([Number.NaN, 1])))).toEqual([0, 1]);
    expect(values(isin(f64([Number.NaN, 1]), [1], true))).toEqual([1, 0]);
  });

  it("supports invert", () => {
    // np.isin([1, 2, 3], [2], invert=True) -> [True, False, True]
    expect(values(isin(f64([1, 2, 3]), [2], true))).toEqual([1, 0, 1]);
  });

  it("compares int64 exactly beyond 2^53", () => {
    const a = 9007199254740992n;
    const b = 9007199254740993n;
    expect(values(isin(int64([a, b]), int64([b])))).toEqual([0, 1]);
    expect(values(isin(int64([a, 3n]), [3, 1.5]))).toEqual([0, 1]);
  });

  it("tests string tensors against string lists and tensors", () => {
    const s = Tensor.fromStringArray({ data: ["a", "b", "c"], shape: [3] });
    expect(values(isin(s, ["b", "z"]))).toEqual([0, 1, 0]);
    expect(values(isin(s, Tensor.fromStringArray({ data: ["c"], shape: [1] }), true))).toEqual([
      1, 1, 0,
    ]);
    expect(values(isin(s, []))).toEqual([0, 0, 0]);
    expect(() => isin(s, [1, 2])).toThrow(DTypeError);
    expect(() => isin(f64([1]), ["a"])).toThrow(DTypeError);
  });

  it("handles empty inputs and strided views", () => {
    expect(isin(tensor([]), [1]).shape).toEqual([0]);
    expect(values(isin(f64([1, 2]), []))).toEqual([0, 0]);
    expect(values(isin(transpose(m23()), [0, 4]))).toEqual([1, 0, 0, 1, 0, 0]);
  });
});

describe("rot90", () => {
  it("matches NumPy for every k, including negative", () => {
    // np.rot90(a, k) for a = [[0, 1, 2], [3, 4, 5]]
    const rot = (k: number) => rot90(m23(), k).toArray();
    expect(rot(0)).toEqual(m23().toArray());
    expect(rot(1)).toEqual([
      [2, 5],
      [1, 4],
      [0, 3],
    ]);
    expect(rot(2)).toEqual([
      [5, 4, 3],
      [2, 1, 0],
    ]);
    expect(rot(3)).toEqual([
      [3, 0],
      [4, 1],
      [5, 2],
    ]);
    expect(rot(-1)).toEqual(rot(3));
    expect(rot(5)).toEqual(rot(1));
  });

  it("rejects a non-integer k", () => {
    expect(() => rot90(m23(), 1.5)).toThrow(InvalidParameterError);
    expect(() => rot90(m23(), Number.NaN)).toThrow(InvalidParameterError);
  });

  it("works on strided views", () => {
    expect(rot90(transpose(m23()), 1).toArray()).toEqual([
      [3, 4, 5],
      [0, 1, 2],
    ]);
  });
});

describe("atleast_1d / atleast_2d", () => {
  it("promote scalars and vectors", () => {
    expect(atleast1d(tensor(3)).shape).toEqual([1]);
    expect(atleast2d(tensor(3)).shape).toEqual([1, 1]);
    expect(atleast2d(tensor([1, 2, 3])).shape).toEqual([1, 3]);
  });
});

describe("cross", () => {
  it("matches NumPy for strided input", () => {
    // np.cross([1, 2, 3], [4, 5, 6]) -> [-3, 6, -3]
    expect(values(cross(f64([1, 2, 3]), f64([4, 5, 6])))).toEqual([-3, 6, -3]);
    const strided = stridedVector([1, 2, 3]);
    expect(strided.strides).toEqual([2]);
    expect(values(cross(strided, f64([4, 5, 6])))).toEqual([-3, 6, -3]);
    expect(values(cross(f64([4, 5, 6]), strided))).toEqual([3, -6, 3]);
  });

  it("rejects wrong shapes and string input", () => {
    expect(() => cross(f64([1, 2]), f64([1, 2, 3]))).toThrow(ShapeError);
    // batched [..., 3] input is valid since cross broadcasts; a trailing length of 2 is not
    expect(() => cross(f64([[1, 2]]), f64([1, 2, 3]))).toThrow(ShapeError);
    expect(() =>
      cross(Tensor.fromStringArray({ data: ["a", "b", "c"], shape: [3] }), f64([1, 2, 3]))
    ).toThrow(DTypeError);
  });
});

describe("meshgrid", () => {
  it("matches NumPy for xy and ij indexing", () => {
    // np.meshgrid([1, 2, 3], [4, 5])
    const [X, Y] = meshgrid(f64([1, 2, 3]), f64([4, 5]));
    expect(X?.toArray()).toEqual([
      [1, 2, 3],
      [1, 2, 3],
    ]);
    expect(Y?.toArray()).toEqual([
      [4, 4, 4],
      [5, 5, 5],
    ]);
    // np.meshgrid([1, 2], [3, 4, 5], indexing='ij')
    const [I, J] = meshgrid(f64([1, 2]), f64([3, 4, 5]), { indexing: "ij" });
    expect(I?.toArray()).toEqual([
      [1, 1, 1],
      [2, 2, 2],
    ]);
    expect(J?.toArray()).toEqual([
      [3, 4, 5],
      [3, 4, 5],
    ]);
  });

  it("matches NumPy for three inputs", () => {
    const grids = meshgrid(f64([0, 1]), f64([0, 1, 2]), f64([0, 1, 2, 3]));
    expect(grids.map((g) => g.shape)).toEqual([
      [3, 2, 4],
      [3, 2, 4],
      [3, 2, 4],
    ]);
    // np.meshgrid(arange(2), arange(3), arange(4))
    expect(grids[0]?.toArray()).toEqual(
      Array.from({ length: 3 }, () => [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
      ])
    );
    expect(grids[1]?.toArray()).toEqual(
      [0, 1, 2].map((v) => [
        [v, v, v, v],
        [v, v, v, v],
      ])
    );
    expect(grids[2]?.toArray()).toEqual(
      Array.from({ length: 3 }, () => [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
      ])
    );
  });

  it("keeps the dtype of each input", () => {
    const [A, B] = meshgrid(tensor([0, 1], { dtype: "int32" }), f64([0, 1, 2]), { indexing: "ij" });
    expect(A?.dtype).toBe("int32");
    expect(B?.dtype).toBe("float64");
    const [C] = meshgrid(int64([5n, 6n]), int64([1n]));
    expect(C?.toArray()).toEqual([[5n, 6n]]);
  });

  it("accepts an empty options object and rejects bad options", () => {
    const grids = meshgrid(f64([0, 1]), f64([0, 1, 2]), {});
    expect(grids.map((g) => g.shape)).toEqual([
      [3, 2],
      [3, 2],
    ]);
    expect(() => meshgrid(f64([0, 1]), { indexing: "zz" as never })).toThrow(InvalidParameterError);
    expect(() => meshgrid(f64([[0, 1]]))).toThrow(ShapeError);
    expect(meshgrid()).toEqual([]);
  });

  it("reads strided inputs and handles empty vectors", () => {
    const [S, T] = meshgrid(stridedVector([1, 2, 3]), f64([10, 20]), { indexing: "ij" });
    expect(S?.toArray()).toEqual([
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    expect(T?.shape).toEqual([3, 2]);
    const [E] = meshgrid(f64([]), f64([1, 2]));
    expect(E?.shape).toEqual([2, 0]);
  });
});

describe("aliases", () => {
  it("camelCase aliases are the same functions", async () => {
    const mod = await import("../../src/ndarray/ops/utils");
    expect(mod.zerosLike).toBe(mod.zeros_like);
    expect(mod.onesLike).toBe(mod.ones_like);
    expect(mod.emptyLike).toBe(mod.empty_like);
    expect(mod.fullLike).toBe(mod.full_like);
    expect(mod.indexSelect).toBe(mod.index_select);
    expect(mod.broadcastTo).toBe(mod.broadcast_to);
    expect(mod.flipLr).toBe(mod.fliplr);
    expect(mod.flipUd).toBe(mod.flipud);
  });
});
