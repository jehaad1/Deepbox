/**
 * Regression tests for ndarray sorting and trigonometry (v1.5.0 review).
 *
 * Reference values come from NumPy 2.4 (`np.argsort(-a, kind="stable")`, `np.tanh`,
 * `np.arctan2`) and PyTorch (`torch.argsort(..., descending=True, stable=True)`).
 */
import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  acosh,
  argsort,
  asinh,
  atan2,
  atanh,
  sort,
  tanh,
  tensor,
  transpose,
} from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor/Tensor";

const f64 = { dtype: "float64" } as const;

function stableDescendingReference(values: number[]): number[] {
  const idx = values.map((_, i) => i);
  idx.sort((a, b) => {
    const va = values[a] as number;
    const vb = values[b] as number;
    const aNaN = Number.isNaN(va);
    const bNaN = Number.isNaN(vb);
    if (aNaN || bNaN) return aNaN === bNaN ? a - b : aNaN ? -1 : 1;
    return va === vb ? a - b : vb - va;
  });
  return idx;
}

describe("argsort descending is stable", () => {
  // np.argsort(-[1,2,2,3,2,1,3], kind="stable") == torch stable descending
  const data = [1, 2, 2, 3, 2, 1, 3];
  const expected = [3, 6, 1, 2, 4, 0, 5];

  it.each([
    "float64",
    "float32",
    "int32",
    "uint8",
    "int64",
    "float16",
  ] as const)("keeps ties in input order for %s", (dtype) => {
    expect(argsort(tensor(data, { dtype }), -1, true).toArray()).toEqual(expected);
  });

  it("keeps NaN first and ties in input order", () => {
    const v = [3, Number.NaN, 1, Number.NaN, 3];
    for (const dtype of ["float64", "float32"] as const) {
      expect(argsort(tensor(v, { dtype }), -1, true).toArray()).toEqual([1, 3, 0, 4, 2]);
      expect(argsort(tensor(v, { dtype }), -1).toArray()).toEqual([2, 0, 4, 1, 3]);
    }
  });

  it("matches a stable reference on long lanes (radix path)", () => {
    const n = 20000;
    const values: number[] = [];
    let seed = 12345;
    for (let i = 0; i < n; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      values.push(seed % 50);
    }
    const t = tensor(values, f64);
    expect(Array.from(argsort(t, -1, true).data as Int32Array)).toEqual(
      stableDescendingReference(values)
    );
    const asc = Array.from(argsort(t).data as Int32Array);
    const ref = values
      .map((_, i) => i)
      .sort((a, b) => (values[a] as number) - (values[b] as number) || a - b);
    expect(asc).toEqual(ref);
  });

  it("works per lane along any axis of an N-D tensor", () => {
    const a = tensor(
      [
        [
          [3, 1],
          [2, 5],
        ],
        [
          [0, 9],
          [4, 4],
        ],
      ],
      f64
    );
    // np.argsort(-a, axis=0, kind="stable")
    expect(argsort(a, 0, true).toArray()).toEqual([
      [
        [0, 1],
        [1, 0],
      ],
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    // np.sort(a, axis=1)[:, ::-1]
    expect(sort(a, 1, true).toArray()).toEqual([
      [
        [3, 5],
        [2, 1],
      ],
      [
        [4, 9],
        [0, 4],
      ],
    ]);
  });
});

describe("sort and argsort on views and edge shapes", () => {
  const m = tensor(
    [
      [3, 1, 2],
      [9, 8, 7],
    ],
    f64
  );

  it("handles transposed (non-contiguous) inputs along both axes", () => {
    const tr = transpose(m);
    // np.sort(m.T, axis=0), np.argsort(-m.T, axis=1, kind="stable")
    expect(sort(tr, 0).toArray()).toEqual([
      [1, 7],
      [2, 8],
      [3, 9],
    ]);
    expect(sort(tr, -1).toArray()).toEqual([
      [3, 9],
      [1, 8],
      [2, 7],
    ]);
    expect(argsort(tr, 1, true).toArray()).toEqual([
      [1, 0],
      [1, 0],
      [1, 0],
    ]);
  });

  it("does not modify the input", () => {
    const src = tensor([3, 1, 2], f64);
    sort(src);
    argsort(src, -1, true);
    expect(src.toArray()).toEqual([3, 1, 2]);
  });

  it("sorts int64 along a non-last axis", () => {
    const t = tensor(
      [
        [3, 1],
        [1, 7],
        [2, 4],
      ],
      { dtype: "int64" }
    );
    expect(sort(t, 0).toArray()).toEqual([
      [1n, 1n],
      [2n, 4n],
      [3n, 7n],
    ]);
    expect(argsort(t, 0, true).toArray()).toEqual([
      [0, 1],
      [2, 2],
      [1, 0],
    ]);
  });

  it("treats a 0-d tensor as one element", () => {
    const s = tensor(3, f64);
    expect(sort(s).toArray()).toBe(3);
    expect(sort(s).shape).toEqual([]);
    expect(argsort(s).toArray()).toBe(0);
    expect(() => sort(s, 1)).toThrow(InvalidParameterError);
  });

  it("returns empty results for empty lanes and empty tensors", () => {
    expect(sort(tensor([[], []], f64), 0).shape).toEqual([2, 0]);
    expect(argsort(tensor([[]], f64), 1).shape).toEqual([1, 0]);
    expect(sort(tensor([], f64)).shape).toEqual([0]);
  });

  it("sorts long float lanes row by row with NaN last (ascending) and first (descending)", () => {
    const n = 9000;
    const rows = [
      Array.from({ length: n }, (_, i) => ((i * 7919) % n) - n / 2),
      Array.from({ length: n }, (_, i) => (i % 97) * 0.5),
    ];
    (rows[0] as number[])[5] = Number.NaN;
    const t = tensor(rows, f64);
    const asc = sort(t).data as Float64Array;
    const desc = sort(t, -1, true).data as Float64Array;
    for (let r = 0; r < 2; r++) {
      const ref = [...(rows[r] as number[])].sort((a, b) =>
        Number.isNaN(a) ? 1 : Number.isNaN(b) ? -1 : a - b
      );
      expect(Array.from(asc.subarray(r * n, (r + 1) * n))).toEqual(ref);
      const d = Array.from(desc.subarray(r * n, (r + 1) * n));
      expect(d).toEqual([...ref].reverse());
    }
    expect(Number.isNaN(asc[n - 1] as number)).toBe(true);
    expect(Number.isNaN(desc[0] as number)).toBe(true);
  });

  it("sorts a strided float32 view along its axis", () => {
    const data = new Float32Array([3, 0, 1, 0, 2, 0, 4]);
    const t = Tensor.fromTypedArray({
      data,
      shape: [4],
      strides: [2],
      dtype: "float32",
      device: "cpu",
    });
    expect(sort(t, 0, true).toArray()).toEqual([4, 3, 2, 1]);
  });

  it("rejects string tensors and bad axes", () => {
    const s = tensor(["b", "a"]);
    expect(() => sort(s)).toThrow(DTypeError);
    expect(() => argsort(s)).toThrow(DTypeError);
    expect(() => sort(m, 2)).toThrow(InvalidParameterError);
    expect(() => argsort(m, -3)).toThrow(InvalidParameterError);
  });
});

describe("tanh accuracy", () => {
  it("is accurate for small magnitudes (np.tanh)", () => {
    const x = [1e-10, 1e-5, 1e-3, -1e-8, 0.5, 25];
    const ref = [1e-10, 9.999999999666668e-6, 0.0009999996666668, -1e-8, 0.46211715726000974, 1.0];
    const out = tanh(tensor(x, f64)).toArray() as number[];
    for (let i = 0; i < x.length; i++) {
      const r = ref[i] as number;
      expect(Math.abs((out[i] as number) - r) / Math.abs(r)).toBeLessThan(4e-16);
    }
  });

  it("saturates and preserves signed values and NaN", () => {
    const out = tanh(tensor([-50, 50, Number.NaN, 0], f64)).toArray() as number[];
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(1);
    expect(Number.isNaN(out[2] as number)).toBe(true);
    expect(out[3]).toBe(0);
  });
});

describe("atan2", () => {
  it("matches np.arctan2 for the four quadrants and the origin", () => {
    const out = atan2(tensor([1, -1, 0, 0], f64), tensor([-1, -1, -1, 0], f64)).toArray();
    expect(out).toEqual([2.356194490192345, -2.356194490192345, Math.PI, 0]);
  });

  it("broadcasts column against row", () => {
    const out = atan2(tensor([[1], [2]], f64), tensor([1, 2, 3], f64)).toArray() as number[][];
    const ref = [
      [0.7853981633974483, 0.4636476090008061, 0.3217505543966422],
      [1.1071487177940904, 0.7853981633974483, 0.5880026035475675],
    ];
    expect(out).toEqual(ref);
  });

  it("broadcasts 0-d operands and reads transposed views", () => {
    expect(atan2(tensor(1, f64), tensor([1, 0], f64)).toArray()).toEqual([
      Math.PI / 4,
      Math.PI / 2,
    ]);
    expect(atan2(tensor([1, 0], f64), tensor(1, f64)).toArray()).toEqual([Math.PI / 4, 0]);
    const m = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const mt = transpose(m);
    expect(atan2(mt, m).toArray()).toEqual([
      [Math.atan2(1, 1), Math.atan2(3, 2)],
      [Math.atan2(2, 3), Math.atan2(4, 4)],
    ]);
  });

  it("accepts int64 operands and mixed dtypes", () => {
    const y = tensor([1, 2], { dtype: "int64" });
    const x = tensor([1, 1], { dtype: "int32" });
    const r = atan2(y, x);
    expect(r.dtype).toBe("float32");
    expect(r.toArray()).toEqual([Math.fround(Math.atan2(1, 1)), Math.fround(Math.atan2(2, 1))]);
  });

  it("rejects strings and incompatible shapes", () => {
    expect(() => atan2(tensor(["a"]), tensor([1]))).toThrow(DTypeError);
    expect(() => atan2(tensor([1, 2, 3], f64), tensor([1, 2], f64))).toThrow(ShapeError);
  });
});

describe("inverse hyperbolic functions", () => {
  it("match NumPy", () => {
    // np.arcsinh(1), np.arccosh(2), np.arctanh(0.5)
    expect(asinh(tensor([1], f64)).toArray()).toEqual([0.881373587019543]);
    expect(acosh(tensor([2, 0.5], f64)).toArray()).toEqual([1.3169578969248166, Number.NaN]);
    // JS Math.atanh(0.5) differs from NumPy's 0.5493061443340549 by one ulp.
    const th = atanh(tensor([0.5, 1], f64)).toArray() as number[];
    expect(th[0]).toBeCloseTo(0.5493061443340549, 15);
    expect(th[1]).toBe(Infinity);
  });
});
