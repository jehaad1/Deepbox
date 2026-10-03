/**
 * Regression tests for src/ndarray/linalg (v1.5.0 review).
 *
 * Reference values come from NumPy 2.4 (`numpy.dot`, `numpy.matmul`, `numpy.tensordot`,
 * `numpy.cov`, `numpy.corrcoef`).
 */
import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import { corrcoef, cov, dot, tensor, tensordot, transpose } from "../../src/ndarray";
import { matmul, dot as strictDot } from "../../src/ndarray/linalg/basic";

/** Row-major values of a tensor as plain numbers. */
function flat(t: { toArray(): unknown }): number[] {
  const out: number[] = [];
  const walk = (v: unknown): void => {
    if (Array.isArray(v)) v.forEach(walk);
    else out.push(Number(v));
  };
  walk(t.toArray());
  return out;
}

describe("dot / matmul: NaN and Infinity propagation", () => {
  it("0 * Infinity gives NaN on non-contiguous (transposed) operands", () => {
    // numpy: [[0, 1], [2, 3]] @ [[inf, 1], [1, 1]] = [[nan, 1], [inf, 5]]
    const aT = tensor(
      [
        [0, 2],
        [1, 3],
      ],
      { dtype: "float64" }
    );
    const a = transpose(aT);
    const b = tensor(
      [
        [Number.POSITIVE_INFINITY, 1],
        [1, 1],
      ],
      { dtype: "float64" }
    );
    const r = flat(dot(a, b));
    expect(Number.isNaN(r[0])).toBe(true);
    expect(r.slice(1)).toEqual([1, Number.POSITIVE_INFINITY, 5]);
  });

  it("gives the same answer for contiguous and transposed layouts", () => {
    const a = tensor(
      [
        [0, 1],
        [2, 3],
      ],
      { dtype: "float32" }
    );
    const b = tensor(
      [
        [Number.NaN, 1],
        [1, 1],
      ],
      { dtype: "float32" }
    );
    const direct = flat(dot(a, b));
    const viaTranspose = flat(dot(transpose(transpose(a)), transpose(transpose(b))));
    expect(direct.map(Number.isNaN)).toEqual(viaTranspose.map(Number.isNaN));
    expect(Number.isNaN(direct[0])).toBe(true);
  });
});

describe("dot / matmul: integer and bool dtypes", () => {
  it("int32 products wrap exactly like NumPy", () => {
    // numpy int32: [[3, -18], [-6, 38]]
    const a = tensor(
      [
        [2147483647, 2147483647, 2147483647],
        [1, 2, 3],
      ],
      { dtype: "int32" }
    );
    const b = tensor(
      [
        [2147483647, 5],
        [2147483647, 6],
        [2147483647, 7],
      ],
      { dtype: "int32" }
    );
    const r = dot(a, b);
    expect(r.dtype).toBe("int32");
    expect(flat(r)).toEqual([3, -18, -6, 38]);
    expect(flat(matmul(a, b))).toEqual([3, -18, -6, 38]);

    // numpy int32: [[-884901888]]
    const w = dot(
      tensor([[2000000000, 2000000000]], { dtype: "int32" }),
      tensor([[3], [3]], { dtype: "int32" })
    );
    expect(flat(w)).toEqual([-884901888]);
  });

  it("int32 vector dot wraps exactly", () => {
    const r = dot(
      tensor([2147483647, 2147483647, 2147483647], { dtype: "int32" }),
      tensor([2147483647, 2147483647, 2147483647], { dtype: "int32" })
    );
    // 3 * (2^31 - 1)^2 mod 2^32 as signed int32
    expect(flat(r)).toEqual([Number(BigInt.asIntN(32, 3n * 2147483647n * 2147483647n))]);
  });

  it("bool matmul is a logical OR of ANDs", () => {
    const t = tensor([[1, 1]], { dtype: "bool" });
    const col = tensor([[1], [1]], { dtype: "bool" });
    const r = dot(t, col);
    expect(r.dtype).toBe("bool");
    expect(flat(r)).toEqual([1]);
    expect(
      flat(dot(tensor([[1, 0]], { dtype: "bool" }), tensor([[0], [1]], { dtype: "bool" })))
    ).toEqual([0]);
  });

  it("bool dot does not wrap when 256 pairs are true", () => {
    const ones = tensor(new Array(256).fill(1), { dtype: "bool" });
    expect(flat(dot(ones, ones))).toEqual([1]);
  });

  it("int64 overflow names the operation", () => {
    const big = tensor([[2 ** 62]], { dtype: "int64" });
    expect(() => matmul(big, big)).toThrow(/int64 matmul overflow/);
    expect(() => dot(big, big)).toThrow(/int64 dot overflow/);
  });
});

describe("dot: shapes, strides and errors", () => {
  it("matrix-vector and vector-matrix products on strided views match NumPy", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    // numpy: m.T @ [1, 1] = [5, 7, 9]; [1, 1] @ m = [5, 7, 9]
    expect(flat(dot(transpose(m), tensor([1, 1], { dtype: "float64" })))).toEqual([5, 7, 9]);
    expect(flat(dot(tensor([1, 1], { dtype: "float64" }), m))).toEqual([5, 7, 9]);
  });

  it("batched product with a shared 2-D matrix on either side", () => {
    const batch = tensor([[[1, 2]], [[3, 4]]], { dtype: "float64" }); // (2, 1, 2)
    const w = tensor(
      [
        [1, 0, 2],
        [0, 1, 3],
      ],
      { dtype: "float64" }
    ); // (2, 3)
    const r = dot(batch, w);
    expect(r.shape).toEqual([2, 1, 3]);
    expect(flat(r)).toEqual([1, 2, 8, 3, 4, 18]);

    const left = tensor([[1, 1]], { dtype: "float64" }); // (1, 2)
    const r2 = dot(
      left,
      tensor(
        [
          [[1], [2]],
          [[3], [4]],
        ],
        { dtype: "float64" }
      )
    );
    expect(r2.shape).toEqual([2, 1, 1]);
    expect(flat(r2)).toEqual([3, 7]);
  });

  it("empty inner dimension gives zeros", () => {
    const r = dot(
      tensor([[]], { dtype: "float64" }).reshape([1, 0]),
      tensor([[]], { dtype: "float64" }).reshape([0, 1])
    );
    expect(r.shape).toEqual([1, 1]);
    expect(flat(r)).toEqual([0]);
  });

  it("reports readable shape errors", () => {
    expect(() => dot(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/\(3,\) and \(2,\) not aligned/);
    expect(() => dot(tensor([[1, 2]]), tensor([[1, 2]]))).toThrow(
      /\(1, 2\) and \(1, 2\) not aligned: 2 \(dim 1\) != 1 \(dim 0\)/
    );
    expect(() => dot(tensor(5), tensor(10))).toThrow(ShapeError);
    expect(() => dot(tensor(5), tensor(10))).toThrow(/0-d/);
    // A 1-D operand now combines with a batched operand like numpy.matmul.
    const r = dot(tensor([1, 2]), tensor([[[1], [2]]]));
    expect(r.shape).toEqual([1, 1]);
    expect(flat(r)).toEqual([5]);
    expect(() => dot(tensor([1, 2, 3]), tensor([[[1], [2]]]))).toThrow(/not aligned/);
  });
});

describe("strict matmul / dot in basic.ts", () => {
  it("matmul matches the general dot for float32 and mixed float dtypes", () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float32" }
    );
    const b = tensor(
      [
        [1, 4],
        [2, 5],
        [3, 6],
      ],
      { dtype: "float64" }
    );
    const r = matmul(a, b);
    expect(r.dtype).toBe("float64");
    expect(flat(r)).toEqual([14, 32, 32, 77]);
  });

  it("strict dot returns a 0-d tensor and keeps dtype checks", () => {
    const r = strictDot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    expect(r.shape).toEqual([]);
    expect(flat(r)).toEqual([32]);
    expect(() => strictDot(tensor([1, 2]), tensor([1, 2], { dtype: "int32" }))).toThrow(DTypeError);
  });
});

describe("tensordot", () => {
  const A = tensor(
    Array.from({ length: 2 }, (_, i) =>
      Array.from({ length: 3 }, (_, j) => Array.from({ length: 4 }, (_, k) => i * 12 + j * 4 + k))
    ),
    { dtype: "float64" }
  );
  const B = tensor(
    Array.from({ length: 4 }, (_, i) =>
      Array.from({ length: 3 }, (_, j) => Array.from({ length: 5 }, (_, k) => i * 15 + j * 5 + k))
    ),
    { dtype: "float64" }
  );
  // numpy.tensordot(A, B, axes=([1, 2], [1, 0]))
  const expected = [2200, 2266, 2332, 2398, 2464, 6160, 6370, 6580, 6790, 7000];

  it("contracts several axes in the given pairing", () => {
    const r = tensordot(A, B, [
      [1, 2],
      [1, 0],
    ]);
    expect(r.shape).toEqual([2, 5]);
    expect(flat(r)).toEqual(expected);
  });

  it("accepts negative axes", () => {
    const r = tensordot(A, B, [
      [-2, -1],
      [-2, -3],
    ]);
    expect(r.shape).toEqual([2, 5]);
    expect(flat(r)).toEqual(expected);
  });

  it("accepts a single axis per operand", () => {
    const r = tensordot(A, B, [2, 0]);
    expect(r.shape).toEqual([2, 3, 3, 5]);
  });

  it("rejects out-of-range, repeated and non-integer axes", () => {
    expect(() => tensordot(A, B, [[3], [0]])).toThrow(InvalidParameterError);
    expect(() => tensordot(A, B, [[-4], [0]])).toThrow(/out of range/);
    expect(() =>
      tensordot(A, B, [
        [1, 1],
        [1, 0],
      ])
    ).toThrow(/repeated/);
    expect(() => tensordot(A, B, [[1.5], [0]])).toThrow(/integers/);
    expect(() => tensordot(A, B, [[1, 2], [0]])).toThrow(/same length/);
  });

  it("rejects a contraction count larger than the operands", () => {
    expect(() => tensordot(tensor([1, 2, 3]), tensor([[1, 2, 3]]), 2)).toThrow(ShapeError);
    expect(() => tensordot(A, B, -1)).toThrow(InvalidParameterError);
    expect(() => tensordot(A, B, 1.5)).toThrow(InvalidParameterError);
  });

  it("reports mismatched contracted sizes", () => {
    expect(() => tensordot(tensor([[1, 2]]), tensor([[1, 2, 3]]), 1)).toThrow(
      /axis 1 of a is 2, axis 0 of b is 1/
    );
  });

  it("keeps float32 and computes int64 exactly", () => {
    const f = tensordot(
      tensor([1, 2], { dtype: "float32" }),
      tensor([3, 4], { dtype: "float32" }),
      1
    );
    expect(f.dtype).toBe("float32");

    // numpy int64: [[1152921504606846997, 8796093022208], [8388608, 1152921504606846981]]
    const a = tensor(
      [
        [2 ** 40, 3],
        [1, 2 ** 20],
      ],
      { dtype: "int64" }
    );
    const b = tensor(
      [
        [2 ** 20, 5],
        [7, 2 ** 40],
      ],
      { dtype: "int64" }
    );
    const r = tensordot(a, b, 1);
    expect(r.dtype).toBe("int64");
    expect(r.toArray()).toEqual([
      [1152921504606846997n, 8796093022208n],
      [8388608n, 1152921504606846981n],
    ]);
  });

  it("promotes mixed dtypes the way NumPy does", () => {
    const r = tensordot(
      tensor([1, 2], { dtype: "int32" }),
      tensor([3, 4], { dtype: "float64" }),
      1
    );
    expect(r.dtype).toBe("float64");
    expect(flat(r)).toEqual([11]);
    const small = tensordot(
      tensor([1, 2], { dtype: "uint8" }),
      tensor([3, 4], { dtype: "int32" }),
      1
    );
    expect(small.dtype).toBe("int32");
  });

  it("mixes int64 with other dtypes through float64", () => {
    // numpy.tensordot(int64 [1, 2], float64 [3, 4], 1) = 11.0
    const r = tensordot(
      tensor([1, 2], { dtype: "int64" }),
      tensor([3, 4], { dtype: "float64" }),
      1
    );
    expect(r.dtype).toBe("float64");
    expect(flat(r)).toEqual([11]);
    const s = tensordot(
      tensor([[1, 2]], { dtype: "int32" }),
      tensor([[3], [4]], { dtype: "int64" }),
      1
    );
    expect(s.dtype).toBe("float64");
    expect(flat(s)).toEqual([11]);
  });

  it("propagates NaN for 0 * Infinity", () => {
    const r = tensordot(
      tensor([[0, 1]], { dtype: "float64" }),
      tensor([[Number.POSITIVE_INFINITY], [1]], { dtype: "float64" }),
      1
    );
    expect(Number.isNaN(flat(r)[0])).toBe(true);
  });

  it("reads non-contiguous views in place", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    // numpy.tensordot(m.T, m, axes=1) == m.T @ m
    const r = tensordot(transpose(m), m, 1);
    expect(r.shape).toEqual([3, 3]);
    expect(flat(r)).toEqual([17, 22, 27, 22, 29, 36, 27, 36, 45]);
  });

  it("handles empty operands", () => {
    const r = tensordot(
      tensor([[]], { dtype: "float64" }).reshape([1, 0]),
      tensor([[]], { dtype: "float64" }).reshape([0, 1]),
      1
    );
    expect(r.shape).toEqual([1, 1]);
    expect(flat(r)).toEqual([0]);
  });
});

describe("cov", () => {
  const x = tensor(
    [
      [1, 2, 4, 7],
      [3, 1, 0, 9],
      [5, 5, 5, 5],
    ],
    { dtype: "float64" }
  );

  it("matches numpy.cov for rows as variables", () => {
    const r = cov(x);
    expect(r.shape).toEqual([3, 3]);
    const want = [7, 7.5, 0, 7.5, 16.25, 0, 0, 0, 0];
    flat(r).forEach((v, i) => {
      expect(v).toBeCloseTo(want[i] as number, 12);
    });
  });

  it("supports ddof and rowvar", () => {
    flat(cov(x, 0)).forEach((v, i) => {
      expect(v).toBeCloseTo([5.25, 5.625, 0, 5.625, 12.1875, 0, 0, 0, 0][i] as number, 12);
    });
    // numpy.cov(x, rowvar=False)
    const cols = cov(x, 1, false);
    expect(cols.shape).toEqual([4, 4]);
    const want = [4, 3, 1, -2, 3, 13 / 3, 4.5, -4, 1, 4.5, 7, -5, -2, -4, -5, 4];
    flat(cols).forEach((v, i) => {
      expect(v).toBeCloseTo(want[i] as number, 12);
    });
    const tall = cov(
      tensor(
        [
          [1, 2],
          [3, 4],
          [5, 7],
        ],
        { dtype: "float64" }
      ),
      1,
      false
    );
    expect(flat(tall)).toEqual([4, 5, 5, 19 / 3]);
  });

  it("is accurate for data with a large offset", () => {
    // numpy.cov([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4]) = 5/3
    const r = cov(tensor([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4], { dtype: "float64" }));
    expect(flat(r)[0]).toBeCloseTo(5 / 3, 14);
  });

  it("treats a 0-d tensor as a single observation", () => {
    expect(flat(cov(tensor(5), 0))).toEqual([0]);
    expect(() => cov(tensor(5))).toThrow(InvalidParameterError);
  });

  it("throws instead of returning zeros when ddof leaves no degrees of freedom", () => {
    expect(() => cov(tensor([1]))).toThrow(
      /ddof=1 must be smaller than the number of observations \(1\)/
    );
    expect(() => cov(tensor([1, 2]), 2)).toThrow(InvalidParameterError);
    expect(() => cov(tensor([]))).toThrow(InvalidParameterError);
  });

  it("validates ddof, dtype, ndim and device", () => {
    expect(() => cov(tensor([1, 2, 3]), -1)).toThrow(InvalidParameterError);
    expect(() => cov(tensor([1, 2, 3]), Number.NaN)).toThrow(InvalidParameterError);
    expect(() => cov(tensor(["a", "b"]))).toThrow(DTypeError);
    expect(() => cov(tensor([[[1, 2]]]))).toThrow(ShapeError);
  });

  it("reads strided views and integer dtypes", () => {
    const m = tensor(
      [
        [1, 3],
        [2, 1],
        [4, 0],
        [7, 9],
      ],
      { dtype: "int32" }
    );
    // m.T has rows [1, 2, 4, 7] and [3, 1, 0, 9]
    const r = cov(transpose(m));
    expect(flat(r).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([7, 7.5, 7.5, 16.25]);
  });
});

describe("corrcoef", () => {
  it("matches numpy.corrcoef and honors rowvar", () => {
    const x = tensor(
      [
        [1, 2, 4, 7],
        [3, 1, 0, 9],
      ],
      { dtype: "float64" }
    );
    const r = flat(corrcoef(x));
    expect(r[0]).toBeCloseTo(1, 15);
    expect(r[1]).toBeCloseTo(0.7032108464077431, 14);
    expect(r[2]).toBeCloseTo(0.7032108464077431, 14);
    expect(r[3]).toBeCloseTo(1, 15);

    const cols = corrcoef(x, false);
    expect(cols.shape).toEqual([4, 4]);
    const c = flat(cols);
    expect(c[1]).toBeCloseTo(-1, 12);
    expect(c[0]).toBe(1);
  });

  it("returns NaN for a constant variable (numpy convention)", () => {
    const x = tensor(
      [
        [1, 2, 4, 7],
        [3, 1, 0, 9],
        [5, 5, 5, 5],
      ],
      { dtype: "float64" }
    );
    const r = flat(corrcoef(x));
    expect(r[0]).toBeCloseTo(1, 15);
    expect(r[1]).toBeCloseTo(0.7032108464077431, 14);
    for (const idx of [2, 5, 6, 7, 8]) expect(Number.isNaN(r[idx])).toBe(true);
    expect(Number.isNaN(flat(corrcoef(tensor([2, 2, 2])))[0])).toBe(true);
  });

  it("does not overflow the product of two large variances", () => {
    // numpy.corrcoef -> [[1, 1], [1, 1]]; var * var would be 1e400 = Infinity.
    const y = tensor(
      [
        [1e100, 2e100, 3e100],
        [2e100, 4e100, 6e100],
      ],
      { dtype: "float64" }
    );
    expect(flat(corrcoef(y))).toEqual([1, 1, 1, 1]);
  });

  it("never exceeds 1 in magnitude", () => {
    const base = [0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.7];
    const x = tensor([base, base.map((v) => v * 3), base.map((v) => -v * 7)], { dtype: "float64" });
    for (const v of flat(corrcoef(x))) {
      expect(Math.abs(v)).toBeLessThanOrEqual(1);
    }
  });

  it("requires at least two observations", () => {
    expect(() => corrcoef(tensor([1]))).toThrow(InvalidParameterError);
    expect(() => corrcoef(tensor([[1], [2]]))).toThrow(/at least 2 observations/);
    expect(() => corrcoef(tensor(["a", "b"]))).toThrow(DTypeError);
  });
});
