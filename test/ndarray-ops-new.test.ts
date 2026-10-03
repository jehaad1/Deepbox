import { describe, expect, it } from "vitest";
import { corrcoef, cov, tensor, tensordot } from "../src/ndarray";
import { atleast_1d, atleast_2d, cross, isin, rot90 } from "../src/ndarray/ops/utils";

describe("rot90", () => {
  it("rotates 2D tensor 90° CCW (k=1)", () => {
    // [[1,2],[3,4]] → [[2,4],[1,3]]
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(t);
    expect(r.shape).toEqual([2, 2]);
    expect(r.toArray()).toEqual([
      [2, 4],
      [1, 3],
    ]);
  });

  it("rotates 180° (k=2)", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(t, 2);
    expect(r.toArray()).toEqual([
      [4, 3],
      [2, 1],
    ]);
  });

  it("rotates 270° CCW (k=3)", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(t, 3);
    expect(r.toArray()).toEqual([
      [3, 1],
      [4, 2],
    ]);
  });

  it("k=0 returns a clone", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(t, 0);
    expect(r.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("k=4 is identity", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(t, 4);
    expect(r.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("negative k rotates clockwise", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    // k=-1 → same as k=3
    expect(rot90(t, -1).toArray()).toEqual(rot90(t, 3).toArray());
  });

  it("works on non-square matrices", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = rot90(t);
    expect(r.shape).toEqual([3, 2]);
    expect(r.toArray()).toEqual([
      [3, 6],
      [2, 5],
      [1, 4],
    ]);
  });

  it("throws on non-2D input", () => {
    expect(() => rot90(tensor([1, 2, 3]))).toThrow("2-D");
  });
});

describe("atleast_1d", () => {
  it("passes through 1-D tensors unchanged", () => {
    const t = tensor([1, 2, 3]);
    const r = atleast_1d(t);
    expect(r.shape).toEqual([3]);
  });

  it("passes through 2-D tensors unchanged", () => {
    const t = tensor([[1, 2]]);
    const r = atleast_1d(t);
    expect(r.shape).toEqual([1, 2]);
  });

  it("converts scalar to [1]", () => {
    const t = tensor(5);
    const r = atleast_1d(t);
    expect(r.shape).toEqual([1]);
    expect(r.toArray()).toEqual([5]);
  });
});

describe("atleast_2d", () => {
  it("passes through 2-D tensors unchanged", () => {
    const t = tensor([[1, 2]]);
    const r = atleast_2d(t);
    expect(r.shape).toEqual([1, 2]);
  });

  it("converts 1-D to [1, n]", () => {
    const t = tensor([1, 2, 3]);
    const r = atleast_2d(t);
    expect(r.shape).toEqual([1, 3]);
    expect(r.toArray()).toEqual([[1, 2, 3]]);
  });

  it("converts scalar to [1, 1]", () => {
    const t = tensor(5);
    const r = atleast_2d(t);
    expect(r.shape).toEqual([1, 1]);
    expect(r.toArray()).toEqual([[5]]);
  });

  it("passes through 3-D tensors unchanged", () => {
    const t = tensor([[[1, 2]]]);
    const r = atleast_2d(t);
    expect(r.shape).toEqual([1, 1, 2]);
  });
});

describe("cross", () => {
  it("computes cross product of standard basis vectors", () => {
    // i × j = k
    const r = cross(tensor([1, 0, 0]), tensor([0, 1, 0]));
    expect(r.toArray()).toEqual([0, 0, 1]);
  });

  it("computes cross product correctly", () => {
    // [1,2,3] × [4,5,6] = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    const r = cross(tensor([1, 2, 3]), tensor([4, 5, 6]));
    expect(r.toArray()).toEqual([-3, 6, -3]);
  });

  it("cross product is anti-commutative", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const ab = cross(a, b).toArray() as number[];
    const ba = cross(b, a).toArray() as number[];
    for (let i = 0; i < 3; i++) {
      expect(ab[i]).toBeCloseTo(-ba[i]!, 10);
    }
  });

  it("cross product of parallel vectors is zero", () => {
    const r = cross(tensor([1, 2, 3]), tensor([2, 4, 6]));
    expect(r.toArray()).toEqual([0, 0, 0]);
  });

  it("throws on wrong dimensions", () => {
    expect(() => cross(tensor([1, 2]), tensor([3, 4]))).toThrow();
    // [[1, 2, 3]] is a valid batch of one vector since cross broadcasts; a trailing length of 2 is not
    expect(() => cross(tensor([[1, 2]]), tensor([4, 5, 6]))).toThrow();
  });
});

describe("isin", () => {
  it("tests membership with number array", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const r = isin(a, [2, 4]);
    expect(r.toArray()).toEqual([0, 1, 0, 1, 0]);
  });

  it("tests membership with tensor values", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const vals = tensor([2, 4]);
    const r = isin(a, vals);
    expect(r.toArray()).toEqual([0, 1, 0, 1, 0]);
  });

  it("works on 2D tensors", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = isin(a, [1, 4]);
    expect(r.shape).toEqual([2, 2]);
    expect(r.toArray()).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  it("returns all false when no matches", () => {
    const a = tensor([1, 2, 3]);
    const r = isin(a, [10, 20]);
    expect(r.toArray()).toEqual([0, 0, 0]);
  });

  it("returns all true when all match", () => {
    const a = tensor([1, 2, 3]);
    const r = isin(a, [1, 2, 3, 4, 5]);
    expect(r.toArray()).toEqual([1, 1, 1]);
  });

  it("handles empty values array", () => {
    const a = tensor([1, 2, 3]);
    const r = isin(a, []);
    expect(r.toArray()).toEqual([0, 0, 0]);
  });
});

describe("tensordot", () => {
  it("axes=0 is outer product", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const r = tensordot(a, b, 0);
    expect(r.shape).toEqual([2, 2]);
    expect(r.toArray()).toEqual([
      [3, 4],
      [6, 8],
    ]);
  });

  it("axes=1 on vectors is dot product", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const r = tensordot(a, b, 1);
    expect(r.shape).toEqual([]);
    const val = r.toArray() as number;
    expect(val).toBeCloseTo(32, 10); // 1*4+2*5+3*6
  });

  it("axes=1 on 2D is matrix multiply", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    const r = tensordot(a, b, 1);
    expect(r.shape).toEqual([2, 2]);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    expect(r.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("custom axes pairs", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]); // (2,3)
    const b = tensor([
      [1, 4],
      [2, 5],
      [3, 6],
    ]); // (3,2)
    // Contract axis 1 of a with axis 0 of b → (2,2)
    const r = tensordot(a, b, [[1], [0]]);
    expect(r.shape).toEqual([2, 2]);
    expect(r.toArray()).toEqual([
      [14, 32],
      [32, 77],
    ]);
  });

  it("throws for mismatched contracted axes", () => {
    const a = tensor([[1, 2]]); // (1,2)
    const b = tensor([[1, 2, 3]]); // (1,3)
    expect(() => tensordot(a, b, 1)).toThrow();
  });
});

describe("corrcoef", () => {
  it("1D input returns [[1]]", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const r = corrcoef(x);
    expect(r.shape).toEqual([1, 1]);
    expect((r.toArray() as number[][])[0]![0]).toBeCloseTo(1.0, 10);
  });

  it("perfectly correlated variables", () => {
    const x = tensor([
      [1, 2, 3, 4],
      [2, 4, 6, 8],
    ]);
    const r = corrcoef(x);
    expect(r.shape).toEqual([2, 2]);
    const arr = r.toArray() as number[][];
    expect(arr[0]![0]).toBeCloseTo(1.0, 10);
    expect(arr[0]![1]).toBeCloseTo(1.0, 10);
    expect(arr[1]![0]).toBeCloseTo(1.0, 10);
    expect(arr[1]![1]).toBeCloseTo(1.0, 10);
  });

  it("negatively correlated variables", () => {
    const x = tensor([
      [1, 2, 3, 4],
      [4, 3, 2, 1],
    ]);
    const r = corrcoef(x);
    const arr = r.toArray() as number[][];
    expect(arr[0]![1]).toBeCloseTo(-1.0, 10);
  });

  it("uncorrelated variables have low correlation", () => {
    const x = tensor([
      [1, 2, 3, 4, 5],
      [5, 1, 4, 2, 3],
    ]);
    const r = corrcoef(x);
    const arr = r.toArray() as number[][];
    expect(Math.abs(arr[0]![1]!)).toBeLessThan(0.5);
  });
});

describe("cov", () => {
  it("1D input returns scalar variance", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const r = cov(x);
    expect(r.shape).toEqual([1, 1]);
    // var with ddof=1: 2.5
    expect((r.toArray() as number[][])[0]![0]).toBeCloseTo(2.5, 10);
  });

  it("2D input returns covariance matrix", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = cov(x);
    expect(r.shape).toEqual([2, 2]);
    const arr = r.toArray() as number[][];
    // var(row0) = var([1,2,3]) = 1.0
    expect(arr[0]![0]).toBeCloseTo(1.0, 10);
    // var(row1) = var([4,5,6]) = 1.0
    expect(arr[1]![1]).toBeCloseTo(1.0, 10);
    // cov(row0, row1) = 1.0 (perfectly correlated)
    expect(arr[0]![1]).toBeCloseTo(1.0, 10);
    expect(arr[1]![0]).toBeCloseTo(1.0, 10);
  });

  it("respects ddof=0", () => {
    const x = tensor([2, 4, 6]);
    const r = cov(x, 0);
    const arr = r.toArray() as number[][];
    // population variance of [2,4,6] = ((2-4)^2+(4-4)^2+(6-4)^2)/3 = 8/3
    expect(arr[0]![0]).toBeCloseTo(8 / 3, 10);
  });

  it("symmetric matrix", () => {
    const x = tensor([
      [1, 2, 3, 4],
      [5, 3, 8, 1],
    ]);
    const r = cov(x);
    const arr = r.toArray() as number[][];
    expect(arr[0]![1]).toBeCloseTo(arr[1]![0]!, 10);
  });
});
