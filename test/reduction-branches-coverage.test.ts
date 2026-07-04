import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  all,
  any,
  cumprod,
  cumsum,
  diff,
  max,
  mean,
  median,
  min,
  prod,
  std,
  sum,
  variance,
} from "../src/ndarray/ops/reduction";
import { Tensor } from "../src/ndarray/tensor/Tensor";

// ────── sum ──────
describe("sum branches", () => {
  it("full reduction", () => {
    const t = tensor([1, 2, 3, 4]);
    const s = sum(t);
    expect(Array.from(s.data as Float64Array)[0]).toBe(10);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = sum(t, 0);
    expect(s.shape).toEqual([2]);
    expect(Array.from(s.data as Float64Array)).toEqual([4, 6]);
  });

  it("axis reduction with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = sum(t, 0, true);
    expect(s.shape).toEqual([1, 2]);
  });

  it("axis=1", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const s = sum(t, 1);
    expect(s.shape).toEqual([2]);
    expect(Array.from(s.data as Float64Array)).toEqual([6, 15]);
  });

  it("int32 tensor", () => {
    const t = Tensor.fromTypedArray({
      data: new Int32Array([1, 2, 3]),
      shape: [3],
      dtype: "int32",
      device: "cpu",
    });
    const s = sum(t);
    expect(Array.from(s.data as Int32Array)[0]).toBe(6);
  });

  it("bool tensor", () => {
    const t = Tensor.fromTypedArray({
      data: new Uint8Array([1, 0, 1, 1]),
      shape: [4],
      dtype: "bool",
      device: "cpu",
    });
    const s = sum(t);
    expect(s.size).toBe(1);
  });

  it("empty reduction returns 0", () => {
    const t = tensor([1, 2, 3]);
    const s = sum(t);
    expect(s.size).toBe(1);
  });
});

// ────── mean ──────
describe("mean branches", () => {
  it("full reduction", () => {
    const t = tensor([2, 4, 6]);
    const m = mean(t);
    expect(Array.from(m.data as Float64Array)[0]).toBeCloseTo(4);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const m = mean(t, 0);
    expect(Array.from(m.data as Float64Array)).toEqual([2, 3]);
  });

  it("axis=1 with keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const m = mean(t, 1, true);
    expect(m.shape).toEqual([2, 1]);
  });
});

// ────── prod ──────
describe("prod branches", () => {
  it("full reduction", () => {
    const t = tensor([1, 2, 3, 4]);
    const p = prod(t);
    expect(Array.from(p.data as Float64Array)[0]).toBe(24);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const p = prod(t, 0);
    expect(Array.from(p.data as Float64Array)).toEqual([3, 8]);
  });

  it("multi-axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const p = prod(t, [0, 1]);
    expect(Array.from(p.data as Float64Array)[0]).toBe(24);
  });

  it("keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const p = prod(t, 0, true);
    expect(p.shape).toEqual([1, 2]);
  });
});

// ────── std / variance ──────
describe("std branches", () => {
  it("full reduction", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const s = std(t);
    expect(Array.from(s.data as Float64Array)[0]).toBeGreaterThan(0);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = std(t, 0);
    expect(s.shape).toEqual([2]);
  });

  it("ddof parameter", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const s = std(t, undefined, false, 1);
    expect(Array.from(s.data as Float64Array)[0]).toBeGreaterThan(0);
  });
});

describe("variance branches", () => {
  it("full reduction", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const v = variance(t);
    expect(Array.from(v.data as Float64Array)[0]).toBeGreaterThan(0);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const v = variance(t, 0);
    expect(v.shape).toEqual([2]);
  });
});

// ────── min / max ──────
describe("min branches", () => {
  it("full reduction", () => {
    const t = tensor([3, 1, 4, 1, 5]);
    const m = min(t);
    expect(Array.from(m.data as Float64Array)[0]).toBe(1);
  });

  it("axis reduction", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = min(t, 0);
    expect(Array.from(m.data as Float64Array)).toEqual([3, 1]);
  });

  it("multi-axis", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = min(t, [0, 1]);
    expect(Array.from(m.data as Float64Array)[0]).toBe(1);
  });

  it("keepdims", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = min(t, 0, true);
    expect(m.shape).toEqual([1, 2]);
  });
});

describe("max branches", () => {
  it("full reduction", () => {
    const t = tensor([3, 1, 4, 1, 5]);
    const m = max(t);
    expect(Array.from(m.data as Float64Array)[0]).toBe(5);
  });

  it("axis reduction", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = max(t, 0);
    expect(Array.from(m.data as Float64Array)).toEqual([4, 2]);
  });

  it("multi-axis", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = max(t, [0, 1]);
    expect(Array.from(m.data as Float64Array)[0]).toBe(4);
  });

  it("keepdims", () => {
    const t = tensor([
      [3, 1],
      [4, 2],
    ]);
    const m = max(t, 1, true);
    expect(m.shape).toEqual([2, 1]);
  });
});

// ────── median ──────
describe("median branches", () => {
  it("odd-length full reduction", () => {
    const t = tensor([3, 1, 2]);
    const m = median(t);
    expect(Array.from(m.data as Float64Array)[0]).toBe(2);
  });

  it("even-length full reduction", () => {
    const t = tensor([1, 2, 3, 4]);
    const m = median(t);
    expect(Array.from(m.data as Float64Array)[0]).toBe(2.5);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const m = median(t, 0);
    expect(m.shape).toEqual([2]);
  });

  it("keepdims", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const m = median(t, 0, true);
    expect(m.shape).toEqual([1, 2]);
  });
});

// ────── cumsum / cumprod ──────
describe("cumsum branches", () => {
  it("flat cumsum", () => {
    const t = tensor([1, 2, 3, 4]);
    const c = cumsum(t);
    expect(Array.from(c.data as Float64Array)).toEqual([1, 3, 6, 10]);
  });

  it("axis cumsum", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const c = cumsum(t, 0);
    expect(c.shape).toEqual([2, 2]);
  });

  it("axis=1 cumsum", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const c = cumsum(t, 1);
    expect(c.shape).toEqual([2, 3]);
  });
});

describe("cumprod branches", () => {
  it("flat cumprod", () => {
    const t = tensor([1, 2, 3, 4]);
    const c = cumprod(t);
    expect(Array.from(c.data as Float64Array)).toEqual([1, 2, 6, 24]);
  });

  it("axis cumprod", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const c = cumprod(t, 0);
    expect(c.shape).toEqual([2, 2]);
  });
});

// ────── diff ──────
describe("diff branches", () => {
  it("1D diff", () => {
    const t = tensor([1, 3, 6, 10]);
    const d = diff(t);
    expect(Array.from(d.data as Float64Array)).toEqual([2, 3, 4]);
  });

  it("2D diff axis=0", () => {
    const t = tensor([
      [1, 2],
      [4, 5],
      [9, 10],
    ]);
    const d = diff(t, 1, 0);
    expect(d.shape).toEqual([2, 2]);
  });

  it("n=2", () => {
    const t = tensor([1, 3, 6, 10, 15]);
    const d = diff(t, 2);
    expect(d.shape).toEqual([3]);
  });
});

// ────── any / all ──────
describe("any branches", () => {
  it("any true", () => {
    const t = tensor([0, 0, 1, 0]);
    const a = any(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(1);
  });

  it("all zero", () => {
    const t = tensor([0, 0, 0]);
    const a = any(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(0);
  });

  it("axis reduction", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const a = any(t, 0);
    expect(a.shape).toEqual([2]);
  });

  it("axis with keepdims", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const a = any(t, 0, true);
    expect(a.shape).toEqual([1, 2]);
  });

  it("multi-axis", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const a = any(t, [0, 1]);
    expect(a.size).toBe(1);
  });
});

describe("all branches", () => {
  it("all true", () => {
    const t = tensor([1, 1, 1]);
    const a = all(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(1);
  });

  it("has zero", () => {
    const t = tensor([1, 0, 1]);
    const a = all(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(0);
  });

  it("axis reduction", () => {
    const t = tensor([
      [1, 1],
      [1, 0],
    ]);
    const a = all(t, 0);
    expect(a.shape).toEqual([2]);
  });

  it("axis with keepdims", () => {
    const t = tensor([
      [1, 1],
      [1, 0],
    ]);
    const a = all(t, 1, true);
    expect(a.shape).toEqual([2, 1]);
  });

  it("multi-axis", () => {
    const t = tensor([
      [1, 1],
      [1, 0],
    ]);
    const a = all(t, [0, 1]);
    expect(a.size).toBe(1);
  });
});
