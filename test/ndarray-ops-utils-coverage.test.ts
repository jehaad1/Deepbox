import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  atleast_1d,
  atleast_2d,
  bincount,
  booleanIndex,
  broadcast_to,
  clone,
  contiguous,
  copy,
  cross,
  detach,
  diag,
  diagonal,
  empty_like,
  fancyIndex,
  flip,
  fliplr,
  flipud,
  full_like,
  histogram,
  isin,
  moveaxis,
  nanmax,
  nanmean,
  nanmin,
  nanstd,
  nansum,
  ones_like,
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
  zeros_like,
} from "../src/ndarray/ops/utils";

// ────── where ──────
describe("where", () => {
  it("selects from x when condition is true", () => {
    const cond = tensor([1, 0, 1, 0]);
    const x = tensor([10, 20, 30, 40]);
    const y = tensor([1, 2, 3, 4]);
    const out = where(cond, x, y);
    expect(Array.from(out.data as Float64Array)).toEqual([10, 2, 30, 4]);
  });

  it("broadcasts shapes", () => {
    const cond = tensor([1, 0]);
    const x = tensor([
      [10, 20],
      [30, 40],
    ]);
    const y = tensor([
      [1, 2],
      [3, 4],
    ]);
    const out = where(cond, x, y);
    expect(out.shape).toEqual([2, 2]);
  });
});

// ────── zeros_like / ones_like / empty_like / full_like ──────
describe("zeros_like", () => {
  it("creates zero tensor of same shape", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const z = zeros_like(a);
    expect(z.shape).toEqual([2, 2]);
    expect(Array.from(z.data as Float64Array)).toEqual([0, 0, 0, 0]);
  });
});

describe("ones_like", () => {
  it("creates ones tensor of same shape", () => {
    const a = tensor([1, 2, 3]);
    const o = ones_like(a);
    expect(o.shape).toEqual([3]);
    expect(Array.from(o.data as Float64Array)).toEqual([1, 1, 1]);
  });
});

describe("empty_like", () => {
  it("creates tensor of same shape", () => {
    const a = tensor([1, 2]);
    const e = empty_like(a);
    expect(e.shape).toEqual([2]);
  });
});

describe("full_like", () => {
  it("creates tensor filled with value", () => {
    const a = tensor([1, 2, 3]);
    const f = full_like(a, 7);
    expect(Array.from(f.data as Float64Array)).toEqual([7, 7, 7]);
  });
});

// ────── clone / detach / contiguous / copy ──────
describe("clone", () => {
  it("creates independent copy", () => {
    const a = tensor([1, 2, 3]);
    const b = clone(a);
    expect(b.shape).toEqual([3]);
    expect(Array.from(b.data as Float64Array)).toEqual([1, 2, 3]);
  });
});

describe("detach", () => {
  it("returns a copy", () => {
    const a = tensor([4, 5]);
    const b = detach(a);
    expect(Array.from(b.data as Float64Array)).toEqual([4, 5]);
  });
});

describe("contiguous", () => {
  it("returns self if already contiguous", () => {
    const a = tensor([1, 2, 3]);
    const b = contiguous(a);
    expect(b).toBe(a);
  });
});

describe("copy", () => {
  it("returns a deep copy", () => {
    const a = tensor([1, 2]);
    const b = copy(a);
    expect(Array.from(b.data as Float64Array)).toEqual([1, 2]);
  });
});

// ────── diag / diagonal ──────
describe("diag", () => {
  it("creates diagonal matrix from 1D", () => {
    const a = tensor([1, 2, 3]);
    const d = diag(a);
    expect(d.shape).toEqual([3, 3]);
  });

  it("extracts diagonal from 2D", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const d = diag(a);
    expect(Array.from(d.data as Float64Array)).toEqual([1, 4]);
  });

  it("supports k offset", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const d = diag(a, 1);
    expect(Array.from(d.data as Float64Array)).toEqual([2, 6]);
  });

  it("throws for 3D", () => {
    const a = tensor([[[1]]]);
    expect(() => diag(a)).toThrow();
  });
});

describe("diagonal", () => {
  it("extracts main diagonal", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const d = diagonal(a);
    expect(Array.from(d.data as Float64Array)).toEqual([1, 5]);
  });

  it("extracts sub-diagonal", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const d = diagonal(a, -1);
    expect(Array.from(d.data as Float64Array)).toEqual([3, 6]);
  });
});

// ────── triu / tril ──────
describe("triu", () => {
  it("upper triangular", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const u = triu(a);
    expect(Array.from(u.data as Float64Array)).toEqual([1, 2, 3, 0, 5, 6, 0, 0, 9]);
  });

  it("upper triangular with offset", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const u = triu(a, 1);
    expect(Array.from(u.data as Float64Array)).toEqual([0, 2, 3, 0, 0, 6, 0, 0, 0]);
  });
});

describe("tril", () => {
  it("lower triangular", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const l = tril(a);
    expect(Array.from(l.data as Float64Array)).toEqual([1, 0, 0, 4, 5, 0, 7, 8, 9]);
  });
});

// ────── flip / fliplr / flipud ──────
describe("flip", () => {
  it("reverses 1D", () => {
    const a = tensor([1, 2, 3]);
    const f = flip(a);
    expect(Array.from(f.data as Float64Array)).toEqual([3, 2, 1]);
  });

  it("reverses specific axis", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const f = flip(a, [1]);
    expect(Array.from(f.data as Float64Array)).toEqual([2, 1, 4, 3]);
  });
});

describe("fliplr", () => {
  it("flips left-right", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const f = fliplr(a);
    expect(Array.from(f.data as Float64Array)).toEqual([2, 1, 4, 3]);
  });
});

describe("flipud", () => {
  it("flips up-down", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const f = flipud(a);
    expect(Array.from(f.data as Float64Array)).toEqual([3, 4, 1, 2]);
  });
});

// ────── roll ──────
describe("roll", () => {
  it("rolls flat", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const r = roll(a, 2);
    expect(Array.from(r.data as Float64Array)).toEqual([4, 5, 1, 2, 3]);
  });

  it("rolls along axis", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = roll(a, 1, 1);
    expect(Array.from(r.data as Float64Array)).toEqual([3, 1, 2, 6, 4, 5]);
  });
});

// ────── pad ──────
describe("pad", () => {
  it("constant pad", () => {
    const a = tensor([1, 2, 3]);
    const p = pad(a, [[2, 1]]);
    expect(Array.from(p.data as Float64Array)).toEqual([0, 0, 1, 2, 3, 0]);
  });

  it("replicate pad", () => {
    const a = tensor([1, 2, 3]);
    const p = pad(a, [[1, 1]], "replicate");
    expect(Array.from(p.data as Float64Array)).toEqual([1, 1, 2, 3, 3]);
  });

  it("reflect pad", () => {
    const a = tensor([1, 2, 3]);
    const p = pad(a, [[1, 1]], "reflect");
    expect(Array.from(p.data as Float64Array)).toEqual([2, 1, 2, 3, 2]);
  });

  it("circular pad", () => {
    const a = tensor([1, 2, 3]);
    const p = pad(a, [[1, 1]], "circular");
    expect(Array.from(p.data as Float64Array)).toEqual([3, 1, 2, 3, 1]);
  });

  it("validates padWidth length", () => {
    const a = tensor([1, 2]);
    expect(() =>
      pad(a, [
        [1, 1],
        [1, 1],
      ])
    ).toThrow(/padWidth/);
  });
});

// ────── moveaxis / swapaxes ──────
describe("moveaxis", () => {
  it("moves axis", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const m = moveaxis(a, 0, 1);
    expect(m.shape).toEqual([2, 3]);
  });

  it("validates length mismatch", () => {
    const a = tensor([[1, 2]]);
    expect(() => moveaxis(a, [0], [0, 1])).toThrow();
  });
});

describe("swapaxes", () => {
  it("swaps axes", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const s = swapaxes(a, 0, 1);
    expect(s.shape).toEqual([3, 2]);
  });
});

// ────── broadcast_to ──────
describe("broadcast_to", () => {
  it("broadcasts to target shape", () => {
    const a = tensor([1, 2, 3]);
    const b = broadcast_to(a, [2, 3]);
    expect(b.shape).toEqual([2, 3]);
  });

  it("throws on incompatible shapes", () => {
    const a = tensor([1, 2, 3]);
    expect(() => broadcast_to(a, [2, 4])).toThrow();
  });

  it("throws when target has fewer dims", () => {
    const a = tensor([[1, 2]]);
    expect(() => broadcast_to(a, [2])).toThrow();
  });
});

// ────── scatter ──────
describe("scatter", () => {
  it("scatters values", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    const idx = tensor([
      [0, 1, 2],
      [2, 0, 1],
    ]);
    const src = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const out = scatter(t, 1, idx, src);
    expect(out.shape).toEqual([2, 3]);
  });
});

// ────── NaN-aware reductions ──────
describe("nansum", () => {
  it("sums ignoring NaN", () => {
    const a = tensor([1, NaN, 3, NaN, 5]);
    const s = nansum(a);
    expect(Array.from(s.data as Float64Array)[0]).toBeCloseTo(9);
  });
});

describe("nanmean", () => {
  it("means ignoring NaN", () => {
    const a = tensor([1, NaN, 3, NaN, 5]);
    const m = nanmean(a);
    expect(Array.from(m.data as Float64Array)[0]).toBeCloseTo(3);
  });
});

describe("nanstd", () => {
  it("std ignoring NaN", () => {
    const a = tensor([1, NaN, 3, NaN, 5]);
    const s = nanstd(a);
    expect(Array.from(s.data as Float64Array)[0]).toBeGreaterThan(0);
  });
});

describe("nanmin", () => {
  it("min ignoring NaN", () => {
    const a = tensor([3, NaN, 1, NaN, 5]);
    const m = nanmin(a);
    expect(Array.from(m.data as Float64Array)[0]).toBe(1);
  });
});

describe("nanmax", () => {
  it("max ignoring NaN", () => {
    const a = tensor([3, NaN, 1, NaN, 5]);
    const m = nanmax(a);
    expect(Array.from(m.data as Float64Array)[0]).toBe(5);
  });
});

// ────── unique ──────
describe("unique", () => {
  it("returns unique values", () => {
    const a = tensor([3, 1, 2, 1, 3, 2]);
    const { values } = unique(a);
    expect(Array.from(values.data as Float64Array)).toEqual([1, 2, 3]);
  });

  it("returns counts when requested", () => {
    const a = tensor([1, 1, 2, 3, 3, 3]);
    const { values, counts } = unique(a, true);
    expect(Array.from(values.data as Float64Array)).toEqual([1, 2, 3]);
    expect(Array.from(counts!.data as Float64Array)).toEqual([2, 1, 3]);
  });
});

// ────── searchsorted ──────
describe("searchsorted", () => {
  it("finds left insertion points", () => {
    const sorted = tensor([1, 3, 5, 7]);
    const vals = tensor([2, 4, 6]);
    const out = searchsorted(sorted, vals);
    expect(Array.from(out.data as Float64Array)).toEqual([1, 2, 3]);
  });

  it("finds right insertion points", () => {
    const sorted = tensor([1, 3, 5, 7]);
    const vals = tensor([3, 5]);
    const out = searchsorted(sorted, vals, "right");
    expect(Array.from(out.data as Float64Array)).toEqual([2, 3]);
  });
});

// ────── histogram ──────
describe("histogram", () => {
  it("computes histogram", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const { counts, binEdges } = histogram(a, 5);
    expect(counts.shape).toEqual([5]);
    expect(binEdges.shape).toEqual([6]);
  });

  it("with range", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const { counts } = histogram(a, 5, [0, 10]);
    expect(counts.shape).toEqual([5]);
  });
});

// ────── bincount ──────
describe("bincount", () => {
  it("counts occurrences", () => {
    const a = tensor([0, 1, 1, 2, 2, 2]);
    const c = bincount(a);
    expect(Array.from(c.data as Float64Array)).toEqual([1, 2, 3]);
  });

  it("validates negative values", () => {
    const a = tensor([-1, 0]);
    expect(() => bincount(a)).toThrow();
  });
});

// ────── booleanIndex ──────
describe("booleanIndex", () => {
  it("selects by mask", () => {
    const a = tensor([10, 20, 30, 40, 50]);
    const mask = tensor([1, 0, 1, 0, 1]);
    const out = booleanIndex(a, mask);
    expect(Array.from(out.data as Float64Array)).toEqual([10, 30, 50]);
  });

  it("validates size mismatch", () => {
    const a = tensor([1, 2, 3]);
    const mask = tensor([1, 0]);
    expect(() => booleanIndex(a, mask)).toThrow();
  });
});

// ────── fancyIndex ──────
describe("fancyIndex", () => {
  it("indexes with integer array", () => {
    const a = tensor([
      [10, 20],
      [30, 40],
      [50, 60],
    ]);
    const idx = tensor([0, 2]);
    const out = fancyIndex(a, idx, 0);
    expect(out.shape).toEqual([2, 2]);
    expect(Array.from(out.data as Float64Array)).toEqual([10, 20, 50, 60]);
  });
});

// ────── rot90 ──────
describe("rot90", () => {
  it("rotates 90 degrees", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(a);
    expect(r.shape).toEqual([2, 2]);
  });

  it("k=0 returns clone", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(a, 0);
    expect(Array.from(r.data as Float64Array)).toEqual([1, 2, 3, 4]);
  });

  it("k=2 rotates 180 degrees", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const r = rot90(a, 2);
    expect(Array.from(r.data as Float64Array)).toEqual([4, 3, 2, 1]);
  });

  it("throws for non-2D", () => {
    const a = tensor([1, 2, 3]);
    expect(() => rot90(a)).toThrow();
  });
});

// ────── atleast_1d / atleast_2d ──────
describe("atleast_1d", () => {
  it("passes through 1D+", () => {
    const a = tensor([1, 2]);
    expect(atleast_1d(a)).toBe(a);
  });
});

describe("atleast_2d", () => {
  it("promotes 1D to 2D", () => {
    const a = tensor([1, 2, 3]);
    const b = atleast_2d(a);
    expect(b.shape).toEqual([1, 3]);
  });

  it("passes through 2D+", () => {
    const a = tensor([[1, 2]]);
    expect(atleast_2d(a)).toBe(a);
  });
});

// ────── cross ──────
describe("cross", () => {
  it("computes cross product", () => {
    const a = tensor([1, 0, 0]);
    const b = tensor([0, 1, 0]);
    const c = cross(a, b);
    expect(Array.from(c.data as Float64Array)).toEqual([0, 0, 1]);
  });

  it("validates shape", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4, 5]);
    expect(() => cross(a, b)).toThrow();
  });
});

// ────── isin ──────
describe("isin", () => {
  it("tests membership with array", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const out = isin(a, [2, 4]);
    expect(Array.from(out.data as Uint8Array)).toEqual([0, 1, 0, 1, 0]);
  });

  it("tests membership with tensor", () => {
    const a = tensor([1, 2, 3]);
    const vals = tensor([3, 1]);
    const out = isin(a, vals);
    expect(Array.from(out.data as Uint8Array)).toEqual([1, 0, 1]);
  });
});
