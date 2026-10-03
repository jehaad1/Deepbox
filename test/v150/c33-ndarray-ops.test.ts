/**
 * Regression tests for ndarray reductions, set operations and signal helpers
 * (v1.5.0 review).
 *
 * Reference values come from NumPy 2.4 / SciPy 1.17 (`np.sum`, `np.var`,
 * `np.median`, `np.cumsum`, `np.diff`, `np.convolve`, `np.correlate`,
 * `np.kaiser`, `scipy.signal.windows`, `np.gcd`, `np.lcm`, `np.union1d`).
 */
import { describe, expect, it } from "vitest";
import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import type { Tensor } from "../../src/ndarray";
import {
  all,
  any,
  bartlettWindow,
  blackmanWindow,
  convolve,
  correlate,
  cumprod,
  cumsum,
  diff,
  gcd,
  hammingWindow,
  hannWindow,
  intersect1d,
  kaiserWindow,
  lcm,
  max,
  mean,
  median,
  min,
  prod,
  setdiff1d,
  std,
  sum,
  tensor,
  transpose,
  union1d,
  variance,
} from "../../src/ndarray";
import { Tensor as TensorClass } from "../../src/ndarray/tensor/Tensor";

function f64(data: number | number[] | number[][] | number[][][]): Tensor {
  return tensor(data, { dtype: "float64" });
}

function close(actual: ArrayLike<number>, expected: number[], tol = 1e-12): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const a = actual[i] as number;
    expect(Math.abs(a - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  }
}

function values(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number>).slice(t.offset, t.offset + t.size);
}

/** A (2, 3, 2) array 0..11, viewed through transpose (2, 0, 1) -> shape (2, 2, 3) is not contiguous. */
function strided3d(): Tensor {
  const a = f64([
    [
      [0, 1],
      [2, 3],
      [4, 5],
    ],
    [
      [6, 7],
      [8, 9],
      [10, 11],
    ],
  ]);
  return transpose(a, [2, 0, 1]);
}

describe("sum / mean dtype handling", () => {
  it("sums float16 and bfloat16 without truncating to int32", () => {
    for (const dtype of ["float16", "bfloat16"] as const) {
      const t = tensor([0.5, 0.25, 1.5], { dtype });
      const s = sum(t);
      expect(s.dtype).toBe(dtype);
      expect(s.toArray()).toBe(2.25);
      expect(mean(t).toArray()).toBe(0.75);
      expect(sum(tensor([[0.5, 0.25]], { dtype }), 1).toArray()).toEqual([0.75]);
    }
  });

  it("mean of int32 accumulates in float64 instead of overflowing", () => {
    const t = tensor([2_000_000_000, 2_000_000_000], { dtype: "int32" });
    expect(() => sum(t)).toThrow(DataValidationError);
    expect(mean(t).toArray()).toBe(2_000_000_000);
    expect(mean(tensor([[2_000_000_000, 2_000_000_000]], { dtype: "int32" }), 1).toArray()).toEqual(
      [2_000_000_000]
    );
  });

  it("mean divides by n instead of multiplying by 1/n", () => {
    // 0.1 + 0.2 + 0 = 0.30000000000000004; /3 = 0.10000000000000002 (NumPy), *(1/3) = 0.1
    expect(mean(f64([0.1, 0.2, 0])).toArray()).toBe(0.10000000000000002);
    expect(mean(f64([[0.1, 0.2, 0]]), 1).toArray()).toEqual([0.10000000000000002]);
  });

  it("mean of an empty reduction is NaN with the right shape", () => {
    expect(mean(f64([])).toArray()).toBeNaN();
    const z = tensor(new Float64Array(0), { dtype: "float64" }).reshape([0, 3]);
    const m0 = mean(z, 0);
    expect(m0.shape).toEqual([3]);
    expect(values(m0).every(Number.isNaN)).toBe(true);
    expect(mean(z, 1).shape).toEqual([0]);
  });

  it("sums large float64 arrays with pairwise accuracy", () => {
    const n = 1_000_000;
    const t = tensor(new Float64Array(n).fill(0.1), { dtype: "float64" });
    // exact sum of the doubles is 100000.00000000000555..., NumPy's pairwise sum gives ...03e-11
    expect(Math.abs((sum(t).toArray() as number) - 100000)).toBeLessThan(1e-9);
  });
});

describe("pairwise summation matches NumPy bit for bit", () => {
  // x[i] = ((i * 7919) % 1009) / 1013, values from NumPy 2.4 (np.sum, np.mean, np.var, np.std).
  function series(n: number): Tensor {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = ((i * 7919) % 1009) / 1013;
    return tensor(x, { dtype: "float64" });
  }

  it("n = 37 (a left-to-right or four-lane sum rounds differently)", () => {
    const t = series(37);
    expect(sum(t).toArray()).toBe(18.935834155972362);
    expect(mean(t).toArray()).toBe(0.5117793015127665);
    expect(variance(t).toArray()).toBe(0.08457558163345258);
    expect(std(t).toArray()).toBe(0.2908188123788634);
  });

  it("n = 5000 (a left-to-right sum rounds differently)", () => {
    const t = series(5000);
    expect(sum(t).toArray()).toBe(2488.193484698914);
    expect(mean(t).toArray()).toBe(0.49763869693978274);
    expect(variance(t).toArray()).toBe(0.0827094312301385);
    expect(std(t).toArray()).toBe(0.28759247422375034);
  });
});

describe("int64 inputs through strided views", () => {
  // logical matrix [[1, 4], [2, 5], [3, 6]], a transposed (non-contiguous) view
  const view = (): Tensor =>
    transpose(
      tensor(new BigInt64Array([1n, 2n, 3n, 4n, 5n, 6n]), { dtype: "int64" }).reshape([2, 3]),
      [1, 0]
    );

  it("mean, variance and median read the logical layout", () => {
    expect(mean(view(), 0).toArray()).toEqual([2, 5]);
    expect(mean(view(), 1).toArray()).toEqual([2.5, 3.5, 4.5]);
    // int64 input computes in float32
    close(values(variance(view(), 0)), [2 / 3, 2 / 3], 1e-6);
    expect(values(variance(view(), 1))).toEqual([2.25, 2.25, 2.25]);
    expect(median(view(), 0).toArray()).toEqual([2, 5]);
    expect(median(view(), 1).toArray()).toEqual([2.5, 3.5, 4.5]);
  });

  it("any and all read the logical layout", () => {
    const t = transpose(
      tensor(new BigInt64Array([0n, 1n, 0n, 0n, 0n, 2n]), { dtype: "int64" }).reshape([2, 3]),
      [1, 0]
    );
    // logical [[0, 0], [1, 0], [0, 2]]
    expect(any(t, 0).toArray()).toEqual([1, 1]);
    expect(any(t, 1).toArray()).toEqual([0, 1, 1]);
    expect(all(t, 1).toArray()).toEqual([0, 0, 0]);
  });
});

describe("multi-axis reductions", () => {
  // y = transpose(a, [2, 0, 1]); shape (2, 2, 3), non-contiguous
  it("sum/mean/min/max over several axes match NumPy", () => {
    const y = strided3d();
    expect(y.shape).toEqual([2, 2, 3]);
    expect(sum(y, [0, 2]).toArray()).toEqual([15, 51]);
    expect(values(sum(y, [0, 2], true))).toEqual([15, 51]);
    expect(sum(y, [0, 2], true).shape).toEqual([1, 2, 1]);
    expect(sum(y, [0, 1, 2]).toArray()).toBe(66);
    expect(sum(y, [2, 0]).toArray()).toEqual([15, 51]);
    expect(mean(y, [0, 1]).toArray()).toEqual([3.5, 5.5, 7.5]);
    expect(max(y, [0, 1]).toArray()).toEqual([7, 9, 11]);
    expect(min(y, [0, 1]).toArray()).toEqual([0, 2, 4]);
  });

  it("variance and std are joint over the listed axes", () => {
    const y = strided3d();
    // NumPy: y.var(axis=(1, 2)) == [11.666666666666666, 11.666666666666666]
    close(values(variance(y, [1, 2])), [11.666666666666666, 11.666666666666666]);
    close(values(std(y, [1, 2])), [Math.sqrt(11.666666666666666), Math.sqrt(11.666666666666666)]);
    // sample variance over all 12 elements, NumPy: np.arange(12).var(ddof=1) == 13
    close(values(variance(y, [0, 1, 2], false, 1)), [13]);
  });

  it("median over several axes", () => {
    expect(median(strided3d(), [0, 1]).toArray()).toEqual([3.5, 5.5, 7.5]);
  });

  it("any/all over several axes with keepdims", () => {
    const t = tensor([
      [
        [0, 0],
        [0, 1],
      ],
      [
        [0, 0],
        [0, 0],
      ],
    ]);
    expect(any(t, [1, 2]).toArray()).toEqual([1, 0]);
    expect(any(t, [0, 2], true).shape).toEqual([1, 2, 1]);
    expect(all(t, [1, 2]).toArray()).toEqual([0, 0]);
    expect(any(t, "rows").dtype).toBe("bool");
  });

  it("rejects duplicate axes with a typed error", () => {
    expect(() => sum(f64([[1, 2]]), [1, -1])).toThrow(InvalidParameterError);
    expect(() => variance(f64([[1, 2]]), [0, 0])).toThrow(/duplicate axis/);
  });

  it("an empty axis list returns a copy, not the input", () => {
    const t = f64([1, 2, 3]);
    const out = max(t, []);
    expect(out).not.toBe(t);
    expect(out.toArray()).toEqual([1, 2, 3]);
    expect(sum(t, []).toArray()).toEqual([1, 2, 3]);
  });
});

describe("prod", () => {
  it("promotes uint8 and bool to int32 instead of wrapping", () => {
    const u = tensor([20, 20], { dtype: "uint8" });
    const p = prod(u);
    expect(p.dtype).toBe("int32");
    expect(p.toArray()).toBe(400);
    expect(prod(tensor([[200, 200]], { dtype: "uint8" }), 1).toArray()).toEqual([40000]);
    expect(prod(tensor([1, 1, 1], { dtype: "bool" })).toArray()).toBe(1);
  });

  it("throws on int32 overflow and survives a later zero", () => {
    expect(() => prod(tensor([65536, 65536], { dtype: "int32" }))).toThrow(/int32 prod overflow/);
    expect(() => prod(tensor([[65536, 65536]], { dtype: "int32" }), 1)).toThrow(
      /int32 prod overflow/
    );
    const many = new Array<number>(60).fill(2_000_000_000);
    many.push(0);
    expect(prod(tensor(many, { dtype: "int32" })).toArray()).toBe(0);
  });

  it("rounds half-precision products to the output dtype", () => {
    // NumPy: np.prod(np.array([0.1, 0.3, 7.0], dtype=np.float16)) == 0.21
    const p = prod(tensor([0.1, 0.3, 7], { dtype: "float16" }));
    expect(p.dtype).toBe("float16");
    expect(Math.abs((p.toArray() as number) - 0.21)).toBeLessThan(1e-3);
  });

  it("supports a list of axes", () => {
    const t = f64([
      [1, 2],
      [3, 4],
    ]);
    expect(prod(t, [0, 1]).toArray()).toBe(24);
    expect(prod(t, 0).toArray()).toEqual([3, 8]);
  });
});

describe("median", () => {
  it("handles runs of equal values in linear time", () => {
    const t = tensor(new Float64Array(1_000_000).fill(1), { dtype: "float64" });
    expect(median(t).toArray()).toBe(1);
    const ties = new Float64Array(200_001);
    for (let i = 0; i < ties.length; i++) ties[i] = i % 2;
    // 100001 zeros and 100000 ones: the median is 0
    expect(median(tensor(ties, { dtype: "float64" })).toArray()).toBe(0);
  });

  it("does not overflow when averaging two huge middle values", () => {
    expect(median(f64([1e308, 1e308])).toArray()).toBe(1e308);
    expect(median(f64([-1e308, 1e308])).toArray()).toBe(0);
  });

  it("propagates NaN and does not mutate the input", () => {
    const t = f64([3, Number.NaN, 1]);
    expect(median(t).toArray()).toBeNaN();
    const u = f64([5, 1, 4, 2, 3, 9]);
    median(u);
    expect(values(u)).toEqual([5, 1, 4, 2, 3, 9]);
    expect(median(u).toArray()).toBe(3.5);
  });
});

describe("variance validation", () => {
  it("rejects non-finite ddof", () => {
    expect(() => variance(f64([1, 2, 3]), undefined, false, Number.NaN)).toThrow(/ddof/);
    expect(() => std(f64([1, 2, 3]), undefined, false, Number.POSITIVE_INFINITY)).toThrow(/ddof/);
  });

  it("reports ddof against the number of reduced elements", () => {
    const t = f64([
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
    ]);
    expect(() => variance(t, [0, 2], false, 3)).toThrow(/ddof=3 >= axis size=3/);
    expect(() => variance(t, undefined, false, 6)).toThrow(/ddof=6 >= size=6/);
  });
});

describe("cumsum / cumprod", () => {
  it("flatten the input when no axis is given", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const c = cumsum(t);
    expect(c.shape).toEqual([6]);
    expect(c.toArray()).toEqual([1, 3, 6, 10, 15, 21]);
    const p = cumprod(t);
    expect(p.shape).toEqual([6]);
    expect(p.toArray()).toEqual([1, 2, 6, 24, 120, 720]);
  });

  it("scan along every axis of a strided view", () => {
    const y = strided3d();
    // NumPy: np.cumsum(y, axis=2) on the same view
    expect(cumsum(y, 2).toArray()).toEqual([
      [
        [0, 2, 6],
        [6, 14, 24],
      ],
      [
        [1, 4, 9],
        [7, 16, 27],
      ],
    ]);
    expect(cumsum(y, 0).toArray()).toEqual([
      [
        [0, 2, 4],
        [6, 8, 10],
      ],
      [
        [1, 5, 9],
        [13, 17, 21],
      ],
    ]);
    expect(cumsum(y, 0).shape).toEqual([2, 2, 3]);
    expect(cumsum(y, -1).shape).toEqual([2, 2, 3]);
  });

  it("throw on int64 overflow instead of wrapping", () => {
    const big = tensor(new BigInt64Array([(1n << 62n) + 1n, 1n << 62n]), { dtype: "int64" });
    expect(() => cumsum(big)).toThrow(/int64 cumsum overflow/);
    const p = tensor(new BigInt64Array([1n << 40n, 1n << 40n]), { dtype: "int64" });
    expect(() => cumprod(p, 0)).toThrow(/int64 cumprod overflow/);
    const ok = tensor(new BigInt64Array([3n, 4n]), { dtype: "int64" });
    expect(cumsum(ok).dtype).toBe("int64");
  });

  it("handle empty tensors", () => {
    expect(cumsum(f64([])).shape).toEqual([0]);
    const z = tensor(new Float64Array(0), { dtype: "float64" }).reshape([0, 3]);
    expect(cumsum(z, 1).shape).toEqual([0, 3]);
  });
});

describe("diff", () => {
  it("validates n", () => {
    expect(() => diff(f64([1, 2, 3]), 1.5)).toThrow(InvalidParameterError);
    expect(() => diff(f64([1, 2, 3]), Number.NaN)).toThrow(InvalidParameterError);
    expect(() => diff(f64([1, 2, 3]), -1)).toThrow(InvalidParameterError);
  });

  it("matches NumPy for higher orders along each axis", () => {
    const t = f64([
      [1, 4, 9, 16],
      [2, 3, 5, 8],
    ]);
    expect(diff(t, 2, 1).toArray()).toEqual([
      [2, 2],
      [1, 1],
    ]);
    const u = f64([
      [1, 4, 9],
      [2, 3, 5],
      [7, 7, 8],
      [0, 1, 1],
    ]);
    expect(diff(u, 2, 0).toArray()).toEqual([
      [4, 5, 7],
      [-12, -10, -10],
    ]);
  });

  it("returns an empty tensor once n reaches the axis length", () => {
    expect(diff(f64([1, 2, 3]), 3).shape).toEqual([0]);
    expect(diff(f64([1, 2, 3]), 10).shape).toEqual([0]);
  });

  it("throws on int64 overflow", () => {
    const t = tensor(new BigInt64Array([-(1n << 63n), (1n << 63n) - 1n]), { dtype: "int64" });
    expect(() => diff(t)).toThrow(/int64 diff overflow/);
  });
});

describe("min / max", () => {
  it("propagate NaN along an axis and keep dtype", () => {
    const t = f64([
      [1, Number.NaN],
      [0, 5],
    ]);
    expect(values(min(t, 0))).toEqual([0, Number.NaN]);
    expect(values(max(t, 1))).toEqual([Number.NaN, 5]);
    expect(min(tensor([3, 1], { dtype: "int32" }), 0).dtype).toBe("int32");
  });

  it("throw a typed error for empty reductions", () => {
    expect(() => max(f64([]))).toThrow(InvalidParameterError);
    const z = tensor(new Float64Array(0), { dtype: "float64" }).reshape([3, 0]);
    expect(() => min(z, 1)).toThrow(/at least one element/);
    expect(min(z, 0).shape).toEqual([0]);
  });
});

describe("string dtype", () => {
  it("is rejected with a DTypeError", () => {
    const s = tensor(["a", "b"]);
    expect(() => sum(s, [0])).toThrow(DTypeError);
    expect(() => median(s)).toThrow(/string/);
    expect(() => any(s)).toThrow(DTypeError);
  });
});

describe("window functions", () => {
  it("periodic windows match scipy.signal.windows(sym=False)", () => {
    close(
      values(hannWindow(5, { periodic: true })),
      [0, 0.34549150281252633, 0.9045084971874737, 0.9045084971874737, 0.34549150281252633]
    );
    close(
      values(hammingWindow(5, { periodic: true })),
      [
        0.08000000000000007, 0.3978521825875243, 0.9121478174124759, 0.9121478174124759,
        0.3978521825875243,
      ]
    );
    close(
      values(blackmanWindow(5, { periodic: true })),
      [
        -1.3877787807814457e-17, 0.2007701432625305, 0.8492298567374694, 0.8492298567374694,
        0.2007701432625305,
      ],
      1e-15
    );
    close(values(bartlettWindow(5, { periodic: true })), [0, 0.4, 0.8, 0.8, 0.4]);
    close(
      values(kaiserWindow(5, 6, { periodic: true })),
      [
        0.014873337104763207, 0.3390180566488852, 0.8954001841928596, 0.8954001841928596,
        0.3390180566488852,
      ]
    );
  });

  it("symmetric windows keep NumPy values", () => {
    close(
      values(hannWindow(6)),
      [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737, 0.3454915028125263, 0]
    );
    close(values(bartlettWindow(6)), [0, 0.4, 0.8, 0.8, 0.4, 0]);
  });

  it("kaiser is accurate for large beta (series needs more than 25 terms)", () => {
    // np.kaiser(6, 40)
    close(
      values(kaiserWindow(6, 40)),
      [
        6.713763812271753e-17, 0.00037536038368440324, 0.45027693174827954, 0.45027693174827954,
        0.00037536038368440324, 6.713763812271753e-17,
      ],
      1e-12
    );
    // np.kaiser(7, 12)
    close(
      values(kaiserWindow(7, 12)),
      [
        5.2773441320097665e-5, 0.054759364131249516, 0.5188447216723606, 1, 0.5188447216723606,
        0.054759364131249516, 5.2773441320097665e-5,
      ]
    );
  });

  it("kaiser stays finite where I0(beta) overflows a double", () => {
    // scipy.special.i0e(x) / i0e(beta) * exp(x - beta) with beta = 1000
    const w = values(kaiserWindow(5, 1000));
    expect(w.every(Number.isFinite)).toBe(true);
    expect(w[0]).toBe(0);
    expect(w[2]).toBe(1);
    expect(Math.abs((w[1] as number) / 7.027732781623501e-59 - 1)).toBeLessThan(1e-9);
  });

  it("kaiser accepts negative beta (I0 is even) and rejects non-finite beta", () => {
    close(values(kaiserWindow(4, -5)), values(kaiserWindow(4, 5)));
    expect(() => kaiserWindow(4, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => kaiserWindow(4, Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
  });

  it("n = 1 is [1] and invalid n throws", () => {
    expect(values(hannWindow(1, { periodic: true }))).toEqual([1]);
    expect(() => hannWindow(2.5)).toThrow(InvalidParameterError);
  });
});

describe("convolve / correlate", () => {
  it("correlate 'same' matches NumPy when the first input is shorter", () => {
    // np.correlate([1, 2], [0, 1, 0.5], mode) for each mode
    expect(values(correlate(f64([1, 2]), f64([0, 1, 0.5]), "same"))).toEqual([2, 2, 0]);
    expect(values(correlate(f64([1, 2]), f64([0, 1, 0.5]), "valid"))).toEqual([2, 2]);
    expect(values(correlate(f64([1, 2]), f64([0, 1, 0.5]), "full"))).toEqual([0.5, 2, 2, 0]);
  });

  it("convolve 'same' keeps the longer input's length", () => {
    expect(values(convolve(f64([1, 2, 3]), f64([0, 1, 0.5]), "same"))).toEqual([1, 2.5, 4]);
    expect(values(convolve(f64([0, 1, 0.5]), f64([1, 2, 3]), "same"))).toEqual([1, 2.5, 4]);
  });

  it("reads strided 1-D inputs correctly", () => {
    // logical values [1, 2, 3] stored with stride 2
    const strided = TensorClass.fromTypedArray({
      data: new Float64Array([1, 99, 2, 99, 3, 99]),
      shape: [3],
      strides: [2],
      dtype: "float64",
      device: "cpu",
    });
    expect(values(convolve(strided, f64([0, 1, 0.5])))).toEqual([0, 1, 2.5, 4, 1.5]);
    expect(values(correlate(strided, f64([0, 1, 0.5])))).toEqual([0.5, 2, 3.5, 3, 0]);
  });

  it("rejects empty inputs and unknown modes with typed errors", () => {
    expect(() => convolve(f64([]), f64([1]))).toThrow(InvalidParameterError);
    expect(() => correlate(f64([1]), f64([]))).toThrow(/cannot be empty/);
    expect(() => convolve(f64([1]), f64([1]), "circular" as never)).toThrow(/mode/);
  });

  it("names the offending argument for non-1-D input", () => {
    const m = f64([[1, 2]]);
    expect(() => convolve(f64([1, 2]), m)).toThrow(/v has ndim 2/);
    expect(() => convolve(m, f64([1, 2]))).toThrow(/a has ndim 2/);
  });
});

describe("gcd / lcm", () => {
  it("broadcast like NumPy", () => {
    const a = tensor([
      [4, 6],
      [12, 10],
    ]);
    const b = tensor([6, 8]);
    expect(lcm(a, b).toArray()).toEqual([
      [12, 24],
      [12, 40],
    ]);
    expect(gcd(a, tensor([[8], [16]])).shape).toEqual([2, 2]);
    expect(gcd(a, tensor(9)).toArray()).toEqual([
      [1, 3],
      [3, 1],
    ]);
  });

  it("reject shapes that do not broadcast, even when the sizes match", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(() => gcd(a, b)).toThrow(ShapeError);
    expect(() => lcm(a, b)).toThrow(ShapeError);
  });

  it("throw on NaN and infinite inputs instead of hanging", () => {
    expect(() => gcd(tensor([Number.NaN]), tensor([3]))).toThrow(DataValidationError);
    expect(() => lcm(tensor([4]), tensor([Number.POSITIVE_INFINITY]))).toThrow(DataValidationError);
  });

  it("throw when an int32 result does not fit instead of wrapping", () => {
    // np.lcm(100000, 99999) == 9999900000, outside int32
    expect(() => lcm(tensor([100000]), tensor([99999]))).toThrow(/int32/);
  });

  it("are exact for int64 inputs", () => {
    const a = tensor(new BigInt64Array([1n << 40n]), { dtype: "int64" });
    const b = tensor(new BigInt64Array([1n << 35n]), { dtype: "int64" });
    const g = gcd(a, b);
    expect(g.dtype).toBe("int64");
    expect(g.toArray()).toEqual([34359738368n]);
    const x = tensor(new BigInt64Array([100000n]), { dtype: "int64" });
    const y = tensor(new BigInt64Array([99999n]), { dtype: "int64" });
    expect(lcm(x, y).toArray()).toEqual([9999900000n]);
    const max64 = tensor(new BigInt64Array([(1n << 62n) + 1n]), { dtype: "int64" });
    const two = tensor(new BigInt64Array([2n]), { dtype: "int64" });
    expect(() => lcm(max64, two)).toThrow(/int64/);
  });

  it("truncate float inputs toward zero", () => {
    expect(gcd(tensor([12.9, -15.5]), tensor([8.2, 10])).toArray()).toEqual([4, 5]);
  });
});

describe("set operations", () => {
  const nan = Number.NaN;

  it("collapse NaN in union1d and sort it last", () => {
    // np.union1d([nan, 1, nan], [nan, 2]) -> [1, 2, nan]
    expect(union1d(f64([nan, 1, nan]), f64([nan, 2])).toArray()).toEqual([1, 2, nan]);
    expect(union1d(f64([3, nan, 1]), f64([2])).toArray()).toEqual([1, 2, 3, nan]);
  });

  it("never match NaN in intersect1d and setdiff1d", () => {
    // np.intersect1d([nan, 1], [nan, 1]) -> [1]; np.setdiff1d([nan, 1, 2], [nan]) -> [1, 2, nan]
    expect(intersect1d(f64([nan, 1]), f64([nan, 1])).toArray()).toEqual([1]);
    expect(setdiff1d(f64([nan, 1, 2]), f64([nan])).toArray()).toEqual([1, 2, nan]);
  });

  it("order unsorted input numerically", () => {
    expect(union1d(f64([10, 9, 100]), f64([-1, 2])).toArray()).toEqual([-1, 2, 9, 10, 100]);
  });

  it("keep int64 values above 2^53 distinct", () => {
    const a = tensor(new BigInt64Array([9007199254740993n, 9007199254740992n]), {
      dtype: "int64",
    });
    const u = union1d(a, a);
    expect(u.dtype).toBe("int64");
    expect(u.toArray()).toEqual([9007199254740992n, 9007199254740993n]);
    const i = intersect1d(a, tensor(new BigInt64Array([9007199254740993n]), { dtype: "int64" }));
    expect(i.toArray()).toEqual([9007199254740993n]);
    const d = setdiff1d(a, tensor(new BigInt64Array([9007199254740993n]), { dtype: "int64" }));
    expect(d.toArray()).toEqual([9007199254740992n]);
  });

  it("read strided 1-D inputs through their strides", () => {
    const strided = TensorClass.fromTypedArray({
      data: new Float64Array([3, 99, 1, 99, 3, 99]),
      shape: [3],
      strides: [2],
      dtype: "float64",
      device: "cpu",
    });
    expect(union1d(strided, f64([2])).toArray()).toEqual([1, 2, 3]);
  });

  it("report which argument is not 1-D", () => {
    expect(() => union1d(f64([1]), f64([[1]]))).toThrow(/b has shape \[1, 1\]/);
    expect(() => setdiff1d(tensor(["a"]), f64([1]))).toThrow(DTypeError);
  });
});
