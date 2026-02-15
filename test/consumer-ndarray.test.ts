import { describe, expect, it } from "vitest";
import {
  abs,
  acos,
  acosh,
  add,
  addScalar,
  all,
  allclose,
  any,
  arange,
  argsort,
  arrayEqual,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  CSRMatrix,
  cbrt,
  ceil,
  clip,
  concatenate,
  cos,
  cosh,
  cumprod,
  cumsum,
  diff,
  div,
  dot,
  dropoutMask,
  elu,
  empty,
  equal,
  exp,
  exp2,
  expandDims,
  expm1,
  eye,
  flatten,
  floor,
  floorDiv,
  full,
  gather,
  gelu,
  geomspace,
  greater,
  greaterEqual,
  im2col,
  isclose,
  isfinite,
  isinf,
  isnan,
  leakyRelu,
  less,
  lessEqual,
  linspace,
  log,
  log1p,
  log2,
  log10,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  logSoftmax,
  logspace,
  max,
  maximum,
  mean,
  median,
  min,
  minimum,
  mish,
  mod,
  mul,
  mulScalar,
  neg,
  notEqual,
  ones,
  pow,
  prod,
  randn,
  reciprocal,
  relu,
  repeat,
  reshape,
  round,
  rsqrt,
  sigmoid,
  sign,
  sin,
  sinh,
  slice,
  softmax,
  softplus,
  sort,
  split,
  sqrt,
  square,
  squeeze,
  stack,
  std,
  sub,
  sum,
  swish,
  tan,
  tanh,
  tensor,
  tile,
  transpose,
  trunc,
  unsqueeze,
  variance,
  zeros,
} from "../src/ndarray";

describe("consumer API: ndarray", () => {
  describe("tensor creation", () => {
    it("creates 2D tensor with correct shape/size/ndim/dtype", () => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(a.shape).toEqual([2, 3]);
      expect(a.size).toBe(6);
      expect(a.ndim).toBe(2);
      expect(a.dtype).toBe("float32");
    });

    it("zeros, ones, full, eye, arange, linspace, logspace, geomspace, empty, randn", () => {
      expect(zeros([3, 3]).at(0, 0)).toBe(0);
      expect(ones([2, 2]).at(1, 1)).toBe(1);
      expect(full([2, 2], 7).at(0, 0)).toBe(7);
      expect(eye(3).at(0, 0)).toBe(1);
      expect(eye(3).at(0, 1)).toBe(0);
      expect(arange(0, 10, 2).size).toBe(5);
      expect(linspace(0, 1, 5).size).toBe(5);
      expect(logspace(0, 2, 3).size).toBe(3);
      expect(geomspace(1, 100, 3).size).toBe(3);
      expect(randn([2, 3]).shape).toEqual([2, 3]);
      expect(empty([2, 3]).shape).toEqual([2, 3]);
    });

    it("respects dtype option", () => {
      expect(tensor([1, 2, 3], { dtype: "float64" }).dtype).toBe("float64");
      expect(tensor([1, 2, 3], { dtype: "int32" }).dtype).toBe("int32");
    });
  });

  describe("arithmetic ops", () => {
    it("basic element-wise arithmetic", () => {
      const x = tensor([1, 2, 3]);
      const y = tensor([4, 5, 6]);
      expect(add(x, y).at(0)).toBe(5);
      expect(sub(y, x).at(0)).toBe(3);
      expect(mul(x, y).at(0)).toBe(4);
      expect(div(y, x).at(0)).toBe(4);
      expect(pow(x, tensor(2)).at(0)).toBe(1);
      expect(sqrt(tensor([4, 9])).at(0)).toBe(2);
      expect(square(tensor([3])).at(0)).toBe(9);
      expect(neg(tensor([5])).at(0)).toBe(-5);
      expect(abs(tensor([-3])).at(0)).toBe(3);
      expect(mod(tensor([7]), tensor([3])).at(0)).toBe(1);
      expect(floorDiv(tensor([7]), tensor([2])).at(0)).toBe(3);
    });

    it("scalar ops", () => {
      expect(addScalar(tensor([1, 2]), 10).at(0)).toBe(11);
      expect(mulScalar(tensor([2, 3]), 5).at(0)).toBe(10);
    });

    it("unary math ops", () => {
      expect(Math.abs(Number(reciprocal(tensor([4])).at(0)) - 0.25)).toBeLessThan(0.001);
      expect(Math.abs(Number(rsqrt(tensor([4])).at(0)) - 0.5)).toBeLessThan(0.001);
      expect(Math.abs(Number(cbrt(tensor([27])).at(0)) - 3)).toBeLessThan(0.01);
      expect(ceil(tensor([1.3])).at(0)).toBe(2);
      expect(floor(tensor([1.7])).at(0)).toBe(1);
      expect(round(tensor([1.5])).at(0)).toBe(2);
      expect(sign(tensor([-5])).at(0)).toBe(-1);
      expect(trunc(tensor([1.9])).at(0)).toBe(1);
    });

    it("exp/log variants", () => {
      expect(Math.abs(Number(exp(tensor([0])).at(0)) - 1)).toBeLessThan(0.001);
      expect(Math.abs(Number(exp2(tensor([3])).at(0)) - 8)).toBeLessThan(0.01);
      expect(typeof expm1(tensor([1])).at(0)).toBe("number");
      expect(Math.abs(Number(log(tensor([1])).at(0)))).toBeLessThan(0.001);
      expect(typeof log1p(tensor([1])).at(0)).toBe("number");
      expect(Math.abs(Number(log2(tensor([8])).at(0)) - 3)).toBeLessThan(0.01);
      expect(Math.abs(Number(log10(tensor([1000])).at(0)) - 3)).toBeLessThan(0.01);
    });
  });

  describe("trig ops", () => {
    it("sin, cos, tan, and inverses", () => {
      expect(Math.abs(Number(sin(tensor([0])).at(0)))).toBeLessThan(0.001);
      expect(Math.abs(Number(cos(tensor([0])).at(0)) - 1)).toBeLessThan(0.001);
      expect(Math.abs(Number(tan(tensor([0])).at(0)))).toBeLessThan(0.001);
      expect(typeof asin(tensor([0.5])).at(0)).toBe("number");
      expect(typeof acos(tensor([0.5])).at(0)).toBe("number");
      expect(typeof atan(tensor([1])).at(0)).toBe("number");
      expect(typeof atan2(tensor([1]), tensor([1])).at(0)).toBe("number");
    });

    it("hyperbolic functions", () => {
      expect(typeof sinh(tensor([1])).at(0)).toBe("number");
      expect(typeof cosh(tensor([1])).at(0)).toBe("number");
      expect(typeof tanh(tensor([1])).at(0)).toBe("number");
      expect(typeof asinh(tensor([1])).at(0)).toBe("number");
      expect(typeof acosh(tensor([2])).at(0)).toBe("number");
      expect(typeof atanh(tensor([0.5])).at(0)).toBe("number");
    });
  });

  describe("reductions", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);

    it("global reductions", () => {
      expect(sum(m).at()).toBe(10);
      expect(mean(m).at()).toBe(2.5);
      expect(max(m).at()).toBe(4);
      expect(min(m).at()).toBe(1);
      expect(prod(tensor([2, 3, 4])).at()).toBe(24);
      expect(Number(std(tensor([2, 4, 4, 4, 5, 5, 7, 9])).at())).toBeGreaterThan(0);
      expect(Number(variance(tensor([2, 4, 4, 4, 5, 5, 7, 9])).at())).toBeGreaterThan(0);
      expect(median(tensor([1, 3, 2])).at()).toBe(2);
    });

    it("axis reductions", () => {
      const s0 = sum(m, 0);
      expect(s0.at(0)).toBe(4);
      expect(s0.at(1)).toBe(6);
      const s1 = sum(m, 1);
      expect(s1.at(0)).toBe(3);
      expect(s1.at(1)).toBe(7);
    });
  });

  describe("comparison ops", () => {
    it("element-wise comparisons", () => {
      expect(greater(tensor([3]), tensor([2])).at(0)).toBe(1);
      expect(less(tensor([1]), tensor([2])).at(0)).toBe(1);
      expect(equal(tensor([2]), tensor([2])).at(0)).toBe(1);
      expect(greaterEqual(tensor([2]), tensor([2])).at(0)).toBe(1);
      expect(lessEqual(tensor([2]), tensor([3])).at(0)).toBe(1);
      expect(notEqual(tensor([1]), tensor([2])).at(0)).toBe(1);
      expect(maximum(tensor([1, 5]), tensor([3, 2])).at(0)).toBe(3);
      expect(minimum(tensor([1, 5]), tensor([3, 2])).at(0)).toBe(1);
      expect(clip(tensor([-1, 5, 10]), 0, 8).at(0)).toBe(0);
      expect(clip(tensor([-1, 5, 10]), 0, 8).at(2)).toBe(8);
    });
  });

  describe("logical ops", () => {
    it("and, or, not, xor, all, any", () => {
      expect(logicalAnd(tensor([1, 0, 1]), tensor([1, 1, 0])).at(0)).toBe(1);
      expect(logicalOr(tensor([1, 0, 0]), tensor([0, 0, 1])).at(0)).toBe(1);
      expect(logicalNot(tensor([1, 0])).at(0)).toBe(0);
      expect(logicalXor(tensor([1, 0, 1]), tensor([0, 0, 1])).at(0)).toBe(1);
      expect(all(tensor([1, 1, 1])).at()).toBe(1);
      expect(all(tensor([1, 0, 1])).at()).toBe(0);
      expect(any(tensor([0, 0, 1])).at()).toBe(1);
      expect(any(tensor([0, 0, 0])).at()).toBe(0);
    });
  });

  describe("sort / scan", () => {
    it("sort, argsort, cumsum, cumprod, diff", () => {
      expect(sort(tensor([3, 1, 4, 1, 5])).at(0)).toBe(1);
      expect(argsort(tensor([3, 1, 4])).at(0)).toBe(1);
      expect(cumsum(tensor([1, 2, 3, 4])).at(3)).toBe(10);
      expect(cumprod(tensor([1, 2, 3, 4])).at(3)).toBe(24);
      const df = diff(tensor([1, 3, 6, 10]));
      expect(df.at(0)).toBe(2);
      expect(df.at(1)).toBe(3);
      expect(df.at(2)).toBe(4);
    });
  });

  describe("manipulation", () => {
    it("flatten, reshape, transpose, squeeze, unsqueeze, expandDims", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(flatten(m).size).toBe(4);
      expect(flatten(m).ndim).toBe(1);
      expect(reshape(tensor([1, 2, 3, 4, 5, 6]), [2, 3]).shape).toEqual([2, 3]);
      expect(transpose(m).at(0, 1)).toBe(3);
      expect(squeeze(tensor([[[1, 2, 3]]]), 0).ndim).toBe(2);
      expect(unsqueeze(tensor([1, 2, 3]), 0).shape[0]).toBe(1);
      expect(expandDims(tensor([1, 2, 3]), 1).shape[1]).toBe(1);
    });

    it("reshape supports -1 inference", () => {
      const t = tensor([1, 2, 3, 4, 5, 6]);
      const r = reshape(t, [-1]);
      expect(r.shape).toEqual([6]);
      const r2 = reshape(t, [2, -1]);
      expect(r2.shape).toEqual([2, 3]);
    });

    it("slice, gather", () => {
      const sliced = slice(tensor([10, 20, 30, 40, 50]), { start: 1, end: 4 });
      expect(sliced.size).toBe(3);
      expect(sliced.at(0)).toBe(20);
      const gath = gather(tensor([10, 20, 30, 40]), tensor([0, 2], { dtype: "int32" }), 0);
      expect(gath.at(0)).toBe(10);
      expect(gath.at(1)).toBe(30);
    });

    it("concatenate, stack, tile, repeat, split", () => {
      expect(concatenate([tensor([1, 2]), tensor([3, 4])]).size).toBe(4);
      const stk = stack([tensor([1, 2]), tensor([3, 4])]);
      expect(stk.shape).toEqual([2, 2]);
      expect(tile(tensor([1, 2, 3]), [2]).size).toBe(6);
      expect(repeat(tensor([1, 2]), 3, 0).size).toBe(6);
      const parts = split(tensor([1, 2, 3, 4, 5, 6]), 3);
      expect(parts.length).toBe(3);
      expect(parts[0].size).toBe(2);
    });
  });

  describe("comparison utilities", () => {
    it("allclose, isclose, arrayEqual, isnan, isinf, isfinite", () => {
      expect(allclose(tensor([1.0, 2.0]), tensor([1.0, 2.0]))).toBe(true);
      expect(allclose(tensor([1.0]), tensor([2.0]))).toBe(false);
      expect(isclose(tensor([1.0, 2.0]), tensor([1.0, 2.001])).at(0)).toBe(1);
      expect(arrayEqual(tensor([1, 2, 3]), tensor([1, 2, 3]))).toBe(true);
      expect(arrayEqual(tensor([1, 2, 3]), tensor([1, 2, 4]))).toBe(false);
      expect(isnan(tensor([NaN, 1, 2])).at(0)).toBe(1);
      expect(isinf(tensor([Infinity, 1])).at(0)).toBe(1);
      expect(isfinite(tensor([1, Infinity])).at(0)).toBe(1);
    });
  });

  describe("activations", () => {
    it("relu, sigmoid, softmax, gelu, leakyRelu, elu, mish, swish, softplus, logSoftmax", () => {
      expect(relu(tensor([-1, 0, 1])).at(0)).toBe(0);
      expect(relu(tensor([-1, 0, 1])).at(2)).toBe(1);
      expect(Math.abs(Number(sigmoid(tensor([0])).at(0)) - 0.5)).toBeLessThan(0.001);
      const sm = softmax(tensor([1, 2, 3]));
      expect(Math.abs(Number(sm.at(0)) + Number(sm.at(1)) + Number(sm.at(2)) - 1)).toBeLessThan(
        0.01
      );
      expect(typeof gelu(tensor([1])).at(0)).toBe("number");
      expect(Number(leakyRelu(tensor([-1])).at(0))).toBeLessThan(0);
      expect(Number(elu(tensor([-1])).at(0))).toBeLessThan(0);
      expect(typeof mish(tensor([1])).at(0)).toBe("number");
      expect(typeof swish(tensor([1])).at(0)).toBe("number");
      expect(Number(softplus(tensor([0])).at(0))).toBeGreaterThan(0);
      const lsm = logSoftmax(tensor([1, 2, 3]));
      expect(lsm.size).toBe(3);
      expect(Number(lsm.at(0))).toBeLessThan(0);
    });
  });

  describe("linalg dot", () => {
    it("1D dot product", () => {
      expect(dot(tensor([1, 2, 3]), tensor([4, 5, 6])).at()).toBe(32);
    });

    it("1D x 2D vector-matrix product", () => {
      const v = tensor([1, 2]);
      const M = tensor([
        [1, 0],
        [0, 1],
      ]);
      const result = dot(v, M);
      expect(result.at(0)).toBe(1);
      expect(result.at(1)).toBe(2);
    });
  });

  describe("CSRMatrix", () => {
    it("fromCOO, nnz, toDense, add, scale, transpose", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 3,
        cols: 3,
        rowIndices: new Int32Array([0, 1, 2]),
        colIndices: new Int32Array([0, 2, 1]),
        values: new Float64Array([1, 2, 3]),
      });
      expect(sparse.nnz).toBe(3);
      expect(sparse.toDense().at(0, 0)).toBe(1);
      expect(sparse.toDense().at(1, 2)).toBe(2);

      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([1, 2]),
      });
      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([3, 4]),
      });
      expect(a.add(b).toDense().at(0, 0)).toBe(4);
      expect(a.scale(2).toDense().at(1, 1)).toBe(4);
      const t = a.transpose();
      expect(t.rows).toBe(2);
      expect(t.cols).toBe(2);
    });
  });

  describe("conv ops & dropout", () => {
    it("im2col produces correct shape", () => {
      const input = tensor([
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
      ]);
      const patches = im2col(input, [2, 2], [1, 1], [0, 0]);
      expect(patches.shape.length).toBeGreaterThanOrEqual(2);
    });

    it("dropoutMask correct shape", () => {
      const mask = dropoutMask([2, 3], 0.5, 2, "float32", "cpu");
      expect(mask.shape).toEqual([2, 3]);
    });
  });

  describe("tensor methods", () => {
    it("toString, toArray, at", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(m.toString().length).toBeGreaterThan(0);
      const arr = m.toArray() as number[][];
      expect(arr.length).toBe(2);
      expect(m.at(1, 0)).toBe(3);
    });
  });
});
