import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  elu,
  gelu,
  leakyRelu,
  logSoftmax,
  mish,
  relu,
  sigmoid,
  softmax,
  swish,
} from "../src/ndarray/ops/activation";
import {
  add,
  addScalar,
  clip,
  div,
  floorDiv,
  maximum,
  minimum,
  mod,
  mul,
  mulScalar,
  neg,
  pow,
  reciprocal,
  sign,
  sub,
} from "../src/ndarray/ops/arithmetic";
import {
  allclose,
  arrayEqual,
  equal,
  greater,
  greaterEqual,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
  lessEqual,
  notEqual,
} from "../src/ndarray/ops/comparison";
import { logicalAnd, logicalNot, logicalOr, logicalXor } from "../src/ndarray/ops/logical";
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
  sum,
  variance,
} from "../src/ndarray/ops/reduction";

describe("deepbox/ndarray - Extra Ops Coverage", () => {
  it("handles arithmetic with broadcasting", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([10, 20]);
    const c = add(a, b);
    expect(c.shape).toEqual([2, 2]);
    const d = sub(a, tensor([[1], [1]]));
    expect(d.shape).toEqual([2, 2]);
    const e = mul(a, tensor(2));
    expect(e.shape).toEqual([2, 2]);
    const f = div(a, tensor(2));
    expect(f.shape).toEqual([2, 2]);
    const g = pow(tensor([2, 3]), tensor([3, 2]));
    expect(g.toArray()).toEqual([8, 9]);
  });

  it("handles int64 arithmetic and comparison branches", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([2, 1], { dtype: "int64" });
    expect(add(a, b).toArray()).toEqual([3n, 3n]);
    const divOut = div(a, b);
    expect(divOut.dtype).toBe("float64");
    expect(divOut.toArray()).toEqual([0.5, 2]);
    expect(equal(a, b).toArray()).toEqual([0, 0]);
    expect(greater(a, b).toArray()).toEqual([0, 1]);
  });

  it("throws on dtype mismatch or non-broadcastable shapes", () => {
    const a = tensor([1], { dtype: "float32" });
    const b = tensor([1], { dtype: "int32" });
    expect(() => add(a, b)).toThrow();
    const c = tensor([
      [1, 2],
      [3, 4],
    ]);
    const d = tensor([1, 2, 3]);
    expect(() => add(c, d)).toThrow();
  });

  it("covers scalar math helpers", () => {
    const t = tensor([1, -2, 3], { dtype: "int32" });
    expect(addScalar(t, 1).toArray()).toEqual([2, -1, 4]);
    expect(mulScalar(t, 2).toArray()).toEqual([2, -4, 6]);
    expect(floorDiv(tensor([5, 7]), tensor([2, 3])).toArray()).toEqual([2, 2]);
    expect(mod(tensor([5, 7]), tensor([2, 3])).toArray()).toEqual([1, 1]);
    expect(neg(t).toArray()).toEqual([-1, 2, -3]);
    expect(sign(t).toArray()).toEqual([1, -1, 1]);
    expect(reciprocal(tensor([2, 4])).toArray()).toEqual([0.5, 0.25]);
    expect(maximum(tensor([1, 5]), tensor([3, 2])).toArray()).toEqual([3, 5]);
    expect(minimum(tensor([1, 5]), tensor([3, 2])).toArray()).toEqual([1, 2]);
    expect(clip(tensor([1, 5, 10]), 2, 8).toArray()).toEqual([2, 5, 8]);
  });

  it("compares tensors and scalars", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1, 0, 3]);
    expect(equal(a, b).toArray()).toEqual([1, 0, 1]);
    expect(notEqual(a, b).toArray()).toEqual([0, 1, 0]);
    expect(greater(a, tensor(2)).toArray()).toEqual([0, 0, 1]);
    expect(greaterEqual(a, tensor(2)).toArray()).toEqual([0, 1, 1]);
    expect(less(a, tensor(2)).toArray()).toEqual([1, 0, 0]);
    expect(lessEqual(a, tensor(2)).toArray()).toEqual([1, 1, 0]);
  });

  it("validates comparison inputs", () => {
    const a = tensor([1, 2]);
    const b = tensor([1, 2, 3]);
    expect(() => equal(a, b)).toThrow();
    const s = tensor(["a", "b"]);
    expect(() => greater(s, s)).toThrow();
  });

  it("computes closeness and finite checks", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1.000001, 2, 3.1]);
    expect(isclose(a, b).toArray()).toEqual([1, 1, 0]);
    expect(allclose(a, b)).toBe(false);
    expect(arrayEqual(a, tensor([1, 2, 3]))).toBe(true);
    const nanInf = tensor([NaN, Infinity, -Infinity, 1]);
    expect(isnan(nanInf).toArray()).toEqual([1, 0, 0, 0]);
    expect(isinf(nanInf).toArray()).toEqual([0, 1, 1, 0]);
    expect(isfinite(nanInf).toArray()).toEqual([0, 0, 0, 1]);
  });

  it("reduces tensors with axis and keepdims", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(sum(x).toArray()).toEqual(10);
    expect(sum(x, 0, true).shape).toEqual([1, 2]);
    expect(mean(x, 1).toArray()).toEqual([1.5, 3.5]);
    expect(min(x).toArray()).toEqual(1);
    expect(max(x).toArray()).toEqual(4);
    expect(median(tensor([1, 3, 2, 4])).toArray()).toEqual(2.5);
    expect(prod(tensor([1, 2, 3])).toArray()).toEqual(6);
    expect(variance(tensor([1, 2, 3])).shape).toEqual([]);
  });

  it("min/max with axis reduce", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(min(x, 0).toArray()).toEqual([1, 2]);
    expect(max(x, 1).toArray()).toEqual([2, 4]);
  });

  it("computes cumulative and diff ops", () => {
    const x = tensor([1, 2, 3]);
    expect(cumsum(x).toArray()).toEqual([1, 3, 6]);
    expect(cumprod(x).toArray()).toEqual([1, 2, 6]);
    expect(diff(tensor([1, 3, 6])).toArray()).toEqual([2, 3]);
  });

  it("handles reduction edge cases and errors", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(sum(x, -1).toArray()).toEqual([3, 7]);
    expect(median(x, 0).toArray()).toEqual([2, 3]);
    expect(prod(x, 0).toArray()).toEqual([3, 8]);
    expect(cumsum(x, 0).toArray()).toEqual([
      [1, 2],
      [4, 6],
    ]);
    expect(cumprod(x, 1).toArray()).toEqual([
      [1, 2],
      [3, 12],
    ]);
    expect(diff(tensor([1, 2, 3]), 2).toArray()).toEqual([0]);
    expect(diff(tensor([1])).toArray()).toEqual([]);
    const mask = tensor([
      [0, 1],
      [0, 0],
    ]);
    expect(any(mask, 0).toArray()).toEqual([0, 1]);
    expect(all(mask, 1).toArray()).toEqual([0, 0]);
  });

  it("handles reduction with string and int64 branches", () => {
    const t = tensor([1, 2], { dtype: "int64" });
    expect(cumsum(t).dtype).toBe("int64");
    expect(cumprod(t).dtype).toBe("int64");
    const s = tensor(["a", "b"]);
    expect(() => any(s)).toThrow();
    expect(() => cumsum(s)).toThrow();
  });

  it("computes any/all", () => {
    const x = tensor([0, 1, 0], { dtype: "int32" });
    expect(any(x).toArray()).toEqual(1);
    expect(all(x).toArray()).toEqual(0);
    const m = tensor([
      [0, 1],
      [2, 0],
    ]);
    expect(any(m, 0).toArray()).toEqual([1, 1]);
    expect(all(m, 1).toArray()).toEqual([0, 0]);
    expect(any(m, 0, true).shape).toEqual([1, 2]);
  });

  it("computes logical ops", () => {
    const a = tensor([1, 0, 1], { dtype: "int32" });
    const b = tensor([1, 1, 0], { dtype: "int32" });
    expect(logicalAnd(a, b).toArray()).toEqual([1, 0, 0]);
    expect(logicalOr(a, b).toArray()).toEqual([1, 1, 1]);
    expect(logicalXor(a, b).toArray()).toEqual([0, 1, 1]);
    expect(logicalNot(a).toArray()).toEqual([0, 1, 0]);
  });

  it("computes activation ops", () => {
    const x = tensor([-1, 0, 1]);
    expect(relu(x).toArray()).toEqual([0, 0, 1]);
    expect(sigmoid(x).shape).toEqual([3]);
    expect(softmax(tensor([1, 2, 3])).shape).toEqual([3]);
    expect(logSoftmax(tensor([1, 2, 3])).shape).toEqual([3]);
    expect(gelu(x).shape).toEqual([3]);
    expect(mish(x).shape).toEqual([3]);
    expect(swish(x).shape).toEqual([3]);
    expect(elu(x).shape).toEqual([3]);
    expect(leakyRelu(x).shape).toEqual([3]);
  });
});
