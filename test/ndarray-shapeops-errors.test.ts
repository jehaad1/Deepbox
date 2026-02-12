import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { reshape, transpose } from "../src/ndarray/tensor/shape";
import { squeeze, unsqueeze } from "../src/ndarray/tensor/shapeOps";

describe("ndarray shape ops error branches", () => {
  it("throws on reshape size mismatch", () => {
    const t = tensor([1, 2, 3, 4]);
    expect(() => reshape(t, [3, 2])).toThrow(/Cannot reshape/);
  });

  it("throws on reshape of non-contiguous tensors", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const tT = transpose(t);
    expect(() => reshape(tT, [6])).toThrow(/contiguous/i);
  });

  it("throws on transpose invalid axes", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => transpose(t, [0])).toThrow(/axes must have length/);
    expect(() => transpose(t, [0, 2])).toThrow(/out of range/);
    expect(() => transpose(t, [0, 0])).toThrow(/duplicate axis/);
  });

  it("supports negative axes in transpose", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const out = transpose(t, [-1, -2]);
    const expected = transpose(t, [1, 0]);
    expect(out.toArray()).toEqual(expected.toArray());
    expect(() => transpose(t, [-3, 0])).toThrow(/out of range/);
  });

  it("throws on squeeze/unsqueeze invalid axes", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => squeeze(t, 2)).toThrow(/out of bounds/);
    expect(() => squeeze(t, 0)).toThrow(/dimension.*must be 1/);
    expect(() => unsqueeze(t, 5)).toThrow(/out of bounds/);
  });

  it("squeezes to scalar when all dims are 1", () => {
    const t = tensor([[[1]]]);
    const out = squeeze(t);
    expect(out.shape).toEqual([]);
  });
});
