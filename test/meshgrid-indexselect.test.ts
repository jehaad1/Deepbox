import { describe, expect, it } from "vitest";
import { index_select, meshgrid, tensor } from "../src/ndarray";

describe("meshgrid", () => {
  it("creates 2D coordinate grids with xy indexing (default)", () => {
    const x = tensor([1, 2, 3]);
    const y = tensor([4, 5]);
    const [X, Y] = meshgrid(x, y);

    // xy indexing: output shape is [len(y), len(x)] = [2, 3]
    expect(X.shape).toEqual([2, 3]);
    expect(Y.shape).toEqual([2, 3]);

    // X should repeat x along rows
    expect(Number(X.data[0])).toBe(1);
    expect(Number(X.data[1])).toBe(2);
    expect(Number(X.data[2])).toBe(3);
    expect(Number(X.data[3])).toBe(1);
    expect(Number(X.data[4])).toBe(2);
    expect(Number(X.data[5])).toBe(3);

    // Y should repeat y along columns
    expect(Number(Y.data[0])).toBe(4);
    expect(Number(Y.data[1])).toBe(4);
    expect(Number(Y.data[2])).toBe(4);
    expect(Number(Y.data[3])).toBe(5);
    expect(Number(Y.data[4])).toBe(5);
    expect(Number(Y.data[5])).toBe(5);
  });

  it("creates 2D coordinate grids with ij indexing", () => {
    const x = tensor([1, 2, 3]);
    const y = tensor([4, 5]);
    const [X, Y] = meshgrid(x, y, { indexing: "ij" });

    // ij indexing: output shape is [len(x), len(y)] = [3, 2]
    expect(X.shape).toEqual([3, 2]);
    expect(Y.shape).toEqual([3, 2]);
  });

  it("returns empty array for no inputs", () => {
    const result = meshgrid();
    expect(result).toEqual([]);
  });

  it("works with single input", () => {
    const x = tensor([1, 2, 3]);
    const [X] = meshgrid(x);
    expect(X.shape).toEqual([3]);
  });

  it("throws on non-1D inputs", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => meshgrid(x)).toThrow();
  });
});

describe("index_select", () => {
  it("selects columns from a 2D tensor", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const idx = tensor([0, 2], { dtype: "int32" });
    const result = index_select(x, 1, idx);

    expect(result.shape).toEqual([2, 2]);
    expect(Number(result.data[0])).toBe(1);
    expect(Number(result.data[1])).toBe(3);
    expect(Number(result.data[2])).toBe(4);
    expect(Number(result.data[3])).toBe(6);
  });

  it("selects rows from a 2D tensor", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const idx = tensor([0, 2], { dtype: "int32" });
    const result = index_select(x, 0, idx);

    expect(result.shape).toEqual([2, 3]);
    expect(Number(result.data[0])).toBe(1);
    expect(Number(result.data[1])).toBe(2);
    expect(Number(result.data[2])).toBe(3);
    expect(Number(result.data[3])).toBe(7);
    expect(Number(result.data[4])).toBe(8);
    expect(Number(result.data[5])).toBe(9);
  });

  it("works with 1D tensor", () => {
    const x = tensor([10, 20, 30, 40]);
    const idx = tensor([1, 3], { dtype: "int32" });
    const result = index_select(x, 0, idx);

    expect(result.shape).toEqual([2]);
    expect(Number(result.data[0])).toBe(20);
    expect(Number(result.data[1])).toBe(40);
  });

  it("supports negative dim", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const idx = tensor([1], { dtype: "int32" });
    const result = index_select(x, -1, idx);

    expect(result.shape).toEqual([2, 1]);
    expect(Number(result.data[0])).toBe(2);
    expect(Number(result.data[1])).toBe(5);
  });

  it("throws on non-1D index", () => {
    const x = tensor([1, 2, 3]);
    const idx = tensor([[0, 1]]);
    expect(() => index_select(x, 0, idx)).toThrow();
  });

  it("throws on out-of-range dim", () => {
    const x = tensor([1, 2, 3]);
    const idx = tensor([0]);
    expect(() => index_select(x, 5, idx)).toThrow();
  });
});
