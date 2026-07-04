import { describe, expect, it } from "vitest";
import { delete_, insert, tensor } from "../src/ndarray";

describe("insert", () => {
  it("inserts a scalar at a single index in a 1D tensor", () => {
    const a = tensor([1, 2, 3, 4]);
    const result = insert(a, 2, 99);
    expect(result.toArray()).toEqual([1, 2, 99, 3, 4]);
    expect(result.shape).toEqual([5]);
  });

  it("inserts at the beginning (index 0)", () => {
    const a = tensor([1, 2, 3]);
    const result = insert(a, 0, 99);
    expect(result.toArray()).toEqual([99, 1, 2, 3]);
  });

  it("inserts at the end (index = size)", () => {
    const a = tensor([1, 2, 3]);
    const result = insert(a, 3, 99);
    expect(result.toArray()).toEqual([1, 2, 3, 99]);
  });

  it("inserts at multiple indices with scalar value", () => {
    const a = tensor([1, 2, 3, 4]);
    const result = insert(a, [1, 3], 99);
    expect(result.toArray()).toEqual([1, 99, 2, 3, 99, 4]);
    expect(result.shape).toEqual([6]);
  });

  it("inserts a tensor value along axis 0 of a 2D tensor", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const vals = tensor([5, 6]);
    const result = insert(a, 1, vals, 0);
    expect(result.shape).toEqual([3, 2]);
    expect(result.toArray()).toEqual([
      [1, 2],
      [5, 6],
      [3, 4],
    ]);
  });

  it("inserts a scalar along axis 1 of a 2D tensor", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = insert(a, 1, 99, 1);
    expect(result.shape).toEqual([2, 3]);
    expect(result.toArray()).toEqual([
      [1, 99, 2],
      [3, 99, 4],
    ]);
  });

  it("flattens when no axis is provided", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = insert(a, 2, 99);
    expect(result.shape).toEqual([5]);
    expect(result.toArray()).toEqual([1, 2, 99, 3, 4]);
  });

  it("preserves dtype", () => {
    const a = tensor([1, 2, 3], { dtype: "float32" });
    const result = insert(a, 1, 99);
    expect(result.dtype).toBe("float32");
  });

  // Error cases
  it("throws on out-of-bounds index", () => {
    const a = tensor([1, 2, 3]);
    expect(() => insert(a, 4, 99, 0)).toThrow();
    // Negative indices are dim-relative (NumPy semantics): -1 inserts before
    // the last element; only indices below -axisSize are out of bounds.
    expect(insert(a, -1, 99, 0).toArray()).toEqual([1, 2, 99, 3]);
    expect(() => insert(a, -4, 99, 0)).toThrow();
  });

  it("throws on non-integer index", () => {
    const a = tensor([1, 2, 3]);
    expect(() => insert(a, 1.5, 99, 0)).toThrow();
  });
});

describe("delete_", () => {
  it("deletes a single element from a 1D tensor", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const result = delete_(a, 2);
    expect(result.toArray()).toEqual([1, 2, 4, 5]);
    expect(result.shape).toEqual([4]);
  });

  it("deletes the first element", () => {
    const a = tensor([1, 2, 3]);
    const result = delete_(a, 0);
    expect(result.toArray()).toEqual([2, 3]);
  });

  it("deletes the last element", () => {
    const a = tensor([1, 2, 3]);
    const result = delete_(a, 2);
    expect(result.toArray()).toEqual([1, 2]);
  });

  it("deletes multiple elements", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const result = delete_(a, [0, 3]);
    expect(result.toArray()).toEqual([2, 3, 5]);
    expect(result.shape).toEqual([3]);
  });

  it("deletes a row along axis 0 of a 2D tensor", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const result = delete_(a, 1, 0);
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });

  it("deletes a column along axis 1 of a 2D tensor", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = delete_(a, 1, 1);
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [1, 3],
      [4, 6],
    ]);
  });

  it("deletes multiple rows along axis 0", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const result = delete_(a, [0, 2], 0);
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [3, 4],
      [7, 8],
    ]);
  });

  it("flattens when no axis is provided", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = delete_(a, 1);
    expect(result.shape).toEqual([3]);
    expect(result.toArray()).toEqual([1, 3, 4]);
  });

  it("supports negative indices", () => {
    const a = tensor([1, 2, 3, 4]);
    const result = delete_(a, -1, 0);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it("preserves dtype", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    const result = delete_(a, 0);
    expect(result.dtype).toBe("float64");
  });

  // Error cases
  it("throws on out-of-bounds index", () => {
    const a = tensor([1, 2, 3]);
    expect(() => delete_(a, 3, 0)).toThrow();
    expect(() => delete_(a, -4, 0)).toThrow();
  });

  it("throws on non-integer index", () => {
    const a = tensor([1, 2, 3]);
    expect(() => delete_(a, 1.5, 0)).toThrow();
  });
});
