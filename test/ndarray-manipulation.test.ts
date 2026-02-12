import { describe, expect, it } from "vitest";
import { concatenate, repeat, split, stack, tensor, tile } from "../src/ndarray";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Manipulation Operations", () => {
  describe("concatenate", () => {
    it("should concatenate 1D tensors along axis 0", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = concatenate([a, b], 0);
      expect(c.shape).toEqual([6]);
      expect(numData(c)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should concatenate 2D tensors along axis 0", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([[5, 6]]);
      const c = concatenate([a, b], 0);
      expect(c.shape).toEqual([3, 2]);
      expect(numData(c)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should concatenate 2D tensors along axis 1", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([[5], [6]]);
      const c = concatenate([a, b], 1);
      expect(c.shape).toEqual([2, 3]);
      expect(numData(c)).toEqual([1, 2, 5, 3, 4, 6]);
    });

    it("should concatenate multiple tensors", () => {
      const a = tensor([1, 2]);
      const b = tensor([3, 4]);
      const c = tensor([5, 6]);
      const d = concatenate([a, b, c], 0);
      expect(d.shape).toEqual([6]);
      expect(numData(d)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should handle negative axis", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([[5, 6]]);
      const c = concatenate([a, b], -2);
      expect(c.shape).toEqual([3, 2]);
    });

    it("should concatenate 3D tensors", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([[[5, 6]], [[7, 8]]]);
      const c = concatenate([a, b], 0);
      expect(c.shape).toEqual([4, 1, 2]);
    });

    it("should return copy for single tensor", () => {
      const a = tensor([1, 2, 3]);
      const b = concatenate([a], 0);
      expect(b.shape).toEqual([3]);
      expect(numData(b)).toEqual([1, 2, 3]);
    });

    it("should throw on empty array", () => {
      expect(() => concatenate([], 0)).toThrow("at least one tensor");
    });

    it("should throw on shape mismatch", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([[5, 6, 7]]);
      expect(() => concatenate([a, b], 0)).toThrow();
    });

    it("should throw on ndim mismatch", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([[4, 5, 6]]);
      expect(() => concatenate([a, b], 0)).toThrow("same ndim");
    });

    it("should throw on dtype mismatch", () => {
      const a = tensor([1, 2, 3], { dtype: "float32" });
      const b = tensor([4, 5, 6], { dtype: "float64" });
      expect(() => concatenate([a, b], 0)).toThrow("same dtype");
    });

    it("should throw on invalid axis", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      expect(() => concatenate([a, b], 5)).toThrow("out of bounds");
    });

    it("should concatenate int32 tensors", () => {
      const a = tensor([1, 2, 3], { dtype: "int32" });
      const b = tensor([4, 5, 6], { dtype: "int32" });
      const c = concatenate([a, b], 0);
      expect(c.dtype).toBe("int32");
      expect(numData(c)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should concatenate with different sizes along concat axis", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
        [9, 10],
      ]);
      const c = concatenate([a, b], 0);
      expect(c.shape).toEqual([5, 2]);
    });

    it("should throw on 0D tensor concatenation (no axes)", () => {
      const a = tensor(5);
      const b = tensor(10);
      expect(() => concatenate([a, b], 0)).toThrow("out of bounds");
    });
  });

  describe("stack", () => {
    it("should stack 1D tensors along axis 0", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = stack([a, b], 0);
      expect(c.shape).toEqual([2, 3]);
      expect(numData(c)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should stack 1D tensors along axis 1", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = stack([a, b], 1);
      expect(c.shape).toEqual([3, 2]);
      expect(numData(c)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it("should stack 2D tensors along axis 0", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const c = stack([a, b], 0);
      expect(c.shape).toEqual([2, 2, 2]);
    });

    it("should stack 2D tensors along axis 2", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const c = stack([a, b], 2);
      expect(c.shape).toEqual([2, 2, 2]);
    });

    it("should handle negative axis", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = stack([a, b], -1);
      expect(c.shape).toEqual([3, 2]);
    });

    it("should stack multiple tensors", () => {
      const a = tensor([1, 2]);
      const b = tensor([3, 4]);
      const c = tensor([5, 6]);
      const d = stack([a, b, c], 0);
      expect(d.shape).toEqual([3, 2]);
      expect(numData(d)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should throw on empty array", () => {
      expect(() => stack([], 0)).toThrow("at least one tensor");
    });

    it("should throw on shape mismatch", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5]);
      expect(() => stack([a, b], 0)).toThrow("same shape");
    });

    it("should throw on ndim mismatch", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([[4, 5, 6]]);
      expect(() => stack([a, b], 0)).toThrow("same ndim");
    });

    it("should throw on dtype mismatch", () => {
      const a = tensor([1, 2, 3], { dtype: "float32" });
      const b = tensor([4, 5, 6], { dtype: "float64" });
      expect(() => stack([a, b], 0)).toThrow("same dtype");
    });

    it("should throw on invalid axis", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      expect(() => stack([a, b], 5)).toThrow("out of bounds");
    });

    it("should stack int32 tensors", () => {
      const a = tensor([1, 2, 3], { dtype: "int32" });
      const b = tensor([4, 5, 6], { dtype: "int32" });
      const c = stack([a, b], 0);
      expect(c.dtype).toBe("int32");
    });

    it("should stack 0D tensors", () => {
      const a = tensor(5);
      const b = tensor(10);
      const c = stack([a, b], 0);
      expect(c.shape).toEqual([2]);
      expect(numData(c)).toEqual([5, 10]);
    });

    it("should stack 3D tensors", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([[[5, 6]], [[7, 8]]]);
      const c = stack([a, b], 0);
      expect(c.shape).toEqual([2, 2, 1, 2]);
    });
  });

  describe("split", () => {
    it("should split 1D tensor into equal parts", () => {
      const t = tensor([1, 2, 3, 4, 5, 6]);
      const parts = split(t, 3, 0);
      expect(parts.length).toBe(3);
      expect(parts[0]?.shape).toEqual([2]);
      expect(numData(parts[0] ?? tensor([]))).toEqual([1, 2]);
      expect(numData(parts[1] ?? tensor([]))).toEqual([3, 4]);
      expect(numData(parts[2] ?? tensor([]))).toEqual([5, 6]);
    });

    it("should split 1D tensor at specified indices", () => {
      const t = tensor([1, 2, 3, 4, 5, 6]);
      const parts = split(t, [2, 4], 0);
      expect(parts.length).toBe(3);
      expect(numData(parts[0] ?? tensor([]))).toEqual([1, 2]);
      expect(numData(parts[1] ?? tensor([]))).toEqual([3, 4]);
      expect(numData(parts[2] ?? tensor([]))).toEqual([5, 6]);
    });

    it("should split 2D tensor along axis 0", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const parts = split(t, 2, 0);
      expect(parts.length).toBe(2);
      expect(parts[0]?.shape).toEqual([2, 2]);
      expect(parts[1]?.shape).toEqual([2, 2]);
    });

    it("should split 2D tensor along axis 1", () => {
      const t = tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const parts = split(t, 2, 1);
      expect(parts.length).toBe(2);
      expect(parts[0]?.shape).toEqual([2, 2]);
      expect(parts[1]?.shape).toEqual([2, 2]);
    });

    it("should handle negative axis", () => {
      const t = tensor([1, 2, 3, 4]);
      const parts = split(t, 2, -1);
      expect(parts.length).toBe(2);
    });

    it("should throw on non-divisible split", () => {
      const t = tensor([1, 2, 3, 4, 5]);
      expect(() => split(t, 2, 0)).toThrow("not divisible");
    });

    it("should throw on invalid axis", () => {
      const t = tensor([1, 2, 3, 4]);
      expect(() => split(t, 2, 5)).toThrow("out of bounds");
    });

    it("should split with single index", () => {
      const t = tensor([1, 2, 3, 4]);
      const parts = split(t, [2], 0);
      expect(parts.length).toBe(2);
      expect(numData(parts[0] ?? tensor([]))).toEqual([1, 2]);
      expect(numData(parts[1] ?? tensor([]))).toEqual([3, 4]);
    });

    it("should split 3D tensor", () => {
      const t = tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]);
      const parts = split(t, 2, 0);
      expect(parts.length).toBe(2);
      expect(parts[0]?.shape).toEqual([2, 1, 2]);
    });

    it("should preserve dtype", () => {
      const t = tensor([1, 2, 3, 4], { dtype: "int32" });
      const parts = split(t, 2, 0);
      expect(parts[0]?.dtype).toBe("int32");
    });
  });

  describe("tile", () => {
    it("should tile 1D tensor", () => {
      const t = tensor([1, 2, 3]);
      const tiled = tile(t, [2]);
      expect(tiled.shape).toEqual([6]);
      expect(numData(tiled)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it("should tile 2D tensor along both axes", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      const tiled = tile(t, [2, 3]);
      expect(tiled.shape).toEqual([4, 6]);
      const expected = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4];
      expect(numData(tiled)).toEqual(expected);
    });

    it("should tile with reps longer than ndim", () => {
      const t = tensor([1, 2]);
      const tiled = tile(t, [2, 3]);
      expect(tiled.shape).toEqual([2, 6]);
    });

    it("should tile with reps shorter than ndim", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      const tiled = tile(t, [2]);
      expect(tiled.shape).toEqual([2, 4]);
    });

    it("should tile with single repetition", () => {
      const t = tensor([1, 2, 3]);
      const tiled = tile(t, [1]);
      expect(tiled.shape).toEqual([3]);
      expect(numData(tiled)).toEqual([1, 2, 3]);
    });

    it("should tile 0D tensor", () => {
      const t = tensor(5);
      const tiled = tile(t, [3]);
      expect(tiled.shape).toEqual([3]);
      expect(numData(tiled)).toEqual([5, 5, 5]);
    });

    it("should throw on empty reps", () => {
      const t = tensor([1, 2, 3]);
      expect(() => tile(t, [])).toThrow("at least one element");
    });

    it("should preserve dtype", () => {
      const t = tensor([1, 2], { dtype: "int32" });
      const tiled = tile(t, [2]);
      expect(tiled.dtype).toBe("int32");
    });

    it("should tile 3D tensor", () => {
      const t = tensor([[[1, 2]]]);
      const tiled = tile(t, [2, 2, 2]);
      expect(tiled.shape).toEqual([2, 2, 4]);
    });
  });

  describe("repeat", () => {
    it("should repeat 1D tensor elements (flattened)", () => {
      const t = tensor([1, 2, 3]);
      const repeated = repeat(t, 2);
      expect(repeated.shape).toEqual([6]);
      expect(numData(repeated)).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it("should repeat along axis 0", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      const repeated = repeat(t, 2, 0);
      expect(repeated.shape).toEqual([4, 2]);
      expect(numData(repeated)).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
    });

    it("should repeat along axis 1", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      const repeated = repeat(t, 2, 1);
      expect(repeated.shape).toEqual([2, 4]);
      expect(numData(repeated)).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });

    it("should handle negative axis", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      const repeated = repeat(t, 2, -1);
      expect(repeated.shape).toEqual([2, 4]);
    });

    it("should repeat 0D tensor", () => {
      const t = tensor(5);
      const repeated = repeat(t, 3);
      expect(repeated.shape).toEqual([3]);
      expect(numData(repeated)).toEqual([5, 5, 5]);
    });

    it("should repeat with count 1", () => {
      const t = tensor([1, 2, 3]);
      const repeated = repeat(t, 1);
      expect(numData(repeated)).toEqual([1, 2, 3]);
    });

    it("should throw on invalid axis", () => {
      const t = tensor([1, 2, 3]);
      expect(() => repeat(t, 2, 5)).toThrow("out of bounds");
    });

    it("should preserve dtype", () => {
      const t = tensor([1, 2, 3], { dtype: "int32" });
      const repeated = repeat(t, 2);
      expect(repeated.dtype).toBe("int32");
    });

    it("should repeat 3D tensor along axis", () => {
      const t = tensor([[[1, 2]], [[3, 4]]]);
      const repeated = repeat(t, 2, 0);
      expect(repeated.shape).toEqual([4, 1, 2]);
    });

    it("should repeat large count", () => {
      const t = tensor([1, 2]);
      const repeated = repeat(t, 5);
      expect(repeated.shape).toEqual([10]);
    });
  });

  describe("edge cases and integration", () => {
    it("should concatenate then split", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = concatenate([a, b], 0);
      const parts = split(c, 2, 0);
      expect(numData(parts[0] ?? tensor([]))).toEqual([1, 2, 3]);
      expect(numData(parts[1] ?? tensor([]))).toEqual([4, 5, 6]);
    });

    it("should stack then split", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = stack([a, b], 0);
      const parts = split(c, 2, 0);
      expect(parts[0]?.shape).toEqual([1, 3]);
      expect(parts[1]?.shape).toEqual([1, 3]);
    });

    it("should tile then split", () => {
      const t = tensor([1, 2]);
      const tiled = tile(t, [3]);
      const parts = split(tiled, 3, 0);
      expect(parts.length).toBe(3);
    });

    it("should repeat then reshape", () => {
      const t = tensor([1, 2]);
      const repeated = repeat(t, 3);
      expect(repeated.shape).toEqual([6]);
    });

    it("should handle large tensors", () => {
      const size = 1000;
      const data = Array.from({ length: size }, (_, i) => i);
      const t = tensor(data);
      const parts = split(t, 10, 0);
      expect(parts.length).toBe(10);
      expect(parts[0]?.shape).toEqual([100]);
    });
  });
});
