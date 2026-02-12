import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { col2im, im2col } from "../src/ndarray/ops/conv";
import { dropoutMask } from "../src/ndarray/ops/random";
import {
  broadcastOffsetFromFlatIndex,
  isContiguous,
  offsetFromFlatIndex,
} from "../src/ndarray/tensor/strides";
import { computeStrides, Tensor } from "../src/ndarray/tensor/Tensor";

// ─── strides.ts ────────────────────────────────────────────────────────────────

describe("strides utilities", () => {
  describe("isContiguous", () => {
    it("returns true for standard row-major layout", () => {
      expect(isContiguous([2, 3], [3, 1])).toBe(true);
      expect(isContiguous([4], [1])).toBe(true);
      expect(isContiguous([], [])).toBe(true);
    });

    it("returns false for non-standard strides", () => {
      expect(isContiguous([2, 3], [1, 2])).toBe(false);
      expect(isContiguous([2, 3], [6, 1])).toBe(false);
    });

    it("returns false when stride length mismatches shape length", () => {
      expect(isContiguous([2, 3], [3])).toBe(false);
    });
  });

  describe("offsetFromFlatIndex", () => {
    it("computes correct offset for contiguous tensors", () => {
      const logicalStrides = computeStrides([2, 3]);
      // flat=0 -> [0,0] -> offset=5 + 0*3 + 0*1 = 5
      expect(offsetFromFlatIndex(0, logicalStrides, [3, 1], 5)).toBe(5);
      // flat=4 -> [1,1] -> offset=5 + 1*3 + 1*1 = 9
      expect(offsetFromFlatIndex(4, logicalStrides, [3, 1], 5)).toBe(9);
    });

    it("computes correct offset for transposed strides", () => {
      // A 2x3 tensor stored in column-major order (strides [1, 2])
      const logicalStrides = computeStrides([2, 3]);
      // flat=0 -> [0,0] -> 0*1 + 0*2 = 0
      expect(offsetFromFlatIndex(0, logicalStrides, [1, 2], 0)).toBe(0);
      // flat=1 -> [0,1] -> 0*1 + 1*2 = 2
      expect(offsetFromFlatIndex(1, logicalStrides, [1, 2], 0)).toBe(2);
      // flat=3 -> [1,0] -> 1*1 + 0*2 = 1
      expect(offsetFromFlatIndex(3, logicalStrides, [1, 2], 0)).toBe(1);
    });

    it("handles scalar (0D) case", () => {
      expect(offsetFromFlatIndex(0, [], [], 7)).toBe(7);
    });
  });

  describe("broadcastOffsetFromFlatIndex", () => {
    it("returns inOffset for scalar input (empty inShape)", () => {
      expect(broadcastOffsetFromFlatIndex(3, [2, 3], [3, 1], [], [], 42)).toBe(42);
    });

    it("broadcasts a 1-element dimension correctly", () => {
      // out [2, 3], in [1, 3] — first dim is broadcast
      const outStrides = computeStrides([2, 3]); // [3, 1]
      const inStrides = computeStrides([1, 3]); // [3, 1]
      // flat=0 -> out [0,0] -> in dim0=1 skip, dim1 coord=0 -> 0
      expect(broadcastOffsetFromFlatIndex(0, [2, 3], outStrides, [1, 3], inStrides, 0)).toBe(0);
      // flat=3 -> out [1,0] -> in dim0=1 skip, dim1 coord=0 -> 0
      expect(broadcastOffsetFromFlatIndex(3, [2, 3], outStrides, [1, 3], inStrides, 0)).toBe(0);
      // flat=4 -> out [1,1] -> in dim1 coord=1 -> 1
      expect(broadcastOffsetFromFlatIndex(4, [2, 3], outStrides, [1, 3], inStrides, 0)).toBe(1);
    });

    it("handles rank difference (lower-rank input)", () => {
      // out [2, 3], in [3] — input has rank 1
      const outStrides = computeStrides([2, 3]); // [3, 1]
      const inStrides = computeStrides([3]); // [1]
      // flat=0 -> out [0,0] -> rankDiff=1, axis=1 maps to in axis 0 -> coord 0
      expect(broadcastOffsetFromFlatIndex(0, [2, 3], outStrides, [3], inStrides, 0)).toBe(0);
      // flat=4 -> out [1,1] -> axis=1 coord=1 -> 1
      expect(broadcastOffsetFromFlatIndex(4, [2, 3], outStrides, [3], inStrides, 0)).toBe(1);
    });

    it("applies inOffset", () => {
      const outStrides = computeStrides([2]);
      const inStrides = computeStrides([2]);
      expect(broadcastOffsetFromFlatIndex(1, [2], outStrides, [2], inStrides, 10)).toBe(11);
    });
  });
});

// ─── conv.ts ───────────────────────────────────────────────────────────────────

describe("conv operations", () => {
  describe("im2col", () => {
    it("unfolds a simple 1x1x3x3 image with 2x2 kernel", () => {
      // 1 batch, 1 channel, 3x3 image, 2x2 kernel, stride 1, padding 0
      const data = tensor(
        [
          [
            [
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
            ],
          ],
        ],
        { dtype: "float64" }
      );
      const result = im2col(data, [2, 2], [1, 1], [0, 0]);
      // outH = (3-2)/1 + 1 = 2, outW = (3-2)/1 + 1 = 2
      // shape: [1, 4, 4] (batch=1, outH*outW=4, C*kH*kW=4)
      expect(result.shape).toEqual([1, 4, 4]);
      // First window [0,0]: elements [1,2,4,5]
      const arr = result.toArray() as number[][][];
      expect(arr[0][0]).toEqual([1, 2, 4, 5]);
      // Second window [0,1]: elements [2,3,5,6]
      expect(arr[0][1]).toEqual([2, 3, 5, 6]);
      // Third window [1,0]: elements [4,5,7,8]
      expect(arr[0][2]).toEqual([4, 5, 7, 8]);
      // Fourth window [1,1]: elements [5,6,8,9]
      expect(arr[0][3]).toEqual([5, 6, 8, 9]);
    });

    it("handles padding", () => {
      const data = tensor(
        [
          [
            [
              [1, 2],
              [3, 4],
            ],
          ],
        ],
        { dtype: "float64" }
      );
      const result = im2col(data, [2, 2], [1, 1], [1, 1]);
      // outH = (2+2-2)/1+1 = 3, outW = 3
      expect(result.shape).toEqual([1, 9, 4]);
      // Top-left window includes zero-padding
      const arr = result.toArray() as number[][][];
      expect(arr[0][0]).toEqual([0, 0, 0, 1]);
    });

    it("handles stride > 1", () => {
      const data = tensor(
        [
          [
            [
              [1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16],
            ],
          ],
        ],
        { dtype: "float64" }
      );
      const result = im2col(data, [2, 2], [2, 2], [0, 0]);
      // outH = (4-2)/2+1 = 2, outW = 2
      expect(result.shape).toEqual([1, 4, 4]);
    });

    it("handles multiple channels and batches", () => {
      // 2 batch, 2 channel, 2x2 image
      const data = Tensor.fromTypedArray({
        data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        shape: [2, 2, 2, 2],
        dtype: "float64",
        device: "cpu",
      });
      const result = im2col(data, [2, 2], [1, 1], [0, 0]);
      expect(result.shape).toEqual([2, 1, 8]);
    });

    it("handles BigInt64Array", () => {
      const data = Tensor.fromTypedArray({
        data: new BigInt64Array([1n, 2n, 3n, 4n]),
        shape: [1, 1, 2, 2],
        dtype: "int64",
        device: "cpu",
      });
      const result = im2col(data, [2, 2], [1, 1], [0, 0]);
      expect(result.shape).toEqual([1, 1, 4]);
      expect(result.dtype).toBe("int64");
    });

    it("throws for non-4D input", () => {
      const t = tensor([1, 2, 3]);
      expect(() => im2col(t, [1, 1], [1, 1], [0, 0])).toThrow("4D");
    });

    it("throws for string dtype", () => {
      const t = Tensor.fromStringArray({
        data: ["a"],
        shape: [1, 1, 1, 1],
        device: "cpu",
      });
      expect(() => im2col(t, [1, 1], [1, 1], [0, 0])).toThrow("string");
    });

    it("throws for invalid kernelSize", () => {
      const t = tensor([[[[1]]]], { dtype: "float64" });
      expect(() => im2col(t, [0, 1], [1, 1], [0, 0])).toThrow("kernelSize");
    });

    it("throws for invalid stride", () => {
      const t = tensor([[[[1]]]], { dtype: "float64" });
      expect(() => im2col(t, [1, 1], [0, 1], [0, 0])).toThrow("stride");
    });

    it("throws for invalid padding", () => {
      const t = tensor([[[[1]]]], { dtype: "float64" });
      expect(() => im2col(t, [1, 1], [1, 1], [-1, 0])).toThrow("padding");
    });

    it("throws when output dimensions are invalid", () => {
      // kernel bigger than image with no padding
      const t = tensor([[[[1]]]], { dtype: "float64" });
      expect(() => im2col(t, [3, 3], [1, 1], [0, 0])).toThrow("dimensions");
    });
  });

  describe("col2im", () => {
    it("is the inverse of im2col for non-overlapping windows", () => {
      // 1x1x2x2, kernel 2x2, stride 2x2 — one window, no overlap
      const input = tensor(
        [
          [
            [
              [1, 2],
              [3, 4],
            ],
          ],
        ],
        { dtype: "float64" }
      );
      const cols = im2col(input, [2, 2], [2, 2], [0, 0]);
      const reconstructed = col2im(cols, [1, 1, 2, 2], [2, 2], [2, 2], [0, 0]);
      expect(reconstructed.shape).toEqual([1, 1, 2, 2]);
      expect(reconstructed.toArray()).toEqual([
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
      ]);
    });

    it("accumulates overlapping windows", () => {
      // With stride 1 and kernel 2x2 on 2x2, windows overlap
      const input = tensor(
        [
          [
            [
              [1, 2],
              [3, 4],
            ],
          ],
        ],
        { dtype: "float64" }
      );
      const cols = im2col(input, [1, 1], [1, 1], [0, 0]);
      const reconstructed = col2im(cols, [1, 1, 2, 2], [1, 1], [1, 1], [0, 0]);
      expect(reconstructed.toArray()).toEqual([
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
      ]);
    });

    it("throws for non-3D cols input", () => {
      const t = tensor([1, 2]);
      expect(() => col2im(t, [1, 1, 2, 2], [1, 1], [1, 1], [0, 0])).toThrow("3D");
    });

    it("throws for non-4 length inputShape", () => {
      const t = tensor([[[1]]]);
      expect(() => col2im(t, [1, 1, 2], [1, 1], [1, 1], [0, 0])).toThrow("length 4");
    });

    it("throws for shape mismatch", () => {
      const cols = tensor([[[1, 2, 3, 4]]], { dtype: "float64" });
      expect(() => col2im(cols, [1, 1, 2, 2], [2, 2], [2, 2], [0, 0])).not.toThrow();
      // Wrong batch count
      expect(() => col2im(cols, [2, 1, 2, 2], [2, 2], [2, 2], [0, 0])).toThrow("mismatch");
    });

    it("handles BigInt64Array", () => {
      const cols = Tensor.fromTypedArray({
        data: new BigInt64Array([1n, 2n, 3n, 4n]),
        shape: [1, 1, 4],
        dtype: "int64",
        device: "cpu",
      });
      const result = col2im(cols, [1, 1, 2, 2], [2, 2], [2, 2], [0, 0]);
      expect(result.dtype).toBe("int64");
      expect(result.shape).toEqual([1, 1, 2, 2]);
    });
  });
});

// ─── random.ts ─────────────────────────────────────────────────────────────────

describe("dropoutMask", () => {
  it("produces correct shape and dtype", () => {
    const mask = dropoutMask([2, 3], 0.5, 2.0, "float64", "cpu");
    expect(mask.shape).toEqual([2, 3]);
    expect(mask.dtype).toBe("float64");
  });

  it("all elements are either 0 or scale", () => {
    const scale = 2.0;
    const mask = dropoutMask([100], 0.3, scale, "float64", "cpu");
    const arr = mask.toArray() as number[];
    for (const v of arr) {
      expect(v === 0 || v === scale).toBe(true);
    }
  });

  it("p=0 keeps all elements", () => {
    const mask = dropoutMask([100], 0, 1.0, "float64", "cpu");
    const arr = mask.toArray() as number[];
    for (const v of arr) {
      expect(v).toBe(1.0);
    }
  });

  it("handles BigInt64Array dtype", () => {
    const mask = dropoutMask([50], 0.5, 2, "int64", "cpu");
    expect(mask.dtype).toBe("int64");
    const data = mask.data as BigInt64Array;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      expect(v === 0n || v === 2n).toBe(true);
    }
  });

  it("throws for invalid p", () => {
    expect(() => dropoutMask([2], -0.1, 1, "float64", "cpu")).toThrow();
    expect(() => dropoutMask([2], 1.0, 1, "float64", "cpu")).toThrow();
    expect(() => dropoutMask([2], NaN, 1, "float64", "cpu")).toThrow();
    expect(() => dropoutMask([2], Infinity, 1, "float64", "cpu")).toThrow();
  });

  it("respects float32 dtype", () => {
    const mask = dropoutMask([10], 0.3, 1.5, "float32", "cpu");
    expect(mask.dtype).toBe("float32");
  });
});
