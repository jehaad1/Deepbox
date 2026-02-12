import { describe, expect, it } from "vitest";
import { DataLoader } from "../src/datasets";
import { tensor } from "../src/ndarray";
import { numRawData } from "./_helpers";

describe("DataLoader - Comprehensive Tests", () => {
  describe("Constructor and Basic Functionality", () => {
    it("should create DataLoader with X only", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X);
      expect(loader).toBeDefined();
      expect(loader.length).toBe(3);
    });

    it("should create DataLoader with X and y", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([0, 1]);
      const loader = new DataLoader(X, y);
      expect(loader).toBeDefined();
      expect(loader.length).toBe(2);
    });

    it("should create DataLoader with custom batch size", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 2 });
      expect(loader.length).toBe(2);
    });

    it("should throw error for invalid batchSize (zero)", () => {
      const X = tensor([[1, 2]]);
      expect(() => new DataLoader(X, undefined, { batchSize: 0 })).toThrow();
    });

    it("should throw error for invalid batchSize (negative)", () => {
      const X = tensor([[1, 2]]);
      expect(() => new DataLoader(X, undefined, { batchSize: -1 })).toThrow();
    });

    it("should throw error for invalid batchSize (non-integer)", () => {
      const X = tensor([[1, 2]]);
      expect(() => new DataLoader(X, undefined, { batchSize: 1.5 })).toThrow();
    });

    it("should throw error for invalid seed values", () => {
      const X = tensor([[1, 2]]);
      expect(() => new DataLoader(X, undefined, { seed: Number.NaN })).toThrow();
      expect(() => new DataLoader(X, undefined, { seed: Number.POSITIVE_INFINITY })).toThrow();
      expect(() => new DataLoader(X, undefined, { seed: 1.5 })).toThrow();
    });

    it("should throw error for scalar X", () => {
      const X = tensor(5);
      expect(() => new DataLoader(X)).toThrow();
    });

    it("should throw for scalar y", () => {
      const X = tensor([[1, 2]]);
      const y = tensor(1);
      expect(() => new DataLoader(X, y)).toThrow();
    });

    it("should throw for empty dataset", () => {
      // Create a tensor with 0 samples by using an empty 2D array
      const X = tensor([]);
      expect(() => new DataLoader(X)).toThrow();
    });

    it("should throw error when X and y have different sample counts", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([1]);
      expect(() => new DataLoader(X, y)).toThrow();
    });
  });

  describe("Batch Iteration", () => {
    it("should iterate over all batches", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const y = tensor([0, 1, 0, 1]);
      const loader = new DataLoader(X, y, { batchSize: 2 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
    });

    it("should yield correct batch shapes", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const y = tensor([0, 1, 0]);
      const loader = new DataLoader(X, y, { batchSize: 2 });

      const batches = Array.from(loader);
      expect(batches[0]?.[0].shape).toEqual([2, 2]);
      expect(batches[0]?.[1]?.shape).toEqual([2]);
      expect(batches[1]?.[0].shape).toEqual([1, 2]);
      expect(batches[1]?.[1]?.shape).toEqual([1]);
    });

    it("should yield X only when y is not provided", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches[0]?.length).toBe(1);
      expect(batches[1]?.length).toBe(1);
    });

    it("should yield X and y when y is provided", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = tensor([0, 1]);
      const loader = new DataLoader(X, y, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches[0]?.length).toBe(2);
      expect(batches[1]?.length).toBe(2);
    });

    it("should handle single batch", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 10 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(2);
    });

    it("should handle batch size equal to dataset size", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 3 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(3);
    });

    it("should handle batch size larger than dataset size", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 10 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(2);
    });
  });

  describe("dropLast Option", () => {
    it("should drop last incomplete batch when dropLast is true", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: true,
      });

      expect(loader.length).toBe(1);
      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(2);
    });

    it("should keep last incomplete batch when dropLast is false", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: false,
      });

      expect(loader.length).toBe(2);
      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
      expect(batches[1]?.[0].shape[0]).toBe(1);
    });

    it("should not drop anything when all batches are complete", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: true,
      });

      expect(loader.length).toBe(2);
      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
    });

    it("should drop all batches if batchSize > nSamples with dropLast", () => {
      const X = tensor([[1, 2]]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: true,
      });

      expect(loader.length).toBe(0);
      const batches = Array.from(loader);
      expect(batches).toHaveLength(0);
    });
  });

  describe("Shuffle Option", () => {
    it("should shuffle data when shuffle is true", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader1 = new DataLoader(X, undefined, {
        batchSize: 1,
        shuffle: false,
      });
      const loader2 = new DataLoader(X, undefined, {
        batchSize: 1,
        shuffle: true,
        seed: 42,
      });

      const batches1 = Array.from(loader1);
      const batches2 = Array.from(loader2);

      const data1 = batches1.map((b) => numRawData(b[0].data));
      const data2 = batches2.map((b) => numRawData(b[0].data));

      // With shuffle, order should be different
      expect(data1).not.toEqual(data2);
    });

    it("should be deterministic with same seed", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader1 = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: true,
        seed: 123,
      });
      const loader2 = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: true,
        seed: 123,
      });

      const batches1 = Array.from(loader1);
      const batches2 = Array.from(loader2);

      const data1 = numRawData(batches1[0]?.[0].data ?? []);
      const data2 = numRawData(batches2[0]?.[0].data ?? []);

      expect(data1).toEqual(data2);
    });

    it("should produce different shuffles with different seeds", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader1 = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: true,
        seed: 123,
      });
      const loader2 = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: true,
        seed: 456,
      });

      const batches1 = Array.from(loader1);
      const batches2 = Array.from(loader2);

      const data1 = numRawData(batches1[0]?.[0].data ?? []);
      const data2 = numRawData(batches2[0]?.[0].data ?? []);

      expect(data1).not.toEqual(data2);
    });

    it("should shuffle both X and y consistently", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const y = tensor([0, 1, 2, 3]);
      const loader = new DataLoader(X, y, {
        batchSize: 1,
        shuffle: true,
        seed: 42,
      });

      const batches = Array.from(loader);

      // Verify that X and y are shuffled together
      for (const batch of batches) {
        const xData = numRawData(batch[0].data);
        const yData = numRawData(batch[1]?.data ?? []);

        // Check that the relationship is maintained
        // e.g., if X is [1, 2], y should be 0
        if (xData[0] === 1 && xData[1] === 2) {
          expect(yData[0]).toBe(0);
        } else if (xData[0] === 3 && xData[1] === 4) {
          expect(yData[0]).toBe(1);
        }
      }
    });

    it("should handle shuffle with single sample", () => {
      const X = tensor([[1, 2]]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 1,
        shuffle: true,
        seed: 42,
      });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
    });

    it("should handle shuffle with empty-like case", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 1,
        shuffle: true,
        seed: 42,
      });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
    });
  });

  describe("Length Property", () => {
    it("should return correct length with even division", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 2 });
      expect(loader.length).toBe(2);
    });

    it("should return correct length with uneven division", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 2 });
      expect(loader.length).toBe(2);
    });

    it("should return correct length with dropLast true", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: true,
      });
      expect(loader.length).toBe(1);
    });

    it("should return correct length with dropLast false", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: false,
      });
      expect(loader.length).toBe(2);
    });

    it("should return 1 when batchSize >= nSamples", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 10 });
      expect(loader.length).toBe(1);
    });

    it("should return 0 when dropLast and incomplete batch", () => {
      const X = tensor([[1, 2]]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        dropLast: true,
      });
      expect(loader.length).toBe(0);
    });
  });

  describe("Multiple Iterations", () => {
    it("should allow multiple iterations over same loader", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 1 });

      const batches1 = Array.from(loader);
      const batches2 = Array.from(loader);

      expect(batches1.length).toBe(batches2.length);
    });

    it("should produce same results on multiple iterations without shuffle", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: false,
      });

      const batches1 = Array.from(loader);
      const batches2 = Array.from(loader);

      const data1 = numRawData(batches1[0]?.[0].data ?? []);
      const data2 = numRawData(batches2[0]?.[0].data ?? []);

      expect(data1).toEqual(data2);
    });

    it("should produce same results on multiple iterations with shuffle and seed", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: true,
        seed: 42,
      });

      const batches1 = Array.from(loader);
      const batches2 = Array.from(loader);

      const data1 = numRawData(batches1[0]?.[0].data ?? []);
      const data2 = numRawData(batches2[0]?.[0].data ?? []);

      expect(data1).toEqual(data2);
    });
  });

  describe("Edge Cases", () => {
    it("should handle 1D feature vectors", () => {
      const X = tensor([1, 2, 3, 4]);
      const loader = new DataLoader(X, undefined, { batchSize: 2 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
      expect(batches[0]?.[0].shape[0]).toBe(2);
    });

    it("should handle 3D tensors", () => {
      const X = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
      expect(batches[0]?.[0].ndim).toBe(3);
    });

    it("should handle large batch sizes", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
      ]);
      const loader = new DataLoader(X, undefined, { batchSize: 1000 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(2);
    });

    it("should handle single sample dataset", () => {
      const X = tensor([[1, 2]]);
      const y = tensor([0]);
      const loader = new DataLoader(X, y, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(1);
      expect(batches[0]?.[0].shape[0]).toBe(1);
      expect(batches[0]?.[1]?.shape[0]).toBe(1);
    });

    it("should handle very large datasets efficiently", () => {
      const size = 10000;
      const data = Array.from({ length: size }, (_, i) => [i, i + 1]);
      const X = tensor(data);
      const loader = new DataLoader(X, undefined, { batchSize: 32 });

      expect(loader.length).toBe(Math.ceil(size / 32));

      let count = 0;
      for (const batch of loader) {
        count++;
        expect(batch[0].shape[0]).toBeGreaterThan(0);
        expect(batch[0].shape[0]).toBeLessThanOrEqual(32);
      }
      expect(count).toBe(loader.length);
    });

    it("should handle different dtypes for X and y", () => {
      const X = tensor([
        [1.5, 2.5],
        [3.5, 4.5],
      ]);
      const y = tensor([0, 1], { dtype: "int32" });
      const loader = new DataLoader(X, y, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);
    });

    it("should preserve data types through batching", () => {
      const X = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float32" }
      );
      const y = tensor([0, 1], { dtype: "int32" });
      const loader = new DataLoader(X, y, { batchSize: 1 });

      const batches = Array.from(loader);
      expect(batches[0]?.[0].dtype).toBe("float32");
      expect(batches[0]?.[1]?.dtype).toBe("int32");
    });
  });

  describe("Integration Tests", () => {
    it("should work with typical training loop pattern", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const y = tensor([0, 1, 0, 1]);
      const loader = new DataLoader(X, y, {
        batchSize: 2,
        shuffle: true,
        seed: 42,
      });

      let totalSamples = 0;
      for (const [xBatch, yBatch] of loader) {
        expect(xBatch.shape[0]).toBe(yBatch.shape[0]);
        totalSamples += xBatch.shape[0];
      }
      expect(totalSamples).toBe(4);
    });

    it("should work with validation loop pattern (no shuffle)", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const y = tensor([0, 1, 0]);
      const loader = new DataLoader(X, y, {
        batchSize: 2,
        shuffle: false,
      });

      const batches = Array.from(loader);
      expect(batches).toHaveLength(2);

      // First batch should be samples 0 and 1
      const firstBatchX = numRawData(batches[0]?.[0].data ?? []);
      expect(firstBatchX).toEqual([1, 2, 3, 4]);
    });

    it("should work with inference pattern (X only, no shuffle)", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        shuffle: false,
      });

      let predictions = 0;
      for (const [xBatch] of loader) {
        predictions += xBatch.shape[0];
      }
      expect(predictions).toBe(3);
    });
  });
});
