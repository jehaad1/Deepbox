import { describe, expect, it } from "vitest";
import { CSRMatrix, tensor } from "../src/ndarray";

describe("deepbox/ndarray - CSRMatrix", () => {
  describe("construction", () => {
    it("should create a CSRMatrix from valid CSR data", () => {
      const sparse = new CSRMatrix({
        data: new Float64Array([1, 2, 3]),
        indices: new Int32Array([0, 2, 1]),
        indptr: new Int32Array([0, 1, 2, 3]),
        shape: [3, 3],
      });

      expect(sparse.rows).toBe(3);
      expect(sparse.cols).toBe(3);
      expect(sparse.nnz).toBe(3);
      expect(sparse.shape).toEqual([3, 3]);
    });

    it("should create an empty CSRMatrix", () => {
      const sparse = new CSRMatrix({
        data: new Float64Array(0),
        indices: new Int32Array(0),
        indptr: new Int32Array([0, 0, 0]),
        shape: [2, 3],
      });

      expect(sparse.nnz).toBe(0);
      expect(sparse.rows).toBe(2);
      expect(sparse.cols).toBe(3);
    });

    it("should throw for non-2D shape", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1]),
            indices: new Int32Array([0]),
            indptr: new Int32Array([0, 1]),
            shape: [1],
          })
      ).toThrow("2D");
    });

    it("should throw for negative shape dimensions", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1]),
            indices: new Int32Array([0]),
            indptr: new Int32Array([0, 1]),
            shape: [-1, 3],
          })
      ).toThrow("non-negative");
    });

    it("should throw for incorrect indptr length", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1]),
            indices: new Int32Array([0]),
            indptr: new Int32Array([0, 1]), // Should be length 4 for 3 rows
            shape: [3, 3],
          })
      ).toThrow("indptr length");
    });

    it("should throw for data/indices length mismatch", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1, 2]),
            indices: new Int32Array([0]),
            indptr: new Int32Array([0, 1, 2]),
            shape: [2, 3],
          })
      ).toThrow("mismatch");
    });

    it("should throw if indptr[0] is not 0", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1]),
            indices: new Int32Array([0]),
            indptr: new Int32Array([1, 2]),
            shape: [1, 3],
          })
      ).toThrow("indptr[0] must be 0");
    });

    it("should throw for decreasing indptr values", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1, 2]),
            indices: new Int32Array([0, 1]),
            indptr: new Int32Array([0, 2, 1, 2]),
            shape: [3, 3],
          })
      ).toThrow("non-decreasing");
    });

    it("should throw for out-of-bounds column indices", () => {
      expect(
        () =>
          new CSRMatrix({
            data: new Float64Array([1]),
            indices: new Int32Array([5]), // col 5 is out of bounds for 3 cols
            indptr: new Int32Array([0, 1]),
            shape: [1, 3],
          })
      ).toThrow("out of bounds");
    });
  });

  describe("fromCOO", () => {
    it("should create CSRMatrix from COO format", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 3,
        cols: 3,
        rowIndices: new Int32Array([0, 1, 2]),
        colIndices: new Int32Array([0, 2, 1]),
        values: new Float64Array([1, 2, 3]),
      });

      expect(sparse.rows).toBe(3);
      expect(sparse.cols).toBe(3);
      expect(sparse.nnz).toBe(3);
      expect(sparse.get(0, 0)).toBe(1);
      expect(sparse.get(1, 2)).toBe(2);
      expect(sparse.get(2, 1)).toBe(3);
    });

    it("should handle unsorted COO input", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([1, 0, 1]),
        colIndices: new Int32Array([0, 1, 1]),
        values: new Float64Array([3, 1, 4]),
      });

      expect(sparse.get(0, 1)).toBe(1);
      expect(sparse.get(1, 0)).toBe(3);
      expect(sparse.get(1, 1)).toBe(4);
    });

    it("should throw for mismatched array lengths", () => {
      expect(() =>
        CSRMatrix.fromCOO({
          rows: 2,
          cols: 2,
          rowIndices: new Int32Array([0, 1]),
          colIndices: new Int32Array([0]),
          values: new Float64Array([1, 2]),
        })
      ).toThrow("same length");
    });

    it("should throw for out-of-bounds row indices", () => {
      expect(() =>
        CSRMatrix.fromCOO({
          rows: 2,
          cols: 2,
          rowIndices: new Int32Array([5]),
          colIndices: new Int32Array([0]),
          values: new Float64Array([1]),
        })
      ).toThrow("out of bounds");
    });

    it("should throw for out-of-bounds column indices", () => {
      expect(() =>
        CSRMatrix.fromCOO({
          rows: 2,
          cols: 2,
          rowIndices: new Int32Array([0]),
          colIndices: new Int32Array([5]),
          values: new Float64Array([1]),
        })
      ).toThrow("out of bounds");
    });
  });

  describe("get", () => {
    it("should retrieve stored values", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 3,
        cols: 3,
        rowIndices: new Int32Array([0, 1, 2]),
        colIndices: new Int32Array([0, 1, 2]),
        values: new Float64Array([1, 2, 3]),
      });

      expect(sparse.get(0, 0)).toBe(1);
      expect(sparse.get(1, 1)).toBe(2);
      expect(sparse.get(2, 2)).toBe(3);
    });

    it("should return 0 for unstored positions", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 3,
        cols: 3,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([1]),
      });

      expect(sparse.get(0, 1)).toBe(0);
      expect(sparse.get(1, 0)).toBe(0);
      expect(sparse.get(2, 2)).toBe(0);
    });

    it("should throw for out-of-bounds indices", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([1]),
      });

      expect(() => sparse.get(-1, 0)).toThrow();
      expect(() => sparse.get(0, -1)).toThrow();
      expect(() => sparse.get(2, 0)).toThrow();
      expect(() => sparse.get(0, 2)).toThrow();
    });
  });

  describe("toDense", () => {
    it("should convert to dense tensor", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([0, 0, 1]),
        colIndices: new Int32Array([0, 2, 1]),
        values: new Float64Array([1, 2, 3]),
      });

      const dense = sparse.toDense();
      expect(dense.shape).toEqual([2, 3]);
      expect(dense.toArray()).toEqual([
        [1, 0, 2],
        [0, 3, 0],
      ]);
    });

    it("should handle empty matrix", () => {
      const sparse = new CSRMatrix({
        data: new Float64Array(0),
        indices: new Int32Array(0),
        indptr: new Int32Array([0, 0]),
        shape: [1, 3],
      });

      const dense = sparse.toDense();
      expect(dense.toArray()).toEqual([[0, 0, 0]]);
    });
  });

  describe("add", () => {
    it("should add two sparse matrices", () => {
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
        colIndices: new Int32Array([1, 0]),
        values: new Float64Array([3, 4]),
      });

      const c = a.add(b);
      expect(c.toDense().toArray()).toEqual([
        [1, 3],
        [4, 2],
      ]);
    });

    it("should handle overlapping entries", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([5]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([3]),
      });

      const c = a.add(b);
      expect(c.get(0, 0)).toBe(8);
    });

    it("should cancel out to zero", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([5]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([-5]),
      });

      const c = a.add(b);
      expect(c.nnz).toBe(0);
    });

    it("should throw for shape mismatch", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 3,
        cols: 2,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      expect(() => a.add(b)).toThrow();
    });
  });

  describe("sub", () => {
    it("should subtract two sparse matrices", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([5, 8]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([2, 3]),
      });

      const c = a.sub(b);
      expect(c.get(0, 0)).toBe(3);
      expect(c.get(1, 1)).toBe(5);
    });

    it("should throw for shape mismatch", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      expect(() => a.sub(b)).toThrow();
    });
  });

  describe("scale", () => {
    it("should multiply all elements by scalar", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([2, 3]),
      });

      const scaled = sparse.scale(10);
      expect(scaled.get(0, 0)).toBe(20);
      expect(scaled.get(1, 1)).toBe(30);
    });

    it("should return empty matrix when scaled by 0", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([2, 3]),
      });

      const scaled = sparse.scale(0);
      expect(scaled.nnz).toBe(0);
    });

    it("should handle negative scalars", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([5]),
      });

      const scaled = sparse.scale(-2);
      expect(scaled.get(0, 0)).toBe(-10);
    });
  });

  describe("multiply (element-wise)", () => {
    it("should compute element-wise product", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 0, 1]),
        colIndices: new Int32Array([0, 1, 1]),
        values: new Float64Array([2, 3, 4]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([5, 6]),
      });

      const c = a.multiply(b);
      expect(c.get(0, 0)).toBe(10);
      expect(c.get(0, 1)).toBe(0); // b has no entry at (0,1)
      expect(c.get(1, 1)).toBe(24);
    });

    it("should throw for shape mismatch", () => {
      const a = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const b = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      expect(() => a.multiply(b)).toThrow();
    });
  });

  describe("matvec", () => {
    it("should compute matrix-vector product", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([0, 0, 1, 1]),
        colIndices: new Int32Array([0, 2, 1, 2]),
        values: new Float64Array([1, 2, 3, 4]),
      });

      const vec = tensor([1, 2, 3]);
      const result = sparse.matvec(vec);

      expect(result.shape).toEqual([2]);
      // Row 0: 1*1 + 0*2 + 2*3 = 7
      // Row 1: 0*1 + 3*2 + 4*3 = 18
      expect(result.toArray()).toEqual([7, 18]);
    });

    it("should accept Float64Array directly", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 1]),
        values: new Float64Array([2, 3]),
      });

      const vec = new Float64Array([4, 5]);
      const result = sparse.matvec(vec);

      expect(result.toArray()).toEqual([8, 15]);
    });

    it("should throw for dimension mismatch", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const vec = tensor([1, 2]); // Wrong size
      expect(() => sparse.matvec(vec)).toThrow();
    });
  });

  describe("matmul", () => {
    it("should compute sparse-dense matrix multiplication", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([0, 2]),
        values: new Float64Array([1, 2]),
      });

      const dense = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);

      const result = sparse.matmul(dense);
      expect(result.shape).toEqual([2, 2]);
      // Row 0: [1*1 + 0*3 + 0*5, 1*2 + 0*4 + 0*6] = [1, 2]
      // Row 1: [0*1 + 0*3 + 2*5, 0*2 + 0*4 + 2*6] = [10, 12]
      expect(result.toArray()).toEqual([
        [1, 2],
        [10, 12],
      ]);
    });

    it("should throw for non-2D dense tensor", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const vec = tensor([1, 2]);
      expect(() => sparse.matmul(vec)).toThrow("2D");
    });

    it("should throw for dimension mismatch", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([]),
        colIndices: new Int32Array([]),
        values: new Float64Array([]),
      });

      const dense = tensor([
        [1, 2],
        [3, 4],
      ]); // 2x2, but sparse has 3 cols
      expect(() => sparse.matmul(dense)).toThrow();
    });
  });

  describe("transpose", () => {
    it("should transpose the matrix", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([0, 0, 1]),
        colIndices: new Int32Array([0, 2, 1]),
        values: new Float64Array([1, 2, 3]),
      });

      const transposed = sparse.transpose();
      expect(transposed.shape).toEqual([3, 2]);
      expect(transposed.get(0, 0)).toBe(1);
      expect(transposed.get(2, 0)).toBe(2);
      expect(transposed.get(1, 1)).toBe(3);
    });

    it("should handle empty matrix", () => {
      const sparse = new CSRMatrix({
        data: new Float64Array(0),
        indices: new Int32Array(0),
        indptr: new Int32Array([0, 0]),
        shape: [1, 3],
      });

      const transposed = sparse.transpose();
      expect(transposed.shape).toEqual([3, 1]);
      expect(transposed.nnz).toBe(0);
    });

    it("should be its own inverse", () => {
      const sparse = CSRMatrix.fromCOO({
        rows: 2,
        cols: 3,
        rowIndices: new Int32Array([0, 1]),
        colIndices: new Int32Array([1, 2]),
        values: new Float64Array([5, 7]),
      });

      const doubleTransposed = sparse.transpose().transpose();
      expect(doubleTransposed.shape).toEqual([2, 3]);
      expect(doubleTransposed.get(0, 1)).toBe(5);
      expect(doubleTransposed.get(1, 2)).toBe(7);
    });
  });

  describe("copy", () => {
    it("should create an independent copy", () => {
      const original = CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: new Int32Array([0]),
        colIndices: new Int32Array([0]),
        values: new Float64Array([5]),
      });

      const copy = original.copy();

      expect(copy.get(0, 0)).toBe(5);
      expect(copy.shape).toEqual(original.shape);
      expect(copy.nnz).toBe(original.nnz);

      // Verify independence (data arrays are different objects)
      expect(copy.data).not.toBe(original.data);
      expect(copy.indices).not.toBe(original.indices);
      expect(copy.indptr).not.toBe(original.indptr);
    });
  });
});
