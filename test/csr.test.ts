import { describe, expect, it } from "vitest";
import { IndexError } from "../src/core";
import { CSRMatrix, tensor } from "../src/ndarray";

describe("CSRMatrix", () => {
  it("should create CSR from COO and roundtrip to dense", () => {
    const csr = CSRMatrix.fromCOO({
      rows: 3,
      cols: 4,
      rowIndices: Int32Array.from([0, 1, 1, 2]),
      colIndices: Int32Array.from([1, 0, 3, 2]),
      values: Float64Array.from([1, 2, 3, 4]),
    });

    expect(csr.shape).toEqual([3, 4]);
    expect(csr.nnz).toBe(4);

    const dense = csr.toDense();
    expect(dense.shape).toEqual([3, 4]);

    // spot check
    expect(Number(dense.data[dense.offset + 0 * 4 + 1])).toBe(1);
    expect(Number(dense.data[dense.offset + 1 * 4 + 0])).toBe(2);
    expect(Number(dense.data[dense.offset + 1 * 4 + 3])).toBe(3);
    expect(Number(dense.data[dense.offset + 2 * 4 + 2])).toBe(4);
  });

  it("should have rows and cols getters", () => {
    const csr = CSRMatrix.fromCOO({
      rows: 3,
      cols: 4,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([0]),
      values: Float64Array.from([1]),
    });
    expect(csr.rows).toBe(3);
    expect(csr.cols).toBe(4);
  });

  it("should add two sparse matrices", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([1, 2]),
    });
    const b = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([1, 0]),
      values: Float64Array.from([3, 4]),
    });

    const c = a.add(b);
    expect(c.nnz).toBe(4);

    const dense = c.toDense();
    expect(Number(dense.data[0])).toBe(1); // (0,0)
    expect(Number(dense.data[1])).toBe(3); // (0,1)
    expect(Number(dense.data[2])).toBe(4); // (1,0)
    expect(Number(dense.data[3])).toBe(2); // (1,1)
  });

  it("should subtract sparse matrices", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([5, 6]),
    });
    const b = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([2, 3]),
    });

    const c = a.sub(b);
    const dense = c.toDense();
    expect(Number(dense.data[0])).toBe(3); // (0,0)
    expect(Number(dense.data[3])).toBe(3); // (1,1)
  });

  it("should scale a sparse matrix", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([2, 4]),
    });

    const scaled = a.scale(3);
    const dense = scaled.toDense();
    expect(Number(dense.data[0])).toBe(6);
    expect(Number(dense.data[3])).toBe(12);
  });

  it("should scale by zero to return empty matrix", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([0]),
      values: Float64Array.from([5]),
    });

    const scaled = a.scale(0);
    expect(scaled.nnz).toBe(0);
  });

  it("should element-wise multiply (Hadamard)", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 0, 1]),
      colIndices: Int32Array.from([0, 1, 1]),
      values: Float64Array.from([2, 3, 4]),
    });
    const b = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([5, 6]),
    });

    const c = a.multiply(b);
    // Only overlapping: (0,0) = 2*5=10, (1,1) = 4*6=24
    expect(c.nnz).toBe(2);
    const dense = c.toDense();
    expect(Number(dense.data[0])).toBe(10);
    expect(Number(dense.data[3])).toBe(24);
  });

  it("should multiply sparse matrix by dense vector (matvec)", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 3,
      rowIndices: Int32Array.from([0, 0, 1]),
      colIndices: Int32Array.from([0, 2, 1]),
      values: Float64Array.from([1, 2, 3]),
    });
    const x = tensor([1, 2, 3], { dtype: "float64" });

    const y = a.matvec(x);
    expect(y.shape).toEqual([2]);
    // row 0: 1*1 + 0*2 + 2*3 = 7
    // row 1: 0*1 + 3*2 + 0*3 = 6
    expect(Number(y.data[0])).toBe(7);
    expect(Number(y.data[1])).toBe(6);
  });

  it("should multiply sparse matrix by dense matrix (matmul)", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0, 1]),
      colIndices: Int32Array.from([0, 1]),
      values: Float64Array.from([2, 3]),
    });
    const b = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float64" }
    );

    const c = a.matmul(b);
    expect(c.shape).toEqual([2, 2]);
    // row 0: [2*1, 2*2] = [2, 4]
    // row 1: [3*3, 3*4] = [9, 12]
    expect(Number(c.data[0])).toBe(2);
    expect(Number(c.data[1])).toBe(4);
    expect(Number(c.data[2])).toBe(9);
    expect(Number(c.data[3])).toBe(12);
  });

  it("should transpose a sparse matrix", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 3,
      rowIndices: Int32Array.from([0, 1, 1]),
      colIndices: Int32Array.from([0, 1, 2]),
      values: Float64Array.from([1, 2, 3]),
    });

    const at = a.transpose();
    expect(at.shape).toEqual([3, 2]);
    expect(at.nnz).toBe(3);

    // Check values
    expect(at.get(0, 0)).toBe(1);
    expect(at.get(1, 1)).toBe(2);
    expect(at.get(2, 1)).toBe(3);
  });

  it("should get element at position", () => {
    const a = CSRMatrix.fromCOO({
      rows: 3,
      cols: 3,
      rowIndices: Int32Array.from([0, 1, 2]),
      colIndices: Int32Array.from([0, 1, 2]),
      values: Float64Array.from([1, 2, 3]),
    });

    expect(a.get(0, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(2);
    expect(a.get(2, 2)).toBe(3);
    expect(a.get(0, 1)).toBe(0); // Not stored = 0
    expect(a.get(1, 0)).toBe(0);
  });

  it("should throw on out of bounds get", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([]),
      colIndices: Int32Array.from([]),
      values: Float64Array.from([]),
    });

    expect(() => a.get(-1, 0)).toThrow(IndexError);
    expect(() => a.get(0, 2)).toThrow(IndexError);
    expect(() => a.get(2, 0)).toThrow(IndexError);
  });

  it("should copy a sparse matrix", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([1]),
      values: Float64Array.from([5]),
    });

    const b = a.copy();
    expect(b.shape).toEqual(a.shape);
    expect(b.nnz).toBe(a.nnz);
    expect(b.get(0, 1)).toBe(5);

    // Verify it's a copy, not same reference
    expect(b.data).not.toBe(a.data);
  });

  it("should throw on shape mismatch for add", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 3,
      rowIndices: Int32Array.from([]),
      colIndices: Int32Array.from([]),
      values: Float64Array.from([]),
    });
    const b = CSRMatrix.fromCOO({
      rows: 3,
      cols: 2,
      rowIndices: Int32Array.from([]),
      colIndices: Int32Array.from([]),
      values: Float64Array.from([]),
    });

    expect(() => a.add(b)).toThrow();
  });
});
