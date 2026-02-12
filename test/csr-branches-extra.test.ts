import { describe, expect, it } from "vitest";
import { CSRMatrix, tensor } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("CSRMatrix branch coverage extras", () => {
  it("validates constructor invariants", () => {
    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([0]),
          indptr: new Int32Array([0, 1]),
          shape: [2],
        })
    ).toThrow(/shape must be 2D/i);

    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([0]),
          indptr: new Int32Array([0, 1]),
          shape: [2, 2],
        })
    ).toThrow(/indptr length/i);

    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([0, 1]),
          indptr: new Int32Array([0, 2, 2]),
          shape: [2, 2],
        })
    ).toThrow(/length mismatch/i);
  });

  it("rejects invalid indptr and index bounds", () => {
    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([0]),
          indptr: new Int32Array([1, 1]),
          shape: [1, 2],
        })
    ).toThrow(/indptr\[0\]/i);

    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1, 2, 3]),
          indices: new Int32Array([0, 1, 0]),
          indptr: new Int32Array([0, 2, 1, 3]),
          shape: [3, 2],
        })
    ).toThrow(/indptr.*non-decreasing/i);

    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([0]),
          indptr: new Int32Array([0, 1, 2]),
          shape: [2, 2],
        })
    ).toThrow(/indptr.*nnz/i);

    expect(
      () =>
        new CSRMatrix({
          data: new Float64Array([1]),
          indices: new Int32Array([2]),
          indptr: new Int32Array([0, 1]),
          shape: [1, 2],
        })
    ).toThrow(/column index.*out of bounds/i);
  });

  it("skips zero entries in add/sub/multiply", () => {
    const a = CSRMatrix.fromCOO({
      rows: 1,
      cols: 1,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([0]),
      values: Float64Array.from([1]),
    });
    const b = CSRMatrix.fromCOO({
      rows: 1,
      cols: 1,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([0]),
      values: Float64Array.from([-1]),
    });
    const sum = a.add(b);
    expect(sum.nnz).toBe(0);

    const diff = a.sub(a);
    expect(diff.nnz).toBe(0);

    const zeroMul = a.multiply(
      CSRMatrix.fromCOO({
        rows: 1,
        cols: 1,
        rowIndices: Int32Array.from([0]),
        colIndices: Int32Array.from([0]),
        values: Float64Array.from([0]),
      })
    );
    expect(zeroMul.nnz).toBe(0);
  });

  it("validates matvec/matmul shapes and tensor conversion", () => {
    const a = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([0]),
      colIndices: Int32Array.from([0]),
      values: Float64Array.from([1]),
    });

    expect(() => a.matvec(tensor([1, 2, 3]))).toThrow(/Vector length/i);
    expect(() => a.matmul(tensor([1, 2, 3]))).toThrow(/2D/);
    expect(() =>
      a.matmul(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ])
      )
    ).toThrow(/Cannot multiply/);

    const s = Tensor.fromStringArray({
      data: ["a", "b"],
      shape: [2],
      device: "cpu",
    });
    expect(() => a.matvec(s)).toThrow(/Cannot convert string tensor/i);
  });

  it("validates fromCOO inputs and supports unsorted data", () => {
    expect(() =>
      CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: Int32Array.from([0]),
        colIndices: Int32Array.from([0, 1]),
        values: Float64Array.from([1, 2]),
      })
    ).toThrow(/same length/i);

    expect(() =>
      CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: Int32Array.from([2]),
        colIndices: Int32Array.from([0]),
        values: Float64Array.from([1]),
      })
    ).toThrow(/row index out of bounds/i);

    expect(() =>
      CSRMatrix.fromCOO({
        rows: 2,
        cols: 2,
        rowIndices: Int32Array.from([0]),
        colIndices: Int32Array.from([2]),
        values: Float64Array.from([1]),
      })
    ).toThrow(/col index out of bounds/i);

    const unsorted = CSRMatrix.fromCOO({
      rows: 2,
      cols: 2,
      rowIndices: Int32Array.from([1, 0]),
      colIndices: Int32Array.from([1, 0]),
      values: Float64Array.from([3, 2]),
      sort: false,
    });
    expect(unsorted.nnz).toBe(2);
    expect(unsorted.get(0, 0)).toBe(2);
    expect(unsorted.get(1, 1)).toBe(3);
  });
});
