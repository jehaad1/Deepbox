/**
 * Regression tests for CSRMatrix (v1.5.0 review).
 *
 * Reference values come from SciPy 1.17 (`scipy.sparse.csr_matrix`).
 */
import { describe, expect, it } from "vitest";
import { DTypeError, IndexError, InvalidParameterError, ShapeError } from "../../src/core";
import { CSRMatrix, tensor, transpose } from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor/Tensor";

function dense(m: CSRMatrix): number[][] {
  const t = m.toDense();
  const d = t.data as Float64Array;
  const [r = 0, c = 0] = t.shape;
  const out: number[][] = [];
  for (let i = 0; i < r; i++) out.push(Array.from(d.subarray(i * c, (i + 1) * c)));
  return out;
}

const A = CSRMatrix.fromDense(
  tensor(
    [
      [1, 0, 2, 0],
      [0, 0, 0, 3],
      [4, 5, 0, 0],
    ],
    { dtype: "float64" }
  )
);

// Hand-built matrix with unsorted columns and a repeated entry: [[2, 0, 6], [0, 0, 0]].
function nonCanonical(): CSRMatrix {
  return new CSRMatrix({
    data: new Float64Array([1, 2, 5]),
    indices: new Int32Array([2, 0, 2]),
    indptr: new Int32Array([0, 3, 3]),
    shape: [2, 3],
  });
}

describe("c36 CSRMatrix: non-canonical input", () => {
  it("reports canonical state", () => {
    expect(A.hasCanonicalFormat).toBe(true);
    expect(nonCanonical().hasCanonicalFormat).toBe(false);
  });

  it("toDense / getRow / get / getCol sum repeated entries like SciPy", () => {
    const m = nonCanonical();
    expect(dense(m)).toEqual([
      [2, 0, 6],
      [0, 0, 0],
    ]);
    expect(Array.from(m.getRow(0))).toEqual([2, 0, 6]);
    expect(m.get(0, 2)).toBe(6);
    expect(m.get(0, 0)).toBe(2);
    expect(Array.from(m.getCol(2))).toEqual([6, 0]);
  });

  it("matvec matches SciPy on repeated entries", () => {
    expect(
      nonCanonical()
        .matvec(new Float64Array([1, 2, 3]))
        .toArray()
    ).toEqual([20, 0]);
  });

  it("multiply sums repeated entries of both operands", () => {
    const ones = CSRMatrix.fromDense(
      tensor(
        [
          [1, 1, 1],
          [1, 1, 1],
        ],
        { dtype: "float64" }
      )
    );
    expect(dense(nonCanonical().multiply(ones))).toEqual([
      [2, 0, 6],
      [0, 0, 0],
    ]);
    expect(dense(ones.multiply(nonCanonical()))).toEqual([
      [2, 0, 6],
      [0, 0, 0],
    ]);
  });

  it("canonicalize sorts and sums (SciPy sum_duplicates)", () => {
    const c = nonCanonical().canonicalize();
    expect(c.hasCanonicalFormat).toBe(true);
    expect(Array.from(c.data)).toEqual([2, 6]);
    expect(Array.from(c.indices)).toEqual([0, 2]);
  });

  it("add produces canonical output from non-canonical input", () => {
    const sum = nonCanonical().add(CSRMatrix.eye(3).sliceRows(0, 2));
    expect(sum.hasCanonicalFormat).toBe(true);
    expect(dense(sum)).toEqual([
      [3, 0, 6],
      [0, 1, 0],
    ]);
  });
});

describe("c36 CSRMatrix: arithmetic against SciPy", () => {
  const B = CSRMatrix.fromDense(
    tensor(
      [
        [0, 1, 0],
        [2, 0, 0],
        [0, 0, 3],
        [1, 0, 1],
      ],
      { dtype: "float64" }
    )
  );

  it("spmm", () => {
    expect(dense(A.spmm(B))).toEqual([
      [0, 1, 6],
      [3, 0, 3],
      [10, 4, 0],
    ]);
    expect(dense(A.transpose().spmm(A))).toEqual([
      [17, 20, 2, 0],
      [20, 25, 0, 0],
      [2, 0, 4, 0],
      [0, 0, 0, 9],
    ]);
  });

  it("multiply", () => {
    const other = CSRMatrix.fromDense(
      tensor(
        [
          [2, 0, 3, 0],
          [0, 0, 0, 0],
          [1, 1, 1, 1],
        ],
        { dtype: "float64" }
      )
    );
    expect(dense(A.multiply(other))).toEqual([
      [2, 0, 6, 0],
      [0, 0, 0, 0],
      [4, 5, 0, 0],
    ]);
  });

  it("matmul matches the dense product", () => {
    const out = A.matmul(
      tensor(
        [
          [1, 2],
          [3, 4],
          [5, 6],
          [7, 8],
        ],
        { dtype: "float64" }
      )
    );
    expect(out.toArray()).toEqual([
      [11, 14],
      [21, 24],
      [19, 28],
    ]);
  });

  it("sub drops exact cancellations and add keeps the pattern sorted", () => {
    expect(A.sub(A).nnz).toBe(0);
    const s = A.add(A.scale(-1));
    expect(s.nnz).toBe(0);
    const t = A.add(CSRMatrix.eye(4).sliceRows(0, 3));
    expect(t.hasCanonicalFormat).toBe(true);
    expect(dense(t)).toEqual([
      [2, 0, 2, 0],
      [0, 1, 0, 3],
      [4, 5, 1, 0],
    ]);
  });
});

describe("c36 CSRMatrix: scale", () => {
  it("scale(0) keeps Infinity/NaN entries as NaN, like 0 * Infinity", () => {
    const m = CSRMatrix.fromDense(tensor([[1, Infinity, 0, 2]], { dtype: "float64" }));
    const z = m.scale(0);
    expect(z.nnz).toBe(1);
    expect(Number.isNaN(z.get(0, 1))).toBe(true);
    expect(A.scale(0).nnz).toBe(0);
    expect(A.scale(0).indptr.length).toBe(4);
  });
});

describe("c36 CSRMatrix: dense conversions", () => {
  it("matvec honours views whose backing buffer is longer than the view", () => {
    const x = tensor([1, 2, 3, 4, 5, 6], { dtype: "float64" });
    const view = x.slice({ start: 0, end: 4 });
    expect(A.matvec(view).toArray()).toEqual([7, 12, 14]);
    const shifted = x.slice({ start: 2, end: 6 });
    expect(A.matvec(shifted).toArray()).toEqual([1 * 3 + 2 * 5, 3 * 6, 4 * 3 + 5 * 4]);
  });

  it("matvec and matmul read strided (transposed) tensors correctly", () => {
    const base = tensor(
      [
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
      ],
      { dtype: "float64" }
    );
    const view = transpose(base); // shape [2, 4], non-contiguous
    expect(view.strides).not.toEqual([4, 1]);
    // A (3x4) times view^T (4x2) equals A times base.
    expect(A.matmul(transpose(view)).toArray()).toEqual([
      [7, 19],
      [12, 24],
      [14, 50],
    ]);
    // Using the strided view as the dense operand of a 2x? product.
    const wide = CSRMatrix.fromDense(tensor([[1, 1]], { dtype: "float64" }));
    expect(wide.matmul(view).toArray()).toEqual([[6, 8, 10, 12]]);
  });

  it("matvec converts int32 and int64 tensors", () => {
    expect(A.matvec(tensor([1, 2, 3, 4], { dtype: "int32" })).toArray()).toEqual([7, 12, 14]);
    expect(A.matvec(tensor([1, 2, 3, 4], { dtype: "int64" })).toArray()).toEqual([7, 12, 14]);
  });

  it("matvec rejects matrices instead of reading them as flat vectors", () => {
    const square = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float64" }
    );
    expect(() => A.matvec(square)).toThrow(ShapeError);
    expect(() => A.matvec(square)).toThrow(/use matmul/);
  });

  it("matvec still accepts column and row vectors with singleton dimensions", () => {
    const col = tensor([[1], [2], [3], [4]], { dtype: "float64" });
    const row = tensor([[1, 2, 3, 4]], { dtype: "float64" });
    expect(A.matvec(col).toArray()).toEqual([7, 12, 14]);
    expect(A.matvec(row).toArray()).toEqual([7, 12, 14]);
  });

  it("string tensors raise DTypeError", () => {
    const s = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [4], device: "cpu" });
    expect(() => A.matvec(s)).toThrow(DTypeError);
    expect(() => A.matvec(s)).toThrow(/Cannot convert string tensor/);
  });

  it("fromDense round-trips, keeps NaN, drops -0", () => {
    const d = tensor(
      [
        [0, NaN, -0],
        [3, 0, 0],
      ],
      { dtype: "float64" }
    );
    const m = CSRMatrix.fromDense(d);
    expect(m.nnz).toBe(2);
    expect(Number.isNaN(m.get(0, 1))).toBe(true);
    expect(m.get(1, 0)).toBe(3);
    expect(() => CSRMatrix.fromDense(tensor([1, 2, 3]))).toThrow(ShapeError);
    const empty = CSRMatrix.fromDense(tensor([[]], { dtype: "float64" }));
    expect(empty.shape).toEqual([1, 0]);
  });
});

describe("c36 CSRMatrix: validation", () => {
  it("constructor rejects fractional and NaN shapes", () => {
    const base = {
      data: new Float64Array(0),
      indices: new Int32Array(0),
    };
    expect(() => new CSRMatrix({ ...base, indptr: new Int32Array(3), shape: [2.5, 2] })).toThrow(
      /non-negative integers/
    );
    expect(
      () => new CSRMatrix({ ...base, indptr: new Int32Array(3), shape: [Number.NaN, 2] })
    ).toThrow(ShapeError);
  });

  it("constructor copies the shape array", () => {
    const shape = [1, 1];
    const m = new CSRMatrix({
      data: new Float64Array(0),
      indices: new Int32Array(0),
      indptr: new Int32Array(2),
      shape,
    });
    shape[0] = 9;
    expect(m.shape).toEqual([1, 1]);
  });

  it("get / getRow / getCol reject fractional and NaN indices", () => {
    expect(() => A.get(0.5, 0)).toThrow(InvalidParameterError);
    expect(() => A.get(0, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => A.getRow(1.5)).toThrow(InvalidParameterError);
    expect(() => A.getCol(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => A.get(3, 0)).toThrow(IndexError);
  });

  it("sliceRows rejects NaN and fractional bounds, accepts Infinity", () => {
    expect(() => A.sliceRows(Number.NaN, 2)).toThrow(InvalidParameterError);
    expect(() => A.sliceRows(0, 1.5)).toThrow(InvalidParameterError);
    expect(A.sliceRows(1, Number.POSITIVE_INFINITY).shape).toEqual([2, 4]);
    expect(dense(A.sliceRows(1, 3))).toEqual([
      [0, 0, 0, 3],
      [4, 5, 0, 0],
    ]);
    expect(A.sliceRows(3, 1).shape).toEqual([0, 4]);
  });

  it("eye rejects invalid sizes", () => {
    expect(() => CSRMatrix.eye(-1)).toThrow(InvalidParameterError);
    expect(() => CSRMatrix.eye(2.5)).toThrow(InvalidParameterError);
    expect(CSRMatrix.eye(0).shape).toEqual([0, 0]);
  });

  it("fromCOO rejects invalid shapes with a typed error", () => {
    const empty = {
      rowIndices: new Int32Array(0),
      colIndices: new Int32Array(0),
      values: new Float64Array(0),
    };
    expect(() => CSRMatrix.fromCOO({ rows: -1, cols: 2, ...empty })).toThrow(ShapeError);
    expect(() => CSRMatrix.fromCOO({ rows: 2, cols: 1.5, ...empty })).toThrow(ShapeError);
  });
});

describe("c36 CSRMatrix: fromCOO ordering", () => {
  it("always returns sorted columns, even with sort: false", () => {
    const m = CSRMatrix.fromCOO({
      rows: 2,
      cols: 4,
      rowIndices: new Int32Array([1, 0, 0, 1, 0]),
      colIndices: new Int32Array([3, 2, 0, 1, 2]),
      values: new Float64Array([4, 1, 2, 5, 10]),
      sort: false,
    });
    expect(m.hasCanonicalFormat).toBe(true);
    expect(Array.from(m.indices)).toEqual([0, 2, 1, 3]);
    expect(Array.from(m.data)).toEqual([2, 11, 5, 4]);
    expect(Array.from(m.indptr)).toEqual([0, 2, 4]);
    expect(m.get(1, 3)).toBe(4);
  });
});
