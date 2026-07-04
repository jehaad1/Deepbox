import { describe, expect, it } from "vitest";
import { CSRMatrix } from "../src/ndarray";

function makeSparse(
  rows: number,
  cols: number,
  rowIdx: number[],
  colIdx: number[],
  vals: number[]
): CSRMatrix {
  return CSRMatrix.fromCOO({
    rows,
    cols,
    rowIndices: new Int32Array(rowIdx),
    colIndices: new Int32Array(colIdx),
    values: new Float64Array(vals),
  });
}

describe("CSRMatrix sparse arithmetic", () => {
  describe("add()", () => {
    it("adds two sparse matrices", () => {
      const a = makeSparse(2, 2, [0, 1], [0, 1], [1, 2]);
      const b = makeSparse(2, 2, [0, 1], [1, 0], [3, 4]);
      const c = a.add(b);
      expect(c.get(0, 0)).toBe(1);
      expect(c.get(0, 1)).toBe(3);
      expect(c.get(1, 0)).toBe(4);
      expect(c.get(1, 1)).toBe(2);
    });

    it("handles overlapping indices", () => {
      const a = makeSparse(2, 2, [0], [0], [5]);
      const b = makeSparse(2, 2, [0], [0], [3]);
      const c = a.add(b);
      expect(c.get(0, 0)).toBe(8);
    });

    it("throws on shape mismatch", () => {
      const a = makeSparse(2, 2, [0], [0], [1]);
      const b = makeSparse(3, 2, [0], [0], [1]);
      expect(() => a.add(b)).toThrow();
    });
  });

  describe("sub()", () => {
    it("subtracts two sparse matrices", () => {
      const a = makeSparse(2, 2, [0, 1], [0, 1], [5, 3]);
      const b = makeSparse(2, 2, [0, 1], [0, 1], [2, 1]);
      const c = a.sub(b);
      expect(c.get(0, 0)).toBe(3);
      expect(c.get(1, 1)).toBe(2);
    });
  });

  describe("scale()", () => {
    it("scales all values", () => {
      const a = makeSparse(2, 2, [0, 1], [0, 1], [2, 3]);
      const b = a.scale(3);
      expect(b.get(0, 0)).toBe(6);
      expect(b.get(1, 1)).toBe(9);
    });

    it("scaling by 0 returns empty", () => {
      const a = makeSparse(2, 2, [0, 1], [0, 1], [2, 3]);
      const b = a.scale(0);
      expect(b.nnz).toBe(0);
    });
  });

  describe("multiply()", () => {
    it("element-wise multiplication", () => {
      const a = makeSparse(2, 2, [0, 0, 1], [0, 1, 0], [2, 3, 4]);
      const b = makeSparse(2, 2, [0, 1], [0, 0], [5, 6]);
      const c = a.multiply(b);
      expect(c.get(0, 0)).toBe(10); // 2*5
      expect(c.get(0, 1)).toBe(0); // 3*0
      expect(c.get(1, 0)).toBe(24); // 4*6
    });
  });

  describe("spmm()", () => {
    it("sparse-sparse matrix multiplication", () => {
      // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
      // C = [[19,22],[43,50]]
      const a = makeSparse(2, 2, [0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4]);
      const b = makeSparse(2, 2, [0, 0, 1, 1], [0, 1, 0, 1], [5, 6, 7, 8]);
      const c = a.spmm(b);
      expect(c.get(0, 0)).toBe(19);
      expect(c.get(0, 1)).toBe(22);
      expect(c.get(1, 0)).toBe(43);
      expect(c.get(1, 1)).toBe(50);
    });

    it("non-square multiplication", () => {
      // A = 2x3, B = 3x2
      const a = makeSparse(2, 3, [0, 0, 1], [0, 2, 1], [1, 2, 3]);
      const b = makeSparse(3, 2, [0, 2], [0, 1], [4, 5]);
      const c = a.spmm(b);
      expect(c.shape).toEqual([2, 2]);
      expect(c.get(0, 0)).toBe(4); // 1*4 + 0*0 + 2*0
      expect(c.get(0, 1)).toBe(10); // 1*0 + 0*0 + 2*5
    });

    it("throws on dimension mismatch", () => {
      const a = makeSparse(2, 3, [], [], []);
      const b = makeSparse(2, 2, [], [], []);
      expect(() => a.spmm(b)).toThrow();
    });

    it("identity multiplication", () => {
      const a = makeSparse(3, 3, [0, 1, 2], [0, 1, 2], [7, 8, 9]);
      const eye = CSRMatrix.eye(3);
      const c = a.spmm(eye);
      expect(c.get(0, 0)).toBe(7);
      expect(c.get(1, 1)).toBe(8);
      expect(c.get(2, 2)).toBe(9);
    });
  });

  describe("sliceRows()", () => {
    it("extracts a row range", () => {
      const a = makeSparse(4, 3, [0, 1, 2, 3], [0, 1, 2, 0], [1, 2, 3, 4]);
      const sub = a.sliceRows(1, 3);
      expect(sub.rows).toBe(2);
      expect(sub.cols).toBe(3);
      expect(sub.get(0, 1)).toBe(2); // was row 1
      expect(sub.get(1, 2)).toBe(3); // was row 2
    });

    it("returns empty on invalid range", () => {
      const a = makeSparse(3, 3, [0], [0], [1]);
      const sub = a.sliceRows(2, 2);
      expect(sub.rows).toBe(0);
      expect(sub.nnz).toBe(0);
    });
  });

  describe("getRow()", () => {
    it("returns dense row", () => {
      const a = makeSparse(3, 4, [0, 0, 1, 2], [0, 3, 1, 2], [1, 2, 3, 4]);
      const row0 = a.getRow(0);
      expect(row0[0]).toBe(1);
      expect(row0[1]).toBe(0);
      expect(row0[2]).toBe(0);
      expect(row0[3]).toBe(2);
    });

    it("throws for out-of-bounds row", () => {
      const a = makeSparse(2, 2, [0], [0], [1]);
      expect(() => a.getRow(5)).toThrow();
    });
  });

  describe("getCol()", () => {
    it("returns dense column", () => {
      const a = makeSparse(3, 3, [0, 1, 2], [1, 1, 1], [10, 20, 30]);
      const col1 = a.getCol(1);
      expect(col1[0]).toBe(10);
      expect(col1[1]).toBe(20);
      expect(col1[2]).toBe(30);
    });

    it("throws for out-of-bounds column", () => {
      const a = makeSparse(2, 2, [0], [0], [1]);
      expect(() => a.getCol(5)).toThrow();
    });
  });

  describe("transpose()", () => {
    it("transposes a sparse matrix", () => {
      const a = makeSparse(2, 3, [0, 0, 1], [0, 2, 1], [1, 2, 3]);
      const at = a.transpose();
      expect(at.shape).toEqual([3, 2]);
      expect(at.get(0, 0)).toBe(1);
      expect(at.get(2, 0)).toBe(2);
      expect(at.get(1, 1)).toBe(3);
    });
  });

  describe("static factories", () => {
    it("eye creates identity", () => {
      const I = CSRMatrix.eye(3);
      expect(I.nnz).toBe(3);
      expect(I.get(0, 0)).toBe(1);
      expect(I.get(1, 1)).toBe(1);
      expect(I.get(2, 2)).toBe(1);
      expect(I.get(0, 1)).toBe(0);
    });

    it("diag creates diagonal matrix", () => {
      const D = CSRMatrix.diag(new Float64Array([2, 0, 5]));
      expect(D.nnz).toBe(2); // zero diagonal entry skipped
      expect(D.get(0, 0)).toBe(2);
      expect(D.get(1, 1)).toBe(0);
      expect(D.get(2, 2)).toBe(5);
    });
  });

  describe("toDense() round-trip", () => {
    it("preserves values through toDense()", () => {
      const a = makeSparse(3, 3, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2], [1, 2, 3, 4, 5]);
      const dense = a.toDense();
      expect(dense.shape).toEqual([3, 3]);
      const data = dense.data as Float64Array;
      expect(data[0]).toBe(1);
      expect(data[2]).toBe(2);
      expect(data[4]).toBe(3);
      expect(data[6]).toBe(4);
      expect(data[8]).toBe(5);
    });
  });
});
