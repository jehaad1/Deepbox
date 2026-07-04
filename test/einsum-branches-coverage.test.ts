import { describe, expect, it } from "vitest";
import { slice, tensor, transpose } from "../src/ndarray";
import { einsum } from "../src/ndarray/ops/einsum";

describe("einsum", () => {
  it("matrix multiply ij,jk->ik", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    const c = einsum("ij,jk->ik", a, b);
    expect(c.shape).toEqual([2, 2]);
    expect(Array.from(c.data as Float64Array)).toEqual([19, 22, 43, 50]);
  });

  it("dot product i,i->", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const c = einsum("i,i->", a, b);
    expect(c.shape).toEqual([]);
  });

  it("trace ii->", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const c = einsum("ii->", a);
    expect(c.shape).toEqual([]);
  });

  it("transpose ij->ji", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const c = einsum("ij->ji", a);
    expect(c.shape).toEqual([3, 2]);
  });

  it("diagonal ii->i", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const c = einsum("ii->i", a);
    expect(c.shape).toEqual([2]);
    expect(Array.from(c.data as Float64Array)).toEqual([1, 4]);
  });

  it("matrix-vector ij,j->i", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([5, 6]);
    const c = einsum("ij,j->i", a, b);
    expect(c.shape).toEqual([2]);
    expect(Array.from(c.data as Float64Array)).toEqual([17, 39]);
  });

  it("implicit output (no arrow)", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([5, 6]);
    const c = einsum("ij,j", a, b);
    expect(c.shape).toEqual([2]);
  });

  it("outer product i,j->ij", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4, 5]);
    const c = einsum("i,j->ij", a, b);
    expect(c.shape).toEqual([2, 3]);
  });

  it("throws for no tensors", () => {
    expect(() => einsum("i")).toThrow();
  });

  it("throws for input count mismatch", () => {
    const a = tensor([1, 2]);
    expect(() => einsum("i,j", a)).toThrow();
  });

  it("throws for dim count mismatch", () => {
    const a = tensor([1, 2]);
    expect(() => einsum("ij", a)).toThrow();
  });

  it("throws for conflicting sizes", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([1, 2]);
    expect(() => einsum("i,i->", a, b)).toThrow(/conflicting/);
  });

  it("honors strides/offset of non-contiguous (transposed) inputs", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const at = transpose(a); // 3x2 view, non-contiguous strides
    // at = [[1,4],[2,5],[3,6]]
    const out = einsum("ij->ji", at); // should give back the original 2x3
    expect(out.shape).toEqual([2, 3]);
    expect(Array.from(out.data as Float64Array)).toEqual([1, 2, 3, 4, 5, 6]);

    // matmul with a transposed operand: at (3x2) @ a (2x3) -> 3x3
    const prod = einsum("ij,jk->ik", at, a);
    expect(prod.shape).toEqual([3, 3]);
    // row 0 of at is [1,4]; [1,4]·columns of a = [1*1+4*4, 1*2+4*5, 1*3+4*6] = [17,22,27]
    expect(Array.from(prod.data as Float64Array).slice(0, 3)).toEqual([17, 22, 27]);
  });

  it("honors offset of sliced inputs", () => {
    const a = tensor([10, 20, 30, 40, 50]);
    const s = slice(a, { start: 2, end: 5 }); // [30,40,50], offset=2
    const b = tensor([1, 1, 1]);
    const out = einsum("i,i->", s, b);
    expect(out.shape).toEqual([]);
    expect((out.data as Float64Array)[0]).toBe(120);
  });
});
