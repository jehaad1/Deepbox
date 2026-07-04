import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { all, any, max, mean, min, prod, sum } from "../src/ndarray/ops/reduction";
import { Tensor } from "../src/ndarray/tensor/Tensor";

// Additional reduction branches targeting BigInt, keepdims, and edge cases

describe("reduction int64 branches", () => {
  it("sum int64", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const s = sum(t);
    expect(s.size).toBe(1);
  });

  it("min int64", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([3n, 1n, 2n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const m = min(t);
    expect(m.size).toBe(1);
  });

  it("max int64", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([3n, 1n, 2n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const m = max(t);
    expect(m.size).toBe(1);
  });

  it("any int64 - has nonzero", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([0n, 1n, 0n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const a = any(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(1);
  });

  it("any int64 - all zero", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([0n, 0n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const a = any(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(0);
  });

  it("all int64 - all nonzero", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const a = all(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(1);
  });

  it("all int64 - has zero", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 0n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const a = all(t);
    expect(Array.from(a.data as Uint8Array)[0]).toBe(0);
  });

  it("prod int64", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, 3n, 4n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const p = prod(t);
    expect(p.size).toBe(1);
  });
});

describe("reduction 3D branches", () => {
  it("sum 3D axis=0", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const s = sum(t, 0);
    expect(s.shape).toEqual([2, 2]);
  });

  it("sum 3D axis=2", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const s = sum(t, 2);
    expect(s.shape).toEqual([2, 2]);
  });

  it("mean 3D axis=1 keepdims", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const m = mean(t, 1, true);
    expect(m.shape).toEqual([2, 1, 2]);
  });

  it("min 3D axis=2 keepdims", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const m = min(t, 2, true);
    expect(m.shape).toEqual([2, 2, 1]);
  });

  it("any 3D axis=1", () => {
    const t = tensor([
      [
        [0, 1],
        [0, 0],
      ],
      [
        [1, 0],
        [0, 0],
      ],
    ]);
    const a = any(t, 1);
    expect(a.shape).toEqual([2, 2]);
  });

  it("all 3D axis=2", () => {
    const t = tensor([
      [
        [1, 1],
        [1, 0],
      ],
      [
        [1, 1],
        [1, 1],
      ],
    ]);
    const a = all(t, 2);
    expect(a.shape).toEqual([2, 2]);
  });
});

describe("reduction keepdims=true scalar branches", () => {
  it("sum keepdims full", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = sum(t, undefined, true);
    expect(s.shape).toEqual([1, 1]);
  });

  it("any keepdims full", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const a = any(t, undefined, true);
    expect(a.shape).toEqual([1, 1]);
  });

  it("all keepdims full", () => {
    const t = tensor([
      [1, 1],
      [1, 1],
    ]);
    const a = all(t, undefined, true);
    expect(a.shape).toEqual([1, 1]);
  });
});
