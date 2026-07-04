import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { computeStrides, isBigIntArray, Tensor } from "../src/ndarray/tensor/Tensor";

// ────── computeStrides ──────
describe("computeStrides", () => {
  it("3D shape", () => {
    expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
  });

  it("1D shape", () => {
    expect(computeStrides([5])).toEqual([1]);
  });

  it("empty shape", () => {
    expect(computeStrides([])).toEqual([]);
  });
});

// ────── isBigIntArray ──────
describe("isBigIntArray", () => {
  it("true for BigInt64Array", () => {
    expect(isBigIntArray(new BigInt64Array(2))).toBe(true);
  });

  it("false for Float64Array", () => {
    expect(isBigIntArray(new Float64Array(2))).toBe(false);
  });
});

// ────── Tensor.fromTypedArray ──────
describe("Tensor.fromTypedArray", () => {
  it("creates from Float32Array", () => {
    const t = Tensor.fromTypedArray({
      data: new Float32Array([1, 2, 3]),
      shape: [3],
      dtype: "float32",
      device: "cpu",
    });
    expect(t.shape).toEqual([3]);
    expect(t.dtype).toBe("float32");
  });

  it("creates from Int32Array", () => {
    const t = Tensor.fromTypedArray({
      data: new Int32Array([1, 2, 3, 4]),
      shape: [2, 2],
      dtype: "int32",
      device: "cpu",
    });
    expect(t.shape).toEqual([2, 2]);
  });

  it("creates from Uint8Array (bool)", () => {
    const t = Tensor.fromTypedArray({
      data: new Uint8Array([0, 1, 1, 0]),
      shape: [4],
      dtype: "bool",
      device: "cpu",
    });
    expect(t.dtype).toBe("bool");
  });

  it("creates from BigInt64Array", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    expect(t.dtype).toBe("int64");
  });

  it("rejects dtype mismatch", () => {
    expect(() =>
      Tensor.fromTypedArray({
        data: new Float32Array([1]),
        shape: [1],
        dtype: "float64",
        device: "cpu",
      })
    ).toThrow();
  });

  it("with custom strides and offset", () => {
    const t = Tensor.fromTypedArray({
      data: new Float64Array([0, 1, 2, 3, 4, 5]),
      shape: [3],
      dtype: "float64",
      device: "cpu",
      offset: 1,
      strides: [2],
    });
    expect(t.at(0)).toBe(1);
    expect(t.at(1)).toBe(3);
    expect(t.at(2)).toBe(5);
  });
});

// ────── Tensor.fromStringArray ──────
describe("Tensor.fromStringArray", () => {
  it("creates string tensor", () => {
    const t = Tensor.fromStringArray({
      data: ["a", "b", "c"],
      shape: [3],
    });
    expect(t.dtype).toBe("string");
    expect(t.at(0)).toBe("a");
  });

  it("2D string tensor", () => {
    const t = Tensor.fromStringArray({
      data: ["a", "b", "c", "d"],
      shape: [2, 2],
    });
    expect(t.at(0, 0)).toBe("a");
    expect(t.at(1, 1)).toBe("d");
  });
});

// ────── Tensor.zeros ──────
describe("Tensor.zeros", () => {
  it("creates numeric zeros", () => {
    const t = Tensor.zeros([2, 3], { dtype: "float64", device: "cpu" });
    expect(t.shape).toEqual([2, 3]);
    expect(t.at(0, 0)).toBe(0);
  });

  it("creates string zeros", () => {
    const t = Tensor.zeros([2], { dtype: "string", device: "cpu" });
    expect(t.dtype).toBe("string");
    expect(t.at(0)).toBe("");
  });
});

// ────── view ──────
describe("Tensor.view", () => {
  it("creates view with same data", () => {
    const t = tensor([1, 2, 3, 4, 5, 6]);
    const v = t.view([2, 3]);
    expect(v.shape).toEqual([2, 3]);
  });

  it("throws on size mismatch", () => {
    const t = tensor([1, 2, 3]);
    expect(() => t.view([2, 3])).toThrow();
  });

  it("string tensor view", () => {
    const t = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [4] });
    const v = t.view([2, 2]);
    expect(v.shape).toEqual([2, 2]);
    expect(v.dtype).toBe("string");
  });
});

// ────── reshape ──────
describe("Tensor.reshape", () => {
  it("reshapes contiguous tensor", () => {
    const t = tensor([1, 2, 3, 4]);
    const r = t.reshape([2, 2]);
    expect(r.shape).toEqual([2, 2]);
  });

  it("throws on incompatible size", () => {
    const t = tensor([1, 2, 3]);
    expect(() => t.reshape([2, 3])).toThrow();
  });

  it("string tensor reshape", () => {
    const t = Tensor.fromStringArray({ data: ["a", "b", "c", "d"], shape: [4] });
    const r = t.reshape([2, 2]);
    expect(r.shape).toEqual([2, 2]);
  });
});

// ────── flatten ──────
describe("Tensor.flatten", () => {
  it("flattens 2D", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const f = t.flatten();
    expect(f.shape).toEqual([4]);
  });
});

// ────── slice ──────
describe("Tensor.slice", () => {
  it("integer index reduces dimension", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const s = t.slice(0);
    expect(s.shape).toEqual([3]);
  });

  it("range slice", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const s = t.slice({ start: 0, end: 1 });
    expect(s.shape).toEqual([1, 3]);
  });

  it("step slice", () => {
    const t = tensor([1, 2, 3, 4, 5, 6]);
    const s = t.slice({ start: 0, end: 6, step: 2 });
    expect(s.shape).toEqual([3]);
  });

  it("throws for too many indices", () => {
    const t = tensor([1, 2, 3]);
    expect(() => t.slice(0, 0)).toThrow();
  });

  it("string tensor slice", () => {
    const t = Tensor.fromStringArray({ data: ["a", "b", "c"], shape: [3] });
    const s = t.slice({ start: 0, end: 2 });
    expect(s.shape).toEqual([2]);
    expect(s.at(0)).toBe("a");
  });
});

// ────── at ──────
describe("Tensor.at", () => {
  it("accesses element", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(t.at(0, 1)).toBe(2);
    expect(t.at(1, 0)).toBe(3);
  });

  it("negative indexing", () => {
    const t = tensor([10, 20, 30]);
    expect(t.at(-1)).toBe(30);
  });

  it("throws for wrong number of indices", () => {
    const t = tensor([1, 2, 3]);
    expect(() => t.at(0, 0)).toThrow();
  });

  it("throws for out of bounds", () => {
    const t = tensor([1, 2, 3]);
    expect(() => t.at(5)).toThrow();
  });
});

// ────── toArray ──────
describe("Tensor.toArray", () => {
  it("1D", () => {
    const t = tensor([1, 2, 3]);
    expect(t.toArray()).toEqual([1, 2, 3]);
  });

  it("2D", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(t.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});

// ────── toString ──────
describe("Tensor.toString", () => {
  it("1D tensor", () => {
    const t = tensor([1, 2, 3]);
    const s = t.toString();
    expect(s).toContain("tensor");
    expect(s).toContain("1");
  });

  it("2D tensor", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = t.toString();
    expect(s).toContain("tensor");
  });

  it("large tensor gets summarized", () => {
    const data = Array.from({ length: 20 }, (_, i) => i);
    const t = tensor(data);
    const s = t.toString(6);
    expect(s).toContain("...");
  });

  it("float formatting", () => {
    const t = tensor([1.23456789]);
    const s = t.toString();
    expect(s).toContain("1.235");
  });

  it("string tensor", () => {
    const t = Tensor.fromStringArray({ data: ["hello", "world"], shape: [2] });
    const s = t.toString();
    expect(s).toContain("hello");
  });

  it("bigint tensor", () => {
    const t = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const s = t.toString();
    expect(s).toContain("1");
  });
});
