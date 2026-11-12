import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError } from "../../src/core";
import {
  arange,
  dropoutGrad,
  eye,
  GradTensor,
  histogram,
  parameter,
  tensor,
} from "../../src/ndarray";

describe("slice range objects", () => {
  it("throws on an unknown key and hints at end for stop", () => {
    const x = arange(10);
    expect(() => x.slice({ start: 3, stop: 5 } as never)).toThrow(InvalidParameterError);
    expect(() => x.slice({ start: 3, stop: 5 } as never)).toThrow(/"stop".*"end"/);
    expect(() => x.slice({ begin: 1 } as never)).toThrow(/unknown key "begin"/);
  });

  it("still accepts start, end and step", () => {
    expect(arange(10).slice({ start: 3, end: 5 }).toArray()).toEqual([3, 4]);
    expect(arange(10).slice({ step: 3 }).toArray()).toEqual([0, 3, 6, 9]);
    expect(arange(10).slice({}).size).toBe(10);
  });
});

describe("histogram bins", () => {
  it("rejects bins that are neither a number nor an array", () => {
    const x = tensor([1, 2]);
    expect(() => histogram(x, { bins: 3 } as never)).toThrow(InvalidParameterError);
    expect(() => histogram(x, "auto" as never)).toThrow(InvalidParameterError);
    expect(() => histogram(x, null as never)).toThrow(InvalidParameterError);
  });

  it("accepts a bin count and a list of edges", () => {
    // numpy.histogram([1, 2], bins=2) -> ([1, 1], [1, 1.5, 2])
    expect(histogram(tensor([1, 2]), 2).counts.toArray()).toEqual([1, 1]);
    expect(histogram(tensor([1, 2]), [0, 1.5, 3]).counts.toArray()).toEqual([1, 1]);
  });
});

describe("dropout with p = 1", () => {
  // torch.nn.functional.dropout(x, p=1.0) returns zeros (the survivors would be scaled by
  // 1 / (1 - p) = inf, so torch multiplies by zero instead).
  it("returns zeros and a zero gradient", () => {
    const x = parameter([1, 2, 3]);
    const y = dropoutGrad(x, 1);
    expect(y.tensor.toArray()).toEqual([0, 0, 0]);
    y.sum().backward();
    expect(x.grad?.toArray()).toEqual([0, 0, 0]);
  });

  it("still rejects p outside [0, 1]", () => {
    const x = parameter([1, 2, 3]);
    expect(() => dropoutGrad(x, 1.0001)).toThrow(InvalidParameterError);
    expect(() => dropoutGrad(x, -0.1)).toThrow(InvalidParameterError);
    expect(() => dropoutGrad(x, Number.NaN)).toThrow(InvalidParameterError);
  });
});

describe("dropoutMask", () => {
  it("is not part of the public ndarray surface", async () => {
    const api = await import("../../src/ndarray");
    expect("dropoutMask" in api).toBe(false);
  });
});

describe("eye with an options object", () => {
  it("accepts options as the second argument", () => {
    const i = eye(3, { dtype: "float64" });
    expect(i.dtype).toBe("float64");
    expect(i.shape).toEqual([3, 3]);
    expect(i.toArray()).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });

  it("keeps the (n, m, k, options) form", () => {
    expect(eye(2, 3, 1, { dtype: "float64" }).toArray()).toEqual([
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(eye(2).shape).toEqual([2, 2]);
  });
});

describe("autograd on integer dtypes", () => {
  // torch: "Only Tensors of floating point dtype can require gradients"
  it("rejects requiresGrad for an integer or bool tensor", () => {
    for (const dtype of ["int32", "int64", "uint8", "bool"] as const) {
      const t = tensor([1, 0], { dtype });
      expect(() => GradTensor.fromTensor(t, { requiresGrad: true })).toThrow(DTypeError);
    }
    expect(() => new GradTensor([1, 2], { dtype: "int32", requiresGrad: true })).toThrow(
      DTypeError
    );
    expect(() => GradTensor.scalar(1, { dtype: "int32", requiresGrad: true })).toThrow(DTypeError);
    const leaf = GradTensor.fromTensor(tensor([1, 2], { dtype: "int32" }));
    expect(() => leaf.setRequiresGrad(true)).toThrow(DTypeError);
  });

  it("accepts integer tensors without requiresGrad and float tensors with it", () => {
    expect(GradTensor.fromTensor(tensor([1, 2], { dtype: "int32" })).requiresGrad).toBe(false);
    expect(GradTensor.fromTensor(tensor([1, 2]), { requiresGrad: true }).requiresGrad).toBe(true);
    expect(
      GradTensor.fromTensor(tensor([1, 2], { dtype: "float16" }), { requiresGrad: true })
    ).toBeDefined();
  });
});
