import { describe, expect, it } from "vitest";
import { GradTensor, noGrad, parameter, tensor } from "../src/ndarray";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Autograd", () => {
  it("should compute gradients for y = sum(x * x)", () => {
    const x = GradTensor.fromTensor(tensor([2, 3], { dtype: "float64" }), {
      requiresGrad: true,
    });

    const y = x.mul(x).sum();
    y.backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2]);
    if (g === null) {
      return;
    }
    expect(numData(g)).toEqual([4, 6]);
  });

  it("should backprop through add and mul", () => {
    const x = GradTensor.fromTensor(tensor([2, 3], { dtype: "float64" }), {
      requiresGrad: true,
    });

    // y = sum((x + x) * x) = sum(2x^2)
    const y = x.add(x).mul(x).sum();
    y.backward();

    // dy/dx = 4x
    const g = x.grad;
    expect(g).not.toBeNull();
    if (g === null) {
      return;
    }
    expect(numData(g)).toEqual([8, 12]);
  });

  it("should not build a graph inside noGrad", () => {
    const x = GradTensor.fromTensor(tensor([2, 3], { dtype: "float64" }), {
      requiresGrad: true,
    });

    const y = noGrad(() => x.mul(x));
    expect(y.requiresGrad).toBe(false);

    // Should be a no-op
    y.backward();
    expect(x.grad).toBeNull();
  });

  it("should backprop through sum(axis)", () => {
    const x = GradTensor.fromTensor(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float64" }
      ),
      { requiresGrad: true }
    );

    // y shape [2] when axis=0
    const y = x.sum(0);
    const z = y.sum();
    z.backward();

    // dz/dx = 1 everywhere
    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 2]);
    if (g === null) {
      return;
    }
    expect(numData(g)).toEqual([1, 1, 1, 1]);
  });

  it("should backprop through slice", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    // Slice first two rows, columns 1 and 2 -> [[2,3],[5,6]]
    const sliced = x.slice({ start: 0, end: 2 }, { start: 1, end: 3 });
    sliced.sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([3, 3]);
    if (g === null) return;
    // Gradient is 1 for selected positions, 0 elsewhere
    expect(numData(g)).toEqual([0, 1, 1, 0, 1, 1, 0, 0, 0]);
  });

  it("should backprop through slice with single index", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Select second row (index 1) -> [4, 5, 6]
    const row = x.slice(1);
    // y = sum(row^2) = 16 + 25 + 36 = 77
    row.mul(row).sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 3]);
    if (g === null) return;
    // d(x^2)/dx = 2x for row 1, 0 for row 0
    expect(numData(g)).toEqual([0, 0, 0, 8, 10, 12]);
  });

  it("should backprop through gather", () => {
    const x = parameter([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices = GradTensor.fromTensor(tensor([0, 2, 1], { dtype: "int32" }));

    // Gather rows 0, 2, 1 -> [[1,2],[5,6],[3,4]]
    const gathered = x.gather(indices, 0);
    gathered.sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([3, 2]);
    if (g === null) return;
    // Each row selected once, gradient is 1 at each selected position
    expect(numData(g)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("should accumulate gradients for gather with duplicate indices", () => {
    const x = parameter([
      [1, 2],
      [3, 4],
    ]);
    const indices = GradTensor.fromTensor(tensor([0, 0, 1], { dtype: "int32" }));

    // Gather rows [0, 0, 1] - row 0 selected twice
    const gathered = x.gather(indices, 0);
    gathered.sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 2]);
    if (g === null) return;
    // Row 0 gathered twice -> gradient 2; row 1 gathered once -> gradient 1
    expect(numData(g)).toEqual([2, 2, 1, 1]);
  });

  it("should backprop through transpose", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    const transposed = x.transpose();
    expect(transposed.tensor.shape).toEqual([3, 2]);
    const y = transposed.mul(transposed).sum();
    y.backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 3]);
    if (g === null) return;
    // d(x^2)/dx = 2x
    expect(numData(g)).toEqual([2, 4, 6, 8, 10, 12]);
  });

  it("should backprop through transpose with custom axes", () => {
    const t = tensor([1, 2, 3, 4, 5, 6, 7, 8], { dtype: "float64" }).reshape([2, 2, 2]);
    const x = GradTensor.fromTensor(t, { requiresGrad: true });

    // Transpose axes [0, 2, 1] -> shape [2, 2, 2]
    const transposed = x.transpose([0, 2, 1]);
    expect(transposed.tensor.shape).toEqual([2, 2, 2]);
    const y = transposed.sum();
    y.backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 2, 2]);
    if (g === null) return;
    expect(numData(g)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
  });
});
