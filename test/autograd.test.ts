import { describe, expect, it } from "vitest";
import { GradTensor, noGrad, parameter, reshape, tensor } from "../src/ndarray";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Autograd", () => {
  // Regression: axis-reduction backward was rewritten allocation-free (reused
  // coordinate odometer + incremental go offset) to remove two per-element JS
  // array allocations on the softmax/cross-entropy/layernorm/attention path.
  // Verify the gradients still match finite differences for every axis and
  // both keepdims settings.
  it("sum-along-axis backward matches finite differences (all axes, keepdims)", () => {
    const mk = (n: number, seed: number): number[] => {
      const out: number[] = [];
      let s = seed;
      for (let i = 0; i < n; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        out.push((s % 1000) / 100 - 5);
      }
      return out;
    };
    const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
    const mkT = (flat: number[], shape: number[]) => {
      const t = tensor(flat, { dtype: "float64" });
      return shape.length <= 1 ? t : reshape(t, shape);
    };
    const lossOf = (
      xFlat: number[],
      shape: number[],
      axis: number,
      keepdims: boolean,
      coefFlat: number[],
      cs: number[]
    ): number => {
      const x = GradTensor.fromTensor(mkT(xFlat, shape), { requiresGrad: true });
      const coef = GradTensor.fromTensor(
        cs.length ? mkT(coefFlat, cs) : tensor(coefFlat[0] as number, { dtype: "float64" }),
        { requiresGrad: false }
      );
      const loss = x.sum(axis, keepdims).mul(coef).sum();
      return numData(loss.tensor)[0] as number;
    };

    const cases: Array<[number[], number, boolean]> = [
      [[3, 4], 0, false],
      [[3, 4], 1, false],
      [[3, 4], 0, true],
      [[3, 4], 1, true],
      [[2, 3, 4], 0, false],
      [[2, 3, 4], 1, false],
      [[2, 3, 4], 2, false],
      [[2, 3, 4], 1, true],
      [[2, 3, 4], 2, true],
    ];
    for (const [shape, axis, keepdims] of cases) {
      const nx = numel(shape);
      const xFlat = mk(nx, 7 + axis + shape.length * 13);
      const cs = keepdims
        ? shape.map((d, i) => (i === axis ? 1 : d))
        : shape.filter((_, i) => i !== axis);
      const coefFlat = mk(cs.length ? numel(cs) : 1, 99);
      const x = GradTensor.fromTensor(mkT(xFlat, shape), { requiresGrad: true });
      const coef = GradTensor.fromTensor(
        cs.length ? mkT(coefFlat, cs) : tensor(coefFlat[0] as number, { dtype: "float64" }),
        { requiresGrad: false }
      );
      x.sum(axis, keepdims).mul(coef).sum().backward();
      const g = numData(x.grad as NonNullable<typeof x.grad>);
      const eps = 1e-5;
      for (let i = 0; i < nx; i++) {
        const plus = xFlat.slice();
        plus[i] = (plus[i] as number) + eps;
        const minus = xFlat.slice();
        minus[i] = (minus[i] as number) - eps;
        const fd =
          (lossOf(plus, shape, axis, keepdims, coefFlat, cs) -
            lossOf(minus, shape, axis, keepdims, coefFlat, cs)) /
          (2 * eps);
        expect(Math.abs(fd - (g[i] as number))).toBeLessThan(1e-6);
      }
    }
  });

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

  it("should backprop through a very deep graph without stack overflow", () => {
    // Regression: backward() built its topological order with recursion, so a
    // deep/unrolled graph (long RNN, deep residual stack) overflowed V8's call
    // stack ("Maximum call stack size exceeded"). The traversal is now
    // iterative and must handle arbitrary depth.
    const x = GradTensor.fromTensor(tensor([1], { dtype: "float64" }), {
      requiresGrad: true,
    });
    let y = x;
    const depth = 50000;
    for (let i = 0; i < depth; i++) {
      y = y.add(x);
    }
    expect(() => y.backward()).not.toThrow();
    // y = (depth + 1) * x  =>  dy/dx = depth + 1
    expect(numData(x.grad as NonNullable<typeof x.grad>)).toEqual([depth + 1]);
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
