import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { GradTensor, noGrad, parameter } from "../src/ndarray/autograd";
import { Tensor } from "../src/ndarray/tensor";

describe("Autograd - branch coverage extras", () => {
  it("disables grad tracking inside noGrad", () => {
    const out = noGrad(() => parameter([1, 2, 3]));
    expect(out.requiresGrad).toBe(false);
  });

  it("throws when creating a GradTensor for string dtype", () => {
    const t = Tensor.fromStringArray({
      data: ["a", "b"],
      shape: [2],
      device: "cpu",
    });
    expect(() => GradTensor.fromTensor(t, { requiresGrad: true })).toThrow(/string/);
  });

  it("setGrad validates requiresGrad and shape", () => {
    const a = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: false });
    expect(() => a.setGrad(tensor([1, 2, 3]))).toThrow(/requiresGrad/);

    const b = parameter([1, 2, 3]);
    expect(() => b.setGrad(tensor([1, 2]))).toThrow(/shape/i);
  });

  it("zeroGrad is a no-op when requiresGrad=false", () => {
    const a = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: false });
    expect(() => a.zeroGrad()).not.toThrow();
    expect(a.grad).toBeNull();
  });

  it("accumulates gradients when the same tensor appears twice", () => {
    const x = parameter([1, 2, 3]);
    const y = x.add(x).sum();
    y.backward();
    const grad = x.grad;
    expect(grad).not.toBeNull();
    if (grad) {
      const g0 = Number(grad.data[grad.offset]);
      const g1 = Number(grad.data[grad.offset + 1]);
      const g2 = Number(grad.data[grad.offset + 2]);
      expect(g0).toBe(2);
      expect(g1).toBe(2);
      expect(g2).toBe(2);
    }
  });

  it("sum backward throws on invalid axis", () => {
    const x = parameter([
      [1, 2],
      [3, 4],
    ]);
    expect(() => x.sum(5)).toThrow(/axis/);
  });

  it("sum backward with keepdims preserves shape", () => {
    const x = parameter([
      [1, 2],
      [3, 4],
    ]);
    const y = x.sum(1, true);
    y.backward();
    expect(x.grad?.shape).toEqual([2, 2]);
  });

  it("slice backward with step scatters gradient correctly", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    // step=2 selects columns 0 and 2 -> [[1,3],[4,6]]
    const s = x.slice({ start: 0, end: 2 }, { start: 0, end: 3, step: 2 });
    s.sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 3]);
    if (g === null) return;
    // Gradient is 1 at columns 0,2 and 0 at column 1
    const data = g.data as { readonly [n: number]: unknown };
    expect(Number(data[g.offset])).toBe(1); // [0,0]
    expect(Number(data[g.offset + 1])).toBe(0); // [0,1]
    expect(Number(data[g.offset + 2])).toBe(1); // [0,2]
    expect(Number(data[g.offset + 3])).toBe(1); // [1,0]
    expect(Number(data[g.offset + 4])).toBe(0); // [1,1]
    expect(Number(data[g.offset + 5])).toBe(1); // [1,2]
  });

  it("gather backward accumulates for duplicate indices", () => {
    const embeddings = parameter([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices = GradTensor.fromTensor(tensor([0, 2, 2]));
    // Gather rows 0, 2, 2 -> [[1,2],[5,6],[5,6]]
    const gathered = embeddings.gather(indices, 0);
    gathered.sum().backward();

    const g = embeddings.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([3, 2]);
    if (g === null) return;
    // Row 0: selected once -> grad 1; Row 1: not selected -> grad 0; Row 2: selected twice -> grad 2
    const data = g.data as { readonly [n: number]: unknown };
    expect(Number(data[g.offset])).toBe(1); // row 0 col 0
    expect(Number(data[g.offset + 1])).toBe(1); // row 0 col 1
    expect(Number(data[g.offset + 2])).toBe(0); // row 1 col 0
    expect(Number(data[g.offset + 3])).toBe(0); // row 1 col 1
    expect(Number(data[g.offset + 4])).toBe(2); // row 2 col 0
    expect(Number(data[g.offset + 5])).toBe(2); // row 2 col 1
  });

  it("transpose backward uses inverse permutation", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const t = x.transpose([1, 0]).sum();
    t.backward();
    expect(x.grad?.shape).toEqual([2, 3]);
  });

  it("pow throws for bigint tensors, relu handles bigint without backward", () => {
    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const g = GradTensor.fromTensor(big, { requiresGrad: true });
    expect(() => g.pow(2)).toThrow("pow() backward is not supported for int64 tensors");
    const r = g.relu();
    expect(r.tensor.dtype).toBe("int64");
  });

  it("detach disables gradients and setRequiresGrad clears grad", () => {
    const x = parameter([1, 2, 3]);
    x.sum().backward();
    expect(x.grad).not.toBeNull();
    x.setRequiresGrad(false);
    expect(x.grad).toBeNull();
    const d = x.detach();
    expect(d.requiresGrad).toBe(false);
  });

  it("backward respects provided gradient", () => {
    const x = parameter([1, 2]);
    const y = x.sum();
    y.backward(tensor([2]));
    const g = x.grad;
    expect(g).not.toBeNull();
    if (g) {
      expect(Number(g.data[g.offset])).toBe(2);
      expect(Number(g.data[g.offset + 1])).toBe(2);
    }
  });

  it("reshape and view validate shapes and propagate gradients", () => {
    const x = parameter([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => x.reshape([5])).toThrow(/shape/i);

    const v = x.view([6]);
    v.sum().backward();

    const g = x.grad;
    expect(g).not.toBeNull();
    expect(g?.shape).toEqual([2, 3]);
    if (g === null) return;
    // sum gradient is 1 everywhere, reshaped back to original shape
    const data = g.data as { readonly [n: number]: unknown };
    for (let i = 0; i < 6; i++) {
      expect(Number(data[g.offset + i])).toBe(1);
    }
  });

  it("gather throws on invalid axis and backward no-ops when requiresGrad=false", () => {
    const base = GradTensor.fromTensor(tensor([1, 2, 3]), {
      requiresGrad: false,
    });
    expect(() => base.gather(GradTensor.fromTensor(tensor([0])), 2)).toThrow(/axis/i);
    expect(() => base.backward()).not.toThrow();
  });
});
