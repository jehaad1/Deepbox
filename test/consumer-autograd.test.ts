import { describe, expect, it } from "vitest";
import { GradTensor, noGrad, parameter, tensor } from "../src/ndarray";

describe("consumer API: autograd", () => {
  it("parameter() creates GradTensor with requiresGrad=true", () => {
    const x = parameter([2, 3]);
    expect(x).toBeInstanceOf(GradTensor);
    expect(x.requiresGrad).toBe(true);
  });

  it("computes gradient for f(x) = x^2", () => {
    const x = parameter([2, 3]);
    const y = x.mul(x).sum();
    y.backward();
    expect(x.grad).not.toBeNull();
    expect(x.grad?.at(0)).toBe(4);
    expect(x.grad?.at(1)).toBe(6);
  });

  it("zeroGrad resets gradients to zero tensor", () => {
    const x = parameter([2, 3]);
    x.mul(x).sum().backward();
    x.zeroGrad();
    expect(x.grad).not.toBeNull();
    expect(x.grad?.at(0)).toBe(0);
  });

  it("supports multi-variable gradients", () => {
    const a = parameter([
      [1, 2],
      [3, 4],
    ]);
    const w = parameter([[0.5], [0.5]]);
    const z = a.matmul(w).sum();
    z.backward();
    expect(a.grad).not.toBeNull();
    expect(w.grad).not.toBeNull();
  });

  it("noGrad disables gradient tracking", () => {
    const q = parameter([1, 2, 3]);
    const result = noGrad(() => {
      const r = q.mul(q);
      expect(r.requiresGrad).toBe(false);
      return r;
    });
    expect(result.requiresGrad).toBe(false);
  });

  it("supports chained operations with backward", () => {
    const p = parameter([1, 2, 3, 4]);
    const scaled = p.mul(GradTensor.fromTensor(tensor([2, 2, 2, 2]), { requiresGrad: false }));
    const activated = scaled.relu();
    const loss = activated.sum();
    loss.backward();
    expect(p.grad).not.toBeNull();
  });

  it("GradTensor.fromTensor static factory works", () => {
    const gt = GradTensor.fromTensor(tensor([1, 2, 3]));
    expect(gt).toBeInstanceOf(GradTensor);
    expect(gt.requiresGrad).toBe(false);

    const gtGrad = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: true });
    expect(gtGrad.requiresGrad).toBe(true);
  });

  it("supports add, sub, sigmoid, tanh, detach", () => {
    const g1 = parameter([1, 2]);
    const g2 = parameter([3, 4]);
    expect(g1.add(g2).tensor.at(0)).toBe(4);
    expect(g1.sub(g2).tensor.at(0)).toBe(-2);
    expect(Math.abs(Number(parameter([0]).sigmoid().tensor.at(0)) - 0.5)).toBeLessThan(0.01);
    expect(Math.abs(Number(parameter([0]).tanh().tensor.at(0)))).toBeLessThan(0.01);

    const detached = g1.detach();
    expect(detached).toBeInstanceOf(GradTensor);
  });
});
