// NN-7: binaryCrossEntropyLoss rejects probabilities outside [0, 1] and clamps the logarithms
// at -100 like torch.nn.functional.binary_cross_entropy (torch 2.15, float64). Gradients are
// the torch formula (p - t) / max(p * (1 - p), 1e-12), checked against torch.autograd.
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { GradTensor, type Tensor, tensor } from "../../src/ndarray";
import { binaryCrossEntropyLoss } from "../../src/nn/losses";

const REF = JSON.parse(
  `{"p":[0.0,1.0,0.3,0.7,0.0,1.0,1e-12,0.999999999999],"t":[1.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0],"cases":[{"reduction":"none","loss":[100.0,100.0,1.203972804326,1.203972804326,0.0,-0.0,27.63102111593,27.63104323789],"grad":[-1000000003996.0,2000000007992.0,-10.0,13.33333333333,0.0,0.0,-7000000000000.0,8000000031958.0]},{"reduction":"sum","loss":[257.6700099625],"grad":[-1000000003996.0,1000000003996.0,-3.333333333333,3.333333333333,0.0,0.0,-1000000000000.0,1000000003995.0]},{"reduction":"mean","loss":[32.20875124531],"grad":[-125000000499.5,125000000499.5,-0.4166666666667,0.4166666666667,0.0,0.0,-125000000000.0,125000000499.4]}]}`
) as {
  p: number[];
  t: number[];
  cases: Array<{ reduction: "none" | "sum" | "mean"; loss: number[]; grad: number[] }>;
};

function values(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number>).slice(t.offset, t.offset + t.size);
}

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-8): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const a = actual[i] as number;
    expect(Math.abs(a - e), `index ${i}: ${a} vs ${e}`).toBeLessThanOrEqual(
      tol * Math.max(1, Math.abs(e))
    );
  }
}

describe("NN-7: binaryCrossEntropyLoss", () => {
  const preds = tensor(REF.p, { dtype: "float64" });
  const targets = tensor(REF.t, { dtype: "float64" });

  it.each(REF.cases.map((c) => [c.reduction, c] as const))("%s matches torch", (_name, c) => {
    expectClose(values(binaryCrossEntropyLoss(preds, targets, c.reduction)), c.loss);

    const x = GradTensor.fromTensor(preds, { requiresGrad: true });
    const loss = binaryCrossEntropyLoss(x, targets, c.reduction);
    expectClose(values(loss.tensor), c.loss);
    if (loss.size === 1) loss.backward();
    else
      loss.backward(
        tensor(
          Array.from({ length: loss.size }, (_, i) => i + 1),
          { dtype: "float64" }
        )
      );
    expectClose(values(x.grad as Tensor), c.grad);
  });

  it("throws for predictions outside [0, 1]", () => {
    for (const bad of [1.2, -0.1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const p = tensor([0.5, bad], { dtype: "float64" });
      const t = tensor([1, 0], { dtype: "float64" });
      expect(() => binaryCrossEntropyLoss(p, t)).toThrow(InvalidParameterError);
      expect(() =>
        binaryCrossEntropyLoss(GradTensor.fromTensor(p, { requiresGrad: true }), t)
      ).toThrow(InvalidParameterError);
    }
  });

  it("gives a finite loss of 100 per element for a fully wrong hard prediction", () => {
    const loss = binaryCrossEntropyLoss(
      tensor([0, 1], { dtype: "float64" }),
      tensor([1, 0], { dtype: "float64" })
    );
    expect(values(loss)[0]).toBeCloseTo(100, 12);
  });
});
