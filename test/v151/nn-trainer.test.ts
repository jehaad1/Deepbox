import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { GradTensor, noGrad, randn, type Tensor } from "../../src/ndarray";
import { Linear, mseLoss, Trainer } from "../../src/nn";
import { SGD } from "../../src/optim";

function asTensor(t: GradTensor | Tensor): Tensor {
  return GradTensor.isGradTensor(t) ? t.tensor : t;
}

describe("NN-1: Trainer rejects a loss that is not connected to the parameters", () => {
  const x = randn([8, 3]);
  const y = randn([8, 1]);

  it("throws when the loss is a plain Tensor", () => {
    const model = new Linear(3, 1);
    const trainer = new Trainer(
      model,
      new SGD(model.parameters(), { lr: 0.1 }),
      (out, target) => noGrad(() => mseLoss(asTensor(out), target)),
      { epochs: 1 }
    );
    expect(() => trainer.fit([[x, y]])).toThrow(InvalidParameterError);
    expect(() => trainer.fit([[x, y]])).toThrow(/GradTensor connected to the model parameters/);
  });

  it("throws when the loss is a GradTensor that does not require grad", () => {
    const model = new Linear(3, 1);
    const trainer = new Trainer(
      model,
      new SGD(model.parameters(), { lr: 0.1 }),
      (out, target) =>
        GradTensor.fromTensor(mseLoss(asTensor(out), target) as Tensor, { requiresGrad: false }),
      { epochs: 1 }
    );
    expect(() => trainer.fit([[x, y]])).toThrow(InvalidParameterError);
  });

  it("still trains with a connected loss", () => {
    const model = new Linear(3, 1);
    const trainer = new Trainer(
      model,
      new SGD(model.parameters(), { lr: 0.1 }),
      (out, target) => mseLoss(out, target),
      { epochs: 2 }
    );
    const result = trainer.fit([[x, y]]);
    expect(result.history).toHaveLength(2);
  });
});
