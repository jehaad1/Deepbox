import { describe, expect, it } from "vitest";
import type { AnyTensor, Tensor } from "../src/ndarray";
import { GradTensor, tensor } from "../src/ndarray";
import { Linear, Module, mseLoss, Trainer } from "../src/nn";

class TinyModel extends Module {
  private fc: Linear;
  constructor() {
    super();
    this.fc = new Linear(2, 1);
    this.registerModule("fc", this.fc);
  }
  forward(x: AnyTensor): AnyTensor {
    if (x instanceof GradTensor) return this.fc.forward(x);
    return this.fc.forward(x as Tensor);
  }
}

describe("Trainer", () => {
  it("runs a basic training loop and returns history", () => {
    const model = new TinyModel();
    const optimizer = {
      step() {},
      zeroGrad() {},
    };
    const lossFn = (output: AnyTensor, target: Tensor) => mseLoss(output as Tensor, target);

    const trainData: Array<readonly [Tensor, Tensor]> = [
      [tensor([[1, 2]]), tensor([3])],
      [tensor([[3, 4]]), tensor([7])],
    ];

    const trainer = new Trainer(model, optimizer, lossFn, { epochs: 3 });
    const result = trainer.fit(trainData);

    expect(result.history.length).toBe(3);
    expect(result.stoppedEarly).toBe(false);
    for (const info of result.history) {
      expect(info.epoch).toBeGreaterThan(0);
      expect(typeof info.trainLoss).toBe("number");
      expect(info.valLoss).toBeUndefined();
    }
  });

  it("supports validation data", () => {
    const model = new TinyModel();
    const optimizer = { step() {}, zeroGrad() {} };
    const lossFn = (output: AnyTensor, target: Tensor) => mseLoss(output as Tensor, target);

    const trainData: Array<readonly [Tensor, Tensor]> = [[tensor([[1, 2]]), tensor([3])]];
    const valData: Array<readonly [Tensor, Tensor]> = [[tensor([[2, 3]]), tensor([5])]];

    const trainer = new Trainer(model, optimizer, lossFn, { epochs: 2 });
    const result = trainer.fit(trainData, valData);

    expect(result.history.length).toBe(2);
    for (const info of result.history) {
      expect(info.valLoss).toBeDefined();
      expect(typeof info.valLoss).toBe("number");
    }
  });

  it("supports early stopping", () => {
    const model = new TinyModel();
    const optimizer = { step() {}, zeroGrad() {} };

    // Loss function that returns constant loss and never improves
    const lossFn = (_output: AnyTensor, _target: Tensor) => tensor([5.0]);

    const trainData: Array<readonly [Tensor, Tensor]> = [[tensor([[1, 2]]), tensor([3])]];

    const trainer = new Trainer(model, optimizer, lossFn, {
      epochs: 100,
      earlyStopping: { patience: 3 },
    });
    const result = trainer.fit(trainData);

    expect(result.stoppedEarly).toBe(true);
    // Should stop after patience + 1 epochs (1 initial + 3 patience)
    expect(result.history.length).toBeLessThanOrEqual(5);
  });

  it("invokes callbacks after each epoch", () => {
    const model = new TinyModel();
    const optimizer = { step() {}, zeroGrad() {} };
    const lossFn = (_output: AnyTensor, _target: Tensor) => tensor([1.0]);

    const trainData: Array<readonly [Tensor, Tensor]> = [[tensor([[1, 2]]), tensor([3])]];

    const epochsLogged: number[] = [];
    const trainer = new Trainer(model, optimizer, lossFn, {
      epochs: 3,
      callbacks: [(info) => epochsLogged.push(info.epoch)],
    });
    trainer.fit(trainData);

    expect(epochsLogged).toEqual([1, 2, 3]);
  });

  it("validates epochs parameter", () => {
    const model = new TinyModel();
    const optimizer = { step() {}, zeroGrad() {} };
    const lossFn = (_output: AnyTensor, _target: Tensor) => tensor([1.0]);

    expect(() => new Trainer(model, optimizer, lossFn, { epochs: 0 })).toThrow(/epochs/);
  });
});
