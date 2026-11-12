import { afterEach, describe, expect, it, vi } from "vitest";
import { InvalidParameterError, type LogEntry, setLogHandler } from "../../src/core";
import { GradTensor, noGrad, randn, type Tensor } from "../../src/ndarray";
import { Dropout, Linear, mseLoss, Sequential, Trainer } from "../../src/nn";
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

describe("NN-17: Trainer restores the model mode and logs through the core logger", () => {
  const x = randn([8, 3]);
  const y = randn([8, 1]);

  afterEach(() => {
    setLogHandler(undefined);
    vi.restoreAllMocks();
  });

  function build(options: { verbose?: boolean } = {}): {
    model: Sequential;
    trainer: Trainer;
  } {
    const model = new Sequential(new Linear(3, 1), new Dropout(0.1));
    const trainer = new Trainer(
      model,
      new SGD(model.parameters(), { lr: 0.01 }),
      (out, target) => mseLoss(out, target),
      { epochs: 2, ...options }
    );
    return { model, trainer };
  }

  it("leaves a model that was in eval mode in eval mode, with or without validation data", () => {
    const { model, trainer } = build();
    model.eval();
    trainer.fit([[x, y]]);
    expect(model.training).toBe(false);
    trainer.fit([[x, y]], [[x, y]]);
    expect(model.training).toBe(false);
  });

  it("leaves a model that was in train mode in train mode", () => {
    const { model, trainer } = build();
    model.train();
    trainer.fit([[x, y]], [[x, y]]);
    expect(model.training).toBe(true);
    trainer.fit([[x, y]]);
    expect(model.training).toBe(true);
  });

  it("restores the mode with fitAsync and when the loop throws", async () => {
    const { model, trainer } = build();
    model.eval();
    await trainer.fitAsync([[x, y]], [[x, y]]);
    expect(model.training).toBe(false);
    expect(() => trainer.fit([])).toThrow(InvalidParameterError);
    expect(model.training).toBe(false);
  });

  it("sends verbose output to the core log handler", () => {
    const entries: LogEntry[] = [];
    setLogHandler((entry) => entries.push(entry));
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    build({ verbose: true }).trainer.fit([[x, y]], [[x, y]]);
    expect(log).not.toHaveBeenCalled();
    expect(entries).toHaveLength(2);
    expect(entries[0]?.source).toBe("Trainer");
    expect(entries[0]?.message).toMatch(/^Epoch 1\/2, train_loss: \d/);
    expect(entries[0]?.message).toMatch(/val_loss: \d/);
  });

  it("still prints to the console by default when verbose", () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    build({ verbose: true }).trainer.fit([[x, y]]);
    expect(log).toHaveBeenCalledTimes(2);
    expect(String(log.mock.calls[0]?.[0])).toContain("Epoch 1/2");
  });

  it("is silent when not verbose", () => {
    const entries: LogEntry[] = [];
    setLogHandler((entry) => entries.push(entry));
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    build().trainer.fit([[x, y]]);
    expect(entries).toHaveLength(0);
    expect(log).not.toHaveBeenCalled();
  });
});
