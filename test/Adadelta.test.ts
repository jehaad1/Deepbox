import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { AdaDelta } from "../src/optim/optimizers/adadelta";
import { getTensorData } from "./optim-test-helpers";

describe("AdaDelta optimizer", () => {
  it("skips parameters without gradients", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    const opt = new AdaDelta([p]);
    const before = Array.from(getTensorData(p, "AdaDelta param"));
    // A parameter whose grad is null is skipped (PyTorch semantics), not an error.
    expect(() => opt.step()).not.toThrow();
    expect(Array.from(getTensorData(p, "AdaDelta param"))).toEqual(before);
  });

  it("applies weight decay when configured", () => {
    const p = parameter(tensor([1, -2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.25], { dtype: "float64" }));
    const opt = new AdaDelta([p], { weightDecay: 0.1, lr: 1.0 });
    const before = Array.from(getTensorData(p, "AdaDelta param"));
    opt.step();
    const after = Array.from(getTensorData(p, "AdaDelta param"));
    expect(after[0]).not.toBe(before[0]);
    expect(after[1]).not.toBe(before[1]);
  });
});
