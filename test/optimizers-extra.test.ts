import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { AdaDelta } from "../src/optim/optimizers/adadelta";
import { Nadam } from "../src/optim/optimizers/nadam";
import { getTensorData } from "./optim-test-helpers";

describe("deepbox/optim - AdaDelta and Nadam", () => {
  it("AdaDelta updates parameters", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.1, -0.2], { dtype: "float64" }));
    const opt = new AdaDelta([p], { lr: 1.0, rho: 0.9, eps: 1e-6 });
    const before = Array.from(getTensorData(p, "AdaDelta param"));
    opt.step();
    const after = Array.from(getTensorData(p, "AdaDelta param"));
    expect(after[0]).not.toBe(before[0]);
    expect(after[1]).not.toBe(before[1]);
  });

  it("Nadam updates parameters with weight decay", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.2, -0.1], { dtype: "float64" }));
    const opt = new Nadam([p], { lr: 0.01, weightDecay: 0.1 });
    const before = Array.from(getTensorData(p, "Nadam param"));
    opt.step();
    const after = Array.from(getTensorData(p, "Nadam param"));
    expect(after[0]).not.toBe(before[0]);
    expect(after[1]).not.toBe(before[1]);
  });
});
