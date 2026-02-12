import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { SGD } from "../src/optim";

describe("deepbox/optim - state serialization", () => {
  it("should round-trip optimizer state with parameter ids", () => {
    const w = parameter(tensor([1, 2, 3]));
    const optimizer = new SGD([w], { lr: 0.1, momentum: 0.9 });

    // Simulate gradients
    w.setGrad(tensor([0.1, 0.2, 0.3]));
    optimizer.step();

    const state = optimizer.stateDict();
    const optimizer2 = new SGD([w], { lr: 0.1, momentum: 0.9 });
    optimizer2.loadStateDict(state);

    const state2 = optimizer2.stateDict();
    expect(state2).toEqual(state);
  });

  it("should load empty optimizer state without errors", () => {
    const optimizer = new SGD([], { lr: 0.1 });
    const state = optimizer.stateDict();
    const optimizer2 = new SGD([], { lr: 0.1 });
    expect(() => optimizer2.loadStateDict(state)).not.toThrow();
    expect(optimizer2.stateDict()).toEqual(state);
  });
});
