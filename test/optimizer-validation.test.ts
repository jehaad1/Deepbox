import { describe, expect, it } from "vitest";
import { DataValidationError } from "../src/core";
import { parameter, tensor } from "../src/ndarray";
import { SGD } from "../src/optim";

describe("deepbox/optim - validation", () => {
  it("should throw DataValidationError when loading state with invalid option types", () => {
    const w = parameter(tensor([1, 2, 3]));
    const optimizer = new SGD([w], { lr: 0.1 });

    // Create a malformed state dict with string instead of number for lr
    const stateDict = {
      paramGroups: [
        {
          params: [w],
          options: {
            lr: "invalid_string_value", // Should be number
            momentum: 0,
            dampening: 0,
            weightDecay: 0,
            nesterov: false,
          },
        },
      ],
    };

    expect(() => {
      optimizer.loadStateDict(stateDict);
    }).toThrow(DataValidationError);

    expect(() => {
      optimizer.loadStateDict(stateDict);
    }).toThrow(/Type mismatch for option 'lr'/);
  });

  it("should throw DataValidationError when loading state with invalid option types (boolean vs number)", () => {
    const w = parameter(tensor([1, 2, 3]));
    const optimizer = new SGD([w], { lr: 0.1 });

    const stateDict = {
      paramGroups: [
        {
          params: [w],
          options: {
            lr: 0.1,
            momentum: 0,
            dampening: 0,
            weightDecay: 0,
            nesterov: 123, // Should be boolean, passing number
          },
        },
      ],
    };

    expect(() => {
      optimizer.loadStateDict(stateDict);
    }).toThrow(DataValidationError);

    expect(() => {
      optimizer.loadStateDict(stateDict);
    }).toThrow(/Type mismatch for option 'nesterov'/);
  });
});
