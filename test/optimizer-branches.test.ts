import { describe, expect, it } from "vitest";
import { DataValidationError } from "../src/core";
import { parameter, tensor } from "../src/ndarray";
import { SGD } from "../src/optim";

describe("Optimizer - Branch Coverage", () => {
  it("covers serialization and parameter checks", () => {
    const param = parameter(tensor([1, 2, 3]));
    const opt = new SGD([param]);
    const state = opt.stateDict();
    expect(state.paramGroups.length).toBe(1);

    const opt2 = new SGD([parameter(tensor([1, 2, 3]))]);
    opt2.loadStateDict(state);
    expect(opt2.stateDict().paramGroups.length).toBe(1);
  });

  it("throws on invalid state dict", () => {
    const param = parameter(tensor([1, 2]));
    const opt = new SGD([param]);
    const state = opt.stateDict();

    const bad = { ...state, paramGroups: [] };
    expect(() => opt.loadStateDict(bad)).toThrow(DataValidationError);
  });

  it("handles param groups and backward-compatible state formats", () => {
    const p1 = parameter([1, 2]);
    const p2 = parameter([3, 4]);
    const opt = new SGD(
      [
        { params: [p1], lr: 0.1 },
        { params: [p2], lr: 0.01 },
      ],
      { lr: 0.5 }
    );

    const state = opt.stateDict();
    expect(state.paramGroups.length).toBe(2);

    const backwardState = {
      ...state,
      state: [
        {
          param: p1,
          state: { momentum: [0, 0] },
        },
      ],
      paramGroups: [
        { params: [p1], options: state.paramGroups[0]?.options },
        { params: [p2], options: state.paramGroups[1]?.options },
      ],
    };
    opt.loadStateDict(backwardState);
    expect(opt.stateDict().state.length).toBe(1);
  });

  it("throws on paramId or param count mismatches", () => {
    const p1 = parameter([1, 2]);
    const p2 = parameter([3, 4]);
    const opt = new SGD([p1, p2]);
    const state = opt.stateDict();

    const badIds = {
      ...state,
      paramGroups: [
        {
          paramIds: [999],
          options: state.paramGroups[0]?.options,
        },
      ],
    };
    expect(() => opt.loadStateDict(badIds)).toThrow(DataValidationError);

    const badState = {
      ...state,
      state: [
        {
          paramId: 123456,
          state: { foo: "bar" },
        },
      ],
    };
    expect(() => opt.loadStateDict(badState)).toThrow(DataValidationError);

    const badCount = {
      ...state,
      paramGroups: [
        {
          paramIds: (() => {
            const firstGroup = state.paramGroups[0];
            if (firstGroup === undefined || firstGroup.paramIds.length === 0) {
              throw new Error("Expected state to include at least one paramId.");
            }
            return [firstGroup.paramIds[0]];
          })(),
          options: state.paramGroups[0]?.options,
        },
      ],
    };
    expect(() => opt.loadStateDict(badCount)).toThrow(DataValidationError);
  });

  it("covers addParamGroup and paramId consistency checks", () => {
    const p1 = parameter([1, 2]);
    const p2 = parameter([3, 4]);
    const opt = new SGD([p1]);
    opt.addParamGroup({ params: [p2], lr: 0.05 });

    const state = opt.stateDict();
    const groups = state.paramGroups;
    expect(groups.length).toBe(2);

    const badParamIds = {
      ...state,
      paramGroups: [
        {
          paramIds: groups[1]?.paramIds ?? [],
          options: groups[0]?.options,
        },
        {
          paramIds: groups[1]?.paramIds ?? [],
          options: groups[1]?.options,
        },
      ],
    };
    expect(() => opt.loadStateDict(badParamIds)).toThrow(DataValidationError);
  });
});
