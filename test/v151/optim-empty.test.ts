// NN-16: an optimizer given no parameters throws, as torch.optim.SGD([]) raises
// "optimizer got an empty parameter list". A parameter group with an empty params list is
// still accepted inside a non-empty list of groups, as in torch.
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { parameter, tensor } from "../../src/ndarray";
import { Linear, ReLU, Sequential } from "../../src/nn";
import { Adam, AdamW, RMSprop, SGD } from "../../src/optim";

describe("NN-16: optimizers reject an empty parameter list", () => {
  const makers: ReadonlyArray<readonly [string, (params: never) => unknown]> = [
    ["SGD", (p) => new SGD(p, { lr: 0.1 })],
    ["Adam", (p) => new Adam(p, { lr: 0.1 })],
    ["AdamW", (p) => new AdamW(p, { lr: 0.1 })],
    ["RMSprop", (p) => new RMSprop(p, { lr: 0.1 })],
  ];

  it.each(makers)("%s throws for an empty array, set and generator", (_name, make) => {
    expect(() => make([] as never)).toThrow(InvalidParameterError);
    expect(() => make(new Set() as never)).toThrow(InvalidParameterError);
    expect(() => make(new Sequential(new ReLU()).parameters() as never)).toThrow(
      /empty parameter list/
    );
  });

  it("accepts parameters and groups", () => {
    const w = parameter(tensor([1, 2]));
    expect(() => new SGD([w], { lr: 0.1 })).not.toThrow();
    expect(() => new SGD(new Linear(2, 2).parameters(), { lr: 0.1 })).not.toThrow();
    expect(() => new SGD([{ params: [w] }, { params: [], lr: 0.2 }], { lr: 0.1 })).not.toThrow();
  });

  it("rejects an empty list of groups", () => {
    expect(() => new SGD([] as never, { lr: 0.1 })).toThrow(InvalidParameterError);
  });
});
