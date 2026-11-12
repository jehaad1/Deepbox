// NN-4, NN-14, NN-15, NN-20: Module registration names, train(mode) validation,
// loadStateDict input validation and non-persistent buffers (torch register_buffer(...,
// persistent=False): the buffer is a module buffer but not part of the state dict).
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { parameter, type Tensor, tensor } from "../../src/ndarray";
import { Linear, Module, ModuleDict, ParameterDict, PositionalEncoding } from "../../src/nn";

class Probe extends Module {
  forward(x: Tensor): Tensor {
    return x;
  }

  addParameter(name: string): void {
    this.registerParameter(name, parameter(tensor([1, 2, 3])));
  }

  addModule(name: string): void {
    this.registerModule(name, new Linear(1, 1));
  }

  addBuffer(name: string, persistent?: boolean): void {
    if (persistent === undefined) this.registerBuffer(name, tensor([1, 2]));
    else this.registerBuffer(name, tensor([1, 2]), { persistent });
  }
}

describe("NN-4: reserved registration names", () => {
  it.each(["__proto__", "constructor", "prototype"])("rejects %s", (name) => {
    const probe = new Probe();
    expect(() => probe.addParameter(name)).toThrow(InvalidParameterError);
    expect(() => probe.addModule(name)).toThrow(InvalidParameterError);
    expect(() => probe.addBuffer(name)).toThrow(InvalidParameterError);
    expect(() => new ParameterDict().set(name, parameter(tensor([1])))).toThrow(
      InvalidParameterError
    );
    expect(() => new ModuleDict().set(name, new Linear(1, 1))).toThrow(InvalidParameterError);
  });

  it("keeps the state dict free of prototype keys", () => {
    const probe = new Probe();
    probe.addParameter("weight");
    const state = probe.stateDict();
    expect(Object.getPrototypeOf(state.parameters)).toBe(Object.prototype);
    expect(Object.keys(state.parameters)).toEqual(["weight"]);
  });
});

describe("NN-14: train(mode) validates its argument", () => {
  it("accepts booleans and rejects everything else", () => {
    const layer = new Linear(2, 2);
    expect(layer.train(false).training).toBe(false);
    expect(layer.train(true).training).toBe(true);
    expect(layer.train().training).toBe(true);
    for (const bad of [0, 1, "false", null, {}]) {
      expect(() => layer.train(bad as unknown as boolean)).toThrow(InvalidParameterError);
    }
    expect(layer.training).toBe(true);
  });
});

describe("NN-15: loadStateDict validates its argument", () => {
  it("rejects null, primitives and arrays", () => {
    const layer = new Linear(2, 2);
    for (const bad of [null, undefined, 3, "state", [], true]) {
      expect(() => layer.loadStateDict(bad as unknown as never)).toThrow(InvalidParameterError);
    }
  });

  it("rejects a parameters or buffers section that is not an object", () => {
    const layer = new Linear(2, 2);
    expect(() => layer.loadStateDict({ parameters: 5 } as unknown as never)).toThrow(
      InvalidParameterError
    );
    expect(() => layer.loadStateDict({ buffers: [] } as unknown as never)).toThrow(
      InvalidParameterError
    );
  });

  it("still round-trips a valid state dict", () => {
    const a = new Linear(2, 2);
    const b = new Linear(2, 2);
    b.loadStateDict(a.stateDict());
    expect(b.stateDict().parameters["weight"]?.data).toEqual(
      a.stateDict().parameters["weight"]?.data
    );
  });
});

describe("NN-20: non-persistent buffers", () => {
  it("leaves a non-persistent buffer out of the state dict but keeps it a buffer", () => {
    const probe = new Probe();
    probe.addBuffer("kept");
    probe.addBuffer("transient", false);
    expect(Object.keys(probe.stateDict().buffers)).toEqual(["kept"]);
    expect(
      Array.from(probe.namedBuffers())
        .map(([name]) => name)
        .sort()
    ).toEqual(["kept", "transient"]);
    const other = new Probe();
    other.addBuffer("kept");
    other.addBuffer("transient", false);
    expect(() => other.loadStateDict(probe.stateDict())).not.toThrow();
  });

  it("ignores an entry for a non-persistent buffer when loading", () => {
    const probe = new Probe();
    probe.addBuffer("transient", false);
    const state = probe.stateDict();
    state.buffers["transient"] = { data: [9, 9], shape: [2], dtype: "float32" };
    expect(() => probe.loadStateDict(state)).not.toThrow();
  });

  it("registers PositionalEncoding.pe as non-persistent", () => {
    const encoding = new PositionalEncoding(8, { maxLen: 16 });
    expect(Object.keys(encoding.stateDict().buffers)).toEqual([]);
    expect(Array.from(encoding.namedBuffers()).map(([name]) => name)).toEqual(["pe"]);
    const legacy = encoding.stateDict();
    legacy.buffers["pe"] = { data: new Array(16 * 8).fill(0), shape: [16, 8], dtype: "float64" };
    expect(() => new PositionalEncoding(8, { maxLen: 16 }).loadStateDict(legacy)).not.toThrow();
  });
});
