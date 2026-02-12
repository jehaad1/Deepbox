import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../src/core";
import { addScalar, GradTensor, mulScalar, type Tensor, tensor } from "../src/ndarray";
import { Linear, Module } from "../src/nn";

// Create a simple test module
class SimpleModel extends Module {
  private fc1: Linear;
  private fc2: Linear;

  constructor() {
    super();
    this.fc1 = new Linear(10, 5);
    this.fc2 = new Linear(5, 2);
    this.registerModule("fc1", this.fc1);
    this.registerModule("fc2", this.fc2);
  }

  forward(x: Tensor): Tensor {
    let out = this.fc1.forward(x);
    out = this.fc2.forward(out);
    return out;
  }
}

class BufferModel extends Module {
  readonly buffer: Tensor;

  constructor() {
    super();
    this.buffer = tensor([1, 2, 3]);
    this.registerBuffer("running", this.buffer);
  }

  forward(x: Tensor): Tensor {
    return x;
  }
}

class TypedBufferModel extends Module {
  readonly stringBuffer: Tensor;
  readonly int64Buffer: Tensor;

  constructor() {
    super();
    this.stringBuffer = tensor(["a", "b", "c"]);
    this.int64Buffer = tensor([1, 2, 3], { dtype: "int64" });
    this.registerBuffer("labels", this.stringBuffer);
    this.registerBuffer("counts", this.int64Buffer);
  }

  forward(x: Tensor): Tensor {
    return x;
  }
}

describe("deepbox/nn - Module", () => {
  describe("module registration", () => {
    it("should register child modules", () => {
      const model = new SimpleModel();
      const modules = Array.from(model.modules());

      // Should include self + 2 child modules
      expect(modules.length).toBe(3);
    });

    it("should iterate named modules", () => {
      const model = new SimpleModel();
      const namedModules = Array.from(model.namedModules());

      expect(namedModules.length).toBe(3);
      expect(namedModules.map(([name]) => name)).toContain("fc1");
      expect(namedModules.map(([name]) => name)).toContain("fc2");
    });

    it("should respect recurse=false for named modules", () => {
      const model = new SimpleModel();
      const namedModules = Array.from(model.namedModules("", false));
      expect(namedModules.length).toBe(1);
      expect(namedModules[0]?.[0]).toBe("");
    });
  });

  describe("parameter management", () => {
    it("should iterate all parameters", () => {
      const model = new SimpleModel();
      const params = Array.from(model.parameters());

      // fc1: weight + bias, fc2: weight + bias = 4 parameters
      expect(params.length).toBe(4);
    });

    it("should iterate named parameters", () => {
      const model = new SimpleModel();
      const namedParams = Array.from(model.namedParameters());

      expect(namedParams.length).toBe(4);
      const names = namedParams.map(([name]) => name);
      expect(names).toContain("fc1.weight");
      expect(names).toContain("fc1.bias");
      expect(names).toContain("fc2.weight");
      expect(names).toContain("fc2.bias");
    });

    it("should support recurse=false parameter iteration", () => {
      const model = new SimpleModel();
      const params = Array.from(model.parameters(false));
      const namedParams = Array.from(model.namedParameters("", false));

      expect(params.length).toBe(0);
      expect(namedParams.length).toBe(0);
    });

    it("should freeze/unfreeze selected parameters only", () => {
      const model = new SimpleModel();
      model.freezeParameters(["fc1.weight"], true);
      for (const [name, param] of model.namedParameters()) {
        if (name === "fc1.weight") {
          expect(param.requiresGrad).toBe(false);
        } else {
          expect(param.requiresGrad).toBe(true);
        }
      }
      model.unfreezeParameters(["fc1.weight"], true);
      for (const [, param] of model.namedParameters()) {
        expect(param.requiresGrad).toBe(true);
      }
    });
  });

  describe("buffer management", () => {
    it("should iterate named buffers with hierarchy", () => {
      const model = new TypedBufferModel();
      const namedBuffers = Array.from(model.namedBuffers());
      const names = namedBuffers.map(([name]) => name);

      expect(names).toContain("labels");
      expect(names).toContain("counts");
      expect(Array.from(model.buffers(false)).length).toBe(2);
    });
  });

  describe("training mode", () => {
    it("should be in training mode by default", () => {
      const model = new SimpleModel();
      expect(model.training).toBe(true);
    });

    it("should switch to eval mode", () => {
      const model = new SimpleModel();
      model.eval();
      expect(model.training).toBe(false);
    });

    it("should propagate mode to children", () => {
      const model = new SimpleModel();
      model.eval();

      const modules = Array.from(model.modules());
      for (const mod of modules) {
        expect(mod.training).toBe(false);
      }
    });
  });

  describe("toString", () => {
    it("should show module hierarchy", () => {
      const model = new SimpleModel();
      const str = model.toString();

      expect(str).toContain("SimpleModel");
      expect(str).toContain("fc1");
      expect(str).toContain("fc2");
      expect(str).toContain("Linear");
    });
  });

  describe("call method", () => {
    it("should call forward method", () => {
      // Requires full dot implementation
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

      // Should not throw
      expect(() => model.call(input)).not.toThrow();
    });
  });
  describe("parameter freezing", () => {
    it("should freeze and unfreeze parameters", () => {
      const model = new SimpleModel();
      model.freezeParameters();
      for (const param of model.parameters()) {
        expect(param.requiresGrad).toBe(false);
      }
      model.unfreezeParameters();
      for (const param of model.parameters()) {
        expect(param.requiresGrad).toBe(true);
      }
    });
  });

  describe("state dict", () => {
    it("should save and load parameters", () => {
      const model = new SimpleModel();
      const state = model.stateDict();

      // Mutate parameters
      for (const [, param] of model.namedParameters()) {
        const data = param.tensor.data;
        if (data instanceof BigInt64Array) {
          data.fill(0n);
        } else if (Array.isArray(data)) {
          data.fill("");
        } else {
          data.fill(0);
        }
      }

      model.loadStateDict(state);

      const restored = model.stateDict();
      expect(restored.parameters).toEqual(state.parameters);
    });

    it("should save and load buffers", () => {
      const model = new BufferModel();
      const state = model.stateDict();

      const buffer = model.buffer;
      const data = buffer.data;
      if (data instanceof BigInt64Array) {
        data.fill(5n);
      } else if (Array.isArray(data)) {
        data.fill("x");
      } else {
        data.fill(5);
      }

      model.loadStateDict(state);
      const restored = model.stateDict();
      expect(restored.buffers).toEqual(state.buffers);
    });

    it("should throw on missing parameter or buffer", () => {
      const model = new SimpleModel();
      const state = model.stateDict();
      delete state.parameters["fc1.weight"];
      expect(() => model.loadStateDict(state)).toThrow(InvalidParameterError);
    });

    it("should throw on missing buffer", () => {
      const model = new BufferModel();
      const state = model.stateDict();
      delete state.buffers["running"];
      expect(() => model.loadStateDict(state)).toThrow(InvalidParameterError);
    });

    it("should round-trip string and int64 buffers", () => {
      const model = new TypedBufferModel();
      const state = model.stateDict();

      const stringData = model.stringBuffer.data as string[];
      stringData[0] = "z";
      stringData[1] = "y";
      stringData[2] = "x";

      const int64Data = model.int64Buffer.data as BigInt64Array;
      int64Data[0] = 0n;
      int64Data[1] = 0n;
      int64Data[2] = 0n;

      model.loadStateDict(state);
      expect(model.stringBuffer.toArray()).toEqual(["a", "b", "c"]);
      expect(model.int64Buffer.toArray()).toEqual([1n, 2n, 3n]);
    });

    it("should throw on dtype/shape/data length mismatch", () => {
      const model = new SimpleModel();
      const state = model.stateDict();
      const entry = state.parameters["fc1.weight"];
      expect(entry).toBeDefined();
      if (!entry) return;

      const badDtype = { ...entry, dtype: "int32" as const };
      state.parameters["fc1.weight"] = badDtype;
      expect(() => model.loadStateDict(state)).toThrow(DTypeError);

      const goodState = model.stateDict();
      const shapeMismatch = {
        ...goodState.parameters["fc1.weight"],
        shape: [999],
      };
      goodState.parameters["fc1.weight"] = shapeMismatch;
      expect(() => model.loadStateDict(goodState)).toThrow(ShapeError);

      const lengthMismatch = {
        ...model.stateDict().parameters["fc1.weight"],
        data: [],
      };
      const state3 = model.stateDict();
      state3.parameters["fc1.weight"] = lengthMismatch;
      expect(() => model.loadStateDict(state3)).toThrow(ShapeError);
    });
  });

  describe("device movement", () => {
    it("should update device metadata for parameters and buffers", () => {
      const model = new SimpleModel();
      model.to("wasm");
      for (const param of model.parameters()) {
        expect(param.tensor.device).toBe("wasm");
      }

      const bufferModel = new BufferModel();
      bufferModel.to("webgpu");
      for (const buffer of bufferModel.buffers()) {
        expect(buffer.device).toBe("webgpu");
      }

      // @ts-expect-error - invalid device should be rejected at runtime
      expect(() => bufferModel.to("gpu")).toThrow(InvalidParameterError);
    });
  });

  describe("zeroGrad", () => {
    it("should clear gradients for parameters", () => {
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
      const gradInput = GradTensor.fromTensor(input, { requiresGrad: true });
      const output = model.call(gradInput);
      if (!(output instanceof GradTensor)) throw new Error("Expected GradTensor");
      output.sum().backward();

      for (const param of model.parameters()) {
        expect(param.grad).not.toBeNull();
      }

      model.zeroGrad();
      for (const param of model.parameters()) {
        const grad = param.grad;
        expect(grad).not.toBeNull();
        if (grad) {
          const data = grad.data;
          if (data instanceof BigInt64Array || Array.isArray(data))
            throw new Error("Unexpected dtype");
          expect(data.every((v) => v === 0)).toBe(true);
        }
      }
    });
  });

  describe("custom initialization", () => {
    it("should apply custom initializer to modules", () => {
      const model = new SimpleModel();
      model.apply((module) => {
        if (module instanceof Linear) {
          const weight = module.getWeight();
          const bias = module.getBias();
          if (weight.data instanceof BigInt64Array) {
            weight.data.fill(1n);
          } else if (!Array.isArray(weight.data)) {
            weight.data.fill(1);
          }
          if (bias) {
            if (bias.data instanceof BigInt64Array) {
              bias.data.fill(1n);
            } else if (!Array.isArray(bias.data)) {
              bias.data.fill(1);
            }
          }
        }
      });

      for (const [, param] of model.namedParameters()) {
        const data = param.tensor.data;
        if (data instanceof BigInt64Array) {
          expect(data.every((v) => v === 1n)).toBe(true);
        } else if (Array.isArray(data)) {
          expect(data.every((v) => v === "1")).toBe(true);
        } else {
          expect(data.every((v) => v === 1)).toBe(true);
        }
      }
    });
  });

  describe("module hooks", () => {
    it("should run forward pre-hooks and hooks", () => {
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

      const removePre = model.registerForwardPreHook((_mod, inputs) => {
        const x = inputs[0];
        if (!x) throw new Error("Missing input");
        const scaled = mulScalar(x instanceof GradTensor ? x.tensor : x, 2);
        return [scaled];
      });

      const removeHook = model.registerForwardHook((_mod, _inputs, output) => {
        const out = output instanceof GradTensor ? output.tensor : output;
        return addScalar(out, 1);
      });

      const rawOut = model.call(input);
      const out = rawOut instanceof GradTensor ? rawOut.tensor : rawOut;
      expect(out.ndim).toBe(2);

      removePre();
      removeHook();
    });

    it("should not modify output when hooks return undefined", () => {
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

      const removePre = model.registerForwardPreHook(() => undefined);
      const removeHook = model.registerForwardHook(() => undefined);

      const rawOut2 = model.call(input);
      const out = rawOut2 instanceof GradTensor ? rawOut2.tensor : rawOut2;
      expect(out.ndim).toBe(2);

      removePre();
      removeHook();
    });

    it("should execute multiple hooks in registration order", () => {
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
      const executionOrder: number[] = [];

      const remove1 = model.registerForwardPreHook(() => {
        executionOrder.push(1);
        return undefined;
      });
      const remove2 = model.registerForwardPreHook(() => {
        executionOrder.push(2);
        return undefined;
      });
      const remove3 = model.registerForwardHook(() => {
        executionOrder.push(3);
        return undefined;
      });
      const remove4 = model.registerForwardHook(() => {
        executionOrder.push(4);
        return undefined;
      });

      model.call(input);
      expect(executionOrder).toEqual([1, 2, 3, 4]);

      remove1();
      remove2();
      remove3();
      remove4();
    });

    it("should allow hook removal", () => {
      const model = new SimpleModel();
      const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
      let hookCalled = false;

      const removeHook = model.registerForwardHook(() => {
        hookCalled = true;
        return undefined;
      });

      model.call(input);
      expect(hookCalled).toBe(true);

      hookCalled = false;
      removeHook();
      model.call(input);
      expect(hookCalled).toBe(false);
    });
  });

  describe("parameter freezing reference stability", () => {
    it("should warn about reference stability - parameters are recreated", () => {
      const model = new SimpleModel();
      const originalParams = Array.from(model.parameters());
      const originalIds = originalParams.map((p) => p);

      model.freezeParameters();
      const newParams = Array.from(model.parameters());

      // After freezing, parameters are NEW objects (reference stability issue)
      // This test documents the current behavior
      for (let i = 0; i < originalIds.length; i++) {
        expect(newParams[i]).not.toBe(originalIds[i]);
      }
    });

    it("should update requiresGrad flag when freezing", () => {
      const model = new SimpleModel();
      model.freezeParameters();

      for (const param of model.parameters()) {
        expect(param.requiresGrad).toBe(false);
      }
    });

    it("should update requiresGrad flag when unfreezing", () => {
      const model = new SimpleModel();
      model.freezeParameters();
      model.unfreezeParameters();

      for (const param of model.parameters()) {
        expect(param.requiresGrad).toBe(true);
      }
    });
  });

  describe("device propagation", () => {
    it("should propagate device to nested modules", () => {
      const model = new SimpleModel();
      model.to("webgpu");

      // Check all parameters in nested modules
      for (const param of model.parameters()) {
        expect(param.tensor.device).toBe("webgpu");
      }
    });

    it("should update device for buffers", () => {
      const bufferModel = new BufferModel();
      bufferModel.to("wasm");

      for (const buffer of bufferModel.buffers()) {
        expect(buffer.device).toBe("wasm");
      }
    });
  });

  describe("state dict with all dtypes", () => {
    it("should handle float64 dtype", () => {
      const model = new Linear(2, 3, { dtype: "float64" });
      const state = model.stateDict();

      expect(state.parameters["weight"]?.dtype).toBe("float64");
      expect(state.parameters["bias"]?.dtype).toBe("float64");

      model.loadStateDict(state);
      const restored = model.stateDict();
      expect(restored.parameters).toEqual(state.parameters);
    });

    it("should handle empty parameters and buffers", () => {
      class EmptyModel extends Module {
        forward(x: Tensor): Tensor {
          return x;
        }
      }

      const model = new EmptyModel();
      const state = model.stateDict();

      expect(Object.keys(state.parameters).length).toBe(0);
      expect(Object.keys(state.buffers).length).toBe(0);

      // Should not throw
      expect(() => model.loadStateDict(state)).not.toThrow();
    });
  });
});
