import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Dropout, Linear, ReLU, Sequential, Sigmoid } from "../src/nn";

describe("deepbox/nn - Sequential Container", () => {
  describe("constructor", () => {
    it("should create Sequential with multiple layers", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2));

      expect(model).toBeDefined();
      expect(model.length).toBe(3);
    });

    it("should throw error when no layers provided", () => {
      expect(() => new Sequential()).toThrow("Sequential requires at least one layer");
    });

    it("should accept single layer", () => {
      const model = new Sequential(new Linear(10, 5));
      expect(model.length).toBe(1);
    });

    it("should register all layers as child modules", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU());

      const modules = Array.from(model.modules());
      expect(modules.length).toBe(3); // Sequential + 2 layers
    });
  });

  describe("forward", () => {
    it("should pass input through all layers sequentially", () => {
      const model = new Sequential(new Linear(10, 5), new Linear(5, 2));

      const input = tensor([Array(10).fill(1)]);
      const output = model.forward(input);

      expect(output.shape).toEqual([1, 2]);
    });

    it("should apply activations correctly", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2), new Sigmoid());

      const input = tensor([Array(10).fill(0.5)]);
      const output = model.forward(input);

      expect(output.shape).toEqual([1, 2]);
      // Sigmoid output should be in (0, 1)
      for (let i = 0; i < output.size; i++) {
        const val = Number(output.data[i]);
        expect(val).toBeGreaterThan(0);
        expect(val).toBeLessThan(1);
      }
    });

    it("should handle 1D input", () => {
      const model = new Sequential(new Linear(5, 3), new ReLU());

      const input = tensor([1, 2, 3, 4, 5]);
      const output = model.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should handle batch inputs", () => {
      const model = new Sequential(new Linear(4, 3), new ReLU(), new Linear(3, 2));

      const input = tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const output = model.forward(input);

      expect(output.shape).toEqual([3, 2]);
    });

    it("should work with dropout layers", () => {
      const model = new Sequential(new Linear(10, 5), new Dropout(0.5), new Linear(5, 2));

      const input = tensor([Array(10).fill(1)]);
      const output = model.forward(input);

      expect(output.shape).toEqual([1, 2]);
    });
  });

  describe("getLayer", () => {
    it("should return layer at specified index", () => {
      const layer1 = new Linear(10, 5);
      const layer2 = new ReLU();
      const model = new Sequential(layer1, layer2);

      expect(model.getLayer(0)).toBe(layer1);
      expect(model.getLayer(1)).toBe(layer2);
    });

    it("should throw error for negative index", () => {
      const model = new Sequential(new Linear(10, 5));
      expect(() => model.getLayer(-1)).toThrow("out of bounds");
    });

    it("should throw error for index >= length", () => {
      const model = new Sequential(new Linear(10, 5));
      expect(() => model.getLayer(1)).toThrow("out of bounds");
    });
  });

  describe("length", () => {
    it("should return correct number of layers", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2));

      expect(model.length).toBe(3);
    });

    it("should return 1 for single layer", () => {
      const model = new Sequential(new Linear(10, 5));
      expect(model.length).toBe(1);
    });
  });

  describe("toString", () => {
    it("should show all layers", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU());

      const str = model.toString();
      expect(str).toContain("Sequential");
      expect(str).toContain("Linear");
      expect(str).toContain("ReLU");
    });

    it("should show layer indices", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU());

      const str = model.toString();
      expect(str).toContain("(0)");
      expect(str).toContain("(1)");
    });
  });

  describe("parameters", () => {
    it("should iterate all parameters from all layers", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2));

      const params = Array.from(model.parameters());
      // Linear1: weight + bias, Linear2: weight + bias = 4 parameters
      expect(params.length).toBe(4);
    });

    it("should include named parameters with correct paths", () => {
      const model = new Sequential(new Linear(10, 5), new Linear(5, 2));

      const namedParams = Array.from(model.namedParameters());
      const names = namedParams.map(([name]) => name);

      expect(names).toContain("0.weight");
      expect(names).toContain("0.bias");
      expect(names).toContain("1.weight");
      expect(names).toContain("1.bias");
    });
  });

  describe("training mode", () => {
    it("should propagate training mode to all layers", () => {
      const model = new Sequential(new Linear(10, 5), new Dropout(0.5), new Linear(5, 2));

      model.eval();

      const modules = Array.from(model.modules());
      for (const mod of modules) {
        expect(mod.training).toBe(false);
      }
    });

    it("should propagate train mode to all layers", () => {
      const model = new Sequential(new Linear(10, 5), new Dropout(0.5));

      model.eval();
      model.train();

      const modules = Array.from(model.modules());
      for (const mod of modules) {
        expect(mod.training).toBe(true);
      }
    });

    it("should affect dropout behavior", () => {
      const model = new Sequential(new Dropout(0.9));

      const input = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

      // Training mode: dropout active
      model.train();
      const trainOutput = model.forward(input);
      let trainZeros = 0;
      for (let i = 0; i < trainOutput.size; i++) {
        if (trainOutput.data[i] === 0) trainZeros++;
      }

      // Eval mode: dropout disabled
      model.eval();
      const evalOutput = model.forward(input);
      let evalZeros = 0;
      for (let i = 0; i < evalOutput.size; i++) {
        if (evalOutput.data[i] === 0) evalZeros++;
      }

      expect(trainZeros).toBeGreaterThan(0);
      expect(evalZeros).toBe(0);
    });
  });

  describe("iterator", () => {
    it("should be iterable", () => {
      const layer1 = new Linear(10, 5);
      const layer2 = new ReLU();
      const model = new Sequential(layer1, layer2);

      const layers = Array.from(model);
      expect(layers).toHaveLength(2);
      expect(layers[0]).toBe(layer1);
      expect(layers[1]).toBe(layer2);
    });

    it("should work with for...of", () => {
      const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2));

      let count = 0;
      for (const layer of model) {
        expect(layer).toBeDefined();
        count++;
      }
      expect(count).toBe(3);
    });
  });

  describe("complex architectures", () => {
    it("should handle deep networks", () => {
      const model = new Sequential(
        new Linear(100, 64),
        new ReLU(),
        new Dropout(0.5),
        new Linear(64, 32),
        new ReLU(),
        new Dropout(0.5),
        new Linear(32, 16),
        new ReLU(),
        new Linear(16, 10)
      );

      const input = tensor([Array(100).fill(0.1)]);
      const output = model.forward(input);

      expect(output.shape).toEqual([1, 10]);
    });

    it("should handle various activation functions", () => {
      const model = new Sequential(
        new Linear(10, 8),
        new ReLU(),
        new Linear(8, 6),
        new Sigmoid(),
        new Linear(6, 4)
      );

      const input = tensor([Array(10).fill(0.5)]);
      const output = model.forward(input);

      expect(output.shape).toEqual([1, 4]);
    });
  });
});
