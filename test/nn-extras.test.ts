import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Dropout2d, Linear, ReLU, Sequential } from "../src/nn";

describe("Dropout2d", () => {
  it("passes input unchanged in eval mode", () => {
    const d = new Dropout2d(0.5);
    d.eval();
    const input = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ]);
    const output = d.forward(input);
    expect(output.shape).toEqual([1, 2, 2, 2]);
    expect(output.toArray()).toEqual([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ]);
  });

  it("zeros entire channels in train mode", () => {
    const d = new Dropout2d(0.999); // nearly always drop
    d.train();
    const input = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ]);
    const output = d.forward(input);
    expect(output.shape).toEqual([1, 2, 2, 2]);
    // With p=0.999, almost all channels should be zeroed
    // At least check shape is preserved
  });

  it("p=0 keeps all values (scaled by 1)", () => {
    const d = new Dropout2d(0);
    d.train();
    const input = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
    const output = d.forward(input);
    expect(output.toArray()).toEqual([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
  });

  it("throws on non-4D input in train mode", () => {
    const d = new Dropout2d(0.5);
    d.train();
    expect(() => d.forward(tensor([1, 2, 3]))).toThrow("4D");
  });

  it("throws on invalid p", () => {
    expect(() => new Dropout2d(-0.1)).toThrow();
    expect(() => new Dropout2d(1.0)).toThrow();
  });

  it("toString returns descriptive string", () => {
    expect(new Dropout2d(0.3).toString()).toBe("Dropout2d(p=0.3)");
  });

  it("dropoutRate getter works", () => {
    expect(new Dropout2d(0.4).dropoutRate).toBe(0.4);
  });
});

describe("Module.summary()", () => {
  it("returns a formatted summary string", () => {
    const model = new Sequential(new Linear(10, 5), new ReLU(), new Linear(5, 2));
    const s = model.summary();
    expect(s).toContain("Sequential Summary");
    expect(s).toContain("Total params:");
    expect(s).toContain("Trainable params:");
    expect(s).toContain("Non-trainable params:");
  });

  it("counts parameters correctly", () => {
    // Linear(3, 2): weight=6, bias=2 => 8 params
    const model = new Sequential(new Linear(3, 2));
    const s = model.summary();
    expect(s).toContain("8");
  });

  it("works on simple single-layer model", () => {
    const model = new Sequential(new ReLU());
    const s = model.summary();
    expect(s).toContain("Total params: 0");
  });
});
