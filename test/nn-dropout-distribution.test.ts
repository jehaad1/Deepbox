import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Dropout } from "../src/nn";

describe("deepbox/nn - Dropout distribution tests", () => {
  it("should use uniform distribution for dropout mask", () => {
    const dropout = new Dropout(0.5);
    const input = tensor(new Array(10000).fill(1));

    dropout.train();
    const output = dropout.forward(input);

    // Count how many elements were kept (non-zero)
    let keptCount = 0;
    for (let i = 0; i < output.size; i++) {
      if (Number(output.data[output.offset + i]) !== 0) {
        keptCount++;
      }
    }

    // With p=0.5, approximately 50% should be kept (allow 5% tolerance)
    const keepRate = keptCount / output.size;
    expect(keepRate).toBeGreaterThan(0.45);
    expect(keepRate).toBeLessThan(0.55);
  });

  it("should maintain correct dropout rate for p=0.3", () => {
    const dropout = new Dropout(0.3);
    const input = tensor(new Array(10000).fill(1));

    dropout.train();
    const output = dropout.forward(input);

    let keptCount = 0;
    for (let i = 0; i < output.size; i++) {
      if (Number(output.data[output.offset + i]) !== 0) {
        keptCount++;
      }
    }

    // With p=0.3, approximately 70% should be kept (allow 5% tolerance)
    const keepRate = keptCount / output.size;
    expect(keepRate).toBeGreaterThan(0.65);
    expect(keepRate).toBeLessThan(0.75);
  });

  it("should scale kept values by 1/(1-p) during training", () => {
    const dropout = new Dropout(0.5);
    const input = tensor([2, 2, 2, 2, 2, 2, 2, 2]);

    dropout.train();
    const output = dropout.forward(input);

    // Kept values should be scaled by 1/(1-0.5) = 2
    for (let i = 0; i < output.size; i++) {
      const val = Number(output.data[output.offset + i]);
      // Value should be either 0 (dropped) or 4 (2 * 2 = scaled)
      expect(val === 0 || val === 4).toBe(true);
    }
  });

  it("should not drop any elements during eval mode", () => {
    const dropout = new Dropout(0.5);
    const input = tensor([1, 2, 3, 4, 5]);

    dropout.eval();
    const output = dropout.forward(input);

    // All values should be preserved in eval mode
    for (let i = 0; i < output.size; i++) {
      expect(output.data[output.offset + i]).toBe(input.data[input.offset + i]);
    }
  });

  it("should drop no elements when p=0", () => {
    const dropout = new Dropout(0);
    const input = tensor([1, 2, 3, 4, 5]);

    dropout.train();
    const output = dropout.forward(input);

    // All values should be preserved when p=0
    for (let i = 0; i < output.size; i++) {
      expect(output.data[output.offset + i]).toBe(input.data[input.offset + i]);
    }
  });
});
