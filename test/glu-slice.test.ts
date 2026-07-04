import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { GLU } from "../src/nn/layers/activations";

describe("GLU activation (slice-based implementation)", () => {
  it("splits input along default dim=-1 and applies sigmoid gate", () => {
    // Input shape [2, 4] → split into [2, 2] halves along dim=-1
    const input = tensor([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    const glu = new GLU();
    const output = glu.forward(input);
    expect(output.shape).toEqual([2, 2]);

    // Output = firstHalf * sigmoid(secondHalf)
    // firstHalf = [[1,2],[5,6]], secondHalf = [[3,4],[7,8]]
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
    expect(Number(output.data[0])).toBeCloseTo(1 * sigmoid(3), 4);
    expect(Number(output.data[1])).toBeCloseTo(2 * sigmoid(4), 4);
    expect(Number(output.data[2])).toBeCloseTo(5 * sigmoid(7), 4);
    expect(Number(output.data[3])).toBeCloseTo(6 * sigmoid(8), 4);
  });

  it("works with dim=0", () => {
    // Input shape [4, 2] → split along dim=0 into [2, 2] halves
    const input = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const glu = new GLU(0);
    const output = glu.forward(input);
    expect(output.shape).toEqual([2, 2]);
  });

  it("throws on odd-sized dimension", () => {
    const input = tensor([[1, 2, 3]]);
    const glu = new GLU();
    expect(() => glu.forward(input)).toThrow();
  });

  it("works with 3D input along dim=1", () => {
    // [2, 4, 3] → split along dim=1 → [2, 2, 3]
    const data = [];
    for (let i = 0; i < 24; i++) data.push(i);
    const input = tensor(data).reshape([2, 4, 3]);
    const glu = new GLU(1);
    const output = glu.forward(input);
    expect(output.shape).toEqual([2, 2, 3]);
  });
});
