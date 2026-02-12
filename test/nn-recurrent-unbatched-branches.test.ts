import { describe, expect, it } from "vitest";
import { parameter, tensor, zeros } from "../src/ndarray";
import { GRU, LSTM, RNN } from "../src/nn/layers/recurrent";

describe("nn recurrent unbatched inputs", () => {
  it("RNN supports 2D unbatched input with state", () => {
    const rnn = new RNN(2, 3, { numLayers: 1 });
    const x = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const hx = tensor([[0.1, 0.2, 0.3]]);
    const [out, h] = rnn.forwardWithState(x, hx);
    expect(out.shape).toEqual([3, 3]);
    expect(h.shape).toEqual([1, 3]);
  });

  it("LSTM supports 2D unbatched input with state", () => {
    const lstm = new LSTM(2, 4, { numLayers: 1 });
    const x = tensor([
      [1, 0],
      [0, 1],
    ]);
    const hx = tensor([[0.1, 0.2, 0.3, 0.4]]);
    const cx = tensor([[0.0, 0.0, 0.0, 0.0]]);
    const [out, [h, c]] = lstm.forwardWithState(x, hx, cx);
    expect(out.shape).toEqual([2, 4]);
    expect(h.shape).toEqual([1, 4]);
    expect(c.shape).toEqual([1, 4]);
  });

  it("GRU supports 2D unbatched input with state", () => {
    const gru = new GRU(2, 3, { numLayers: 1 });
    const x = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const hx = tensor([[0.2, 0.1, 0.0]]);
    const [out, h] = gru.forwardWithState(x, hx);
    expect(out.shape).toEqual([3, 3]);
    expect(h.shape).toEqual([1, 3]);
  });

  it("validates constructor parameters", () => {
    expect(() => new RNN(0, 2)).toThrow(/positive integer/i);
    expect(() => new LSTM(2, 0)).toThrow(/positive integer/i);
    expect(() => new GRU(2, 2, { numLayers: 0 })).toThrow(/positive integer/i);
  });

  it("rejects non-float and string inputs", () => {
    const rnn = new RNN(2, 2);
    const intInput = tensor(new Int32Array([1, 2, 3, 4]), {
      dtype: "int32",
    }).reshape([2, 2]);
    expect(() => rnn.forward(intInput)).toThrow(/float32 or float64/i);

    const strInput = tensor([["a", "b"]]);
    expect(() => rnn.forward(strInput)).toThrow(/string dtype/i);
  });

  it("rejects invalid input shapes and sizes", () => {
    const rnn = new RNN(2, 2);
    expect(() => rnn.forward(tensor([1, 2, 3]))).toThrow(/2D or 3D/);
    expect(() => rnn.forward(zeros([0, 2]))).toThrow(/Sequence length must be positive/);

    const rnnBatch = new RNN(2, 2, { batchFirst: true });
    expect(() => rnnBatch.forward(zeros([0, 2, 2]))).toThrow(/Batch size must be positive/);
  });

  it("accepts float32 inputs and returns float32 outputs", () => {
    const rnn = new RNN(2, 2);
    const floatInput = tensor(new Float32Array([1, 0, 0, 1]), {
      dtype: "float32",
    }).reshape([2, 2]);
    const out = rnn.forward(floatInput);
    expect(out.dtype).toBe("float32");
  });

  it("accepts GradTensor inputs", () => {
    const rnn = new RNN(2, 2);
    const gradInput = parameter(
      tensor([
        [1, 0],
        [0, 1],
      ])
    );
    const out = rnn.forward(gradInput);
    expect(out.shape).toEqual([2, 2]);
  });

  it("validates forward input counts", () => {
    const rnn = new RNN(2, 2);
    expect(() => rnn.forward()).toThrow(/expects 1 or 2 inputs/);
    expect(() => rnn.forward(undefined as unknown as never)).toThrow(/requires an input/);

    const lstm = new LSTM(2, 2);
    expect(() => lstm.forward()).toThrow(/expects 1 to 3 inputs/);

    const gru = new GRU(2, 2);
    expect(() => gru.forward()).toThrow(/expects 1 or 2 inputs/);
  });
});
