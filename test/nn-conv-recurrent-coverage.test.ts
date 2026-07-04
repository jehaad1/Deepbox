import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  MultiheadAttention,
  PositionalEncoding,
  TransformerEncoderLayer,
} from "../src/nn/layers/attention";
import {
  AdaptiveAvgPool1d,
  AdaptiveAvgPool2d,
  AdaptiveMaxPool1d,
  AdaptiveMaxPool2d,
  AvgPool1d,
  AvgPool2d,
  Conv1d,
  Conv2d,
  ConvTranspose1d,
  ConvTranspose2d,
  MaxPool1d,
  MaxPool2d,
} from "../src/nn/layers/conv";
import { EmbeddingBag } from "../src/nn/layers/embedding";
import { GRU, LSTM, RNN } from "../src/nn/layers/recurrent";

const f32 = { dtype: "float32" as const };

// ────── Conv1d ──────
describe("Conv1d", () => {
  it("forward produces correct output shape", () => {
    const conv = new Conv1d(3, 8, 3, { padding: 1 });
    const x = tensor(
      Array.from({ length: 30 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 3, 10]);
    const out = conv.forward(x);
    expect(out.shape[0]).toBe(1);
    expect(out.shape[1]).toBe(8);
  });

  it("forward without bias", () => {
    const conv = new Conv1d(2, 4, 3, { bias: false });
    const x = tensor(
      Array.from({ length: 12 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 2, 6]);
    const out = conv.forward(x);
    expect(out.shape[1]).toBe(4);
  });

  it("validates non-3D input", () => {
    const conv = new Conv1d(2, 4, 3);
    expect(() => conv.forward(tensor([[1, 2]], f32))).toThrow();
  });

  it("validates channel mismatch", () => {
    const conv = new Conv1d(3, 4, 3);
    const x = tensor(
      Array.from({ length: 10 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 2, 5]);
    expect(() => conv.forward(x)).toThrow();
  });

  it("validates constructor params", () => {
    expect(() => new Conv1d(0, 4, 3)).toThrow();
    expect(() => new Conv1d(3, 0, 3)).toThrow();
    expect(() => new Conv1d(3, 4, 0)).toThrow();
  });

  it("toString", () => {
    expect(new Conv1d(3, 8, 3).toString()).toContain("Conv1d");
  });
});

// ────── Conv2d ──────
describe("Conv2d", () => {
  it("forward produces correct output shape", () => {
    const conv = new Conv2d(1, 4, 3, { padding: 1 });
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = conv.forward(x);
    expect(out.shape[0]).toBe(1);
    expect(out.shape[1]).toBe(4);
  });

  it("forward without bias", () => {
    const conv = new Conv2d(1, 2, 3, { bias: false });
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = conv.forward(x);
    expect(out.shape[1]).toBe(2);
  });

  it("validates non-4D input", () => {
    const conv = new Conv2d(1, 4, 3);
    expect(() => conv.forward(tensor([[1, 2]], f32))).toThrow();
  });

  it("validates constructor params", () => {
    expect(() => new Conv2d(0, 4, 3)).toThrow();
    expect(() => new Conv2d(1, 0, 3)).toThrow();
  });

  it("toString", () => {
    expect(new Conv2d(1, 4, 3).toString()).toContain("Conv2d");
  });
});

// ────── ConvTranspose1d ──────
describe("ConvTranspose1d", () => {
  it("forward produces correct output shape", () => {
    const ct = new ConvTranspose1d(4, 2, 3);
    const x = tensor(
      Array.from({ length: 20 }, (_, i) => i * 0.1),
      f32
    ).reshape([1, 4, 5]);
    const out = ct.forward(x);
    expect(out.shape[0]).toBe(1);
    expect(out.shape[1]).toBe(2);
  });

  it("validates constructor params", () => {
    expect(() => new ConvTranspose1d(0, 2, 3)).toThrow();
  });

  it("toString", () => {
    expect(new ConvTranspose1d(4, 2, 3).toString()).toContain("ConvTranspose1d");
  });
});

// ────── ConvTranspose2d ──────
describe("ConvTranspose2d", () => {
  it("forward produces correct output shape", () => {
    const ct = new ConvTranspose2d(4, 2, 3, { bias: false });
    const x = tensor(
      Array.from({ length: 64 }, (_, i) => i * 0.01),
      f32
    ).reshape([1, 4, 4, 4]);
    const out = ct.forward(x);
    expect(out.shape[0]).toBe(1);
    expect(out.shape[1]).toBe(2);
  });

  it("validates constructor params", () => {
    expect(() => new ConvTranspose2d(0, 2, 3)).toThrow();
  });

  it("toString", () => {
    expect(new ConvTranspose2d(4, 2, 3).toString()).toContain("ConvTranspose2d");
  });
});

// ────── MaxPool2d ──────
describe("MaxPool2d", () => {
  it("forward produces correct output shape", () => {
    const pool = new MaxPool2d(2);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2]);
  });

  it("validates non-4D input", () => {
    const pool = new MaxPool2d(2);
    expect(() => pool.forward(tensor([[1, 2]], f32))).toThrow();
  });

  it("toString", () => {
    expect(new MaxPool2d(2).toString()).toContain("MaxPool2d");
  });
});

// ────── AvgPool2d ──────
describe("AvgPool2d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AvgPool2d(2);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2]);
  });

  it("toString", () => {
    expect(new AvgPool2d(2).toString()).toContain("AvgPool2d");
  });
});

// ────── MaxPool1d ──────
describe("MaxPool1d", () => {
  it("forward produces correct output shape", () => {
    const pool = new MaxPool1d(2);
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i),
      f32
    ).reshape([1, 1, 8]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 4]);
  });

  it("validates non-3D input", () => {
    const pool = new MaxPool1d(2);
    expect(() => pool.forward(tensor([[1, 2]], f32))).toThrow();
  });

  it("toString", () => {
    expect(new MaxPool1d(2).toString()).toContain("MaxPool1d");
  });
});

// ────── AvgPool1d ──────
describe("AvgPool1d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AvgPool1d(2);
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i),
      f32
    ).reshape([1, 1, 8]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 4]);
  });

  it("toString", () => {
    expect(new AvgPool1d(2).toString()).toContain("AvgPool1d");
  });
});

// ────── AdaptiveAvgPool2d ──────
describe("AdaptiveAvgPool2d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AdaptiveAvgPool2d([2, 2]);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2]);
  });

  it("validates non-4D input", () => {
    const pool = new AdaptiveAvgPool2d([2, 2]);
    expect(() => pool.forward(tensor([[1, 2]], f32))).toThrow();
  });

  it("toString", () => {
    expect(new AdaptiveAvgPool2d([2, 2]).toString()).toContain("AdaptiveAvgPool2d");
  });
});

// ────── AdaptiveMaxPool2d ──────
describe("AdaptiveMaxPool2d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AdaptiveMaxPool2d([2, 2]);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i),
      f32
    ).reshape([1, 1, 4, 4]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2]);
  });

  it("toString", () => {
    expect(new AdaptiveMaxPool2d([2, 2]).toString()).toContain("AdaptiveMaxPool2d");
  });
});

// ────── AdaptiveAvgPool1d ──────
describe("AdaptiveAvgPool1d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AdaptiveAvgPool1d(4);
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i),
      f32
    ).reshape([1, 1, 8]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 4]);
  });

  it("toString", () => {
    expect(new AdaptiveAvgPool1d(4).toString()).toContain("AdaptiveAvgPool1d");
  });
});

// ────── AdaptiveMaxPool1d ──────
describe("AdaptiveMaxPool1d", () => {
  it("forward produces correct output shape", () => {
    const pool = new AdaptiveMaxPool1d(4);
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i),
      f32
    ).reshape([1, 1, 8]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 4]);
  });

  it("toString", () => {
    expect(new AdaptiveMaxPool1d(4).toString()).toContain("AdaptiveMaxPool1d");
  });
});

// ────── RNN ──────
describe("RNN", () => {
  it("forward with batch_first=true", () => {
    const rnn = new RNN(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forwardWithState returns output and hidden", () => {
    const rnn = new RNN(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const [output, hn] = rnn.forwardWithState(x);
    expect(output.shape).toEqual([2, 3, 8]);
    expect(hn.shape).toEqual([1, 2, 8]);
  });

  it("forward with batch_first=false", () => {
    const rnn = new RNN(4, 8, { batchFirst: false });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 2, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([3, 2, 8]);
  });

  it("forward unbatched (2D)", () => {
    const rnn = new RNN(4, 8);
    const x = tensor(
      Array.from({ length: 12 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([3, 8]);
  });

  it("forward with numLayers=2", () => {
    const rnn = new RNN(4, 8, { numLayers: 2, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forward with initial hidden state", () => {
    const rnn = new RNN(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const h0 = tensor(
      Array.from({ length: 16 }, () => 0),
      f32
    ).reshape([1, 2, 8]);
    const output = rnn.forward(x, h0);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forward with relu nonlinearity", () => {
    const rnn = new RNN(4, 8, { nonlinearity: "relu", batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forward without bias", () => {
    const rnn = new RNN(4, 8, { bias: false, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = rnn.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("validates constructor params", () => {
    expect(() => new RNN(0, 8)).toThrow(/positive integer/);
    expect(() => new RNN(4, 0)).toThrow(/positive integer/);
    expect(() => new RNN(4, 8, { numLayers: 0 })).toThrow(/positive integer/);
  });

  it("toString", () => {
    expect(new RNN(4, 8).toString()).toContain("RNN");
  });
});

// ────── LSTM ──────
describe("LSTM", () => {
  it("forward with batch_first=true", () => {
    const lstm = new LSTM(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = lstm.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forwardWithState returns output, h, c", () => {
    const lstm = new LSTM(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const [output, [hn, cn]] = lstm.forwardWithState(x);
    expect(output.shape).toEqual([2, 3, 8]);
    expect(hn.shape).toEqual([1, 2, 8]);
    expect(cn.shape).toEqual([1, 2, 8]);
  });

  it("forward with batch_first=false", () => {
    const lstm = new LSTM(4, 8);
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 2, 4]);
    const output = lstm.forward(x);
    expect(output.shape).toEqual([3, 2, 8]);
  });

  it("forward unbatched (2D)", () => {
    const lstm = new LSTM(4, 8);
    const x = tensor(
      Array.from({ length: 12 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 4]);
    const output = lstm.forward(x);
    expect(output.shape).toEqual([3, 8]);
  });

  it("forward with numLayers=2", () => {
    const lstm = new LSTM(4, 8, { numLayers: 2, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = lstm.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forward without bias", () => {
    const lstm = new LSTM(4, 8, { bias: false, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = lstm.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("validates constructor params", () => {
    expect(() => new LSTM(0, 8)).toThrow(/positive integer/);
    expect(() => new LSTM(4, 0)).toThrow(/positive integer/);
  });

  it("toString", () => {
    expect(new LSTM(4, 8).toString()).toContain("LSTM");
  });
});

// ────── GRU ──────
describe("GRU", () => {
  it("forward with batch_first=true", () => {
    const gru = new GRU(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = gru.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forwardWithState returns output and hidden", () => {
    const gru = new GRU(4, 8, { batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const [output, hn] = gru.forwardWithState(x);
    expect(output.shape).toEqual([2, 3, 8]);
    expect(hn.shape).toEqual([1, 2, 8]);
  });

  it("forward with batch_first=false", () => {
    const gru = new GRU(4, 8, { batchFirst: false });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 2, 4]);
    const output = gru.forward(x);
    expect(output.shape).toEqual([3, 2, 8]);
  });

  it("forward unbatched (2D)", () => {
    const gru = new GRU(4, 8);
    const x = tensor(
      Array.from({ length: 12 }, (_, i) => i * 0.01),
      f32
    ).reshape([3, 4]);
    const output = gru.forward(x);
    expect(output.shape).toEqual([3, 8]);
  });

  it("forward with numLayers=2", () => {
    const gru = new GRU(4, 8, { numLayers: 2, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = gru.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("forward without bias", () => {
    const gru = new GRU(4, 8, { bias: false, batchFirst: true });
    const x = tensor(
      Array.from({ length: 24 }, (_, i) => i * 0.01),
      f32
    ).reshape([2, 3, 4]);
    const output = gru.forward(x);
    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("validates constructor params", () => {
    expect(() => new GRU(0, 8)).toThrow(/positive integer/);
    expect(() => new GRU(4, 0)).toThrow(/positive integer/);
  });

  it("toString", () => {
    expect(new GRU(4, 8).toString()).toContain("GRU");
  });
});

// ────── EmbeddingBag ──────
describe("EmbeddingBag", () => {
  it("forward with mean mode", () => {
    const bag = new EmbeddingBag(10, 3, { mode: "mean" });
    const indices = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
    const offsets = tensor([0, 3], { dtype: "int32" });
    const out = bag.forward(indices, offsets);
    expect(out.shape).toEqual([2, 3]);
  });

  it("forward with sum mode", () => {
    const bag = new EmbeddingBag(10, 3, { mode: "sum" });
    const indices = tensor([1, 2, 3], { dtype: "int32" });
    const offsets = tensor([0], { dtype: "int32" });
    const out = bag.forward(indices, offsets);
    expect(out.shape).toEqual([1, 3]);
  });

  it("forward with max mode", () => {
    const bag = new EmbeddingBag(10, 3, { mode: "max" });
    const indices = tensor([1, 2, 3], { dtype: "int32" });
    const offsets = tensor([0], { dtype: "int32" });
    const out = bag.forward(indices, offsets);
    expect(out.shape).toEqual([1, 3]);
  });

  it("handles empty bag in max mode", () => {
    const bag = new EmbeddingBag(10, 3, { mode: "max" });
    const indices = tensor([1, 2], { dtype: "int32" });
    const offsets = tensor([0, 2], { dtype: "int32" });
    const out = bag.forward(indices, offsets);
    expect(out.shape).toEqual([2, 3]);
  });

  it("handles paddingIdx", () => {
    const bag = new EmbeddingBag(10, 3, { paddingIdx: 0 });
    const indices = tensor([0, 1, 2], { dtype: "int32" });
    const offsets = tensor([0], { dtype: "int32" });
    const out = bag.forward(indices, offsets);
    expect(out.shape).toEqual([1, 3]);
  });

  it("throws without offsets", () => {
    const bag = new EmbeddingBag(10, 3);
    expect(() => bag.forward(tensor([1, 2], { dtype: "int32" }))).toThrow(/offsets/);
  });

  it("validates constructor params", () => {
    expect(() => new EmbeddingBag(0, 3)).toThrow(/numEmbeddings/);
    expect(() => new EmbeddingBag(10, 0)).toThrow(/embeddingDim/);
    expect(() => new EmbeddingBag(10, 3, { paddingIdx: -1 })).toThrow(/paddingIdx/);
  });

  it("validates out-of-range index", () => {
    const bag = new EmbeddingBag(5, 3);
    const indices = tensor([10], { dtype: "int32" });
    const offsets = tensor([0], { dtype: "int32" });
    expect(() => bag.forward(indices, offsets)).toThrow(/out of range/);
  });

  it("toString", () => {
    expect(new EmbeddingBag(10, 3).toString()).toContain("EmbeddingBag");
  });

  it("weight getter", () => {
    const bag = new EmbeddingBag(10, 3);
    expect(bag.weight.shape).toEqual([10, 3]);
  });
});

// ────── MultiheadAttention ──────
describe("MultiheadAttention", () => {
  it("validates constructor params", () => {
    expect(() => new MultiheadAttention(0, 2)).toThrow();
    expect(() => new MultiheadAttention(8, 0)).toThrow();
    expect(() => new MultiheadAttention(7, 2)).toThrow(/divisible/);
  });

  it("toString", () => {
    expect(new MultiheadAttention(8, 2).toString()).toContain("MultiheadAttention");
  });
});

// ────── TransformerEncoderLayer ──────
describe("TransformerEncoderLayer", () => {
  it("toString", () => {
    expect(new TransformerEncoderLayer(8, 2).toString()).toContain("TransformerEncoderLayer");
  });
});

// ────── PositionalEncoding ──────
describe("PositionalEncoding", () => {
  it("toString", () => {
    expect(new PositionalEncoding(8).toString()).toContain("PositionalEncoding");
  });
});
