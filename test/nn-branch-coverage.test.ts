import { describe, expect, it } from "vitest";
import { randn } from "../src/ndarray";
import {
  FullTransformer,
  GRU,
  Linear,
  LSTM,
  MultiheadAttention,
  PositionalEncoding,
  ReLU,
  RNN,
  Sequential,
  TransformerDecoder,
  TransformerDecoderLayer,
  TransformerEncoder,
  TransformerEncoderLayer,
} from "../src/nn";
import { setSeed } from "../src/random";

describe("Bidirectional GRU coverage", () => {
  it("forward pass with bidirectional=true, batchFirst=true", () => {
    setSeed(42);
    const gru = new GRU(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([2, 5, 4]); // batch=2, seq=5, input=4
    const output = gru.forward(input);
    // Output should have hidden*2 = 6 features due to bidirectional
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("forward pass with bidirectional=true, batchFirst=false", () => {
    setSeed(42);
    const gru = new GRU(4, 3, { bidirectional: true, batchFirst: false });
    const input = randn([5, 2, 4]); // seq=5, batch=2, input=4
    const output = gru.forward(input);
    expect(output.shape).toEqual([5, 2, 6]);
  });

  it("bidirectional GRU with numLayers=2", () => {
    setSeed(42);
    const gru = new GRU(4, 3, {
      bidirectional: true,
      numLayers: 2,
      batchFirst: true,
    });
    const input = randn([2, 5, 4]);
    const output = gru.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional GRU with no bias", () => {
    setSeed(42);
    const gru = new GRU(4, 3, {
      bidirectional: true,
      bias: false,
      batchFirst: true,
    });
    const input = randn([2, 5, 4]);
    const output = gru.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional GRU with unbatched 2D input", () => {
    setSeed(42);
    const gru = new GRU(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([5, 4]); // seq=5, input=4 (no batch dim)
    const output = gru.forward(input);
    expect(output.shape).toEqual([5, 6]);
  });

  it("bidirectional GRU forwardWithState returns hidden state", () => {
    setSeed(42);
    const gru = new GRU(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([2, 5, 4]);
    const [output, h] = gru.forwardWithState(input);
    expect(output.shape).toEqual([2, 5, 6]);
    // h should have numLayers*2 directions
    expect(h.shape[0]).toBe(2); // numLayers * numDirections
  });

  it("GRU toString", () => {
    const gru = new GRU(4, 3, { numLayers: 2 });
    expect(gru.toString()).toContain("GRU");
  });
});

describe("Bidirectional LSTM coverage", () => {
  it("forward pass with bidirectional=true", () => {
    setSeed(42);
    const lstm = new LSTM(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([2, 5, 4]);
    const output = lstm.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional LSTM with numLayers=2", () => {
    setSeed(42);
    const lstm = new LSTM(4, 3, {
      bidirectional: true,
      numLayers: 2,
      batchFirst: true,
    });
    const input = randn([2, 5, 4]);
    const output = lstm.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional LSTM with no bias", () => {
    setSeed(42);
    const lstm = new LSTM(4, 3, {
      bidirectional: true,
      bias: false,
      batchFirst: true,
    });
    const input = randn([2, 5, 4]);
    const output = lstm.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional LSTM unbatched", () => {
    setSeed(42);
    const lstm = new LSTM(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([5, 4]); // unbatched
    const output = lstm.forward(input);
    expect(output.shape).toEqual([5, 6]);
  });
});

describe("Bidirectional RNN coverage", () => {
  it("forward pass with bidirectional=true", () => {
    setSeed(42);
    const rnn = new RNN(4, 3, { bidirectional: true, batchFirst: true });
    const input = randn([2, 5, 4]);
    const output = rnn.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });

  it("bidirectional RNN with numLayers=2", () => {
    setSeed(42);
    const rnn = new RNN(4, 3, {
      bidirectional: true,
      numLayers: 2,
      batchFirst: true,
    });
    const input = randn([2, 5, 4]);
    const output = rnn.forward(input);
    expect(output.shape).toEqual([2, 5, 6]);
  });
});

describe("FullTransformer coverage", () => {
  it("forward pass with small dimensions", () => {
    setSeed(42);
    const transformer = new FullTransformer(8, 2, 1, 1, 16, { dropout: 0 });
    transformer.eval();
    const src = randn([1, 3, 8]); // batch=1, seq=3, dModel=8
    const tgt = randn([1, 3, 8]);
    const output = transformer.forward(src, tgt);
    expect(output.shape).toEqual([1, 3, 8]);
  });

  it("throws when tgt is missing", () => {
    setSeed(42);
    const transformer = new FullTransformer(8, 2, 1, 1, 16);
    const src = randn([1, 3, 8]);
    expect(() => transformer.forward(src)).toThrow();
  });

  it("toString", () => {
    const transformer = new FullTransformer(8, 2, 1, 1, 16);
    const str = transformer.toString();
    expect(str).toContain("Transformer");
    expect(str).toContain("Encoder");
    expect(str).toContain("Decoder");
  });
});

describe("TransformerDecoder coverage", () => {
  it("forward pass", () => {
    setSeed(42);
    const layer = new TransformerDecoderLayer(8, 2, 16, { dropout: 0 });
    const decoder = new TransformerDecoder(layer, 2);
    decoder.eval();
    const tgt = randn([1, 3, 8]);
    const memory = randn([1, 3, 8]);
    const output = decoder.forward(tgt, memory);
    expect(output.shape).toEqual([1, 3, 8]);
  });

  it("throws without memory", () => {
    setSeed(42);
    const layer = new TransformerDecoderLayer(8, 2, 16);
    const decoder = new TransformerDecoder(layer, 1);
    const tgt = randn([1, 3, 8]);
    expect(() => decoder.forward(tgt)).toThrow();
  });

  it("throws for invalid numLayers", () => {
    const layer = new TransformerDecoderLayer(8, 2, 16);
    expect(() => new TransformerDecoder(layer, 0)).toThrow();
    expect(() => new TransformerDecoder(layer, -1)).toThrow();
  });

  it("toString", () => {
    const layer = new TransformerDecoderLayer(8, 2, 16);
    const decoder = new TransformerDecoder(layer, 3);
    expect(decoder.toString()).toContain("TransformerDecoder");
  });
});

describe("PositionalEncoding coverage", () => {
  it("forward with 3D input", () => {
    setSeed(42);
    const pe = new PositionalEncoding(8, { dropout: 0, maxLen: 100 });
    pe.eval();
    const input = randn([2, 5, 8], { dtype: "float64" }); // batch=2, seq=5, dModel=8
    const output = pe.forward(input);
    expect(output.shape).toEqual([2, 5, 8]);
  });

  it("forward with 2D input", () => {
    setSeed(42);
    const pe = new PositionalEncoding(8, { dropout: 0, maxLen: 100 });
    pe.eval();
    const input = randn([5, 8], { dtype: "float64" }); // seq=5, dModel=8
    const output = pe.forward(input);
    expect(output.shape).toEqual([5, 8]);
  });

  it("throws for 1D input", () => {
    const pe = new PositionalEncoding(8);
    const input = randn([8]);
    expect(() => pe.forward(input)).toThrow();
  });

  it("throws for invalid dModel", () => {
    expect(() => new PositionalEncoding(0)).toThrow();
    expect(() => new PositionalEncoding(-1)).toThrow();
  });

  it("throws when seqLen exceeds maxLen", () => {
    const pe = new PositionalEncoding(8, { maxLen: 3 });
    const input = randn([5, 8]); // seq=5 > maxLen=3
    expect(() => pe.forward(input)).toThrow();
  });

  it("toString", () => {
    const pe = new PositionalEncoding(16, { maxLen: 200 });
    expect(pe.toString()).toContain("PositionalEncoding");
  });
});

describe("TransformerEncoderLayer edge cases", () => {
  it("throws for invalid dModel", () => {
    expect(() => new TransformerEncoderLayer(0, 2, 16)).toThrow();
  });

  it("throws for invalid nHead", () => {
    expect(() => new TransformerEncoderLayer(8, 0, 16)).toThrow();
  });

  it("toString", () => {
    const layer = new TransformerEncoderLayer(8, 2, 16);
    expect(layer.toString()).toContain("TransformerEncoderLayer");
  });
});

describe("TransformerEncoder edge cases", () => {
  it("throws for invalid numLayers", () => {
    const layer = new TransformerEncoderLayer(8, 2, 16);
    expect(() => new TransformerEncoder(layer, 0)).toThrow();
  });

  it("toString", () => {
    const layer = new TransformerEncoderLayer(8, 2, 16);
    const encoder = new TransformerEncoder(layer, 2);
    expect(encoder.toString()).toContain("TransformerEncoder");
  });
});

describe("Sequential edge cases", () => {
  it("constructor throws for no layers", () => {
    expect(() => new Sequential()).toThrow();
  });

  it("forward through chain", () => {
    setSeed(42);
    const seq = new Sequential(new Linear(4, 3), new ReLU(), new Linear(3, 2));
    const input = randn([2, 4]);
    const output = seq.forward(input);
    expect(output.shape).toEqual([2, 2]);
  });

  it("toString", () => {
    const seq = new Sequential(new Linear(4, 3), new ReLU());
    const str = seq.toString();
    expect(str).toContain("Sequential");
    expect(str).toContain("Linear");
    expect(str).toContain("ReLU");
  });

  it("getLayer returns module by index", () => {
    const lin = new Linear(4, 3);
    const relu = new ReLU();
    const seq = new Sequential(lin, relu);
    expect(seq.getLayer(0)).toBe(lin);
    expect(seq.getLayer(1)).toBe(relu);
  });

  it("getLayer throws for invalid index", () => {
    const seq = new Sequential(new Linear(4, 3));
    expect(() => seq.getLayer(5)).toThrow();
    expect(() => seq.getLayer(-1)).toThrow();
  });

  it("length returns number of modules", () => {
    const seq = new Sequential(new Linear(4, 3), new ReLU(), new Linear(3, 2));
    expect(seq.length).toBe(3);
  });
});

describe("MultiheadAttention edge cases", () => {
  it("throws for invalid embedDim", () => {
    expect(() => new MultiheadAttention(0, 2)).toThrow();
  });

  it("throws for invalid numHeads", () => {
    expect(() => new MultiheadAttention(8, 0)).toThrow();
  });

  it("throws when embedDim not divisible by numHeads", () => {
    expect(() => new MultiheadAttention(7, 3)).toThrow();
  });
});
