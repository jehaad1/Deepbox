import { describe, expect, it } from "vitest";
import { tensor, transpose } from "../src/ndarray";
import { GRU, LSTM, RNN } from "../src/nn/layers/recurrent";
import { expectNumber, expectNumberArray3D } from "./nn-test-utils";

describe("deepbox/nn - Recurrent Layers", () => {
  it("RNN forward returns expected shape", () => {
    const rnn = new RNN(2, 3, { numLayers: 2, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
      [
        [1, 1],
        [0, 0],
      ],
    ]);
    const out = rnn.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);

    const [out2, h] = rnn.forwardWithState(x);
    expect(out2.shape).toEqual([2, 2, 3]);
    expect(h.shape).toEqual([2, 2, 3]);
  });

  it("RNN forwardWithState returns final hidden state", () => {
    const rnn = new RNN(2, 3, { numLayers: 1, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const [out, h] = rnn.forwardWithState(x);
    expect(out.shape).toEqual([1, 2, 3]);
    expect(h.shape).toEqual([1, 1, 3]);
    for (let j = 0; j < 3; j++) {
      const lastOut = expectNumber(out.at(0, 1, j), "RNN output");
      const hVal = expectNumber(h.at(0, 0, j), "RNN hidden state");
      expect(lastOut).toBeCloseTo(hVal, 6);
    }
  });

  it("RNN forwardWithState returns all layer states", () => {
    const rnn = new RNN(2, 2, { numLayers: 2, batchFirst: true });
    for (const [name, param] of rnn.namedParameters()) {
      const data = param.tensor.data;
      if (
        data instanceof Float32Array ||
        data instanceof Float64Array ||
        data instanceof Int32Array ||
        data instanceof Uint8Array
      ) {
        if (name.includes("l1")) {
          data.fill(0);
        } else if (name === "weight_ih_l0") {
          data.fill(1);
        } else {
          data.fill(0);
        }
      }
    }

    const x = tensor([
      [
        [1, 1],
        [1, 1],
      ],
    ]);
    const [out, h] = rnn.forwardWithState(x);
    const outArr = expectNumberArray3D(out.toArray(), "RNN output");
    for (const row of outArr[0] ?? []) {
      expect(Math.abs(row[0] ?? 0)).toBeLessThan(1e-6);
      expect(Math.abs(row[1] ?? 0)).toBeLessThan(1e-6);
    }

    const hArr = expectNumberArray3D(h.toArray(), "RNN hidden state");
    const layer0 = hArr[0]?.[0] ?? [];
    const layer1 = hArr[1]?.[0] ?? [];
    expect(layer0.some((v) => Math.abs(v) > 1e-6)).toBe(true);
    expect(layer1.every((v) => Math.abs(v) < 1e-6)).toBe(true);
  });

  it("RNN handles batchFirst=false", () => {
    const rnn = new RNN(2, 3, { batchFirst: false });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
      [
        [1, 1],
        [0, 0],
      ],
    ]);
    const out = rnn.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("LSTM forward returns expected shape", () => {
    const lstm = new LSTM(2, 4, { numLayers: 1, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const out = lstm.forward(x);
    expect(out.shape).toEqual([1, 2, 4]);

    const [out2, [h, c]] = lstm.forwardWithState(x);
    expect(out2.shape).toEqual([1, 2, 4]);
    expect(h.shape).toEqual([1, 1, 4]);
    expect(c.shape).toEqual([1, 1, 4]);
  });

  it("LSTM forwardWithState returns final hidden state", () => {
    const lstm = new LSTM(2, 3, { numLayers: 1, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const [out, [h]] = lstm.forwardWithState(x);
    expect(out.shape).toEqual([1, 2, 3]);
    expect(h.shape).toEqual([1, 1, 3]);
    for (let j = 0; j < 3; j++) {
      const lastOut = expectNumber(out.at(0, 1, j), "LSTM output");
      const hVal = expectNumber(h.at(0, 0, j), "LSTM hidden state");
      expect(lastOut).toBeCloseTo(hVal, 6);
    }
  });

  it("LSTM forwardWithState returns non-zero cell state when biased", () => {
    const lstm = new LSTM(2, 2, { numLayers: 1, batchFirst: true });
    for (const [name, param] of lstm.namedParameters()) {
      const data = param.tensor.data;
      if (
        data instanceof Float32Array ||
        data instanceof Float64Array ||
        data instanceof Int32Array ||
        data instanceof Uint8Array
      ) {
        if (name === "weight_ih_l0" || name === "weight_hh_l0") {
          data.fill(0);
        } else if (name === "bias_ih_l0") {
          data.fill(0);
          data[0] = 1;
          data[1] = 1;
          data[2] = 0;
          data[3] = 0;
          data[4] = 1;
          data[5] = 1;
          data[6] = 1;
          data[7] = 1;
        } else if (name === "bias_hh_l0") {
          data.fill(0);
        }
      }
    }

    const x = tensor([[[0, 0]]]);
    const [, [, c]] = lstm.forwardWithState(x);
    const cArr = expectNumberArray3D(c.toArray(), "LSTM cell state");
    const cell = cArr[0]?.[0] ?? [];
    expect(cell.some((v) => Math.abs(v) > 1e-4)).toBe(true);
  });

  it("GRU forward returns expected shape", () => {
    const gru = new GRU(2, 3, { numLayers: 1, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const out = gru.forward(x);
    expect(out.shape).toEqual([1, 2, 3]);

    const [out2, h] = gru.forwardWithState(x);
    expect(out2.shape).toEqual([1, 2, 3]);
    expect(h.shape).toEqual([1, 1, 3]);
  });

  it("GRU forwardWithState returns final hidden state", () => {
    const gru = new GRU(2, 3, { numLayers: 1, batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const [out, h] = gru.forwardWithState(x);
    expect(out.shape).toEqual([1, 2, 3]);
    expect(h.shape).toEqual([1, 1, 3]);
    for (let j = 0; j < 3; j++) {
      const lastOut = expectNumber(out.at(0, 1, j), "GRU output");
      const hVal = expectNumber(h.at(0, 0, j), "GRU hidden state");
      expect(lastOut).toBeCloseTo(hVal, 6);
    }
  });

  it("GRU forwardWithState returns all layer states", () => {
    const gru = new GRU(2, 2, { numLayers: 2, batchFirst: true });
    for (const [name, param] of gru.namedParameters()) {
      const data = param.tensor.data;
      if (
        data instanceof Float32Array ||
        data instanceof Float64Array ||
        data instanceof Int32Array ||
        data instanceof Uint8Array
      ) {
        if (name.includes("l1")) {
          data.fill(0);
        } else if (name === "weight_ih_l0") {
          data.fill(1);
        } else {
          data.fill(0);
        }
      }
    }

    const x = tensor([
      [
        [1, 1],
        [1, 1],
      ],
    ]);
    const [out, h] = gru.forwardWithState(x);
    const outArr = expectNumberArray3D(out.toArray(), "GRU output");
    for (const row of outArr[0] ?? []) {
      expect(Math.abs(row[0] ?? 0)).toBeLessThan(1e-6);
      expect(Math.abs(row[1] ?? 0)).toBeLessThan(1e-6);
    }

    const hArr = expectNumberArray3D(h.toArray(), "GRU hidden state");
    const layer0 = hArr[0]?.[0] ?? [];
    const layer1 = hArr[1]?.[0] ?? [];
    expect(layer0.some((v) => Math.abs(v) > 1e-6)).toBe(true);
    expect(layer1.every((v) => Math.abs(v) < 1e-6)).toBe(true);
  });

  it("throws when input size mismatch", () => {
    const rnn = new RNN(3, 2);
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    expect(() => rnn.forward(x)).toThrow();
  });

  it("RNN supports relu nonlinearity, no bias, and provided hidden state", () => {
    const rnn = new RNN(2, 2, {
      nonlinearity: "relu",
      bias: false,
      batchFirst: false,
    });
    const x = tensor([[[1, 0]], [[0, 1]]]);
    const hx = tensor([[[0.1, 0.2]]]);
    const out = rnn.forward(x, hx);
    expect(out.shape).toEqual([2, 1, 2]);
    expect(rnn.toString()).toContain("RNN(2, 2");
  });

  it("RNN validates provided hidden state shape", () => {
    const rnn = new RNN(2, 2, { batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const hx = tensor([
      [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
    ]);
    expect(() => rnn.forward(x, hx)).toThrow();
  });

  it("LSTM supports provided hidden/cell state and batchFirst=false", () => {
    const lstm = new LSTM(2, 2, { bias: false, batchFirst: false });
    const x = tensor([[[1, 0]], [[0, 1]]]);
    const hx = tensor([[[0.1, 0.2]]]);
    const cx = tensor([[[0.0, 0.0]]]);
    const out = lstm.forward(x, hx, cx);
    expect(out.shape).toEqual([2, 1, 2]);

    const [out2, [h, c]] = lstm.forwardWithState(x, hx, cx);
    expect(out2.shape).toEqual([2, 1, 2]);
    expect(h.shape).toEqual([1, 1, 2]);
    expect(c.shape).toEqual([1, 1, 2]);
    expect(lstm.toString()).toContain("LSTM(2, 2");
  });

  it("LSTM validates provided hidden/cell state shapes", () => {
    const lstm = new LSTM(2, 2, { batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const hx = tensor([
      [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
    ]);
    const cx = tensor([
      [
        [0.0, 0.0],
        [0.0, 0.0],
      ],
    ]);
    expect(() => lstm.forward(x, hx, cx)).toThrow();
    expect(() => lstm.forward(x, hx)).toThrow();
  });

  it("LSTM handles non-contiguous batchFirst=false input", () => {
    const lstm = new LSTM(2, 3, { batchFirst: false });
    const xBatchFirst = tensor([
      [
        [1, 0],
        [0, 1],
      ],
      [
        [1, 1],
        [0, 0],
      ],
    ]);
    const x = transpose(xBatchFirst, [1, 0, 2]);
    const out = lstm.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("GRU supports provided hidden state and batchFirst=false", () => {
    const gru = new GRU(2, 2, { bias: false, batchFirst: false });
    const x = tensor([[[1, 0]], [[0, 1]]]);
    const hx = tensor([[[0.2, 0.1]]]);
    const out = gru.forward(x, hx);
    expect(out.shape).toEqual([2, 1, 2]);

    const [out2, h] = gru.forwardWithState(x, hx);
    expect(out2.shape).toEqual([2, 1, 2]);
    expect(h.shape).toEqual([1, 1, 2]);
    expect(gru.toString()).toContain("GRU(2, 2");
  });

  it("GRU validates provided hidden state shape", () => {
    const gru = new GRU(2, 2, { batchFirst: true });
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const hx = tensor([
      [
        [0.2, 0.1],
        [0.3, 0.4],
      ],
    ]);
    expect(() => gru.forward(x, hx)).toThrow();
  });

  it("GRU handles non-contiguous batchFirst=false input", () => {
    const gru = new GRU(2, 3, { batchFirst: false });
    const xBatchFirst = tensor([
      [
        [1, 0],
        [0, 1],
      ],
      [
        [1, 1],
        [0, 0],
      ],
    ]);
    const x = transpose(xBatchFirst, [1, 0, 2]);
    const out = gru.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("throws when LSTM/GRU input size mismatch", () => {
    const lstm = new LSTM(3, 2);
    const gru = new GRU(3, 2);
    const x = tensor([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    expect(() => lstm.forward(x)).toThrow();
    expect(() => gru.forward(x)).toThrow();
  });
});
