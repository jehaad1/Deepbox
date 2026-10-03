/**
 * Recurrent neural-network layers: RNN, LSTM, GRU.
 *
 * These layers are fully differentiable: the forward pass is built from
 * composable GradTensor operations (matmul/add/tanh/sigmoid/slice/stack/
 * concat), so backpropagation-through-time flows to every registered weight.
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import {
  type AnyTensor,
  concatGrad,
  GradTensor,
  parameter,
  stackGrad,
  type Tensor,
  zeros,
} from "../../ndarray";
import { Module } from "../module/Module";
import { allPlain, resolveLayerDtype, settle, uniformTensor } from "./_shared";

function validatePositiveInt(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer`, name, value);
  }
}

function validateDtype(dtype: "float32" | "float64" | undefined): void {
  const value: string | undefined = dtype;
  if (value !== undefined && value !== "float32" && value !== "float64") {
    throw new InvalidParameterError("dtype must be 'float32' or 'float64'", "dtype", dtype);
  }
}

function asGrad(x: AnyTensor): GradTensor {
  return GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);
}

/** The input is cast to the parameter dtype, so every numeric dtype is accepted. */
function ensureNumericInput(x: GradTensor, context: string): void {
  if (x.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
}

/**
 * Normalize a recurrent input to a batched [batch, seqLen, feat] GradTensor.
 * Records how to restore the caller's layout for the output.
 */
interface NormalizedSeq {
  readonly x: GradTensor; // [batch, seqLen, feat]
  readonly batch: number;
  readonly seqLen: number;
  readonly feat: number;
  readonly isUnbatched: boolean;
}

function normalizeSeqInput(input: GradTensor, batchFirst: boolean): NormalizedSeq {
  if (input.ndim === 2) {
    const seqLen = input.shape[0] ?? 0;
    const feat = input.shape[1] ?? 0;
    return { x: input.reshape([1, seqLen, feat]), batch: 1, seqLen, feat, isUnbatched: true };
  }
  if (input.ndim !== 3) {
    throw new ShapeError(`Recurrent layers expect 2D or 3D input; got ndim=${input.ndim}`);
  }
  if (batchFirst) {
    return {
      x: input,
      batch: input.shape[0] ?? 0,
      seqLen: input.shape[1] ?? 0,
      feat: input.shape[2] ?? 0,
      isUnbatched: false,
    };
  }
  // [seqLen, batch, feat] -> [batch, seqLen, feat]
  return {
    x: input.transpose([1, 0, 2]),
    batch: input.shape[1] ?? 0,
    seqLen: input.shape[0] ?? 0,
    feat: input.shape[2] ?? 0,
    isUnbatched: false,
  };
}

/** Slice timestep t out of a [batch, seqLen, feat] sequence -> [batch, feat]. */
function timestep(seq: GradTensor, t: number): GradTensor {
  return seq.slice({}, t);
}

/** Restore output from internal [batch, seqLen, hidden] to the caller's layout. */
function restoreOutput(out: GradTensor, batchFirst: boolean, isUnbatched: boolean): GradTensor {
  if (isUnbatched) {
    // [1, seqLen, hidden] -> [seqLen, hidden]
    const seqLen = out.shape[1] ?? 0;
    const hidden = out.shape[2] ?? 0;
    return out.reshape([seqLen, hidden]);
  }
  if (batchFirst) return out;
  // [batch, seqLen, hidden] -> [seqLen, batch, hidden]
  return out.transpose([1, 0, 2]);
}

type Params = {
  readonly wIh: GradTensor;
  readonly wHh: GradTensor;
  readonly bIh?: GradTensor | undefined;
  readonly bHh?: GradTensor | undefined;
};

/** Linear projection y = x @ W^T (+ b). x: [batch, in], W: [out, in]. */
function linear(x: GradTensor, w: GradTensor, b?: GradTensor): GradTensor {
  let y = x.matmul(w.transpose());
  if (b) y = y.add(b);
  return y;
}

export type RNNNonlinearity = "tanh" | "relu";

/**
 * Elman RNN layer with `tanh` or `relu` nonlinearity.
 *
 * Computes `h_t = act(W_ih x_t + b_ih + W_hh h_(t-1) + b_hh)` for every layer and
 * direction. Input is `(batch, seq, feature)` (or `(seq, batch, feature)` with
 * `batchFirst: false`), or unbatched `(seq, feature)`. As in PyTorch, all weights and
 * biases are drawn from the uniform distribution `U(-1/sqrt(hiddenSize),
 * 1/sqrt(hiddenSize))`. Parameters are named like PyTorch's
 * (`weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`, and a `_reverse`
 * suffix for the backward direction). The layer computes in the parameter dtype and
 * casts the input (and the initial state) to it.
 *
 * `forward` returns the output sequence `(batch, seq, hidden * directions)`;
 * `forwardWithState` also returns the final hidden state
 * `(layers * directions, batch, hidden)`. A `GradTensor` input gives `GradTensor`
 * results. A plain `Tensor` input gives `GradTensor` results that track the weights while
 * they require grad and gradient tracking is on, and plain tensors otherwise (inside
 * `noGrad()` or with frozen weights).
 *
 * @example
 * ```ts
 * import { RNN } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const rnn = new RNN(4, 8, { numLayers: 2 });
 * const x = randn([2, 5, 4]); // (batch, seq, feature)
 * const output = rnn.forward(x);
 * ```
 */
export class RNN extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  /** Number of features of each input step. */
  readonly inputSize: number;
  /** Number of features of the hidden state. */
  readonly hiddenSize: number;
  /** Number of stacked layers. */
  readonly numLayers: number;
  /** Activation applied at every step. */
  readonly nonlinearity: RNNNonlinearity;
  private readonly bias: boolean;
  /** Whether input and output are `(batch, seq, feature)` rather than `(seq, batch, feature)`. */
  readonly batchFirst: boolean;
  /** Whether each layer also runs over the sequence backwards. */
  readonly bidirectional: boolean;

  /**
   * @param inputSize - Number of input features
   * @param hiddenSize - Number of hidden features
   * @param options.numLayers - Number of stacked layers (default: 1)
   * @param options.nonlinearity - `'tanh'` or `'relu'` (default: `'tanh'`)
   * @param options.bias - Learn input and hidden biases (default: true)
   * @param options.batchFirst - Use `(batch, seq, feature)` layout (default: true)
   * @param options.bidirectional - Add a backward pass over the sequence (default: false)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly nonlinearity?: RNNNonlinearity;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
      readonly dtype?: "float32" | "float64";
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);
    validateDtype(options.dtype);
    const nonlinearity: string = options.nonlinearity ?? "tanh";
    if (nonlinearity !== "tanh" && nonlinearity !== "relu") {
      throw new InvalidParameterError(
        "nonlinearity must be 'tanh' or 'relu'",
        "nonlinearity",
        options.nonlinearity
      );
    }

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.nonlinearity = nonlinearity;
    this.bias = options.bias ?? true;
    this.batchFirst = options.batchFirst ?? true;
    this.bidirectional = options.bidirectional ?? false;

    initRecurrentParams((n, pr) => this.registerParameter(n, pr), {
      gateMul: 1,
      inputSize,
      hiddenSize,
      numLayers,
      bias: this.bias,
      bidirectional: this.bidirectional,
      dtype: options.dtype,
    });
  }

  private paramsFor(layer: number, reverse: boolean): Params {
    return gatherParams((n) => this.getParameter(n), layer, reverse, this.bias);
  }

  private cell(xt: GradTensor, hPrev: GradTensor, p: Params): GradTensor {
    const pre = linear(xt, p.wIh, p.bIh).add(linear(hPrev, p.wHh, p.bHh));
    return this.nonlinearity === "tanh" ? pre.tanh() : pre.relu();
  }

  private runAll(input: GradTensor, hx?: GradTensor): { output: GradTensor; h: GradTensor } {
    ensureNumericInput(input, "RNN");
    const norm = normalizeSeqInput(input, this.batchFirst);
    if (norm.feat !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${norm.feat}`);
    }
    if (norm.seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", norm.seqLen);
    }
    if (!norm.isUnbatched && norm.batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", norm.batch);
    }
    const numDir = this.bidirectional ? 2 : 1;
    const paramDtype = this.paramsFor(0, false).wIh.dtype === "float64" ? "float64" : "float32";
    const h0 = parseInitialState(
      hx,
      this.numLayers * numDir,
      norm.batch,
      this.hiddenSize,
      paramDtype
    );

    let layerInput = norm.x.astype(paramDtype); // [batch, seqLen, feat]
    const finalStates: GradTensor[] = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const dirOutputs: GradTensor[] = [];
      for (let dir = 0; dir < numDir; dir++) {
        const reverse = dir === 1;
        const stateIdx = layer * numDir + dir;
        const p = this.paramsFor(layer, reverse);
        let hCur = h0[stateIdx]!;
        const perStep: GradTensor[] = new Array(norm.seqLen);
        for (let s = 0; s < norm.seqLen; s++) {
          const t = reverse ? norm.seqLen - 1 - s : s;
          hCur = this.cell(timestep(layerInput, t), hCur, p);
          perStep[t] = hCur;
        }
        // [seqLen, batch, hidden] -> [batch, seqLen, hidden]
        dirOutputs.push(stackGrad(perStep).transpose([1, 0, 2]));
        finalStates[stateIdx] = hCur;
      }
      layerInput = numDir === 1 ? dirOutputs[0]! : concatGrad(dirOutputs, 2);
    }

    const output = restoreOutput(layerInput, this.batchFirst, norm.isUnbatched);
    const h = packStates(finalStates, norm.isUnbatched);
    return { output, h };
  }

  forward(input: GradTensor, hx?: AnyTensor): GradTensor;
  forward(input: Tensor, hx?: AnyTensor): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor {
    return settle(this.runForward(inputs), allPlain(...inputs));
  }

  private runForward(inputs: AnyTensor[]): GradTensor {
    if (inputs.length < 1 || inputs.length > 2) {
      throw new InvalidParameterError("RNN.forward expects 1 or 2 inputs", "inputs", inputs.length);
    }
    if (inputs[0] === undefined) {
      throw new InvalidParameterError("RNN.forward requires an input tensor", "input", inputs[0]);
    }
    const input = asGrad(inputs[0]);
    const hx = inputs[1] === undefined ? undefined : asGrad(inputs[1]);
    return this.runAll(input, hx).output;
  }

  /**
   * Run the layer and also return the final hidden state.
   *
   * @param input - Input sequence
   * @param hx - Optional initial hidden state `(layers * directions, batch, hidden)`
   *   (or `(layers * directions, hidden)` for unbatched input); zeros when omitted
   * @returns `[output, hN]`
   */
  forwardWithState(input: GradTensor, hx?: AnyTensor): [GradTensor, GradTensor];
  forwardWithState(input: Tensor, hx?: AnyTensor): [AnyTensor, AnyTensor];
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [AnyTensor, AnyTensor];
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [AnyTensor, AnyTensor] {
    const { output, h } = this.runAll(asGrad(input), hx === undefined ? undefined : asGrad(hx));
    const plain = allPlain(...(hx === undefined ? [input] : [input, hx]));
    return [settle(output, plain), settle(h, plain)];
  }

  override toString(): string {
    return `RNN(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * LSTM (Long Short-Term Memory) layer.
 *
 * Gate weights are stacked in PyTorch's order (input, forget, cell, output).
 * Input and output layouts, parameter names, initialization and dtype handling
 * follow {@link RNN}. `forwardWithState` returns the output and the final
 * hidden and cell states `[output, [hN, cN]]`.
 *
 * @example
 * ```ts
 * import { LSTM } from 'deepbox/nn';
 *
 * const lstm = new LSTM(4, 8);
 * const [output, [hN, cN]] = lstm.forwardWithState(x);
 * ```
 */
export class LSTM extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  /** Number of features of each input step. */
  readonly inputSize: number;
  /** Number of features of the hidden state. */
  readonly hiddenSize: number;
  /** Number of stacked layers. */
  readonly numLayers: number;
  private readonly bias: boolean;
  /** Whether input and output are `(batch, seq, feature)` rather than `(seq, batch, feature)`. */
  readonly batchFirst: boolean;
  /** Whether each layer also runs over the sequence backwards. */
  readonly bidirectional: boolean;

  /**
   * @param inputSize - Number of input features
   * @param hiddenSize - Number of hidden features
   * @param options.numLayers - Number of stacked layers (default: 1)
   * @param options.bias - Learn input and hidden biases (default: true)
   * @param options.batchFirst - Use `(batch, seq, feature)` layout (default: true)
   * @param options.bidirectional - Add a backward pass over the sequence (default: false)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
      readonly dtype?: "float32" | "float64";
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);
    validateDtype(options.dtype);

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.bias = options.bias ?? true;
    this.batchFirst = options.batchFirst ?? true;
    this.bidirectional = options.bidirectional ?? false;

    initRecurrentParams((n, pr) => this.registerParameter(n, pr), {
      gateMul: 4,
      inputSize,
      hiddenSize,
      numLayers,
      bias: this.bias,
      bidirectional: this.bidirectional,
      dtype: options.dtype,
    });
  }

  private paramsFor(layer: number, reverse: boolean): Params {
    return gatherParams((n) => this.getParameter(n), layer, reverse, this.bias);
  }

  private cell(
    xt: GradTensor,
    hPrev: GradTensor,
    cPrev: GradTensor,
    p: Params
  ): { h: GradTensor; c: GradTensor } {
    const H = this.hiddenSize;
    const gates = linear(xt, p.wIh, p.bIh).add(linear(hPrev, p.wHh, p.bHh)); // [batch, 4H]
    const i = gates.slice({}, { start: 0, end: H }).sigmoid();
    const f = gates.slice({}, { start: H, end: 2 * H }).sigmoid();
    const g = gates.slice({}, { start: 2 * H, end: 3 * H }).tanh();
    const o = gates.slice({}, { start: 3 * H, end: 4 * H }).sigmoid();
    const c = f.mul(cPrev).add(i.mul(g));
    const h = o.mul(c.tanh());
    return { h, c };
  }

  private runAll(
    input: GradTensor,
    hx?: GradTensor,
    cx?: GradTensor
  ): { output: GradTensor; h: GradTensor; c: GradTensor } {
    ensureNumericInput(input, "LSTM");
    const norm = normalizeSeqInput(input, this.batchFirst);
    if (norm.feat !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${norm.feat}`);
    }
    if (norm.seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", norm.seqLen);
    }
    if (!norm.isUnbatched && norm.batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", norm.batch);
    }
    const numDir = this.bidirectional ? 2 : 1;
    const total = this.numLayers * numDir;
    const paramDtype = this.paramsFor(0, false).wIh.dtype === "float64" ? "float64" : "float32";
    const h0 = parseInitialState(hx, total, norm.batch, this.hiddenSize, paramDtype);
    const c0 = parseInitialState(cx, total, norm.batch, this.hiddenSize, paramDtype);

    let layerInput = norm.x.astype(paramDtype);
    const finalH: GradTensor[] = [];
    const finalC: GradTensor[] = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const dirOutputs: GradTensor[] = [];
      for (let dir = 0; dir < numDir; dir++) {
        const reverse = dir === 1;
        const stateIdx = layer * numDir + dir;
        const p = this.paramsFor(layer, reverse);
        let hCur = h0[stateIdx]!;
        let cCur = c0[stateIdx]!;
        const perStep: GradTensor[] = new Array(norm.seqLen);
        for (let s = 0; s < norm.seqLen; s++) {
          const t = reverse ? norm.seqLen - 1 - s : s;
          const res = this.cell(timestep(layerInput, t), hCur, cCur, p);
          hCur = res.h;
          cCur = res.c;
          perStep[t] = hCur;
        }
        dirOutputs.push(stackGrad(perStep).transpose([1, 0, 2]));
        finalH[stateIdx] = hCur;
        finalC[stateIdx] = cCur;
      }
      layerInput = numDir === 1 ? dirOutputs[0]! : concatGrad(dirOutputs, 2);
    }

    return {
      output: restoreOutput(layerInput, this.batchFirst, norm.isUnbatched),
      h: packStates(finalH, norm.isUnbatched),
      c: packStates(finalC, norm.isUnbatched),
    };
  }

  forward(input: GradTensor, hx?: AnyTensor, cx?: AnyTensor): GradTensor;
  forward(input: Tensor, hx?: AnyTensor, cx?: AnyTensor): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor {
    return settle(this.runForward(inputs), allPlain(...inputs));
  }

  private runForward(inputs: AnyTensor[]): GradTensor {
    if (inputs.length < 1 || inputs.length > 3) {
      throw new InvalidParameterError(
        "LSTM.forward expects 1 to 3 inputs",
        "inputs",
        inputs.length
      );
    }
    if (inputs[0] === undefined) {
      throw new InvalidParameterError("LSTM.forward requires an input tensor", "input", inputs[0]);
    }
    const input = asGrad(inputs[0]);
    const hx = inputs[1] === undefined ? undefined : asGrad(inputs[1]);
    const cx = inputs[2] === undefined ? undefined : asGrad(inputs[2]);
    return this.runAll(input, hx, cx).output;
  }

  /**
   * Run the layer and also return the final hidden and cell states.
   *
   * @param input - Input sequence
   * @param hx - Optional initial hidden state `(layers * directions, batch, hidden)`
   * @param cx - Optional initial cell state, same shape as `hx`
   * @returns `[output, [hN, cN]]`
   */
  forwardWithState(
    input: GradTensor,
    hx?: AnyTensor,
    cx?: AnyTensor
  ): [GradTensor, [GradTensor, GradTensor]];
  forwardWithState(
    input: Tensor,
    hx?: AnyTensor,
    cx?: AnyTensor
  ): [AnyTensor, [AnyTensor, AnyTensor]];
  forwardWithState(
    input: AnyTensor,
    hx?: AnyTensor,
    cx?: AnyTensor
  ): [AnyTensor, [AnyTensor, AnyTensor]];
  forwardWithState(
    input: AnyTensor,
    hx?: AnyTensor,
    cx?: AnyTensor
  ): [AnyTensor, [AnyTensor, AnyTensor]] {
    const { output, h, c } = this.runAll(
      asGrad(input),
      hx === undefined ? undefined : asGrad(hx),
      cx === undefined ? undefined : asGrad(cx)
    );
    const given: AnyTensor[] = [input];
    if (hx !== undefined) given.push(hx);
    if (cx !== undefined) given.push(cx);
    const plain = allPlain(...given);
    return [settle(output, plain), [settle(h, plain), settle(c, plain)]];
  }

  override toString(): string {
    return `LSTM(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * GRU (Gated Recurrent Unit) layer.
 *
 * Gate weights are stacked in PyTorch's order (reset, update, new) and the reset
 * gate is applied to the hidden projection only, as in PyTorch. Input and output
 * layouts, parameter names, initialization and dtype handling follow {@link RNN}.
 *
 * @example
 * ```ts
 * import { GRU } from 'deepbox/nn';
 *
 * const gru = new GRU(4, 8);
 * const output = gru.forward(x);
 * ```
 */
export class GRU extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  /** Number of features of each input step. */
  readonly inputSize: number;
  /** Number of features of the hidden state. */
  readonly hiddenSize: number;
  /** Number of stacked layers. */
  readonly numLayers: number;
  private readonly bias: boolean;
  /** Whether input and output are `(batch, seq, feature)` rather than `(seq, batch, feature)`. */
  readonly batchFirst: boolean;
  /** Whether each layer also runs over the sequence backwards. */
  readonly bidirectional: boolean;

  /**
   * @param inputSize - Number of input features
   * @param hiddenSize - Number of hidden features
   * @param options.numLayers - Number of stacked layers (default: 1)
   * @param options.bias - Learn input and hidden biases (default: true)
   * @param options.batchFirst - Use `(batch, seq, feature)` layout (default: true)
   * @param options.bidirectional - Add a backward pass over the sequence (default: false)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
      readonly dtype?: "float32" | "float64";
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);
    validateDtype(options.dtype);

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.bias = options.bias ?? true;
    this.batchFirst = options.batchFirst ?? true;
    this.bidirectional = options.bidirectional ?? false;

    initRecurrentParams((n, pr) => this.registerParameter(n, pr), {
      gateMul: 3,
      inputSize,
      hiddenSize,
      numLayers,
      bias: this.bias,
      bidirectional: this.bidirectional,
      dtype: options.dtype,
    });
  }

  private paramsFor(layer: number, reverse: boolean): Params {
    return gatherParams((n) => this.getParameter(n), layer, reverse, this.bias);
  }

  private cell(xt: GradTensor, hPrev: GradTensor, p: Params): GradTensor {
    const H = this.hiddenSize;
    // Separate input and hidden projections: the new-gate applies the reset
    // gate only to the hidden contribution (PyTorch GRU convention).
    const gih = linear(xt, p.wIh, p.bIh); // [batch, 3H]
    const ghh = linear(hPrev, p.wHh, p.bHh); // [batch, 3H]
    const rGate = gih
      .slice({}, { start: 0, end: H })
      .add(ghh.slice({}, { start: 0, end: H }))
      .sigmoid();
    const zGate = gih
      .slice({}, { start: H, end: 2 * H })
      .add(ghh.slice({}, { start: H, end: 2 * H }))
      .sigmoid();
    const nGate = gih
      .slice({}, { start: 2 * H, end: 3 * H })
      .add(rGate.mul(ghh.slice({}, { start: 2 * H, end: 3 * H })))
      .tanh();
    // h = (1 - z) * n + z * hPrev
    const one = GradTensor.scalar(1, { dtype: zGate.dtype === "float64" ? "float64" : "float32" });
    return one.sub(zGate).mul(nGate).add(zGate.mul(hPrev));
  }

  private runAll(input: GradTensor, hx?: GradTensor): { output: GradTensor; h: GradTensor } {
    ensureNumericInput(input, "GRU");
    const norm = normalizeSeqInput(input, this.batchFirst);
    if (norm.feat !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${norm.feat}`);
    }
    if (norm.seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", norm.seqLen);
    }
    if (!norm.isUnbatched && norm.batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", norm.batch);
    }
    const numDir = this.bidirectional ? 2 : 1;
    const paramDtype = this.paramsFor(0, false).wIh.dtype === "float64" ? "float64" : "float32";
    const h0 = parseInitialState(
      hx,
      this.numLayers * numDir,
      norm.batch,
      this.hiddenSize,
      paramDtype
    );

    let layerInput = norm.x.astype(paramDtype);
    const finalStates: GradTensor[] = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const dirOutputs: GradTensor[] = [];
      for (let dir = 0; dir < numDir; dir++) {
        const reverse = dir === 1;
        const stateIdx = layer * numDir + dir;
        const p = this.paramsFor(layer, reverse);
        let hCur = h0[stateIdx]!;
        const perStep: GradTensor[] = new Array(norm.seqLen);
        for (let s = 0; s < norm.seqLen; s++) {
          const t = reverse ? norm.seqLen - 1 - s : s;
          hCur = this.cell(timestep(layerInput, t), hCur, p);
          perStep[t] = hCur;
        }
        dirOutputs.push(stackGrad(perStep).transpose([1, 0, 2]));
        finalStates[stateIdx] = hCur;
      }
      layerInput = numDir === 1 ? dirOutputs[0]! : concatGrad(dirOutputs, 2);
    }

    return {
      output: restoreOutput(layerInput, this.batchFirst, norm.isUnbatched),
      h: packStates(finalStates, norm.isUnbatched),
    };
  }

  forward(input: GradTensor, hx?: AnyTensor): GradTensor;
  forward(input: Tensor, hx?: AnyTensor): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor {
    return settle(this.runForward(inputs), allPlain(...inputs));
  }

  private runForward(inputs: AnyTensor[]): GradTensor {
    if (inputs.length < 1 || inputs.length > 2) {
      throw new InvalidParameterError("GRU.forward expects 1 or 2 inputs", "inputs", inputs.length);
    }
    if (inputs[0] === undefined) {
      throw new InvalidParameterError("GRU.forward requires an input tensor", "input", inputs[0]);
    }
    const input = asGrad(inputs[0]);
    const hx = inputs[1] === undefined ? undefined : asGrad(inputs[1]);
    return this.runAll(input, hx).output;
  }

  /**
   * Run the layer and also return the final hidden state.
   *
   * @param input - Input sequence
   * @param hx - Optional initial hidden state `(layers * directions, batch, hidden)`
   *   (or `(layers * directions, hidden)` for unbatched input); zeros when omitted
   * @returns `[output, hN]`
   */
  forwardWithState(input: GradTensor, hx?: AnyTensor): [GradTensor, GradTensor];
  forwardWithState(input: Tensor, hx?: AnyTensor): [AnyTensor, AnyTensor];
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [AnyTensor, AnyTensor];
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [AnyTensor, AnyTensor] {
    const { output, h } = this.runAll(asGrad(input), hx === undefined ? undefined : asGrad(hx));
    const plain = allPlain(...(hx === undefined ? [input] : [input, hx]));
    return [settle(output, plain), settle(h, plain)];
  }

  override toString(): string {
    return `GRU(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

// ─── Shared parameter helpers ────────────────────────────────────────────────

interface RecurrentInit {
  readonly gateMul: number;
  readonly inputSize: number;
  readonly hiddenSize: number;
  readonly numLayers: number;
  readonly bias: boolean;
  readonly bidirectional: boolean;
  readonly dtype: "float32" | "float64" | undefined;
}

/**
 * Register all weight/bias parameters for a recurrent module via a supplied
 * `register` callback (bound to the module's protected registerParameter).
 */
function initRecurrentParams(
  register: (name: string, p: GradTensor) => void,
  cfg: RecurrentInit
): void {
  const { gateMul, inputSize, hiddenSize, numLayers, bias, bidirectional } = cfg;
  const stdv = 1.0 / Math.sqrt(hiddenSize);
  const numDir = bidirectional ? 2 : 1;
  const gateSize = gateMul * hiddenSize;

  // PyTorch default: every weight and bias ~ U(-1/sqrt(hidden), 1/sqrt(hidden)).
  const opts = { dtype: resolveLayerDtype(cfg.dtype) };
  const reg = (name: string, shape: number[]) =>
    register(name, parameter(uniformTensor(shape, stdv, opts)));

  for (let layer = 0; layer < numLayers; layer++) {
    const inputDim = layer === 0 ? inputSize : hiddenSize * numDir;
    reg(`weight_ih_l${layer}`, [gateSize, inputDim]);
    reg(`weight_hh_l${layer}`, [gateSize, hiddenSize]);
    if (bias) {
      reg(`bias_ih_l${layer}`, [gateSize]);
      reg(`bias_hh_l${layer}`, [gateSize]);
    }
    if (bidirectional) {
      reg(`weight_ih_l${layer}_reverse`, [gateSize, inputDim]);
      reg(`weight_hh_l${layer}_reverse`, [gateSize, hiddenSize]);
      if (bias) {
        reg(`bias_ih_l${layer}_reverse`, [gateSize]);
        reg(`bias_hh_l${layer}_reverse`, [gateSize]);
      }
    }
  }
}

function gatherParams(
  get: (name: string) => GradTensor | undefined,
  layer: number,
  reverse: boolean,
  bias: boolean
): Params {
  const suffix = reverse ? "_reverse" : "";
  const wIh = get(`weight_ih_l${layer}${suffix}`);
  const wHh = get(`weight_hh_l${layer}${suffix}`);
  if (!wIh || !wHh) throw new ShapeError("Internal error: missing recurrent weights");
  return {
    wIh,
    wHh,
    ...(bias
      ? { bIh: get(`bias_ih_l${layer}${suffix}`), bHh: get(`bias_hh_l${layer}${suffix}`) }
      : {}),
  };
}

/** Build per-(layer,dir) initial state list, each [batch, hidden]. */
function parseInitialState(
  state: GradTensor | undefined,
  total: number,
  batch: number,
  hidden: number,
  dtype: "float32" | "float64"
): GradTensor[] {
  const result: GradTensor[] = [];
  if (state === undefined) {
    for (let i = 0; i < total; i++) {
      result.push(GradTensor.fromTensor(zeros([batch, hidden], { dtype })));
    }
    return result;
  }
  // Accept [total, hidden] (unbatched) or [total, batch, hidden] (batched)
  if (state.ndim === 2) {
    if ((state.shape[0] ?? -1) !== total || (state.shape[1] ?? -1) !== hidden) {
      throw new ShapeError(
        `Expected initial state shape [${total}, ${hidden}], got [${state.shape.join(", ")}]`
      );
    }
    if (batch !== 1) {
      throw new ShapeError(
        `A 2D initial state is only valid for unbatched input; expected shape ` +
          `[${total}, ${batch}, ${hidden}], got [${state.shape.join(", ")}]`
      );
    }
    for (let i = 0; i < total; i++) {
      result.push(state.slice(i).reshape([1, hidden]).astype(dtype));
    }
    return result;
  }
  if (state.ndim === 3) {
    if (
      (state.shape[0] ?? -1) !== total ||
      (state.shape[1] ?? -1) !== batch ||
      (state.shape[2] ?? -1) !== hidden
    ) {
      throw new ShapeError(
        `Expected initial state shape [${total}, ${batch}, ${hidden}], got [${state.shape.join(", ")}]`
      );
    }
    for (let i = 0; i < total; i++) {
      result.push(state.slice(i).astype(dtype));
    }
    return result;
  }
  throw new ShapeError(`Expected initial state with 2 or 3 dimensions, got ${state.ndim}`);
}

/**
 * Pack per-(layer,dir) final states [batch, hidden] into the reported state
 * tensor: [total, batch, hidden] batched, or [total, hidden] unbatched.
 */
function packStates(states: GradTensor[], isUnbatched: boolean): GradTensor {
  const stacked = stackGrad(states); // [total, batch, hidden]
  if (isUnbatched) {
    const total = stacked.shape[0] ?? 0;
    const hidden = stacked.shape[2] ?? 0;
    return stacked.reshape([total, hidden]);
  }
  return stacked;
}
