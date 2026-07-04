/**
 * Recurrent neural-network layers: RNN, LSTM, GRU.
 *
 * These layers are fully differentiable — the forward pass is built from
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
  mulScalar,
  parameter,
  randn,
  stackGrad,
  Tensor,
  zeros,
} from "../../ndarray";
import { Module } from "../module/Module";

function validatePositiveInt(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer`, name, value);
  }
}

function asGrad(x: AnyTensor): GradTensor {
  return GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);
}

function ensureFloatDtype(x: GradTensor, context: string): void {
  if (x.dtype !== "float32" && x.dtype !== "float64") {
    throw new DTypeError(`${context} expects float32 or float64 input`);
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
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly nonlinearity: RNNNonlinearity;
  private readonly bias: boolean;
  private readonly batchFirst: boolean;
  private readonly bidirectional: boolean;

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly nonlinearity?: RNNNonlinearity;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.nonlinearity = options.nonlinearity ?? "tanh";
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
    ensureFloatDtype(input, "RNN");
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

  forward(...inputs: AnyTensor[]): GradTensor {
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

  forwardWithState(input: AnyTensor, hx?: AnyTensor): [GradTensor, GradTensor] {
    const { output, h } = this.runAll(asGrad(input), hx === undefined ? undefined : asGrad(hx));
    return [output, h];
  }

  override toString(): string {
    return `RNN(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * LSTM (Long Short-Term Memory) layer.
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
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly bias: boolean;
  private readonly batchFirst: boolean;
  private readonly bidirectional: boolean;

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);

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
    ensureFloatDtype(input, "LSTM");
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

  forward(...inputs: AnyTensor[]): GradTensor {
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

  forwardWithState(
    input: AnyTensor,
    hx?: AnyTensor,
    cx?: AnyTensor
  ): [GradTensor, [GradTensor, GradTensor]] {
    const { output, h, c } = this.runAll(
      asGrad(input),
      hx === undefined ? undefined : asGrad(hx),
      cx === undefined ? undefined : asGrad(cx)
    );
    return [output, [h, c]];
  }

  override toString(): string {
    return `LSTM(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * GRU (Gated Recurrent Unit) layer.
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
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly bias: boolean;
  private readonly batchFirst: boolean;
  private readonly bidirectional: boolean;

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
      readonly bidirectional?: boolean;
    } = {}
  ) {
    super();
    validatePositiveInt("inputSize", inputSize);
    validatePositiveInt("hiddenSize", hiddenSize);
    const numLayers = options.numLayers ?? 1;
    validatePositiveInt("numLayers", numLayers);

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
    const oneMinusZ = onesLike(zGate).sub(zGate);
    return oneMinusZ.mul(nGate).add(zGate.mul(hPrev));
  }

  private runAll(input: GradTensor, hx?: GradTensor): { output: GradTensor; h: GradTensor } {
    ensureFloatDtype(input, "GRU");
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

  forward(...inputs: AnyTensor[]): GradTensor {
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

  forwardWithState(input: AnyTensor, hx?: AnyTensor): [GradTensor, GradTensor] {
    const { output, h } = this.runAll(asGrad(input), hx === undefined ? undefined : asGrad(hx));
    return [output, h];
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

  const reg = (name: string, t: Tensor) => register(name, parameter(t));

  for (let layer = 0; layer < numLayers; layer++) {
    const inputDim = layer === 0 ? inputSize : hiddenSize * numDir;
    reg(`weight_ih_l${layer}`, mulScalar(randn([gateSize, inputDim]), stdv));
    reg(`weight_hh_l${layer}`, mulScalar(randn([gateSize, hiddenSize]), stdv));
    if (bias) {
      reg(`bias_ih_l${layer}`, zeros([gateSize]));
      reg(`bias_hh_l${layer}`, zeros([gateSize]));
    }
    if (bidirectional) {
      reg(`weight_ih_l${layer}_reverse`, mulScalar(randn([gateSize, inputDim]), stdv));
      reg(`weight_hh_l${layer}_reverse`, mulScalar(randn([gateSize, hiddenSize]), stdv));
      if (bias) {
        reg(`bias_ih_l${layer}_reverse`, zeros([gateSize]));
        reg(`bias_hh_l${layer}_reverse`, zeros([gateSize]));
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

/** GradTensor of ones matching a reference GradTensor's shape/dtype. */
function onesLike(ref: GradTensor): GradTensor {
  const dtype = ref.dtype === "float64" ? "float64" : "float32";
  const data =
    dtype === "float64" ? new Float64Array(ref.size).fill(1) : new Float32Array(ref.size).fill(1);
  return GradTensor.fromTensor(
    Tensor.fromTypedArray({ data, shape: ref.shape, dtype, device: ref.tensor.device })
  );
}
