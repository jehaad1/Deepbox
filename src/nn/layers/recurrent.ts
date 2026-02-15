import {
  DTypeError,
  getElementAsNumber,
  InvalidParameterError,
  type Shape,
  ShapeError,
} from "../../core";
import {
  type AnyTensor,
  GradTensor,
  mulScalar,
  parameter,
  randn,
  Tensor,
  zeros,
} from "../../ndarray";
import { Module } from "../module/Module";

function ensureFloatTensor(
  t: Tensor,
  context: string
): asserts t is Tensor<Shape, "float32" | "float64"> {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  if (t.dtype !== "float32" && t.dtype !== "float64") {
    throw new DTypeError(`${context} expects float32 or float64 dtype`);
  }
}

function readNumeric(t: Tensor, offset: number): number {
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("String tensors are not supported");
  }
  return getElementAsNumber(data, offset);
}

function createFloatBuffer(
  size: number,
  dtype: "float32" | "float64"
): Float32Array | Float64Array {
  return dtype === "float64" ? new Float64Array(size) : new Float32Array(size);
}

function validatePositiveInt(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer`, name, value);
  }
}

type ParsedInput = {
  readonly batch: number;
  readonly seqLen: number;
  readonly inputDim: number;
  readonly isUnbatched: boolean;
  readonly batchStride: number;
  readonly seqStride: number;
  readonly featStride: number;
};

function parseInput(input: Tensor, batchFirst: boolean): ParsedInput {
  if (input.ndim === 2) {
    const seqLen = input.shape[0] ?? 0;
    const inputDim = input.shape[1] ?? 0;
    return {
      batch: 1,
      seqLen,
      inputDim,
      isUnbatched: true,
      batchStride: 0,
      seqStride: input.strides[0] ?? 0,
      featStride: input.strides[1] ?? 0,
    };
  }

  if (input.ndim !== 3) {
    throw new ShapeError(`Recurrent layers expect 2D or 3D input; got ndim=${input.ndim}`);
  }

  if (batchFirst) {
    return {
      batch: input.shape[0] ?? 0,
      seqLen: input.shape[1] ?? 0,
      inputDim: input.shape[2] ?? 0,
      isUnbatched: false,
      batchStride: input.strides[0] ?? 0,
      seqStride: input.strides[1] ?? 0,
      featStride: input.strides[2] ?? 0,
    };
  }

  return {
    batch: input.shape[1] ?? 0,
    seqLen: input.shape[0] ?? 0,
    inputDim: input.shape[2] ?? 0,
    isUnbatched: false,
    batchStride: input.strides[1] ?? 0,
    seqStride: input.strides[0] ?? 0,
    featStride: input.strides[2] ?? 0,
  };
}

function outputIndex(
  batchFirst: boolean,
  isUnbatched: boolean,
  batch: number,
  seqLen: number,
  hiddenSize: number,
  b: number,
  t: number,
  j: number
): number {
  if (isUnbatched) {
    return t * hiddenSize + j;
  }
  if (batchFirst) {
    return b * (seqLen * hiddenSize) + t * hiddenSize + j;
  }
  return t * (batch * hiddenSize) + b * hiddenSize + j;
}

function extractTensor(arg: AnyTensor, _name: string): Tensor {
  if (GradTensor.isGradTensor(arg)) {
    return arg.tensor;
  }
  return arg;
}

function buildState(
  state: Tensor | undefined,
  numLayers: number,
  batch: number,
  hiddenSize: number,
  isUnbatched: boolean,
  name: string
): Float64Array[] {
  const result = new Array<Float64Array>(numLayers);
  for (let l = 0; l < numLayers; l++) {
    result[l] = new Float64Array(batch * hiddenSize);
  }

  if (!state) {
    return result;
  }

  ensureFloatTensor(state, name);

  if (state.ndim === 2) {
    if (!isUnbatched) {
      throw new ShapeError(`Expected ${name} with 3 dimensions for batched input`);
    }
    if ((state.shape[0] ?? 0) !== numLayers || (state.shape[1] ?? 0) !== hiddenSize) {
      throw new ShapeError(
        `Expected ${name} shape [${numLayers}, ${hiddenSize}], got [${state.shape.join(", ")}]`
      );
    }

    const stride0 = state.strides[0] ?? 0;
    const stride1 = state.strides[1] ?? 0;
    for (let l = 0; l < numLayers; l++) {
      const layerState = result[l];
      if (!layerState) {
        throw new ShapeError(`Internal error: missing ${name} layer state`);
      }
      const base = state.offset + l * stride0;
      for (let j = 0; j < hiddenSize; j++) {
        layerState[j] = readNumeric(state, base + j * stride1);
      }
    }

    return result;
  }

  if (state.ndim !== 3) {
    throw new ShapeError(`Expected ${name} with 2 or 3 dimensions; got ndim=${state.ndim}`);
  }

  const expectedBatch = isUnbatched ? 1 : batch;
  if (
    (state.shape[0] ?? 0) !== numLayers ||
    (state.shape[1] ?? 0) !== expectedBatch ||
    (state.shape[2] ?? 0) !== hiddenSize
  ) {
    const expected = isUnbatched ? [numLayers, 1, hiddenSize] : [numLayers, batch, hiddenSize];
    throw new ShapeError(
      `Expected ${name} shape [${expected.join(", ")}], got [${state.shape.join(", ")}]`
    );
  }

  const stride0 = state.strides[0] ?? 0;
  const stride1 = state.strides[1] ?? 0;
  const stride2 = state.strides[2] ?? 0;

  for (let l = 0; l < numLayers; l++) {
    const layerState = result[l];
    if (!layerState) {
      throw new ShapeError(`Internal error: missing ${name} layer state`);
    }
    const baseLayer = state.offset + l * stride0;
    for (let b = 0; b < batch; b++) {
      const baseBatch = baseLayer + b * stride1;
      for (let j = 0; j < hiddenSize; j++) {
        layerState[b * hiddenSize + j] = readNumeric(state, baseBatch + j * stride2);
      }
    }
  }

  return result;
}

function packState(
  state: Float64Array[],
  numLayers: number,
  batch: number,
  hiddenSize: number,
  dtype: "float32" | "float64",
  device: Tensor["device"],
  isUnbatched: boolean
): Tensor {
  const size = isUnbatched ? numLayers * hiddenSize : numLayers * batch * hiddenSize;
  const data = createFloatBuffer(size, dtype);

  if (isUnbatched) {
    for (let l = 0; l < numLayers; l++) {
      const layer = state[l];
      if (!layer) {
        throw new ShapeError("Internal error: missing packed state layer");
      }
      for (let j = 0; j < hiddenSize; j++) {
        data[l * hiddenSize + j] = layer[j] ?? 0;
      }
    }
    return Tensor.fromTypedArray({
      data,
      shape: [numLayers, hiddenSize],
      dtype,
      device,
    });
  }

  for (let l = 0; l < numLayers; l++) {
    const layer = state[l];
    if (!layer) {
      throw new ShapeError("Internal error: missing packed state layer");
    }
    const layerOffset = l * batch * hiddenSize;
    for (let b = 0; b < batch; b++) {
      const batchOffset = layerOffset + b * hiddenSize;
      for (let j = 0; j < hiddenSize; j++) {
        data[batchOffset + j] = layer[b * hiddenSize + j] ?? 0;
      }
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: [numLayers, batch, hiddenSize],
    dtype,
    device,
  });
}

/**
 * Simple RNN layer.
 *
 * Applies a simple recurrent neural network to an input sequence.
 *
 * **Formula**: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
 *
 * @example
 * ```ts
 * import { RNN } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const rnn = new RNN(10, 20);
 * const x = tensor([[[1, 2, 3]]]);  // (batch, seq_len, input_size)
 * const output = rnn.forward(x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-recurrent | Deepbox Recurrent Layers}
 */
export class RNN extends Module {
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly nonlinearity: "tanh" | "relu";
  private readonly bias: boolean;
  private readonly batchFirst: boolean;

  private weightsIh: Tensor[];
  private weightsHh: Tensor[];
  private biasIh: Tensor[];
  private biasHh: Tensor[];

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly nonlinearity?: "tanh" | "relu";
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
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

    const stdv = 1.0 / Math.sqrt(hiddenSize);

    this.weightsIh = [];
    this.weightsHh = [];
    this.biasIh = [];
    this.biasHh = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const inputDim = layer === 0 ? inputSize : hiddenSize;
      const wIh = mulScalar(randn([hiddenSize, inputDim]), stdv);
      const wHh = mulScalar(randn([hiddenSize, hiddenSize]), stdv);

      this.weightsIh.push(wIh);
      this.weightsHh.push(wHh);
      this.registerParameter(`weight_ih_l${layer}`, parameter(wIh));
      this.registerParameter(`weight_hh_l${layer}`, parameter(wHh));

      if (this.bias) {
        const bIh = zeros([hiddenSize]);
        const bHh = zeros([hiddenSize]);
        this.biasIh.push(bIh);
        this.biasHh.push(bHh);
        this.registerParameter(`bias_ih_l${layer}`, parameter(bIh));
        this.registerParameter(`bias_hh_l${layer}`, parameter(bHh));
      }
    }
  }

  private activation(x: number): number {
    return this.nonlinearity === "tanh" ? Math.tanh(x) : Math.max(0, x);
  }

  private run(input: Tensor, hx?: Tensor): { output: Tensor; h: Tensor } {
    ensureFloatTensor(input, "RNN");
    const parsed = parseInput(input, this.batchFirst);
    const { batch, seqLen, inputDim, isUnbatched, batchStride, seqStride, featStride } = parsed;

    if (inputDim !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${inputDim}`);
    }
    if (seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", seqLen);
    }
    if (!isUnbatched && batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", batch);
    }

    const h = buildState(hx, this.numLayers, batch, this.hiddenSize, isUnbatched, "hx");
    const outSize = (isUnbatched ? seqLen : batch * seqLen) * this.hiddenSize;
    const out = createFloatBuffer(outSize, input.dtype);

    const inputVec = new Float64Array(inputDim);

    for (let t = 0; t < seqLen; t++) {
      for (let b = 0; b < batch; b++) {
        const baseOffset = input.offset + b * batchStride + t * seqStride;
        for (let i = 0; i < inputDim; i++) {
          inputVec[i] = readNumeric(input, baseOffset + i * featStride);
        }

        let layerInput = inputVec;

        for (let l = 0; l < this.numLayers; l++) {
          const wIh = this.weightsIh[l];
          const wHh = this.weightsHh[l];
          if (!wIh || !wHh) {
            throw new ShapeError("Internal error: missing RNN weights");
          }

          const curInputSize = l === 0 ? this.inputSize : this.hiddenSize;
          const newH = new Float64Array(this.hiddenSize);
          const hLayer = h[l];
          if (!hLayer) {
            throw new ShapeError("Internal error: missing RNN hidden state");
          }

          const wIhStride0 = wIh.strides[0] ?? 0;
          const wIhStride1 = wIh.strides[1] ?? 0;
          const wHhStride0 = wHh.strides[0] ?? 0;
          const wHhStride1 = wHh.strides[1] ?? 0;
          const biasIh = this.biasIh[l];
          const biasHh = this.biasHh[l];
          const biasIhStride = biasIh ? (biasIh.strides[0] ?? 0) : 0;
          const biasHhStride = biasHh ? (biasHh.strides[0] ?? 0) : 0;

          for (let j = 0; j < this.hiddenSize; j++) {
            let sum = 0;
            const wIhBase = wIh.offset + j * wIhStride0;
            for (let k = 0; k < curInputSize; k++) {
              sum += (layerInput[k] ?? 0) * readNumeric(wIh, wIhBase + k * wIhStride1);
            }

            const wHhBase = wHh.offset + j * wHhStride0;
            for (let k = 0; k < this.hiddenSize; k++) {
              sum +=
                (hLayer[b * this.hiddenSize + k] ?? 0) * readNumeric(wHh, wHhBase + k * wHhStride1);
            }

            if (this.bias && biasIh && biasHh) {
              sum += readNumeric(biasIh, biasIh.offset + j * biasIhStride);
              sum += readNumeric(biasHh, biasHh.offset + j * biasHhStride);
            }

            newH[j] = this.activation(sum);
          }

          for (let j = 0; j < this.hiddenSize; j++) {
            hLayer[b * this.hiddenSize + j] = newH[j] ?? 0;
          }

          layerInput = newH;
        }

        for (let j = 0; j < this.hiddenSize; j++) {
          const idx = outputIndex(
            this.batchFirst,
            isUnbatched,
            batch,
            seqLen,
            this.hiddenSize,
            b,
            t,
            j
          );
          out[idx] = layerInput[j] ?? 0;
        }
      }
    }

    const outShape = isUnbatched
      ? [seqLen, this.hiddenSize]
      : this.batchFirst
        ? [batch, seqLen, this.hiddenSize]
        : [seqLen, batch, this.hiddenSize];

    return {
      output: Tensor.fromTypedArray({
        data: out,
        shape: outShape,
        dtype: input.dtype,
        device: input.device,
      }),
      h: packState(
        h,
        this.numLayers,
        batch,
        this.hiddenSize,
        input.dtype,
        input.device,
        isUnbatched
      ),
    };
  }

  forward(...inputs: AnyTensor[]): Tensor {
    if (inputs.length < 1 || inputs.length > 2) {
      throw new InvalidParameterError("RNN.forward expects 1 or 2 inputs", "inputs", inputs.length);
    }
    const inputArg = inputs[0];
    if (inputArg === undefined) {
      throw new InvalidParameterError("RNN.forward requires an input tensor", "input", inputArg);
    }
    const input = extractTensor(inputArg, "input");
    const hxArg = inputs.length === 2 ? inputs[1] : undefined;
    const hx = hxArg === undefined ? undefined : extractTensor(hxArg, "hx");
    return this.run(input, hx).output;
  }

  /**
   * Forward pass returning both output and hidden state.
   * Use this method when you need the hidden state.
   */
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [Tensor, Tensor] {
    const inputTensor = extractTensor(input, "input");
    const hxTensor = hx === undefined ? undefined : extractTensor(hx, "hx");
    const { output, h } = this.run(inputTensor, hxTensor);
    return [output, h];
  }

  override toString(): string {
    return `RNN(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * LSTM (Long Short-Term Memory) layer.
 *
 * Applies a multi-layer LSTM to an input sequence.
 *
 * **Gates**:
 * - Input gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
 * - Forget gate: f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
 * - Cell gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
 * - Output gate: o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
 * - Cell state: c_t = f_t * c_{t-1} + i_t * g_t
 * - Hidden state: h_t = o_t * tanh(c_t)
 *
 * @see {@link https://deepbox.dev/docs/nn-recurrent | Deepbox Recurrent Layers}
 */
export class LSTM extends Module {
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly bias: boolean;
  private readonly batchFirst: boolean;

  private weightsIh: Tensor[];
  private weightsHh: Tensor[];
  private biasIh: Tensor[];
  private biasHh: Tensor[];

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
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

    const stdv = 1.0 / Math.sqrt(hiddenSize);

    this.weightsIh = [];
    this.weightsHh = [];
    this.biasIh = [];
    this.biasHh = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const inputDim = layer === 0 ? inputSize : hiddenSize;

      const wIh = mulScalar(randn([4 * hiddenSize, inputDim]), stdv);
      const wHh = mulScalar(randn([4 * hiddenSize, hiddenSize]), stdv);

      this.weightsIh.push(wIh);
      this.weightsHh.push(wHh);
      this.registerParameter(`weight_ih_l${layer}`, parameter(wIh));
      this.registerParameter(`weight_hh_l${layer}`, parameter(wHh));

      if (this.bias) {
        const bIh = zeros([4 * hiddenSize]);
        const bHh = zeros([4 * hiddenSize]);
        this.biasIh.push(bIh);
        this.biasHh.push(bHh);
        this.registerParameter(`bias_ih_l${layer}`, parameter(bIh));
        this.registerParameter(`bias_hh_l${layer}`, parameter(bHh));
      }
    }
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private run(input: Tensor, hx?: Tensor, cx?: Tensor): { output: Tensor; h: Tensor; c: Tensor } {
    ensureFloatTensor(input, "LSTM");
    const parsed = parseInput(input, this.batchFirst);
    const { batch, seqLen, inputDim, isUnbatched, batchStride, seqStride, featStride } = parsed;

    if (inputDim !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${inputDim}`);
    }
    if (seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", seqLen);
    }
    if (!isUnbatched && batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", batch);
    }

    const h = buildState(hx, this.numLayers, batch, this.hiddenSize, isUnbatched, "hx");
    const c = buildState(cx, this.numLayers, batch, this.hiddenSize, isUnbatched, "cx");

    const outSize = (isUnbatched ? seqLen : batch * seqLen) * this.hiddenSize;
    const out = createFloatBuffer(outSize, input.dtype);

    const inputVec = new Float64Array(inputDim);
    const gates = new Float64Array(4 * this.hiddenSize);

    for (let t = 0; t < seqLen; t++) {
      for (let b = 0; b < batch; b++) {
        const baseOffset = input.offset + b * batchStride + t * seqStride;
        for (let i = 0; i < inputDim; i++) {
          inputVec[i] = readNumeric(input, baseOffset + i * featStride);
        }

        let layerInput = inputVec;

        for (let l = 0; l < this.numLayers; l++) {
          const wIh = this.weightsIh[l];
          const wHh = this.weightsHh[l];
          if (!wIh || !wHh) {
            throw new ShapeError("Internal error: missing LSTM weights");
          }

          const curInputSize = l === 0 ? this.inputSize : this.hiddenSize;
          const hLayer = h[l];
          const cLayer = c[l];
          if (!hLayer || !cLayer) {
            throw new ShapeError("Internal error: missing LSTM state");
          }

          const wIhStride0 = wIh.strides[0] ?? 0;
          const wIhStride1 = wIh.strides[1] ?? 0;
          const wHhStride0 = wHh.strides[0] ?? 0;
          const wHhStride1 = wHh.strides[1] ?? 0;
          const biasIh = this.biasIh[l];
          const biasHh = this.biasHh[l];
          const biasIhStride = biasIh ? (biasIh.strides[0] ?? 0) : 0;
          const biasHhStride = biasHh ? (biasHh.strides[0] ?? 0) : 0;

          for (let g = 0; g < 4 * this.hiddenSize; g++) {
            let sum = 0;
            const wIhBase = wIh.offset + g * wIhStride0;
            for (let k = 0; k < curInputSize; k++) {
              sum += (layerInput[k] ?? 0) * readNumeric(wIh, wIhBase + k * wIhStride1);
            }
            const wHhBase = wHh.offset + g * wHhStride0;
            for (let k = 0; k < this.hiddenSize; k++) {
              sum +=
                (hLayer[b * this.hiddenSize + k] ?? 0) * readNumeric(wHh, wHhBase + k * wHhStride1);
            }
            if (this.bias && biasIh && biasHh) {
              sum += readNumeric(biasIh, biasIh.offset + g * biasIhStride);
              sum += readNumeric(biasHh, biasHh.offset + g * biasHhStride);
            }
            gates[g] = sum;
          }

          const newH = new Float64Array(this.hiddenSize);
          const newC = new Float64Array(this.hiddenSize);

          for (let j = 0; j < this.hiddenSize; j++) {
            const iGate = this.sigmoid(gates[j] ?? 0);
            const fGate = this.sigmoid(gates[this.hiddenSize + j] ?? 0);
            const gGate = Math.tanh(gates[2 * this.hiddenSize + j] ?? 0);
            const oGate = this.sigmoid(gates[3 * this.hiddenSize + j] ?? 0);

            const prevC = cLayer[b * this.hiddenSize + j] ?? 0;
            const nextC = fGate * prevC + iGate * gGate;
            const nextH = oGate * Math.tanh(nextC);

            newC[j] = nextC;
            newH[j] = nextH;
          }

          for (let j = 0; j < this.hiddenSize; j++) {
            hLayer[b * this.hiddenSize + j] = newH[j] ?? 0;
            cLayer[b * this.hiddenSize + j] = newC[j] ?? 0;
          }

          layerInput = newH;
        }

        for (let j = 0; j < this.hiddenSize; j++) {
          const idx = outputIndex(
            this.batchFirst,
            isUnbatched,
            batch,
            seqLen,
            this.hiddenSize,
            b,
            t,
            j
          );
          out[idx] = layerInput[j] ?? 0;
        }
      }
    }

    const outShape = isUnbatched
      ? [seqLen, this.hiddenSize]
      : this.batchFirst
        ? [batch, seqLen, this.hiddenSize]
        : [seqLen, batch, this.hiddenSize];

    return {
      output: Tensor.fromTypedArray({
        data: out,
        shape: outShape,
        dtype: input.dtype,
        device: input.device,
      }),
      h: packState(
        h,
        this.numLayers,
        batch,
        this.hiddenSize,
        input.dtype,
        input.device,
        isUnbatched
      ),
      c: packState(
        c,
        this.numLayers,
        batch,
        this.hiddenSize,
        input.dtype,
        input.device,
        isUnbatched
      ),
    };
  }

  forward(...inputs: AnyTensor[]): Tensor {
    if (inputs.length < 1 || inputs.length > 3) {
      throw new InvalidParameterError(
        "LSTM.forward expects 1 to 3 inputs",
        "inputs",
        inputs.length
      );
    }
    const inputArg = inputs[0];
    if (inputArg === undefined) {
      throw new InvalidParameterError("LSTM.forward requires an input tensor", "input", inputArg);
    }
    const input = extractTensor(inputArg, "input");
    const hxArg = inputs.length >= 2 ? inputs[1] : undefined;
    const cxArg = inputs.length >= 3 ? inputs[2] : undefined;
    const hx = hxArg === undefined ? undefined : extractTensor(hxArg, "hx");
    const cx = cxArg === undefined ? undefined : extractTensor(cxArg, "cx");
    return this.run(input, hx, cx).output;
  }

  /**
   * Forward pass returning output, hidden state, and cell state.
   * Use this method when you need the hidden/cell states.
   */
  forwardWithState(input: AnyTensor, hx?: AnyTensor, cx?: AnyTensor): [Tensor, [Tensor, Tensor]] {
    const inputTensor = extractTensor(input, "input");
    const hxTensor = hx === undefined ? undefined : extractTensor(hx, "hx");
    const cxTensor = cx === undefined ? undefined : extractTensor(cx, "cx");
    const { output, h, c } = this.run(inputTensor, hxTensor, cxTensor);
    return [output, [h, c]];
  }

  override toString(): string {
    return `LSTM(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}

/**
 * GRU (Gated Recurrent Unit) layer.
 *
 * Applies a multi-layer GRU to an input sequence.
 *
 * **Gates**:
 * - Reset gate: r_t = σ(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
 * - Update gate: z_t = σ(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
 * - New gate: n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
 * - Hidden: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
 *
 * @see {@link https://deepbox.dev/docs/nn-recurrent | Deepbox Recurrent Layers}
 */
export class GRU extends Module {
  private readonly inputSize: number;
  private readonly hiddenSize: number;
  private readonly numLayers: number;
  private readonly bias: boolean;
  private readonly batchFirst: boolean;

  private weightsIh: Tensor[];
  private weightsHh: Tensor[];
  private biasIh: Tensor[];
  private biasHh: Tensor[];

  constructor(
    inputSize: number,
    hiddenSize: number,
    options: {
      readonly numLayers?: number;
      readonly bias?: boolean;
      readonly batchFirst?: boolean;
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

    const stdv = 1.0 / Math.sqrt(hiddenSize);

    this.weightsIh = [];
    this.weightsHh = [];
    this.biasIh = [];
    this.biasHh = [];

    for (let layer = 0; layer < this.numLayers; layer++) {
      const inputDim = layer === 0 ? inputSize : hiddenSize;

      const wIh = mulScalar(randn([3 * hiddenSize, inputDim]), stdv);
      const wHh = mulScalar(randn([3 * hiddenSize, hiddenSize]), stdv);

      this.weightsIh.push(wIh);
      this.weightsHh.push(wHh);
      this.registerParameter(`weight_ih_l${layer}`, parameter(wIh));
      this.registerParameter(`weight_hh_l${layer}`, parameter(wHh));

      if (this.bias) {
        const bIh = zeros([3 * hiddenSize]);
        const bHh = zeros([3 * hiddenSize]);
        this.biasIh.push(bIh);
        this.biasHh.push(bHh);
        this.registerParameter(`bias_ih_l${layer}`, parameter(bIh));
        this.registerParameter(`bias_hh_l${layer}`, parameter(bHh));
      }
    }
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private run(input: Tensor, hx?: Tensor): { output: Tensor; h: Tensor } {
    ensureFloatTensor(input, "GRU");
    const parsed = parseInput(input, this.batchFirst);
    const { batch, seqLen, inputDim, isUnbatched, batchStride, seqStride, featStride } = parsed;

    if (inputDim !== this.inputSize) {
      throw new ShapeError(`Expected input size ${this.inputSize}, got ${inputDim}`);
    }
    if (seqLen <= 0) {
      throw new InvalidParameterError("Sequence length must be positive", "seqLen", seqLen);
    }
    if (!isUnbatched && batch <= 0) {
      throw new InvalidParameterError("Batch size must be positive", "batch", batch);
    }

    const h = buildState(hx, this.numLayers, batch, this.hiddenSize, isUnbatched, "hx");
    const outSize = (isUnbatched ? seqLen : batch * seqLen) * this.hiddenSize;
    const out = createFloatBuffer(outSize, input.dtype);

    const inputVec = new Float64Array(inputDim);
    const gatesIh = new Float64Array(3 * this.hiddenSize);
    const gatesHh = new Float64Array(3 * this.hiddenSize);

    for (let t = 0; t < seqLen; t++) {
      for (let b = 0; b < batch; b++) {
        const baseOffset = input.offset + b * batchStride + t * seqStride;
        for (let i = 0; i < inputDim; i++) {
          inputVec[i] = readNumeric(input, baseOffset + i * featStride);
        }

        let layerInput = inputVec;

        for (let l = 0; l < this.numLayers; l++) {
          const wIh = this.weightsIh[l];
          const wHh = this.weightsHh[l];
          if (!wIh || !wHh) {
            throw new ShapeError("Internal error: missing GRU weights");
          }

          const curInputSize = l === 0 ? this.inputSize : this.hiddenSize;
          const hLayer = h[l];
          if (!hLayer) {
            throw new ShapeError("Internal error: missing GRU hidden state");
          }

          const wIhStride0 = wIh.strides[0] ?? 0;
          const wIhStride1 = wIh.strides[1] ?? 0;
          const wHhStride0 = wHh.strides[0] ?? 0;
          const wHhStride1 = wHh.strides[1] ?? 0;
          const biasIh = this.biasIh[l];
          const biasHh = this.biasHh[l];
          const biasIhStride = biasIh ? (biasIh.strides[0] ?? 0) : 0;
          const biasHhStride = biasHh ? (biasHh.strides[0] ?? 0) : 0;

          for (let g = 0; g < 3 * this.hiddenSize; g++) {
            let sumIh = 0;
            let sumHh = 0;
            const wIhBase = wIh.offset + g * wIhStride0;
            for (let k = 0; k < curInputSize; k++) {
              sumIh += (layerInput[k] ?? 0) * readNumeric(wIh, wIhBase + k * wIhStride1);
            }
            const wHhBase = wHh.offset + g * wHhStride0;
            for (let k = 0; k < this.hiddenSize; k++) {
              sumHh +=
                (hLayer[b * this.hiddenSize + k] ?? 0) * readNumeric(wHh, wHhBase + k * wHhStride1);
            }
            if (this.bias && biasIh && biasHh) {
              sumIh += readNumeric(biasIh, biasIh.offset + g * biasIhStride);
              sumHh += readNumeric(biasHh, biasHh.offset + g * biasHhStride);
            }
            gatesIh[g] = sumIh;
            gatesHh[g] = sumHh;
          }

          const newH = new Float64Array(this.hiddenSize);
          for (let j = 0; j < this.hiddenSize; j++) {
            const r = this.sigmoid((gatesIh[j] ?? 0) + (gatesHh[j] ?? 0));
            const z = this.sigmoid(
              (gatesIh[this.hiddenSize + j] ?? 0) + (gatesHh[this.hiddenSize + j] ?? 0)
            );
            const n = Math.tanh(
              (gatesIh[2 * this.hiddenSize + j] ?? 0) + r * (gatesHh[2 * this.hiddenSize + j] ?? 0)
            );
            newH[j] = (1 - z) * n + z * (hLayer[b * this.hiddenSize + j] ?? 0);
          }

          for (let j = 0; j < this.hiddenSize; j++) {
            hLayer[b * this.hiddenSize + j] = newH[j] ?? 0;
          }

          layerInput = newH;
        }

        for (let j = 0; j < this.hiddenSize; j++) {
          const idx = outputIndex(
            this.batchFirst,
            isUnbatched,
            batch,
            seqLen,
            this.hiddenSize,
            b,
            t,
            j
          );
          out[idx] = layerInput[j] ?? 0;
        }
      }
    }

    const outShape = isUnbatched
      ? [seqLen, this.hiddenSize]
      : this.batchFirst
        ? [batch, seqLen, this.hiddenSize]
        : [seqLen, batch, this.hiddenSize];

    return {
      output: Tensor.fromTypedArray({
        data: out,
        shape: outShape,
        dtype: input.dtype,
        device: input.device,
      }),
      h: packState(
        h,
        this.numLayers,
        batch,
        this.hiddenSize,
        input.dtype,
        input.device,
        isUnbatched
      ),
    };
  }

  forward(...inputs: AnyTensor[]): Tensor {
    if (inputs.length < 1 || inputs.length > 2) {
      throw new InvalidParameterError("GRU.forward expects 1 or 2 inputs", "inputs", inputs.length);
    }
    const inputArg = inputs[0];
    if (inputArg === undefined) {
      throw new InvalidParameterError("GRU.forward requires an input tensor", "input", inputArg);
    }
    const input = extractTensor(inputArg, "input");
    const hxArg = inputs.length === 2 ? inputs[1] : undefined;
    const hx = hxArg === undefined ? undefined : extractTensor(hxArg, "hx");
    return this.run(input, hx).output;
  }

  /**
   * Forward pass returning both output and hidden state.
   * Use this method when you need the hidden state.
   */
  forwardWithState(input: AnyTensor, hx?: AnyTensor): [Tensor, Tensor] {
    const inputTensor = extractTensor(input, "input");
    const hxTensor = hx === undefined ? undefined : extractTensor(hx, "hx");
    const { output, h } = this.run(inputTensor, hxTensor);
    return [output, h];
  }

  override toString(): string {
    return `GRU(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers})`;
  }
}
