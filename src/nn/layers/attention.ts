import { DTypeError, ensureNumericDType, InvalidParameterError, ShapeError } from "../../core";
import {
  type AnyTensor,
  dropoutGrad,
  GradTensor,
  mulScalar,
  parameter,
  randn,
  softmaxGrad,
  zeros,
} from "../../ndarray";
import { Tensor } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { Dropout } from "./dropout";
import { Linear } from "./linear";
import { LayerNorm } from "./normalization";

/**
 * Multi-Head Attention mechanism.
 *
 * Allows the model to jointly attend to information from different representation
 * subspaces at different positions. This is the core building block of Transformers.
 *
 * **Mathematical Formulation**:
 * ```
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
 * where head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)
 * ```
 *
 * @example
 * ```ts
 * import { MultiheadAttention } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const mha = new MultiheadAttention(512, 8);
 * const x = tensor([[/* ... sequence data ... *\/]]);
 * const output = mha.forward(x, x, x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 * @see Vaswani et al. (2017) "Attention Is All You Need"
 */
/**
 * Build an additive causal (autoregressive) attention mask of shape [L, L]:
 * 0 on and below the diagonal, a large negative value above it. Adding this
 * to attention scores before softmax prevents each position from attending to
 * future positions.
 */
export function causalMask(seqLen: number): Tensor {
  const data = new Float64Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      data[i * seqLen + j] = j <= i ? 0 : -1e9;
    }
  }
  return Tensor.fromTypedArray({
    data,
    shape: [seqLen, seqLen],
    dtype: "float64",
    device: "cpu",
  });
}

export class MultiheadAttention extends Module {
  /** Embedding dimension */
  private readonly embedDim: number;

  /** Number of attention heads */
  private readonly numHeads: number;

  /** Dimension of each head */
  private readonly headDim: number;

  /** Scaling factor for dot product attention */
  private readonly scale: number;

  /** Whether to add bias to projections */
  private readonly useBias: boolean;

  /** Dropout probability applied to attention weights */
  private readonly dropout: number;

  /** Query projection weights (embedDim, embedDim) */
  private wQ: GradTensor;
  private bQ?: GradTensor;

  /** Key projection weights (embedDim, embedDim) */
  private wK: GradTensor;
  private bK?: GradTensor;

  /** Value projection weights (embedDim, embedDim) */
  private wV: GradTensor;
  private bV?: GradTensor;

  /** Output projection weights (embedDim, embedDim) */
  private wO: GradTensor;
  private bO?: GradTensor;

  /**
   * Create a new MultiheadAttention layer.
   *
   * @param embedDim - Total dimension of the model (must be divisible by numHeads)
   * @param numHeads - Number of parallel attention heads
   * @param options - Configuration options
   * @param options.bias - Whether to add bias to projections (default: true)
   * @param options.dropout - Dropout probability applied to attention weights (default: 0.0)
   */
  constructor(
    embedDim: number,
    numHeads: number,
    options: {
      readonly bias?: boolean;
      readonly dropout?: number;
    } = {}
  ) {
    super();

    if (!Number.isInteger(embedDim) || embedDim <= 0) {
      throw new InvalidParameterError("embedDim must be a positive integer", "embedDim", embedDim);
    }
    if (!Number.isInteger(numHeads) || numHeads <= 0) {
      throw new InvalidParameterError("numHeads must be a positive integer", "numHeads", numHeads);
    }
    if (embedDim % numHeads !== 0) {
      throw new InvalidParameterError(
        `embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`,
        "embedDim",
        embedDim
      );
    }

    const dropout = options.dropout ?? 0.0;
    if (!Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
      throw new InvalidParameterError("dropout must be in [0, 1)", "dropout", dropout);
    }

    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.scale = Math.sqrt(this.headDim);
    this.useBias = options.bias ?? true;
    this.dropout = dropout;

    // Initialize projection weights using Xavier/Glorot initialization
    const stdDev = Math.sqrt(2.0 / (embedDim + embedDim));

    // Query, Key, Value projections
    // We use GradTensor parameter directly
    this.wQ = parameter(mulScalar(randn([embedDim, embedDim]), stdDev));
    this.wK = parameter(mulScalar(randn([embedDim, embedDim]), stdDev));
    this.wV = parameter(mulScalar(randn([embedDim, embedDim]), stdDev));
    this.wO = parameter(mulScalar(randn([embedDim, embedDim]), stdDev));

    this.registerParameter("in_proj_weight_q", this.wQ);
    this.registerParameter("in_proj_weight_k", this.wK);
    this.registerParameter("in_proj_weight_v", this.wV);
    this.registerParameter("out_proj_weight", this.wO);

    if (this.useBias) {
      this.bQ = parameter(zeros([embedDim]));
      this.bK = parameter(zeros([embedDim]));
      this.bV = parameter(zeros([embedDim]));
      this.bO = parameter(zeros([embedDim]));

      this.registerParameter("in_proj_bias_q", this.bQ);
      this.registerParameter("in_proj_bias_k", this.bK);
      this.registerParameter("in_proj_bias_v", this.bV);
      this.registerParameter("out_proj_bias", this.bO);
    }
  }

  /**
   * Forward pass of multi-head attention.
   *
   * @param query - Query tensor of shape (batch, seqLen, embedDim)
   * @param key - Key tensor of shape (batch, seqLen, embedDim)
   * @param value - Value tensor of shape (batch, seqLen, embedDim)
   * @returns Output tensor of same shape as query
   */
  /**
   * @param inputs query, [key], [value], and an optional 4th additive
   *   attention mask broadcastable to (…, Lq, Lk). Masked positions should be
   *   -Infinity (they receive ~0 weight after softmax); use
   *   {@link causalMask} for autoregressive self-attention.
   */
  forward(...inputs: AnyTensor[]): GradTensor {
    if (inputs.length < 1 || inputs.length > 4) {
      throw new InvalidParameterError(
        "MultiheadAttention.forward expects 1 to 4 input tensors (query, key, value, attnMask)",
        "inputs",
        inputs.length
      );
    }
    const attnMaskInput = inputs[3];
    const attnMask =
      attnMaskInput === undefined
        ? undefined
        : GradTensor.isGradTensor(attnMaskInput)
          ? attnMaskInput
          : GradTensor.fromTensor(attnMaskInput);

    const queryInput = inputs[0];
    if (queryInput === undefined) {
      throw new InvalidParameterError("Query tensor is required", "query", queryInput);
    }

    // Auto-convert to GradTensor
    const query = GradTensor.isGradTensor(queryInput)
      ? queryInput
      : GradTensor.fromTensor(queryInput);

    const keyInput = inputs[1] ?? queryInput;
    const key = GradTensor.isGradTensor(keyInput) ? keyInput : GradTensor.fromTensor(keyInput);

    const valueInput = inputs[2] ?? queryInput;
    const value = GradTensor.isGradTensor(valueInput)
      ? valueInput
      : GradTensor.fromTensor(valueInput);

    if (query.dtype === "string") throw new DTypeError("String tensors are not supported");
    if (query.ndim !== key.ndim || query.ndim !== value.ndim) {
      throw new ShapeError("query, key, and value must have same rank");
    }
    if (query.ndim !== 2 && query.ndim !== 3) {
      throw new ShapeError(`Query must be 2D or 3D; got ndim=${query.ndim}`);
    }
    if (key.ndim !== 2 && key.ndim !== 3) {
      throw new ShapeError(`Key must be 2D or 3D; got ndim=${key.ndim}`);
    }
    if (value.ndim !== 2 && value.ndim !== 3) {
      throw new ShapeError(`Value must be 2D or 3D; got ndim=${value.ndim}`);
    }

    // Shape convention: (Batch, SeqLen, EmbedDim) for 3D inputs.
    // If 2D (SeqLen, EmbedDim), we treat as (1, SeqLen, EmbedDim).

    let q = query;
    let k = key;
    let v = value;

    if (q.ndim === 2) q = q.reshape([1, q.shape[0] ?? 0, q.shape[1] ?? 0]);
    if (k.ndim === 2) k = k.reshape([1, k.shape[0] ?? 0, k.shape[1] ?? 0]);
    if (v.ndim === 2) v = v.reshape([1, v.shape[0] ?? 0, v.shape[1] ?? 0]);

    const batchSize = q.shape[0] ?? 0;
    const seqLenQ = q.shape[1] ?? 0;
    const seqLenK = k.shape[1] ?? 0;
    const seqLenV = v.shape[1] ?? 0;
    const embedDim = q.shape[2] ?? 0;

    if (embedDim !== this.embedDim) {
      throw new ShapeError(`Query embedDim mismatch: expected ${this.embedDim}, got ${embedDim}`);
    }
    if (k.shape[2] !== this.embedDim) {
      throw new ShapeError(`Key embedDim mismatch: expected ${this.embedDim}, got ${k.shape[2]}`);
    }
    if (v.shape[2] !== this.embedDim) {
      throw new ShapeError(`Value embedDim mismatch: expected ${this.embedDim}, got ${v.shape[2]}`);
    }
    if (k.shape[0] !== batchSize || v.shape[0] !== batchSize) {
      throw new ShapeError(
        `batch size mismatch: query=${batchSize}, key=${k.shape[0]}, value=${v.shape[0]}`
      );
    }
    if (seqLenK !== seqLenV) {
      throw new ShapeError(`Key/value sequence length mismatch: key=${seqLenK}, value=${seqLenV}`);
    }

    // Linear projections
    // Q * WQ^T + bQ
    // (B, L, E) @ (E, E) -> (B, L, E)
    let Q = q.matmul(this.wQ.transpose());
    if (this.bQ) Q = Q.add(this.bQ);

    let K = k.matmul(this.wK.transpose());
    if (this.bK) K = K.add(this.bK);

    let V = v.matmul(this.wV.transpose());
    if (this.bV) V = V.add(this.bV);

    // Split heads
    // (B, L, E) -> (B, L, H, D) -> (B, H, L, D)
    const H = this.numHeads;
    const D = this.headDim;

    Q = Q.reshape([batchSize, seqLenQ, H, D]).transpose([0, 2, 1, 3]);
    K = K.reshape([batchSize, seqLenK, H, D]).transpose([0, 2, 1, 3]);
    V = V.reshape([batchSize, seqLenV, H, D]).transpose([0, 2, 1, 3]);

    // Scaled Dot-Product Attention
    // Scores = Q @ K^T / sqrt(D)
    // (B, H, Lq, D) @ (B, H, D, Lk) -> (B, H, Lq, Lk)
    let scores = Q.matmul(K.transpose([0, 1, 3, 2]));
    scores = scores.div(GradTensor.scalar(this.scale));

    // Additive attention mask (e.g. causal / padding): broadcast-add before
    // softmax so masked positions (-inf) get ~0 weight.
    if (attnMask) {
      scores = scores.add(attnMask.astype(ensureNumericDType(scores.dtype, "attnMask")));
    }

    // Softmax
    let attn = softmaxGrad(scores, -1);

    // Dropout
    attn = dropoutGrad(attn, this.dropout, this.training);

    // Weighted Sum
    // (B, H, Lq, Lk) @ (B, H, Lv, D) -> (B, H, Lq, D)
    // Note: Lk == Lv usually
    const context = attn.matmul(V);

    // Concat heads
    // (B, H, Lq, D) -> (B, Lq, H, D) -> (B, Lq, E)
    const contextDtype = ensureNumericDType(context.dtype, "MultiheadAttention");
    const contextReshaped = context
      .transpose([0, 2, 1, 3])
      .mul(GradTensor.scalar(1, { dtype: contextDtype }))
      .reshape([batchSize, seqLenQ, this.embedDim]);

    // Output projection
    let output = contextReshaped.matmul(this.wO.transpose());
    if (this.bO) output = output.add(this.bO);

    // If input was 2D, squeeze back
    if (query.ndim === 2) {
      output = output.reshape([seqLenQ, this.embedDim]);
    }

    return output;
  }

  override toString(): string {
    return `MultiheadAttention(embed_dim=${this.embedDim}, num_heads=${this.numHeads})`;
  }
}

/**
 * Transformer Encoder Layer.
 *
 * A single layer of the Transformer encoder, consisting of:
 * 1. Multi-head self-attention
 * 2. Add & Norm (residual connection + layer normalization)
 * 3. Feed-forward network (FFN)
 * 4. Add & Norm
 *
 * @example
 * ```ts
 * import { TransformerEncoderLayer } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const layer = new TransformerEncoderLayer(512, 8, 2048);
 * const x = tensor([[/* sequence data *\/]]);
 * const output = layer.forward(x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */
export class TransformerEncoderLayer extends Module {
  private readonly dModel: number;
  private readonly nHead: number;
  private readonly dFF: number;
  private readonly eps: number;

  private readonly selfAttn: MultiheadAttention;
  private readonly linear1: Linear;
  private readonly linear2: Linear;
  private readonly norm1: LayerNorm;
  private readonly norm2: LayerNorm;

  private readonly dropout: number;
  // We use functional dropout in forward, or could use Dropout module.
  // Using Dropout module is cleaner.
  private readonly dropout1: Dropout;
  private readonly dropout2: Dropout;
  private readonly dropout3: Dropout;

  constructor(
    dModelOrOpts:
      | number
      | {
          readonly dModel: number;
          readonly nHead: number;
          readonly dimFeedforward?: number;
          readonly dFF?: number;
          readonly dropout?: number;
          readonly eps?: number;
        },
    nHead?: number,
    dFFOrOptions?:
      | number
      | {
          readonly dimFeedforward?: number;
          readonly dFF?: number;
          readonly dropout?: number;
          readonly eps?: number;
        },
    options: {
      readonly dropout?: number;
      readonly eps?: number;
    } = {}
  ) {
    super();

    let resolvedDModel: number;
    let resolvedNHead: number;
    let resolvedDFF: number;
    let resolvedDropout: number | undefined;
    let resolvedEps: number | undefined;

    if (typeof dModelOrOpts === "object") {
      resolvedDModel = dModelOrOpts.dModel;
      resolvedNHead = dModelOrOpts.nHead;
      resolvedDFF = dModelOrOpts.dFF ?? dModelOrOpts.dimFeedforward ?? 2048;
      resolvedDropout = dModelOrOpts.dropout;
      resolvedEps = dModelOrOpts.eps;
    } else if (typeof dFFOrOptions === "object") {
      resolvedDModel = dModelOrOpts;
      resolvedNHead = nHead ?? 1;
      resolvedDFF = dFFOrOptions.dFF ?? dFFOrOptions.dimFeedforward ?? 2048;
      resolvedDropout = dFFOrOptions.dropout;
      resolvedEps = dFFOrOptions.eps;
    } else {
      resolvedDModel = dModelOrOpts;
      resolvedNHead = nHead ?? 1;
      resolvedDFF = dFFOrOptions ?? 2048;
      resolvedDropout = options.dropout;
      resolvedEps = options.eps;
    }

    const dModel = resolvedDModel;
    if (!Number.isInteger(dModel) || dModel <= 0) {
      throw new InvalidParameterError("dModel must be a positive integer", "dModel", dModel);
    }
    if (!Number.isInteger(resolvedNHead) || resolvedNHead <= 0) {
      throw new InvalidParameterError("nHead must be a positive integer", "nHead", resolvedNHead);
    }
    if (dModel % resolvedNHead !== 0) {
      throw new InvalidParameterError(
        `dModel (${dModel}) must be divisible by nHead (${resolvedNHead})`,
        "dModel",
        dModel
      );
    }
    if (!Number.isInteger(resolvedDFF) || resolvedDFF <= 0) {
      throw new InvalidParameterError("dFF must be a positive integer", "dFF", resolvedDFF);
    }

    const dropout = resolvedDropout ?? 0.1;
    const epsVal = resolvedEps ?? 1e-5;

    this.dModel = dModel;
    this.nHead = resolvedNHead;
    this.dFF = resolvedDFF;
    this.eps = epsVal;
    this.dropout = dropout;

    this.selfAttn = new MultiheadAttention(dModel, resolvedNHead, { dropout });
    this.linear1 = new Linear(dModel, resolvedDFF);
    this.linear2 = new Linear(resolvedDFF, dModel);
    this.norm1 = new LayerNorm(dModel, { eps: epsVal });
    this.norm2 = new LayerNorm(dModel, { eps: epsVal });
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);

    this.registerModule("self_attn", this.selfAttn);
    this.registerModule("linear1", this.linear1);
    this.registerModule("linear2", this.linear2);
    this.registerModule("norm1", this.norm1);
    this.registerModule("norm2", this.norm2);
    this.registerModule("dropout1", this.dropout1);
    this.registerModule("dropout2", this.dropout2);
    this.registerModule("dropout3", this.dropout3);
  }

  clone(): TransformerEncoderLayer {
    return new TransformerEncoderLayer(this.dModel, this.nHead, this.dFF, {
      dropout: this.dropout,
      eps: this.eps,
    });
  }

  /**
   * Forward pass of the Transformer encoder layer.
   *
   * @param src - Source sequence of shape (batch, seqLen, dModel)
   * @returns Output of same shape as input
   */
  forward(src: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(src) ? src : GradTensor.fromTensor(src);

    if (input.dtype === "string") {
      throw new DTypeError("TransformerEncoderLayer does not support string dtype");
    }

    // 1. Self Attention
    // src2 = self_attn(src, src, src)
    let src2 = this.selfAttn.forward(input, input, input);

    // src = src + dropout(src2)
    src2 = this.dropout1.forward(src2);
    let out = input.add(src2);

    // src = norm1(src)
    out = this.norm1.forward(out);

    // 2. Feed Forward
    // src2 = linear2(dropout(relu(linear1(src))))
    // We implement FFN manually with modules
    let ffn = this.linear1.forward(out);
    ffn = ffn.relu();
    ffn = this.dropout2.forward(ffn);
    ffn = this.linear2.forward(ffn);

    // src = src + dropout(src2)
    ffn = this.dropout3.forward(ffn);
    out = out.add(ffn);

    // src = norm2(src)
    out = this.norm2.forward(out);

    return out;
  }

  override toString(): string {
    return `TransformerEncoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dFF}, dropout=${this.dropout})`;
  }
}

/**
 * Transformer Decoder Layer.
 *
 * A single layer of the Transformer decoder, consisting of:
 * 1. Masked multi-head self-attention
 * 2. Add & Norm
 * 3. Multi-head cross-attention (attending to encoder output)
 * 4. Add & Norm
 * 5. Feed-forward network
 * 6. Add & Norm
 *
 * @example
 * ```ts
 * const layer = new TransformerDecoderLayer(512, 8, 2048);
 * const tgt = tensor([[/* target sequence *\/]]);
 * const memory = tensor([[/* encoder output *\/]]);
 * const output = layer.forward(tgt, memory);
 * ```
 */
export class TransformerDecoderLayer extends Module {
  private readonly dModel: number;
  private readonly nHead: number;
  private readonly dFF: number;
  private readonly dropoutRate: number;
  private readonly eps: number;

  private readonly selfAttn: MultiheadAttention;
  private readonly crossAttn: MultiheadAttention;
  private readonly linear1: Linear;
  private readonly linear2: Linear;
  private readonly norm1: LayerNorm;
  private readonly norm2: LayerNorm;
  private readonly norm3: LayerNorm;
  private readonly dropout1: Dropout;
  private readonly dropout2: Dropout;
  private readonly dropout3: Dropout;
  private readonly dropout4: Dropout;

  constructor(
    dModel: number,
    nHead: number,
    dFF = 2048,
    options: {
      readonly dropout?: number;
      readonly eps?: number;
    } = {}
  ) {
    super();

    if (!Number.isInteger(dModel) || dModel <= 0) {
      throw new InvalidParameterError("dModel must be a positive integer", "dModel", dModel);
    }
    if (!Number.isInteger(nHead) || nHead <= 0) {
      throw new InvalidParameterError("nHead must be a positive integer", "nHead", nHead);
    }
    if (dModel % nHead !== 0) {
      throw new InvalidParameterError(
        `dModel (${dModel}) must be divisible by nHead (${nHead})`,
        "dModel",
        dModel
      );
    }
    if (!Number.isInteger(dFF) || dFF <= 0) {
      throw new InvalidParameterError("dFF must be a positive integer", "dFF", dFF);
    }

    const dropout = options.dropout ?? 0.1;
    const eps = options.eps ?? 1e-5;

    this.dModel = dModel;
    this.nHead = nHead;
    this.dFF = dFF;
    this.dropoutRate = dropout;
    this.eps = eps;

    this.selfAttn = new MultiheadAttention(dModel, nHead, { dropout });
    this.crossAttn = new MultiheadAttention(dModel, nHead, { dropout });
    this.linear1 = new Linear(dModel, dFF);
    this.linear2 = new Linear(dFF, dModel);
    this.norm1 = new LayerNorm(dModel, { eps });
    this.norm2 = new LayerNorm(dModel, { eps });
    this.norm3 = new LayerNorm(dModel, { eps });
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);
    this.dropout4 = new Dropout(dropout);

    this.registerModule("self_attn", this.selfAttn);
    this.registerModule("multihead_attn", this.crossAttn);
    this.registerModule("linear1", this.linear1);
    this.registerModule("linear2", this.linear2);
    this.registerModule("norm1", this.norm1);
    this.registerModule("norm2", this.norm2);
    this.registerModule("norm3", this.norm3);
    this.registerModule("dropout1", this.dropout1);
    this.registerModule("dropout2", this.dropout2);
    this.registerModule("dropout3", this.dropout3);
    this.registerModule("dropout4", this.dropout4);
  }

  clone(): TransformerDecoderLayer {
    return new TransformerDecoderLayer(this.dModel, this.nHead, this.dFF, {
      dropout: this.dropoutRate,
      eps: this.eps,
    });
  }

  /**
   * Forward pass of the Transformer decoder layer.
   *
   * @param tgt - Target sequence of shape (batch, seqLen, dModel)
   * @param memory - Encoder output (memory) from the second argument
   * @returns Output of same shape as tgt
   */
  forward(tgt: AnyTensor, ...rest: AnyTensor[]): GradTensor {
    const memoryInput = rest[0];
    if (!memoryInput) {
      throw new InvalidParameterError(
        "TransformerDecoderLayer requires memory (encoder output) as second argument",
        "memory",
        undefined
      );
    }

    const input = GradTensor.isGradTensor(tgt) ? tgt : GradTensor.fromTensor(tgt);
    const memory = GradTensor.isGradTensor(memoryInput)
      ? memoryInput
      : GradTensor.fromTensor(memoryInput);

    if (input.dtype === "string") {
      throw new DTypeError("TransformerDecoderLayer does not support string dtype");
    }

    // 1. Causally-masked self-attention: each target position may only attend
    // to itself and earlier positions (prevents information leakage from
    // future tokens during autoregressive decoding).
    const seqLenTgt = input.shape[input.ndim - 2] ?? 0;
    let tgt2 = this.selfAttn.forward(input, input, input, causalMask(seqLenTgt));
    tgt2 = this.dropout1.forward(tgt2);
    let out = input.add(tgt2);
    out = this.norm1.forward(out);

    // 2. Cross-attention (query from decoder, key/value from encoder)
    let cross = this.crossAttn.forward(out, memory, memory);
    cross = this.dropout2.forward(cross);
    out = out.add(cross);
    out = this.norm2.forward(out);

    // 3. Feed-forward network
    let ffn = this.linear1.forward(out);
    ffn = ffn.relu();
    ffn = this.dropout3.forward(ffn);
    ffn = this.linear2.forward(ffn);
    ffn = this.dropout4.forward(ffn);
    out = out.add(ffn);
    out = this.norm3.forward(out);

    return out;
  }

  override toString(): string {
    return `TransformerDecoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dFF}, dropout=${this.dropoutRate})`;
  }
}

/**
 * Transformer Encoder — a stack of N encoder layers.
 *
 * @example
 * ```ts
 * const encoderLayer = new TransformerEncoderLayer(512, 8, 2048);
 * const encoder = new TransformerEncoder(encoderLayer, 6);
 * const output = encoder.forward(src);
 * ```
 */
export class TransformerEncoder extends Module {
  private readonly layers: TransformerEncoderLayer[];
  private readonly norm: LayerNorm | undefined;

  constructor(
    encoderLayer: TransformerEncoderLayer,
    numLayers: number,
    options: {
      readonly norm?: LayerNorm;
    } = {}
  ) {
    super();

    if (!Number.isInteger(numLayers) || numLayers <= 0) {
      throw new InvalidParameterError(
        "numLayers must be a positive integer",
        "numLayers",
        numLayers
      );
    }

    // Clone the layer template by reusing its state for each layer
    // In practice we create separate layers that share the same architecture
    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      const layer = i === 0 ? encoderLayer : encoderLayer.clone();
      this.layers.push(layer);
      this.registerModule(`layers.${i}`, layer);
    }

    this.norm = options.norm;
    if (this.norm) {
      this.registerModule("norm", this.norm);
    }
  }

  forward(src: AnyTensor): GradTensor {
    let out = GradTensor.isGradTensor(src) ? src : GradTensor.fromTensor(src);

    for (const layer of this.layers) {
      out = layer.forward(out);
    }

    if (this.norm) {
      out = this.norm.forward(out);
    }

    return out;
  }

  override toString(): string {
    return `TransformerEncoder(num_layers=${this.layers.length})`;
  }
}

/**
 * Transformer Decoder — a stack of N decoder layers.
 *
 * @example
 * ```ts
 * const decoderLayer = new TransformerDecoderLayer(512, 8, 2048);
 * const decoder = new TransformerDecoder(decoderLayer, 6);
 * const output = decoder.forward(tgt, memory);
 * ```
 */
export class TransformerDecoder extends Module {
  private readonly layers: TransformerDecoderLayer[];
  private readonly norm: LayerNorm | undefined;

  constructor(
    decoderLayer: TransformerDecoderLayer,
    numLayers: number,
    options: {
      readonly norm?: LayerNorm;
    } = {}
  ) {
    super();

    if (!Number.isInteger(numLayers) || numLayers <= 0) {
      throw new InvalidParameterError(
        "numLayers must be a positive integer",
        "numLayers",
        numLayers
      );
    }

    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      const layer = i === 0 ? decoderLayer : decoderLayer.clone();
      this.layers.push(layer);
      this.registerModule(`layers.${i}`, layer);
    }

    this.norm = options.norm;
    if (this.norm) {
      this.registerModule("norm", this.norm);
    }
  }

  forward(tgt: AnyTensor, ...rest: AnyTensor[]): GradTensor {
    const memoryInput = rest[0];
    if (!memoryInput) {
      throw new InvalidParameterError(
        "TransformerDecoder requires memory (encoder output) as second argument",
        "memory",
        undefined
      );
    }

    let out = GradTensor.isGradTensor(tgt) ? tgt : GradTensor.fromTensor(tgt);
    const memory = GradTensor.isGradTensor(memoryInput)
      ? memoryInput
      : GradTensor.fromTensor(memoryInput);

    for (const layer of this.layers) {
      out = layer.forward(out, memory);
    }

    if (this.norm) {
      out = this.norm.forward(out);
    }

    return out;
  }

  override toString(): string {
    return `TransformerDecoder(num_layers=${this.layers.length})`;
  }
}

/**
 * Full Transformer model with encoder and decoder stacks.
 *
 * @example
 * ```ts
 * const transformer = new Transformer(512, 8, 6, 6, 2048);
 * const src = tensor([[/* source *\/]]);
 * const tgt = tensor([[/* target *\/]]);
 * const output = transformer.forward(src, tgt);
 * ```
 */
export class FullTransformer extends Module {
  private readonly encoder: TransformerEncoder;
  private readonly decoder: TransformerDecoder;

  constructor(
    dModel = 512,
    nHead = 8,
    numEncoderLayers = 6,
    numDecoderLayers = 6,
    dFF = 2048,
    options: {
      readonly dropout?: number;
      readonly eps?: number;
    } = {}
  ) {
    super();

    const encLayer = new TransformerEncoderLayer(dModel, nHead, dFF, options);
    const decLayer = new TransformerDecoderLayer(dModel, nHead, dFF, options);

    this.encoder = new TransformerEncoder(encLayer, numEncoderLayers);
    this.decoder = new TransformerDecoder(decLayer, numDecoderLayers);

    this.registerModule("encoder", this.encoder);
    this.registerModule("decoder", this.decoder);
  }

  /**
   * Forward pass of the full Transformer.
   *
   * @param src - Source sequence
   * @param tgt - Target sequence (second positional argument)
   * @returns Decoder output
   */
  forward(src: AnyTensor, ...rest: AnyTensor[]): GradTensor {
    const tgtInput = rest[0];
    if (!tgtInput) {
      throw new InvalidParameterError(
        "Transformer requires target tensor as second argument",
        "tgt",
        undefined
      );
    }

    const memory = this.encoder.forward(src);
    return this.decoder.forward(tgtInput, memory);
  }

  override toString(): string {
    return `Transformer(\n  ${this.encoder.toString()}\n  ${this.decoder.toString()}\n)`;
  }
}

/**
 * Sinusoidal Positional Encoding.
 *
 * Adds sinusoidal positional information to input embeddings using the
 * formula from "Attention Is All You Need":
 * ```
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * ```
 *
 * @example
 * ```ts
 * const pe = new PositionalEncoding(512, { maxLen: 5000 });
 * const x = tensor([[/* embedding *\/]]);
 * const output = pe.forward(x); // x + positional encoding
 * ```
 */
export class PositionalEncoding extends Module {
  private readonly dModel: number;
  private readonly dropoutModule: Dropout;
  private readonly maxLen: number;
  private peBuffer: Tensor;

  constructor(
    dModel: number,
    options: {
      readonly dropout?: number;
      readonly maxLen?: number;
    } = {}
  ) {
    super();

    if (!Number.isInteger(dModel) || dModel <= 0) {
      throw new InvalidParameterError("dModel must be a positive integer", "dModel", dModel);
    }

    const dropout = options.dropout ?? 0.1;
    const maxLen = options.maxLen ?? 5000;

    this.dModel = dModel;
    this.maxLen = maxLen;
    this.dropoutModule = new Dropout(dropout);
    this.registerModule("dropout", this.dropoutModule);

    // Precompute the positional encoding table
    const peData = new Float64Array(maxLen * dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        const angle = pos / 10000 ** ((2 * Math.floor(i / 2)) / dModel);
        if (i % 2 === 0) {
          peData[pos * dModel + i] = Math.sin(angle);
        } else {
          peData[pos * dModel + i] = Math.cos(angle);
        }
      }
    }

    this.peBuffer = Tensor.fromTypedArray({
      data: peData,
      shape: [maxLen, dModel],
      dtype: "float64",
      device: "cpu",
    });
    this.registerBuffer("pe", this.peBuffer);
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("PositionalEncoding does not support string dtype");
    }

    // x shape: (batch, seqLen, dModel) or (seqLen, dModel)
    let seqLen: number;
    if (input.ndim === 3) {
      seqLen = input.shape[1] ?? 0;
    } else if (input.ndim === 2) {
      seqLen = input.shape[0] ?? 0;
    } else {
      throw new ShapeError(`PositionalEncoding expects 2D or 3D input; got ${input.ndim}D`);
    }

    if (seqLen > this.maxLen) {
      throw new InvalidParameterError(
        `Sequence length ${seqLen} exceeds maxLen ${this.maxLen}`,
        "seqLen",
        seqLen
      );
    }

    // Slice PE to match sequence length: (seqLen, dModel)
    const peSlice = this.peBuffer.slice({ start: 0, end: seqLen });
    const peGrad = GradTensor.fromTensor(peSlice, { requiresGrad: false });

    // Add positional encoding to input
    const out = input.add(peGrad);
    return this.dropoutModule.forward(out);
  }

  override toString(): string {
    return `PositionalEncoding(d_model=${this.dModel}, max_len=${this.maxLen})`;
  }
}
