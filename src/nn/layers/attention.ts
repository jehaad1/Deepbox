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
 * @see {@link https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html | PyTorch MultiheadAttention}
 * @see Vaswani et al. (2017) "Attention Is All You Need"
 */
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
  forward(...inputs: AnyTensor[]): GradTensor {
    if (inputs.length < 1 || inputs.length > 3) {
      throw new InvalidParameterError(
        "MultiheadAttention.forward expects 1 to 3 input tensors",
        "inputs",
        inputs.length
      );
    }

    const queryInput = inputs[0];
    if (queryInput === undefined) {
      throw new InvalidParameterError("Query tensor is required", "query", queryInput);
    }

    // Auto-convert to GradTensor
    const query = queryInput instanceof GradTensor ? queryInput : GradTensor.fromTensor(queryInput);

    const keyInput = inputs[1] ?? queryInput;
    const key = keyInput instanceof GradTensor ? keyInput : GradTensor.fromTensor(keyInput);

    const valueInput = inputs[2] ?? queryInput;
    const value = valueInput instanceof GradTensor ? valueInput : GradTensor.fromTensor(valueInput);

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
 * @see {@link https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html | PyTorch TransformerEncoderLayer}
 */
export class TransformerEncoderLayer extends Module {
  private readonly dModel: number;
  private readonly nHead: number;
  private readonly dFF: number;

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
    dModel: number,
    nHead: number,
    dFF: number,
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
    this.dropout = dropout;

    this.selfAttn = new MultiheadAttention(dModel, nHead, { dropout });
    this.linear1 = new Linear(dModel, dFF);
    this.linear2 = new Linear(dFF, dModel);
    this.norm1 = new LayerNorm(dModel, { eps });
    this.norm2 = new LayerNorm(dModel, { eps });
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

  /**
   * Forward pass of the Transformer encoder layer.
   *
   * @param src - Source sequence of shape (batch, seqLen, dModel)
   * @returns Output of same shape as input
   */
  forward(src: AnyTensor): GradTensor {
    const input = src instanceof GradTensor ? src : GradTensor.fromTensor(src);

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
