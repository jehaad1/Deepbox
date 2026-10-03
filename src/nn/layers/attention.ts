/**
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */

import { type DType, DTypeError, InvalidParameterError, ShapeError } from "../../core";
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
import { readNumbers } from "../../ndarray/ops/_internal";
import { Tensor } from "../../ndarray/tensor/Tensor";
import { ModuleList } from "../containers/ModuleList";
import { Module } from "../module/Module";
import { allPlain, type LayerDType, resolveLayerDtype, settle } from "./_shared";
import { Dropout } from "./dropout";
import { Linear } from "./linear";
import { LayerNorm } from "./normalization";

/** Additive score for a masked position. Large enough to give ~0 weight after softmax. */
const MASKED_SCORE = -1e9;

/**
 * Build an additive causal (autoregressive) attention mask of shape [L, L]:
 * 0 on and below the diagonal, -1e9 above it. Adding this to attention scores
 * before softmax prevents each position from attending to future positions.
 *
 * @param seqLen - Sequence length L (non-negative integer)
 * @returns float64 tensor of shape `[seqLen, seqLen]`
 * @throws {InvalidParameterError} If `seqLen` is not a non-negative integer
 *
 * @example
 * ```ts
 * import { causalMask } from 'deepbox/nn';
 *
 * causalMask(3).toArray();
 * // [[0, -1e9, -1e9], [0, 0, -1e9], [0, 0, 0]]
 * ```
 */
export function causalMask(seqLen: number): Tensor {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new InvalidParameterError("seqLen must be a non-negative integer", "seqLen", seqLen);
  }
  const data = new Float64Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    const row = i * seqLen;
    for (let j = i + 1; j < seqLen; j++) {
      data[row + j] = MASKED_SCORE;
    }
  }
  return Tensor.fromTypedArray({
    data,
    shape: [seqLen, seqLen],
    dtype: "float64",
    device: "cpu",
  });
}

/**
 * Cast to the parameter dtype of the layer, so that float64, float16 and integer inputs
 * work the way they do in `Linear`. The cast is differentiable.
 */
function toParamDtype(t: GradTensor, dtype: LayerDType): GradTensor {
  return t.dtype === dtype ? t : t.astype(dtype);
}

function asGrad(t: AnyTensor): GradTensor {
  return GradTensor.isGradTensor(t) ? t : GradTensor.fromTensor(t);
}

/**
 * Activation of the feed-forward network inside a Transformer layer: `"relu"` or the exact
 * (erf) `"gelu"`.
 *
 * @example
 * ```ts
 * import { TransformerEncoderLayer } from 'deepbox/nn';
 *
 * const layer = new TransformerEncoderLayer(64, 8, 256, { activation: 'gelu', normFirst: true });
 * ```
 */
export type TransformerActivation = "relu" | "gelu";

/** Validate the `activation` option of a Transformer layer. */
function resolveActivation(activation: TransformerActivation | undefined): TransformerActivation {
  const resolved: unknown = activation ?? "relu";
  if (resolved !== "relu" && resolved !== "gelu") {
    throw new InvalidParameterError(
      `activation must be "relu" or "gelu"; received ${String(resolved)}`,
      "activation",
      resolved
    );
  }
  return resolved;
}

/** Apply the feed-forward activation. `"gelu"` is the exact erf form, like PyTorch's `F.gelu`. */
function applyActivation(x: GradTensor, activation: TransformerActivation): GradTensor {
  return activation === "gelu" ? x.gelu("none") : x.relu();
}

/**
 * Turn a user attention mask into an additive float mask. Boolean masks follow the
 * PyTorch convention for `nn.MultiheadAttention`: `true` marks a position that may
 * not be attended to. Numeric masks are used as given (added to the scores).
 */
function resolveAttnMask(mask: AnyTensor): GradTensor {
  const raw = GradTensor.isGradTensor(mask) ? mask.tensor : mask;
  if (raw.dtype === "bool") {
    const flags = readNumbers(raw, "MultiheadAttention");
    const additive = new Float32Array(raw.size);
    for (let i = 0; i < additive.length; i++) {
      additive[i] = flags[i] !== 0 ? MASKED_SCORE : 0;
    }
    return GradTensor.fromTensor(
      Tensor.fromTypedArray({
        data: additive,
        shape: raw.shape,
        dtype: "float32",
        device: raw.device,
      })
    );
  }
  return asGrad(mask);
}

/** True when `size` is 1 or equals `target` (NumPy broadcasting rule for one axis). */
function broadcastsTo(size: number | undefined, target: number): boolean {
  return size === 1 || size === target;
}

/**
 * Options of {@link MultiheadAttention.forward}, passed as the last argument.
 *
 * @example
 * ```ts
 * import { MultiheadAttention } from 'deepbox/nn';
 * import { randn, tensor } from 'deepbox/ndarray';
 *
 * const mha = new MultiheadAttention(8, 2);
 * const x = randn([2, 5, 8]);
 * const keyPaddingMask = tensor([[false, false, false, true, true], [false, false, false, false, false]]);
 * const [out, weights] = mha.forward(x, x, x, undefined, { needWeights: true, keyPaddingMask });
 * // out: [2, 5, 8], weights: [2, 5, 5] (averaged over the 2 heads)
 * ```
 */
export type MultiheadAttentionForwardOptions = {
  /**
   * Also return the attention weights, as `[output, weights]` (default: false). The weights
   * are the softmax scores after attention dropout.
   */
  readonly needWeights?: boolean;
  /**
   * With `needWeights`, average the weights over the heads (default: true). The weights then
   * have shape `(batch, seqLenQ, seqLenK)`; with `false` they keep the head axis,
   * `(batch, numHeads, seqLenQ, seqLenK)`. Inputs without a batch axis give the same shapes
   * without the leading batch axis.
   */
  readonly averageAttnWeights?: boolean;
  /**
   * Marks the key positions to ignore, shape `(batch, seqLenK)` (or `(seqLenK)` for inputs
   * without a batch axis). A boolean mask uses `true` for a padded position; a float mask is
   * added to the scores before softmax.
   */
  readonly keyPaddingMask?: AnyTensor;
};

/** Tensor kind a layer returns for a query: a `GradTensor` stays one, a plain tensor may become one. */
type ForwardOutput<Q extends AnyTensor> = Q extends GradTensor ? GradTensor : AnyTensor;

/** Return type of {@link MultiheadAttention.forward} with options. */
type ForwardResult<Q extends AnyTensor, O extends MultiheadAttentionForwardOptions> = O extends {
  readonly needWeights: true;
}
  ? [ForwardOutput<Q>, ForwardOutput<Q>]
  : O extends { readonly needWeights?: false | undefined }
    ? ForwardOutput<Q>
    : ForwardOutput<Q> | [ForwardOutput<Q>, ForwardOutput<Q>];

/** True for the trailing options object of `forward` (anything that is not a tensor). */
function isForwardOptions(value: unknown): value is MultiheadAttentionForwardOptions {
  return (
    typeof value === "object" &&
    value !== null &&
    !(value instanceof Tensor) &&
    !GradTensor.isGradTensor(value)
  );
}

/**
 * Turn a key padding mask of shape `(batch, seqLenK)` (or `(seqLenK)`) into an additive
 * float mask of shape `(batch, 1, 1, seqLenK)`.
 */
function resolveKeyPaddingMask(
  mask: AnyTensor,
  batchSize: number,
  seqLenK: number,
  unbatched: boolean
): GradTensor {
  if (mask.dtype === "string") {
    throw new DTypeError("keyPaddingMask does not support string dtype");
  }
  const additive = resolveAttnMask(mask);
  const shape = additive.shape;
  const expected = unbatched ? `(${seqLenK})` : `(${batchSize}, ${seqLenK})`;
  const fits = unbatched
    ? (shape.length === 1 && shape[0] === seqLenK) ||
      (shape.length === 2 && shape[0] === 1 && shape[1] === seqLenK)
    : shape.length === 2 && shape[0] === batchSize && shape[1] === seqLenK;
  if (!fits) {
    throw new ShapeError(
      `keyPaddingMask shape [${shape.join(", ")}] does not match the expected ${expected}`
    );
  }
  return additive.reshape([unbatched ? 1 : batchSize, 1, 1, seqLenK]);
}

/**
 * Copy parameters, buffers, trainability and train/eval mode from `source` to
 * `target` (same architecture), so the target is an independent deep copy.
 */
function copyModuleState<T extends Module>(source: Module, target: T): T {
  target.loadStateDict(source.stateDict());
  const targetParams = new Map(target.namedParameters());
  for (const [name, param] of source.namedParameters()) {
    const copy = targetParams.get(name);
    if (copy && copy.requiresGrad !== param.requiresGrad) {
      copy.setRequiresGrad(param.requiresGrad);
    }
  }
  target.train(source.training);
  return target;
}

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
 * Parameters are float32 unless the `dtype` option (or the global default dtype) says
 * otherwise. The layer computes in the parameter dtype and casts its inputs to it. A
 * `GradTensor` input gives a `GradTensor`; plain tensors give a `GradTensor` that tracks the
 * weights while they require grad and gradient tracking is on, and a plain `Tensor`
 * otherwise (inside `noGrad()` or with frozen weights).
 *
 * @example
 * ```ts
 * import { MultiheadAttention, causalMask } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const mha = new MultiheadAttention(64, 8);
 * const x = randn([2, 10, 64]); // (batch, seqLen, embedDim)
 * const out = mha.forward(x, x, x); // shape [2, 10, 64]
 * const masked = mha.forward(x, x, x, causalMask(10)); // each step sees only earlier steps
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
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

  /** Dtype of the parameters; inputs are cast to it */
  private readonly paramDtype: LayerDType;

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
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    embedDim: number,
    numHeads: number,
    options: {
      readonly bias?: boolean;
      readonly dropout?: number;
      readonly dtype?: LayerDType;
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
    this.paramDtype = resolveLayerDtype(options.dtype);
    const dtypeOpt = { dtype: this.paramDtype };

    // Initialize projection weights using Xavier/Glorot initialization
    const stdDev = Math.sqrt(2.0 / (embedDim + embedDim));

    // Query, Key, Value projections
    // We use GradTensor parameter directly
    this.wQ = parameter(mulScalar(randn([embedDim, embedDim], dtypeOpt), stdDev));
    this.wK = parameter(mulScalar(randn([embedDim, embedDim], dtypeOpt), stdDev));
    this.wV = parameter(mulScalar(randn([embedDim, embedDim], dtypeOpt), stdDev));
    this.wO = parameter(mulScalar(randn([embedDim, embedDim], dtypeOpt), stdDev));

    this.registerParameter("in_proj_weight_q", this.wQ);
    this.registerParameter("in_proj_weight_k", this.wK);
    this.registerParameter("in_proj_weight_v", this.wV);
    this.registerParameter("out_proj_weight", this.wO);

    if (this.useBias) {
      this.bQ = parameter(zeros([embedDim], dtypeOpt));
      this.bK = parameter(zeros([embedDim], dtypeOpt));
      this.bV = parameter(zeros([embedDim], dtypeOpt));
      this.bO = parameter(zeros([embedDim], dtypeOpt));

      this.registerParameter("in_proj_bias_q", this.bQ);
      this.registerParameter("in_proj_bias_k", this.bK);
      this.registerParameter("in_proj_bias_v", this.bV);
      this.registerParameter("out_proj_bias", this.bO);
    }
  }

  /**
   * Forward pass of multi-head attention.
   *
   * @param inputs - `query`, then optionally `key`, `value` and `attnMask`, then optionally an
   *   options object ({@link MultiheadAttentionForwardOptions}). Pass `undefined` to skip a
   *   tensor before the options:
   *   - `query` of shape (batch, seqLenQ, embedDim), or (seqLenQ, embedDim) for a single sequence
   *   - `key` of shape (batch, seqLenK, embedDim); defaults to `query`
   *   - `value` of shape (batch, seqLenK, embedDim); defaults to `key`
   *   - `attnMask`, broadcastable to (batch, numHeads, seqLenQ, seqLenK). A float mask is added
   *     to the scores before softmax, so masked positions should hold a large negative number
   *     (`-1e9`, or `-Infinity`; a row that is masked everywhere then gives NaN). A boolean mask
   *     marks the positions that must NOT be attended to with `true`. {@link causalMask}
   *     builds the mask for autoregressive self-attention.
   *   - options: `needWeights` (return `[output, weights]`), `averageAttnWeights` (default
   *     true) and `keyPaddingMask`
   * @returns Output of the same shape as `query`, in the parameter dtype; with
   *   `needWeights: true` the pair `[output, weights]`
   * @throws {InvalidParameterError} If fewer than 1 or more than 4 tensors are passed
   * @throws {ShapeError} If ranks, embedding sizes, batch sizes or a mask shape do not match
   * @throws {DTypeError} For string tensors
   *
   * @example
   * ```ts
   * const [out, weights] = mha.forward(x, x, x, undefined, { needWeights: true });
   * // weights: (batch, seqLen, seqLen), averaged over the heads
   * const padded = mha.forward(x, x, x, undefined, { keyPaddingMask: padMask }); // (batch, seqLen) bool
   * ```
   */
  forward<Q extends AnyTensor, O extends MultiheadAttentionForwardOptions>(
    query: Q,
    key: AnyTensor | undefined,
    value: AnyTensor | undefined,
    attnMask: AnyTensor | undefined,
    options: O
  ): ForwardResult<Q, O>;
  forward<Q extends AnyTensor, O extends MultiheadAttentionForwardOptions>(
    query: Q,
    key: AnyTensor | undefined,
    value: AnyTensor | undefined,
    options: O
  ): ForwardResult<Q, O>;
  forward<Q extends AnyTensor, O extends MultiheadAttentionForwardOptions>(
    query: Q,
    options: O
  ): ForwardResult<Q, O>;
  forward(query: GradTensor, ...rest: AnyTensor[]): GradTensor;
  forward(query: Tensor, ...rest: AnyTensor[]): AnyTensor;
  forward(...inputs: AnyTensor[]): AnyTensor;
  forward(
    ...args: Array<AnyTensor | MultiheadAttentionForwardOptions | undefined>
  ): AnyTensor | [AnyTensor, AnyTensor] {
    let options: MultiheadAttentionForwardOptions = {};
    const inputs = args.slice() as Array<AnyTensor | undefined>;
    const last = args[args.length - 1];
    if (args.length > 1 && isForwardOptions(last)) {
      options = last;
      inputs.pop();
    }
    // Gradient tracking follows the query, key and value; a mask never decides it.
    const plain = allPlain(...inputs.slice(0, 3).filter((t): t is AnyTensor => t !== undefined));
    const { output, weights } = this.run(inputs, options);
    if (options.needWeights === true) {
      return [settle(output, plain), settle(weights, plain)];
    }
    return settle(output, plain);
  }

  private run(
    inputs: Array<AnyTensor | undefined>,
    options: MultiheadAttentionForwardOptions
  ): { output: GradTensor; weights: GradTensor } {
    if (inputs.length < 1 || inputs.length > 4) {
      throw new InvalidParameterError(
        "MultiheadAttention.forward expects 1 to 4 input tensors (query, key, value, attnMask)",
        "inputs",
        inputs.length
      );
    }
    const queryInput = inputs[0];
    if (queryInput === undefined) {
      throw new InvalidParameterError("Query tensor is required", "query", queryInput);
    }
    const keyInput = inputs[1] ?? queryInput;
    const valueInput = inputs[2] ?? keyInput;
    const attnMaskInput = inputs[3];

    for (const [name, t] of [
      ["query", queryInput],
      ["key", keyInput],
      ["value", valueInput],
    ] as const) {
      if (t.dtype === "string") {
        throw new DTypeError(`MultiheadAttention does not support string dtype (${name})`);
      }
    }

    // Auto-convert to GradTensor (in the parameter dtype)
    const query = toParamDtype(asGrad(queryInput), this.paramDtype);
    const key = toParamDtype(asGrad(keyInput), this.paramDtype);
    const value = toParamDtype(asGrad(valueInput), this.paramDtype);

    if (query.ndim !== key.ndim || query.ndim !== value.ndim) {
      throw new ShapeError(
        `query, key, and value must have same rank; got ${query.ndim}, ${key.ndim}, ${value.ndim}`
      );
    }
    if (query.ndim !== 2 && query.ndim !== 3) {
      throw new ShapeError(`Query, key and value must be 2D or 3D; got ndim=${query.ndim}`);
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

    const H = this.numHeads;
    const D = this.headDim;

    let attnMask: GradTensor | undefined;
    if (attnMaskInput !== undefined) {
      attnMask = resolveAttnMask(attnMaskInput);
      const ms = attnMask.shape;
      const nd = ms.length;
      const ok =
        nd >= 2 &&
        nd <= 4 &&
        broadcastsTo(ms[nd - 1], seqLenK) &&
        broadcastsTo(ms[nd - 2], seqLenQ) &&
        (nd < 3 || broadcastsTo(ms[nd - 3], H)) &&
        (nd < 4 || broadcastsTo(ms[0], batchSize));
      if (!ok) {
        throw new ShapeError(
          `attnMask shape [${ms.join(", ")}] cannot broadcast to ` +
            `(batch=${batchSize}, heads=${H}, seqLenQ=${seqLenQ}, seqLenK=${seqLenK})`
        );
      }
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
    Q = Q.reshape([batchSize, seqLenQ, H, D]).transpose([0, 2, 1, 3]);
    K = K.reshape([batchSize, seqLenK, H, D]).transpose([0, 2, 1, 3]);
    V = V.reshape([batchSize, seqLenV, H, D]).transpose([0, 2, 1, 3]);

    // Scaled Dot-Product Attention
    // Scores = Q @ K^T / sqrt(D)
    // (B, H, Lq, D) @ (B, H, D, Lk) -> (B, H, Lq, Lk)
    let scores = Q.matmul(K.transpose([0, 1, 3, 2]));
    scores = scores.div(GradTensor.scalar(this.scale, { dtype: this.paramDtype }));

    // Additive attention mask (e.g. causal / padding): broadcast-add before
    // softmax so masked positions get ~0 weight.
    if (attnMask) {
      scores = scores.add(attnMask.astype(this.paramDtype));
    }
    if (options.keyPaddingMask !== undefined) {
      const padding = resolveKeyPaddingMask(
        options.keyPaddingMask,
        batchSize,
        seqLenK,
        query.ndim === 2
      );
      scores = scores.add(padding.astype(this.paramDtype));
    }

    // Softmax
    let attn = softmaxGrad(scores, -1);

    // Dropout
    attn = dropoutGrad(attn, this.dropout, this.training);

    // Weighted Sum
    // (B, H, Lq, Lk) @ (B, H, Lk, D) -> (B, H, Lq, D)
    const context = attn.matmul(V);

    // Concat heads
    // (B, H, Lq, D) -> (B, Lq, H, D) -> (B, Lq, E)
    const contextReshaped = context
      .transpose([0, 2, 1, 3])
      .reshape([batchSize, seqLenQ, this.embedDim]);

    // Output projection
    let output = contextReshaped.matmul(this.wO.transpose());
    if (this.bO) output = output.add(this.bO);

    // If input was 2D, squeeze back
    const unbatched = query.ndim === 2;
    if (unbatched) {
      output = output.reshape([seqLenQ, this.embedDim]);
    }

    let weights = attn;
    if (options.needWeights === true) {
      if (options.averageAttnWeights ?? true) weights = weights.mean(1);
      if (unbatched) weights = weights.reshape(weights.shape.slice(1));
    }

    return { output, weights };
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
 * Uses the post-norm layout and a ReLU feed-forward network by default, like PyTorch. Pass
 * `activation: 'gelu'` and `normFirst: true` for a GELU network and the pre-norm layout.
 * Parameters are float32 unless the `dtype` option (or the global default dtype) says
 * otherwise; inputs of other dtypes are cast to the parameter dtype. Gradient tracking
 * follows the rule of {@link MultiheadAttention}.
 *
 * @example
 * ```ts
 * import { TransformerEncoderLayer } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const layer = new TransformerEncoderLayer(64, 8, 256);
 * const x = randn([2, 10, 64]); // (batch, seqLen, dModel)
 * const output = layer.forward(x); // shape [2, 10, 64]
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
  private readonly dropout1: Dropout;
  private readonly dropout2: Dropout;
  private readonly dropout3: Dropout;
  private readonly paramDtype: LayerDType;
  private readonly activation: TransformerActivation;
  private readonly normFirst: boolean;

  /**
   * @param dModelOrOpts - Model dimension, or an options object
   *   `{ dModel, nHead, dFF?, dropout?, eps?, dtype?, activation?, normFirst? }`
   *   (`dimFeedforward` is accepted as an alias of `dFF`)
   * @param nHead - Number of attention heads (must divide `dModel`)
   * @param dFFOrOptions - Feed-forward width (default: 2048), or an options object
   * @param options - `{ dropout?: number (default 0.1), eps?: number (default 1e-5),
   *   dtype?: 'float32' | 'float64', activation?: 'relu' | 'gelu' (default 'relu'),
   *   normFirst?: boolean (default false) }`. `activation: 'gelu'` is the exact erf GELU.
   *   `normFirst: true` normalizes before each sub-block (pre-norm) instead of after it.
   * @throws {InvalidParameterError} If a dimension is not a positive integer, `nHead` does not divide
   *   `dModel`, `dropout` is outside [0, 1), or `activation` is not `'relu'` or `'gelu'`
   */
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
          readonly dtype?: LayerDType;
          readonly activation?: TransformerActivation;
          readonly normFirst?: boolean;
        },
    nHead?: number,
    dFFOrOptions?:
      | number
      | {
          readonly dimFeedforward?: number;
          readonly dFF?: number;
          readonly dropout?: number;
          readonly eps?: number;
          readonly dtype?: LayerDType;
          readonly activation?: TransformerActivation;
          readonly normFirst?: boolean;
        },
    options: {
      readonly dropout?: number;
      readonly eps?: number;
      readonly dtype?: LayerDType;
      readonly activation?: TransformerActivation;
      readonly normFirst?: boolean;
    } = {}
  ) {
    super();

    let resolvedDModel: number;
    let resolvedNHead: number;
    let resolvedDFF: number;
    let resolvedDropout: number | undefined;
    let resolvedEps: number | undefined;
    let resolvedDtype: LayerDType | undefined;
    let resolvedActivation: TransformerActivation | undefined;
    let resolvedNormFirst: boolean | undefined;

    if (typeof dModelOrOpts === "object") {
      resolvedDModel = dModelOrOpts.dModel;
      resolvedNHead = dModelOrOpts.nHead;
      resolvedDFF = dModelOrOpts.dFF ?? dModelOrOpts.dimFeedforward ?? 2048;
      resolvedDropout = dModelOrOpts.dropout;
      resolvedEps = dModelOrOpts.eps;
      resolvedDtype = dModelOrOpts.dtype;
      resolvedActivation = dModelOrOpts.activation;
      resolvedNormFirst = dModelOrOpts.normFirst;
    } else if (typeof dFFOrOptions === "object") {
      resolvedDModel = dModelOrOpts;
      resolvedNHead = nHead ?? 1;
      resolvedDFF = dFFOrOptions.dFF ?? dFFOrOptions.dimFeedforward ?? 2048;
      resolvedDropout = dFFOrOptions.dropout;
      resolvedEps = dFFOrOptions.eps;
      resolvedDtype = dFFOrOptions.dtype;
      resolvedActivation = dFFOrOptions.activation;
      resolvedNormFirst = dFFOrOptions.normFirst;
    } else {
      resolvedDModel = dModelOrOpts;
      resolvedNHead = nHead ?? 1;
      resolvedDFF = dFFOrOptions ?? 2048;
      resolvedDropout = options.dropout;
      resolvedEps = options.eps;
      resolvedDtype = options.dtype;
      resolvedActivation = options.activation;
      resolvedNormFirst = options.normFirst;
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
    this.activation = resolveActivation(resolvedActivation);
    this.normFirst = resolvedNormFirst ?? false;
    this.paramDtype = resolveLayerDtype(resolvedDtype);
    const dtype = this.paramDtype;

    this.selfAttn = new MultiheadAttention(dModel, resolvedNHead, { dropout, dtype });
    this.linear1 = new Linear(dModel, resolvedDFF, { dtype });
    this.linear2 = new Linear(resolvedDFF, dModel, { dtype });
    this.norm1 = new LayerNorm(dModel, { eps: epsVal, dtype });
    this.norm2 = new LayerNorm(dModel, { eps: epsVal, dtype });
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
   * Create an independent deep copy: same architecture, copied weights, buffers,
   * trainability and train/eval mode.
   */
  clone(): TransformerEncoderLayer {
    return copyModuleState(
      this,
      new TransformerEncoderLayer(this.dModel, this.nHead, this.dFF, {
        dropout: this.dropout,
        eps: this.eps,
        dtype: this.paramDtype,
        activation: this.activation,
        normFirst: this.normFirst,
      })
    );
  }

  /**
   * Forward pass of the Transformer encoder layer.
   *
   * @param src - Source sequence of shape (batch, seqLen, dModel), or (seqLen, dModel)
   * @param mask - Optional attention mask for the self-attention, see
   *   {@link MultiheadAttention.forward} (additive float mask or boolean mask)
   * @returns Output of same shape as input
   * @throws {DTypeError} For string tensors
   */
  forward(src: GradTensor, mask?: AnyTensor): GradTensor;
  forward(src: Tensor, mask?: AnyTensor): AnyTensor;
  forward(src: AnyTensor, mask?: AnyTensor): AnyTensor;
  forward(src: AnyTensor, mask?: AnyTensor): AnyTensor {
    return settle(this.run(src, mask), allPlain(src));
  }

  private run(src: AnyTensor, mask?: AnyTensor): GradTensor {
    if (src.dtype === "string") {
      throw new DTypeError("TransformerEncoderLayer does not support string dtype");
    }
    const input = toParamDtype(asGrad(src), this.paramDtype);

    // Self-attention block: dropout(self_attn(x, x, x, mask))
    const attend = (x: GradTensor): GradTensor => {
      const a =
        mask === undefined ? this.selfAttn.forward(x, x, x) : this.selfAttn.forward(x, x, x, mask);
      return this.dropout1.forward(a);
    };
    // Feed-forward block: dropout3(linear2(dropout2(activation(linear1(x)))))
    const feedForward = (x: GradTensor): GradTensor => {
      let ffn = this.linear1.forward(x);
      ffn = applyActivation(ffn, this.activation);
      ffn = this.dropout2.forward(ffn);
      ffn = this.linear2.forward(ffn);
      return this.dropout3.forward(ffn);
    };

    if (this.normFirst) {
      // Pre-norm: x = x + sa(norm1(x)); x = x + ff(norm2(x))
      let out = input.add(attend(this.norm1.forward(input)));
      out = out.add(feedForward(this.norm2.forward(out)));
      return out;
    }

    // Post-norm: x = norm1(x + sa(x)); x = norm2(x + ff(x))
    let out = this.norm1.forward(input.add(attend(input)));
    out = this.norm2.forward(out.add(feedForward(out)));
    return out;
  }

  override toString(): string {
    return `TransformerEncoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dFF}, dropout=${this.dropout}, activation=${this.activation}, norm_first=${this.normFirst})`;
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
 * By default the self-attention is causal: position i only attends to positions
 * up to i. Pass `tgtMask` to `forward` to use a different mask. The layout is post-norm with
 * a ReLU feed-forward network unless `normFirst` and `activation` say otherwise.
 *
 * Parameters are float32 unless the `dtype` option (or the global default dtype) says
 * otherwise; inputs of other dtypes are cast to the parameter dtype. Gradient tracking
 * follows the rule of {@link MultiheadAttention}.
 *
 * @example
 * ```ts
 * import { TransformerDecoderLayer } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const layer = new TransformerDecoderLayer(64, 8, 256);
 * const tgt = randn([2, 7, 64]); // (batch, tgtLen, dModel)
 * const memory = randn([2, 10, 64]); // encoder output (batch, srcLen, dModel)
 * const output = layer.forward(tgt, memory); // shape [2, 7, 64]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
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
  private readonly paramDtype: LayerDType;
  private readonly activation: TransformerActivation;
  private readonly normFirst: boolean;

  /**
   * @param dModel - Model dimension
   * @param nHead - Number of attention heads (must divide `dModel`)
   * @param dFF - Feed-forward width (default: 2048)
   * @param options - `{ dropout?: number (default 0.1), eps?: number (default 1e-5),
   *   dtype?: 'float32' | 'float64', activation?: 'relu' | 'gelu' (default 'relu'),
   *   normFirst?: boolean (default false) }`. `activation: 'gelu'` is the exact erf GELU.
   *   `normFirst: true` normalizes before each sub-block (pre-norm) instead of after it.
   * @throws {InvalidParameterError} If a dimension is not a positive integer, `nHead` does not divide
   *   `dModel`, `dropout` is outside [0, 1), or `activation` is not `'relu'` or `'gelu'`
   */
  constructor(
    dModel: number,
    nHead: number,
    dFF = 2048,
    options: {
      readonly dropout?: number;
      readonly eps?: number;
      readonly dtype?: LayerDType;
      readonly activation?: TransformerActivation;
      readonly normFirst?: boolean;
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
    this.activation = resolveActivation(options.activation);
    this.normFirst = options.normFirst ?? false;
    this.paramDtype = resolveLayerDtype(options.dtype);
    const dtype = this.paramDtype;

    this.selfAttn = new MultiheadAttention(dModel, nHead, { dropout, dtype });
    this.crossAttn = new MultiheadAttention(dModel, nHead, { dropout, dtype });
    this.linear1 = new Linear(dModel, dFF, { dtype });
    this.linear2 = new Linear(dFF, dModel, { dtype });
    this.norm1 = new LayerNorm(dModel, { eps, dtype });
    this.norm2 = new LayerNorm(dModel, { eps, dtype });
    this.norm3 = new LayerNorm(dModel, { eps, dtype });
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

  /**
   * Create an independent deep copy: same architecture, copied weights, buffers,
   * trainability and train/eval mode.
   */
  clone(): TransformerDecoderLayer {
    return copyModuleState(
      this,
      new TransformerDecoderLayer(this.dModel, this.nHead, this.dFF, {
        dropout: this.dropoutRate,
        eps: this.eps,
        dtype: this.paramDtype,
        activation: this.activation,
        normFirst: this.normFirst,
      })
    );
  }

  /**
   * Forward pass of the Transformer decoder layer.
   *
   * @param tgt - Target sequence of shape (batch, tgtLen, dModel), or (tgtLen, dModel)
   * @param memory - Encoder output of shape (batch, srcLen, dModel); required
   * @param tgtMask - Optional mask for the self-attention (see {@link MultiheadAttention.forward}).
   *   When omitted, a causal mask is used. When given, it replaces the causal mask.
   * @param memoryMask - Optional mask for the cross-attention, broadcastable to
   *   (batch, numHeads, tgtLen, srcLen)
   * @returns Output of same shape as `tgt`
   * @throws {InvalidParameterError} If `memory` is missing
   * @throws {DTypeError} For string tensors
   */
  forward(
    tgt: GradTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor;
  forward(tgt: Tensor, memory?: AnyTensor, tgtMask?: AnyTensor, memoryMask?: AnyTensor): AnyTensor;
  forward(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor;
  forward(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor {
    const plain = memory === undefined ? allPlain(tgt) : allPlain(tgt, memory);
    return settle(this.run(tgt, memory, tgtMask, memoryMask), plain);
  }

  private run(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor {
    if (!memory) {
      throw new InvalidParameterError(
        "TransformerDecoderLayer requires memory (encoder output) as second argument",
        "memory",
        undefined
      );
    }
    if (tgt.dtype === "string" || memory.dtype === "string") {
      throw new DTypeError("TransformerDecoderLayer does not support string dtype");
    }

    const input = toParamDtype(asGrad(tgt), this.paramDtype);
    const memoryGrad = toParamDtype(asGrad(memory), this.paramDtype);

    // 1. Self-attention. By default each target position may only attend to itself
    // and earlier positions (prevents information leakage from future tokens
    // during autoregressive decoding).
    const selfMask = tgtMask ?? causalMask(input.shape[input.ndim - 2] ?? 0);
    const selfAttend = (x: GradTensor): GradTensor =>
      this.dropout1.forward(this.selfAttn.forward(x, x, x, selfMask));
    // Cross-attention: query from the decoder, key and value from the encoder
    const crossAttend = (x: GradTensor): GradTensor => {
      const c =
        memoryMask === undefined
          ? this.crossAttn.forward(x, memoryGrad, memoryGrad)
          : this.crossAttn.forward(x, memoryGrad, memoryGrad, memoryMask);
      return this.dropout2.forward(c);
    };
    const feedForward = (x: GradTensor): GradTensor => {
      let ffn = this.linear1.forward(x);
      ffn = applyActivation(ffn, this.activation);
      ffn = this.dropout3.forward(ffn);
      ffn = this.linear2.forward(ffn);
      return this.dropout4.forward(ffn);
    };

    if (this.normFirst) {
      // Pre-norm: x = x + sa(norm1(x)); x = x + ca(norm2(x)); x = x + ff(norm3(x))
      let out = input.add(selfAttend(this.norm1.forward(input)));
      out = out.add(crossAttend(this.norm2.forward(out)));
      out = out.add(feedForward(this.norm3.forward(out)));
      return out;
    }

    // Post-norm: x = norm1(x + sa(x)); x = norm2(x + ca(x)); x = norm3(x + ff(x))
    let out = this.norm1.forward(input.add(selfAttend(input)));
    out = this.norm2.forward(out.add(crossAttend(out)));
    out = this.norm3.forward(out.add(feedForward(out)));
    return out;
  }

  override toString(): string {
    return `TransformerDecoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dFF}, dropout=${this.dropoutRate}, activation=${this.activation}, norm_first=${this.normFirst})`;
  }
}

/**
 * Transformer Encoder, a stack of N encoder layers.
 *
 * The first layer is the `encoderLayer` you pass in; the other N - 1 layers are
 * independent copies of it (see {@link TransformerEncoderLayer.clone}), so every
 * layer starts from the same weights and then trains separately.
 *
 * @example
 * ```ts
 * import { TransformerEncoder, TransformerEncoderLayer } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const encoderLayer = new TransformerEncoderLayer(64, 8, 256);
 * const encoder = new TransformerEncoder(encoderLayer, 6);
 * const output = encoder.forward(randn([2, 10, 64])); // shape [2, 10, 64]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */
export class TransformerEncoder extends Module {
  private readonly layers: TransformerEncoderLayer[];
  private readonly norm: LayerNorm | undefined;

  /**
   * @param encoderLayer - Layer used as the first layer and as the template for the others
   * @param numLayers - Number of layers (positive integer)
   * @param options - `norm`: optional final normalization applied after the last layer
   * @throws {InvalidParameterError} If `numLayers` is not a positive integer
   */
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

    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      this.layers.push(i === 0 ? encoderLayer : encoderLayer.clone());
    }
    // A real ModuleList child named "layers": state dict keys stay `layers.<i>.<name>`.
    this.registerModule("layers", new ModuleList(this.layers));

    this.norm = options.norm;
    if (this.norm) {
      this.registerModule("norm", this.norm);
    }
  }

  /**
   * @param src - Source sequence of shape (batch, seqLen, dModel), or (seqLen, dModel)
   * @param mask - Optional self-attention mask applied in every layer
   * @returns Encoded sequence of the same shape
   */
  forward(src: GradTensor, mask?: AnyTensor): GradTensor;
  forward(src: Tensor, mask?: AnyTensor): AnyTensor;
  forward(src: AnyTensor, mask?: AnyTensor): AnyTensor;
  forward(src: AnyTensor, mask?: AnyTensor): AnyTensor {
    return settle(this.run(src, mask), allPlain(src));
  }

  private run(src: AnyTensor, mask?: AnyTensor): GradTensor {
    let out = asGrad(src);

    for (const layer of this.layers) {
      out = layer.forward(out, mask);
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
 * Transformer Decoder, a stack of N decoder layers.
 *
 * The first layer is the `decoderLayer` you pass in; the other N - 1 layers are
 * independent copies of it (see {@link TransformerDecoderLayer.clone}).
 *
 * @example
 * ```ts
 * import { TransformerDecoder, TransformerDecoderLayer } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const decoderLayer = new TransformerDecoderLayer(64, 8, 256);
 * const decoder = new TransformerDecoder(decoderLayer, 6);
 * const output = decoder.forward(randn([2, 7, 64]), randn([2, 10, 64])); // shape [2, 7, 64]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */
export class TransformerDecoder extends Module {
  private readonly layers: TransformerDecoderLayer[];
  private readonly norm: LayerNorm | undefined;

  /**
   * @param decoderLayer - Layer used as the first layer and as the template for the others
   * @param numLayers - Number of layers (positive integer)
   * @param options - `norm`: optional final normalization applied after the last layer
   * @throws {InvalidParameterError} If `numLayers` is not a positive integer
   */
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
      this.layers.push(i === 0 ? decoderLayer : decoderLayer.clone());
    }
    // A real ModuleList child named "layers": state dict keys stay `layers.<i>.<name>`.
    this.registerModule("layers", new ModuleList(this.layers));

    this.norm = options.norm;
    if (this.norm) {
      this.registerModule("norm", this.norm);
    }
  }

  /**
   * @param tgt - Target sequence of shape (batch, tgtLen, dModel), or (tgtLen, dModel)
   * @param memory - Encoder output; required
   * @param tgtMask - Optional self-attention mask; defaults to a causal mask
   * @param memoryMask - Optional cross-attention mask
   * @returns Decoded sequence of the same shape as `tgt`
   * @throws {InvalidParameterError} If `memory` is missing
   */
  forward(
    tgt: GradTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor;
  forward(tgt: Tensor, memory?: AnyTensor, tgtMask?: AnyTensor, memoryMask?: AnyTensor): AnyTensor;
  forward(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor;
  forward(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor {
    const plain = memory === undefined ? allPlain(tgt) : allPlain(tgt, memory);
    return settle(this.run(tgt, memory, tgtMask, memoryMask), plain);
  }

  private run(
    tgt: AnyTensor,
    memory?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor {
    if (!memory) {
      throw new InvalidParameterError(
        "TransformerDecoder requires memory (encoder output) as second argument",
        "memory",
        undefined
      );
    }

    let out = asGrad(tgt);
    const memoryGrad = asGrad(memory);

    for (const layer of this.layers) {
      out = layer.forward(out, memoryGrad, tgtMask, memoryMask);
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
 * Post-norm layers with ReLU feed-forward networks by default (see the `normFirst` and
 * `activation` options). Unlike PyTorch's `nn.Transformer`,
 * the encoder and decoder stacks have no final LayerNorm unless you ask for it with
 * `options.finalNorm`. Parameters follow the `dtype` option (default: the global default
 * dtype, `float32` unless changed).
 *
 * @example
 * ```ts
 * import { FullTransformer } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const transformer = new FullTransformer(64, 8, 2, 2, 256);
 * const src = randn([2, 10, 64]); // (batch, srcLen, dModel)
 * const tgt = randn([2, 7, 64]); // (batch, tgtLen, dModel)
 * const output = transformer.forward(src, tgt); // shape [2, 7, 64]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */
export class FullTransformer extends Module {
  private readonly encoder: TransformerEncoder;
  private readonly decoder: TransformerDecoder;

  /**
   * @param dModel - Model dimension (default: 512)
   * @param nHead - Number of attention heads (default: 8)
   * @param numEncoderLayers - Encoder depth (default: 6)
   * @param numDecoderLayers - Decoder depth (default: 6)
   * @param dFF - Feed-forward width (default: 2048)
   * @param options - `dropout` (default 0.1), `eps` (default 1e-5), `dtype` of the parameters,
   *   `activation` (`'relu'` or `'gelu'`, default `'relu'`), `normFirst` (pre-norm layers,
   *   default false), and `finalNorm`: add a LayerNorm after the encoder stack and after the
   *   decoder stack (default: false)
   */
  constructor(
    dModel = 512,
    nHead = 8,
    numEncoderLayers = 6,
    numDecoderLayers = 6,
    dFF = 2048,
    options: {
      readonly dropout?: number;
      readonly eps?: number;
      readonly dtype?: LayerDType;
      readonly finalNorm?: boolean;
      readonly activation?: TransformerActivation;
      readonly normFirst?: boolean;
    } = {}
  ) {
    super();

    const { finalNorm, ...layerOptions } = options;
    const encLayer = new TransformerEncoderLayer(dModel, nHead, dFF, layerOptions);
    const decLayer = new TransformerDecoderLayer(dModel, nHead, dFF, layerOptions);
    const eps = options.eps ?? 1e-5;
    const dtype = resolveLayerDtype(options.dtype);

    this.encoder = new TransformerEncoder(
      encLayer,
      numEncoderLayers,
      finalNorm ? { norm: new LayerNorm(dModel, { eps, dtype }) } : {}
    );
    this.decoder = new TransformerDecoder(
      decLayer,
      numDecoderLayers,
      finalNorm ? { norm: new LayerNorm(dModel, { eps, dtype }) } : {}
    );

    this.registerModule("encoder", this.encoder);
    this.registerModule("decoder", this.decoder);
  }

  /**
   * Forward pass of the full Transformer.
   *
   * @param src - Source sequence of shape (batch, srcLen, dModel)
   * @param tgt - Target sequence of shape (batch, tgtLen, dModel); required
   * @param srcMask - Optional encoder self-attention mask
   * @param tgtMask - Optional decoder self-attention mask; defaults to a causal mask
   * @param memoryMask - Optional decoder cross-attention mask
   * @returns Decoder output of the same shape as `tgt`
   * @throws {InvalidParameterError} If `tgt` is missing
   */
  forward(
    src: GradTensor,
    tgt?: AnyTensor,
    srcMask?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor;
  forward(
    src: Tensor,
    tgt?: AnyTensor,
    srcMask?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor;
  forward(
    src: AnyTensor,
    tgt?: AnyTensor,
    srcMask?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor;
  forward(
    src: AnyTensor,
    tgt?: AnyTensor,
    srcMask?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): AnyTensor {
    const plain = tgt === undefined ? allPlain(src) : allPlain(src, tgt);
    return settle(this.run(src, tgt, srcMask, tgtMask, memoryMask), plain);
  }

  private run(
    src: AnyTensor,
    tgt?: AnyTensor,
    srcMask?: AnyTensor,
    tgtMask?: AnyTensor,
    memoryMask?: AnyTensor
  ): GradTensor {
    if (!tgt) {
      throw new InvalidParameterError(
        "Transformer requires target tensor as second argument",
        "tgt",
        undefined
      );
    }

    const memory = this.encoder.forward(asGrad(src), srcMask);
    return this.decoder.forward(asGrad(tgt), memory, tgtMask, memoryMask);
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
 * For an odd `dModel` the last column pairs with the same frequency as the one
 * before it. The table is stored as the `pe` buffer and cast to the input's float
 * dtype on each call; integer inputs are cast to float32.
 *
 * @example
 * ```ts
 * import { PositionalEncoding } from 'deepbox/nn';
 * import { randn } from 'deepbox/ndarray';
 *
 * const pe = new PositionalEncoding(64, { maxLen: 500 });
 * const x = randn([2, 10, 64]); // (batch, seqLen, dModel)
 * const output = pe.forward(x); // x + positional encoding, then dropout (training mode)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-attention | Deepbox Attention}
 */
export class PositionalEncoding extends Module {
  private readonly dModel: number;
  private readonly dropoutModule: Dropout;
  private readonly maxLen: number;
  private readonly peBuffer: Tensor;

  /**
   * @param dModel - Embedding dimension (positive integer)
   * @param options - `dropout` (default 0.1) and `maxLen`, the longest supported sequence
   *   (positive integer, default 5000)
   * @throws {InvalidParameterError} If `dModel` or `maxLen` is not a positive integer, or `dropout`
   *   is outside [0, 1)
   */
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
    if (!Number.isInteger(maxLen) || maxLen <= 0) {
      throw new InvalidParameterError("maxLen must be a positive integer", "maxLen", maxLen);
    }

    this.dModel = dModel;
    this.maxLen = maxLen;
    this.dropoutModule = new Dropout(dropout);
    this.registerModule("dropout", this.dropoutModule);

    // Precompute the positional encoding table. The 10000^(2i/d) divisors depend
    // only on the column, so compute them once.
    const divTerm = new Float64Array(dModel);
    for (let i = 0; i < dModel; i++) {
      divTerm[i] = 10000 ** ((2 * Math.floor(i / 2)) / dModel);
    }
    const peData = new Float64Array(maxLen * dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      const row = pos * dModel;
      for (let i = 0; i < dModel; i++) {
        const angle = pos / (divTerm[i] as number);
        peData[row + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
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

  /**
   * @param x - Embeddings of shape (batch, seqLen, dModel) or (seqLen, dModel)
   * @returns `x` plus the positional encoding, after dropout (float dtype of `x`)
   * @throws {DTypeError} For string tensors
   * @throws {ShapeError} If `x` is not 2D or 3D, or its last axis is not `dModel`
   * @throws {InvalidParameterError} If the sequence is longer than `maxLen`
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): Tensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    if (x.dtype === "string") {
      throw new DTypeError("PositionalEncoding does not support string dtype");
    }
    let input = asGrad(x);

    // x shape: (batch, seqLen, dModel) or (seqLen, dModel)
    let seqLen: number;
    if (input.ndim === 3) {
      seqLen = input.shape[1] ?? 0;
    } else if (input.ndim === 2) {
      seqLen = input.shape[0] ?? 0;
    } else {
      throw new ShapeError(`PositionalEncoding expects 2D or 3D input; got ${input.ndim}D`);
    }

    const lastDim = input.shape[input.ndim - 1];
    if (lastDim !== this.dModel) {
      throw new ShapeError(
        `PositionalEncoding expects last dimension ${this.dModel}; got ${lastDim}`
      );
    }

    if (seqLen > this.maxLen) {
      throw new InvalidParameterError(
        `Sequence length ${seqLen} exceeds maxLen ${this.maxLen}`,
        "seqLen",
        seqLen
      );
    }

    // Integer inputs become float32; float inputs keep their dtype.
    const floatDtypes: readonly DType[] = ["float16", "bfloat16", "float32", "float64"];
    if (!floatDtypes.includes(input.dtype)) input = input.astype("float32");

    // Slice PE to match sequence length: (seqLen, dModel), in the input's dtype
    const peSlice = this.peBuffer.slice({ start: 0, end: seqLen });
    const peGrad = GradTensor.fromTensor(peSlice, { requiresGrad: false }).astype(
      input.dtype as "float16" | "bfloat16" | "float32" | "float64"
    );

    // Add positional encoding to input
    const out = input.add(peGrad);
    return this.dropoutModule.forward(out);
  }

  override toString(): string {
    return `PositionalEncoding(d_model=${this.dModel}, max_len=${this.maxLen})`;
  }
}
