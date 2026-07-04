/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

// Module base class

// Gradient clipping
export {
  clip_grad_norm_,
  clip_grad_value_,
  clipGradNorm,
  clipGradNorm_,
  clipGradValue,
  clipGradValue_,
} from "./clip";
// Containers
export { ModuleDict, ModuleList } from "./containers/ModuleList";
export { ParameterDict, ParameterList } from "./containers/ParameterList";
export { Sequential } from "./containers/Sequential";
// Weight initialization
export {
  constant,
  constant_,
  kaiming_normal_,
  kaiming_uniform_,
  kaimingNormal,
  kaimingNormal_,
  kaimingUniform,
  kaimingUniform_,
  normal_,
  ones,
  ones_,
  orthogonal,
  orthogonal_,
  sparse_,
  uniform_,
  xavier_normal_,
  xavier_uniform_,
  xavierNormal,
  xavierNormal_,
  xavierUniform,
  xavierUniform_,
  zeros,
  zeros_,
} from "./init";
// Activation layers
export {
  ELU,
  GELU,
  GLU,
  Hardsigmoid,
  Hardswish,
  Hardtanh,
  LeakyReLU,
  LogSoftmax,
  Mish,
  PReLU,
  ReLU,
  SELU,
  Sigmoid,
  SiLU,
  Softmax,
  Softmax2d,
  Softmin,
  Softplus,
  Softsign,
  Swish,
  Tanh,
  Tanhshrink,
} from "./layers/activations";
// Attention layers
export {
  causalMask,
  FullTransformer,
  MultiheadAttention,
  PositionalEncoding,
  TransformerDecoder,
  TransformerDecoderLayer,
  TransformerEncoder,
  TransformerEncoderLayer,
} from "./layers/attention";
// Convolutional layers
export {
  AdaptiveAvgPool1d,
  AdaptiveAvgPool2d,
  AdaptiveMaxPool1d,
  AdaptiveMaxPool2d,
  AvgPool1d,
  AvgPool2d,
  AvgPool3d,
  Conv1d,
  Conv2d,
  Conv3d,
  ConvTranspose1d,
  ConvTranspose2d,
  MaxPool1d,
  MaxPool2d,
  MaxPool3d,
} from "./layers/conv";
// Regularization layers
export { AlphaDropout, Dropout, Dropout2d } from "./layers/dropout";
// Embedding layers
export { Embedding, EmbeddingBag } from "./layers/embedding";
// Layers - fully connected / dense layers
export { Linear } from "./layers/linear";
// Normalization layers
export {
  BatchNorm1d,
  BatchNorm2d,
  BatchNorm3d,
  GroupNorm,
  InstanceNorm,
  InstanceNorm1d,
  InstanceNorm2d,
  InstanceNorm3d,
  LayerNorm,
  LocalResponseNorm,
  RMSNorm,
} from "./layers/normalization";
// Packed sequences
export {
  type PackedSequence,
  packPaddedSequence,
  packSequence,
  padPackedSequence,
  unpackSequence,
} from "./layers/packed_sequence";
// Padding layers
export {
  ConstantPad2d,
  ReflectionPad2d,
  ReplicationPad2d,
  ZeroPad2d,
} from "./layers/padding";
// Recurrent layers
export { GRU, LSTM, RNN } from "./layers/recurrent";
// Spectral normalization
export { SpectralNorm } from "./layers/spectral_norm";
// Upsampling
export { Upsample } from "./layers/upsample";
// Utility layers
export { Flatten, Identity, Unflatten } from "./layers/utility";
// Loss functions
export {
  binaryCrossEntropyLoss,
  binaryCrossEntropyWithLogitsLoss,
  cosineEmbeddingLoss,
  crossEntropyLoss,
  ctcLoss,
  gaussianNLLLoss,
  huberLoss,
  klDivLoss,
  maeLoss,
  marginRankingLoss,
  mseLoss,
  nllLoss,
  poissonNLLLoss,
  rmseLoss,
  smoothL1Loss,
  tripletMarginLoss,
} from "./losses/index";
export type { ForwardHook, ForwardPreHook } from "./module/Module";
export { Module } from "./module/Module";
// Trainer
export type {
  EpochInfo,
  LossFn,
  TrainerCallback,
  TrainerOptions,
  TrainerResult,
} from "./Trainer";
export { Trainer } from "./Trainer";
// Training utilities
export {
  EarlyStopping,
  GradientAccumulator,
  ModelCheckpoint,
} from "./training";
