/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

// Base optimizer class
export type { ParamGroup } from "./Optimizer";
export { Optimizer } from "./Optimizer";

// Optimizers
export { AdaDelta } from "./optimizers/adadelta";
export { Adagrad } from "./optimizers/adagrad";
export { Adam } from "./optimizers/adam";
export { Adamax } from "./optimizers/adamax";
export { AdamW } from "./optimizers/adamw";
export { ASGD } from "./optimizers/asgd";
export { LAMB } from "./optimizers/lamb";
export { LARS } from "./optimizers/lars";
export { LBFGS } from "./optimizers/lbfgs";
export { Lion } from "./optimizers/lion";
export { Nadam } from "./optimizers/nadam";
export { RAdam } from "./optimizers/radam";
export { RMSprop } from "./optimizers/rmsprop";
export { Rprop } from "./optimizers/rprop";
export { SGD } from "./optimizers/sgd";
export { SparseAdam } from "./optimizers/sparse_adam";

// Learning rate schedulers
export {
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  CyclicLR,
  ExponentialLR,
  LambdaLR,
  LinearLR,
  LRScheduler,
  MultiStepLR,
  OneCycleLR,
  type PlateauStateDict,
  PolynomialLR,
  ReduceLROnPlateau,
  type SchedulerStateDict,
  SequentialLR,
  StepLR,
  WarmupLR,
} from "./schedulers";
