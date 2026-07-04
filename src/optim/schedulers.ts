// Learning rate schedulers for optimizers
import { InvalidParameterError } from "../core";

/**
 * Interface for optimizer-like objects that schedulers can work with.
 * This allows schedulers to work with different optimizer implementations.
 * Parameter groups may expose `lr` directly or via `options.lr`.
 */
interface SchedulerOptimizer {
  paramGroups: SchedulerParamGroup[];
}

type SchedulerParamGroup = {
  params: unknown[];
  lr?: number;
  options?: Record<string, unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function resolveGroupLr(group: SchedulerParamGroup, index: number) {
  const options = isRecord(group.options) ? group.options : undefined;
  const lrValue = group.lr ?? options?.["lr"];
  if (typeof lrValue !== "number" || !Number.isFinite(lrValue) || lrValue < 0) {
    throw new InvalidParameterError(
      `optimizer.paramGroups[${index}].lr must be finite and >= 0`,
      `optimizer.paramGroups[${index}].lr`,
      lrValue
    );
  }
  return lrValue;
}

function setGroupLr(group: SchedulerParamGroup, lr: number) {
  if (isRecord(group.options)) {
    group.options["lr"] = lr;
  }
  if ("lr" in group) {
    group.lr = lr;
  }
  if (!("lr" in group) && !isRecord(group.options)) {
    group.lr = lr;
  }
}

function validateLastEpoch(value: number) {
  if (!Number.isInteger(value) || value < -1) {
    throw new InvalidParameterError("lastEpoch must be an integer >= -1", "lastEpoch", value);
  }
  return value;
}

function validateFiniteNumber(value: number, name: string) {
  if (!Number.isFinite(value)) {
    throw new InvalidParameterError(`${name} must be finite`, name, value);
  }
  return value;
}

function validatePositiveNumber(value: number, name: string) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be > 0`, name, value);
  }
  return value;
}

function validatePositiveInteger(value: number, name: string) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer`, name, value);
  }
  return value;
}

function validateNonNegativeNumber(value: number, name: string) {
  if (!Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(`${name} must be >= 0`, name, value);
  }
  return value;
}

function validateNonNegativeInteger(value: number, name: string) {
  if (!Number.isInteger(value) || value < 0) {
    throw new InvalidParameterError(`${name} must be a non-negative integer`, name, value);
  }
  return value;
}

function validateOptimizer(optimizer: SchedulerOptimizer) {
  if (!optimizer || typeof optimizer !== "object" || !Array.isArray(optimizer.paramGroups)) {
    throw new InvalidParameterError(
      "optimizer must expose paramGroups array",
      "optimizer",
      optimizer
    );
  }
  if (optimizer.paramGroups.length === 0) {
    throw new InvalidParameterError(
      "optimizer.paramGroups must contain at least one group",
      "optimizer.paramGroups",
      optimizer.paramGroups
    );
  }
  for (let i = 0; i < optimizer.paramGroups.length; i++) {
    const group = optimizer.paramGroups[i];
    if (!group || typeof group !== "object") {
      throw new InvalidParameterError(
        `optimizer.paramGroups[${i}] must be an object`,
        "optimizer.paramGroups",
        group
      );
    }
    if (!Array.isArray(group.params)) {
      throw new InvalidParameterError(
        `optimizer.paramGroups[${i}].params must be an array`,
        `optimizer.paramGroups[${i}].params`,
        group.params
      );
    }
    resolveGroupLr(group, i);
  }
}

function validateMilestones(milestones: number[]) {
  if (!Array.isArray(milestones) || milestones.length === 0) {
    throw new InvalidParameterError(
      "milestones must be a non-empty array of non-negative integers",
      "milestones",
      milestones
    );
  }

  const sorted = [...milestones].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) {
    const value = sorted[i];
    if (value === undefined || !Number.isInteger(value) || value < 0) {
      throw new InvalidParameterError(
        "milestones must contain non-negative integers only",
        "milestones",
        milestones
      );
    }
    if (i > 0) {
      const prev = sorted[i - 1];
      if (prev !== undefined && value <= prev) {
        throw new InvalidParameterError(
          "milestones must be strictly increasing",
          "milestones",
          milestones
        );
      }
    }
  }
  return sorted;
}

/**
 * Base class for learning rate schedulers.
 *
 * Learning rate schedulers adjust the learning rate during training according
 * to a predefined schedule. This can help improve convergence and prevent
 * overshooting optimal solutions.
 *
 * @example
 * ```ts
 * import { SGD, StepLR } from 'deepbox/optim';
 *
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new StepLR(optimizer, { stepSize: 10, gamma: 0.1 });
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   scheduler.step();
 * }
 * ```
 *
 * @category Optimization
 */
export abstract class LRScheduler {
  protected optimizer: SchedulerOptimizer;
  protected lastEpoch: number;
  protected baseLrs: number[];

  constructor(optimizer: SchedulerOptimizer, lastEpoch: number = -1) {
    validateOptimizer(optimizer);
    this.lastEpoch = validateLastEpoch(lastEpoch);
    this.optimizer = optimizer;

    // Store base learning rates from all parameter groups
    this.baseLrs = optimizer.paramGroups.map((group, index) => resolveGroupLr(group, index));
  }

  protected initializeFromLastEpoch(lastEpoch: number): void {
    const validated = validateLastEpoch(lastEpoch);
    this.lastEpoch = -1;
    if (validated < 0) {
      // PyTorch's _LRScheduler.__init__ performs one step() at construction,
      // so epoch 0 runs at getLr(0) — not the optimizer's raw base LR. Omitting
      // this shifts every scheduler by one epoch (StepLR decays one epoch late,
      // OneCycleLR/WarmupLR run the first epoch at the wrong LR entirely).
      this.step();
      return;
    }
    for (let i = 0; i <= validated; i++) {
      this.step();
    }
  }

  /**
   * Compute the learning rate for the current epoch.
   * Must be implemented by subclasses.
   *
   * @returns Array of learning rates for each parameter group
   */
  abstract getLr(): number[];

  /**
   * Perform a scheduler step, updating learning rates.
   *
   * Should be called once per epoch after the optimizer step.
   */
  step(): void {
    this.lastEpoch++;
    const newLrs = this.getLr();

    for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
      const group = this.optimizer.paramGroups[i];
      if (group) {
        const next = newLrs[i];
        if (next !== undefined) {
          setGroupLr(group, next);
        }
      }
    }
  }

  /**
   * Get the current learning rates for all parameter groups.
   */
  getLastLr(): number[] {
    return this.optimizer.paramGroups.map((group, index) => resolveGroupLr(group, index));
  }

  /**
   * Get current epoch number.
   */
  get epoch() {
    return this.lastEpoch;
  }
}

/**
 * Step learning rate scheduler.
 *
 * Decays the learning rate by gamma every stepSize epochs.
 * lr = baseLr * gamma^(epoch // stepSize)
 *
 * @example
 * ```ts
 * const scheduler = new StepLR(optimizer, { stepSize: 30, gamma: 0.1 });
 * // lr = 0.1 for epochs 0-29
 * // lr = 0.01 for epochs 30-59
 * // lr = 0.001 for epochs 60-89
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class StepLR extends LRScheduler {
  private stepSize: number;
  private gamma: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: { stepSize: number; gamma?: number; lastEpoch?: number }
  ) {
    const stepSize = validatePositiveInteger(options.stepSize, "stepSize");
    const gamma = validatePositiveNumber(options.gamma ?? 0.1, "gamma");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.stepSize = stepSize;
    this.gamma = gamma;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    const factor = this.gamma ** Math.floor(this.lastEpoch / this.stepSize);
    return this.baseLrs.map((lr) => lr * factor);
  }
}

/**
 * Exponential learning rate scheduler.
 *
 * Decays the learning rate exponentially every epoch.
 * lr = baseLr * gamma^epoch
 *
 * @example
 * ```ts
 * const scheduler = new ExponentialLR(optimizer, { gamma: 0.95 });
 * // lr *= 0.95 each epoch
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class ExponentialLR extends LRScheduler {
  private gamma: number;

  constructor(optimizer: SchedulerOptimizer, options: { gamma: number; lastEpoch?: number }) {
    const gamma = validatePositiveNumber(options.gamma, "gamma");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.gamma = gamma;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    return this.baseLrs.map((lr) => lr * this.gamma ** this.lastEpoch);
  }
}

/**
 * Cosine annealing learning rate scheduler.
 *
 * Sets the learning rate using a cosine annealing schedule.
 * lr = etaMin + (baseLr - etaMin) * (1 + cos(π * epoch / T_max)) / 2
 *
 * @example
 * ```ts
 * const scheduler = new CosineAnnealingLR(optimizer, { T_max: 100, etaMin: 0.001 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class CosineAnnealingLR extends LRScheduler {
  private T_max: number;
  private etaMin: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      T_max?: number;
      tMax?: number;
      etaMin?: number;
      lastEpoch?: number;
    }
  ) {
    const rawTMax = options.T_max ?? options.tMax;
    if (rawTMax === undefined) {
      throw new InvalidParameterError("T_max or tMax must be provided", "T_max");
    }
    const tMax = validatePositiveInteger(rawTMax, "T_max");
    const etaMin = validateNonNegativeNumber(options.etaMin ?? 0, "etaMin");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.T_max = tMax;
    this.etaMin = etaMin;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    return this.baseLrs.map((baseLr) => {
      return (
        this.etaMin +
        ((baseLr - this.etaMin) * (1 + Math.cos((Math.PI * this.lastEpoch) / this.T_max))) / 2
      );
    });
  }
}

/**
 * Multi-step learning rate scheduler.
 *
 * Decays the learning rate by gamma once the epoch reaches one of the milestones.
 *
 * @example
 * ```ts
 * const scheduler = new MultiStepLR(optimizer, { milestones: [30, 80], gamma: 0.1 });
 * // lr = 0.1 for epochs 0-29
 * // lr = 0.01 for epochs 30-79
 * // lr = 0.001 for epochs 80+
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class MultiStepLR extends LRScheduler {
  private sortedMilestones: number[];
  private gamma: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: { milestones: number[]; gamma?: number; lastEpoch?: number }
  ) {
    const milestones = validateMilestones(options.milestones);
    const gamma = validatePositiveNumber(options.gamma ?? 0.1, "gamma");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.sortedMilestones = milestones;
    this.gamma = gamma;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    // Count how many milestones we've passed
    let numDecays = 0;
    for (const milestone of this.sortedMilestones) {
      if (this.lastEpoch >= milestone) {
        numDecays++;
      }
    }
    const factor = this.gamma ** numDecays;
    return this.baseLrs.map((lr) => lr * factor);
  }
}

/**
 * Linear learning rate scheduler.
 *
 * Linearly interpolates the learning rate multiplicative factor from startFactor
 * to endFactor over totalIters epochs. After totalIters, the factor remains at endFactor.
 *
 * lr = baseLr * (startFactor + (endFactor - startFactor) * epoch / totalIters)
 *
 * @example
 * ```ts
 * const scheduler = new LinearLR(optimizer, {
 *   startFactor: 0.1,
 *   endFactor: 0.01,
 *   totalIters: 100
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class LinearLR extends LRScheduler {
  private startFactor: number;
  private endFactor: number;
  private totalIters: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      startFactor?: number;
      endFactor?: number;
      totalIters: number;
      lastEpoch?: number;
    }
  ) {
    const startFactor = validatePositiveNumber(options.startFactor ?? 1 / 3, "startFactor");
    const endFactor = validatePositiveNumber(options.endFactor ?? 1.0, "endFactor");
    const totalIters = validatePositiveInteger(options.totalIters, "totalIters");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.startFactor = startFactor;
    this.endFactor = endFactor;
    this.totalIters = totalIters;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    if (this.lastEpoch >= this.totalIters) {
      return this.baseLrs.map((lr) => lr * this.endFactor);
    }

    const factor =
      this.startFactor + (this.endFactor - this.startFactor) * (this.lastEpoch / this.totalIters);
    return this.baseLrs.map((lr) => lr * factor);
  }
}

/**
 * Reduce learning rate on plateau.
 *
 * Reduces learning rate when a metric has stopped improving.
 * This scheduler reads a metric value and if no improvement is seen
 * for 'patience' epochs, the learning rate is reduced.
 *
 * @example
 * ```ts
 * const scheduler = new ReduceLROnPlateau(optimizer, {
 *   mode: 'min',
 *   factor: 0.1,
 *   patience: 10
 * });
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   const valLoss = validate();
 *   scheduler.step(valLoss);
 * }
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class ReduceLROnPlateau {
  private optimizer: SchedulerOptimizer;
  private mode: "min" | "max";
  private factor: number;
  private patience: number;
  private threshold: number;
  private thresholdMode: "rel" | "abs";
  private cooldown: number;
  private minLr: number;
  private best: number;
  private numBadEpochs: number;
  private cooldownCounter: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      mode?: "min" | "max";
      factor?: number;
      patience?: number;
      threshold?: number;
      thresholdMode?: "rel" | "abs";
      cooldown?: number;
      minLr?: number;
    } = {}
  ) {
    this.optimizer = optimizer;
    validateOptimizer(optimizer);
    this.mode = options.mode ?? "min";
    this.thresholdMode = options.thresholdMode ?? "rel";
    if (this.thresholdMode !== "rel" && this.thresholdMode !== "abs") {
      throw new InvalidParameterError(
        "thresholdMode must be 'rel' or 'abs'",
        "thresholdMode",
        options.thresholdMode
      );
    }
    if (this.mode !== "min" && this.mode !== "max") {
      throw new InvalidParameterError("mode must be 'min' or 'max'", "mode", options.mode);
    }
    this.factor = validateFiniteNumber(options.factor ?? 0.1, "factor");
    if (this.factor <= 0 || this.factor >= 1) {
      throw new InvalidParameterError(
        "factor must be in the interval (0, 1)",
        "factor",
        this.factor
      );
    }
    this.patience = validateNonNegativeInteger(options.patience ?? 10, "patience");
    this.threshold = validateNonNegativeNumber(options.threshold ?? 1e-4, "threshold");
    this.cooldown = validateNonNegativeInteger(options.cooldown ?? 0, "cooldown");
    this.minLr = validateNonNegativeNumber(options.minLr ?? 0, "minLr");
    this.best = this.mode === "min" ? Infinity : -Infinity;
    this.numBadEpochs = 0;
    this.cooldownCounter = 0;
  }

  /**
   * Check if metric improved.
   */
  private isBetter(current: number): boolean {
    // scikit/PyTorch default threshold_mode='rel': compare against a relative
    // margin, not an absolute one — an absolute threshold silently never fires
    // on metrics whose magnitude differs from ~1.
    if (this.thresholdMode === "rel") {
      if (this.mode === "min") return current < this.best * (1 - this.threshold);
      return current > this.best * (1 + this.threshold);
    }
    if (this.mode === "min") return current < this.best - this.threshold;
    return current > this.best + this.threshold;
  }

  /**
   * Perform a scheduler step based on the metric value.
   *
   * @param metric - Current value of the metric being monitored
   */
  step(metric: number): void {
    if (!Number.isFinite(metric)) {
      throw new InvalidParameterError("metric must be finite", "metric", metric);
    }
    // Ordering matches PyTorch: evaluate improvement first, then handle
    // cooldown (previously cooldown was decremented before the bad-epoch
    // bookkeeping, so the epoch cooldown expired on already counted as bad).
    if (this.isBetter(metric)) {
      this.best = metric;
      this.numBadEpochs = 0;
    } else {
      this.numBadEpochs++;
    }

    if (this.cooldownCounter > 0) {
      this.cooldownCounter--;
      this.numBadEpochs = 0;
    }

    if (this.numBadEpochs > this.patience) {
      this.reduceLr();
      this.cooldownCounter = this.cooldown;
      this.numBadEpochs = 0;
    }
  }

  /**
   * Reduce learning rate for all parameter groups.
   */
  private reduceLr(): void {
    for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
      const group = this.optimizer.paramGroups[i];
      if (!group) {
        throw new InvalidParameterError(
          `optimizer.paramGroups[${i}] is missing`,
          "optimizer.paramGroups",
          group
        );
      }
      const currentLr = resolveGroupLr(group, i);
      const newLr = Math.max(currentLr * this.factor, this.minLr);
      setGroupLr(group, newLr);
    }
  }

  /**
   * Get the current learning rates for all parameter groups.
   */
  getLastLr(): number[] {
    return this.optimizer.paramGroups.map((group, index) => resolveGroupLr(group, index));
  }
}

/**
 * Warmup scheduler that wraps another scheduler.
 *
 * Linearly increases the learning rate from 0 to the base lr over warmupEpochs,
 * then delegates to the wrapped scheduler.
 *
 * @example
 * ```ts
 * const baseScheduler = new CosineAnnealingLR(optimizer, { T_max: 100 });
 * const scheduler = new WarmupLR(optimizer, baseScheduler, { warmupEpochs: 5 });
 * ```
 */
export class WarmupLR extends LRScheduler {
  private warmupEpochs: number;
  private afterScheduler: LRScheduler | null;

  constructor(
    optimizer: SchedulerOptimizer,
    afterScheduler: LRScheduler | null,
    options: { warmupEpochs: number; lastEpoch?: number }
  ) {
    const warmupEpochs = validatePositiveInteger(options.warmupEpochs, "warmupEpochs");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.warmupEpochs = warmupEpochs;
    this.afterScheduler = afterScheduler;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    if (this.lastEpoch < this.warmupEpochs) {
      // Linear warmup
      const factor = (this.lastEpoch + 1) / this.warmupEpochs;
      return this.baseLrs.map((lr) => lr * factor);
    }

    if (this.afterScheduler) {
      // Delegate to wrapped scheduler
      return this.afterScheduler.getLr();
    }

    return this.baseLrs;
  }

  override step(): void {
    this.lastEpoch++;

    if (this.lastEpoch < this.warmupEpochs) {
      const newLrs = this.getLr();
      for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
        const group = this.optimizer.paramGroups[i];
        if (group) {
          const next = newLrs[i];
          if (next !== undefined) {
            setGroupLr(group, next);
          }
        }
      }
      return;
    }

    if (this.afterScheduler) {
      this.afterScheduler.step();
      const newLrs = this.afterScheduler.getLr();
      for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
        const group = this.optimizer.paramGroups[i];
        if (group) {
          const next = newLrs[i];
          if (next !== undefined) {
            setGroupLr(group, next);
          }
        }
      }
    } else {
      const newLrs = this.getLr();
      for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
        const group = this.optimizer.paramGroups[i];
        if (group) {
          const next = newLrs[i];
          if (next !== undefined) {
            setGroupLr(group, next);
          }
        }
      }
    }
  }
}

/**
 * One-cycle learning rate scheduler.
 *
 * Implements the 1cycle policy: lr starts at maxLr/divFactor, increases to maxLr
 * over pctStart of the training, then decreases to maxLr/finalDivFactor.
 *
 * @example
 * ```ts
 * const scheduler = new OneCycleLR(optimizer, {
 *   maxLr: 0.1,
 *   totalSteps: 1000,
 *   pctStart: 0.3
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class OneCycleLR extends LRScheduler {
  private maxLr: number;
  private totalSteps: number;
  private pctStart: number;
  private divFactor: number;
  private finalDivFactor: number;
  private annealStrategy: "cos" | "linear";

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      maxLr: number;
      totalSteps: number;
      pctStart?: number;
      divFactor?: number;
      finalDivFactor?: number;
      annealStrategy?: "cos" | "linear";
      lastEpoch?: number;
    }
  ) {
    const maxLr = validatePositiveNumber(options.maxLr, "maxLr");
    const totalSteps = validatePositiveInteger(options.totalSteps, "totalSteps");
    const pctStart = validateFiniteNumber(options.pctStart ?? 0.3, "pctStart");
    if (pctStart <= 0 || pctStart >= 1) {
      throw new InvalidParameterError(
        "pctStart must be in the interval (0, 1)",
        "pctStart",
        pctStart
      );
    }
    const divFactor = validatePositiveNumber(options.divFactor ?? 25, "divFactor");
    const finalDivFactor = validatePositiveNumber(options.finalDivFactor ?? 1e4, "finalDivFactor");
    const annealStrategy = options.annealStrategy ?? "cos";
    if (annealStrategy !== "cos" && annealStrategy !== "linear") {
      throw new InvalidParameterError(
        "annealStrategy must be 'cos' or 'linear'",
        "annealStrategy",
        annealStrategy
      );
    }
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.maxLr = maxLr;
    this.totalSteps = totalSteps;
    this.pctStart = pctStart;
    this.divFactor = divFactor;
    this.finalDivFactor = finalDivFactor;
    this.annealStrategy = annealStrategy;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    const stepNum = this.lastEpoch;
    const upSteps = Math.max(1, Math.floor(this.totalSteps * this.pctStart));
    const downSteps = Math.max(1, this.totalSteps - upSteps);

    const initialLr = this.maxLr / this.divFactor;
    const minLr = this.maxLr / this.finalDivFactor;

    let lr: number;

    if (stepNum >= this.totalSteps) {
      lr = minLr;
    } else if (stepNum < upSteps) {
      // Increasing phase
      const pct = stepNum / upSteps;
      lr = initialLr + (this.maxLr - initialLr) * pct;
    } else {
      // Decreasing phase
      const pct = (stepNum - upSteps) / downSteps;
      if (this.annealStrategy === "cos") {
        lr = minLr + ((this.maxLr - minLr) * (1 + Math.cos(Math.PI * pct))) / 2;
      } else {
        lr = this.maxLr - (this.maxLr - minLr) * pct;
      }
    }

    // Scale for each param group based on their base lr ratio
    const baseRef = this.baseLrs[0] ?? 0;
    return this.baseLrs.map((baseLr) => {
      if (baseRef === 0) {
        return baseLr === 0 ? 0 : lr;
      }
      return lr * (baseLr / baseRef);
    });
  }
}

/**
 * Cosine annealing with warm restarts.
 *
 * The learning rate follows a cosine curve from etaMax to etaMin over T_0 epochs,
 * then restarts. After each restart the period is multiplied by T_mult.
 *
 * lr = etaMin + 0.5 * (etaMax - etaMin) * (1 + cos(pi * T_cur / T_i))
 *
 * @category Optimization
 */
export class CosineAnnealingWarmRestarts extends LRScheduler {
  private readonly t0: number;
  private readonly tMult: number;
  private readonly etaMin: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      T_0: number;
      T_mult?: number;
      etaMin?: number;
      lastEpoch?: number;
    }
  ) {
    const t0 = validatePositiveInteger(options.T_0, "T_0");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.t0 = t0;
    this.tMult = validatePositiveNumber(options.T_mult ?? 1, "T_mult");
    this.etaMin = validateNonNegativeNumber(options.etaMin ?? 0, "etaMin");
    this.initializeFromLastEpoch(lastEpoch);
  }

  /** Incremental restart-period tracking (getLr is amortized O(1)). */
  private cachedEpoch = -2;
  private cachedTCur = 0;
  private cachedTI = 0;

  getLr(): number[] {
    let tCur: number;
    let tI: number;

    if (this.lastEpoch === this.cachedEpoch + 1) {
      // Sequential step: advance the cached period instead of replaying
      // every epoch since 0 (which made long runs O(total^2)).
      tCur = this.cachedTCur + 1;
      tI = this.cachedTI;
    } else {
      tCur = this.lastEpoch;
      tI = this.t0;
    }

    // Find current restart period
    while (tCur >= tI) {
      tCur -= tI;
      tI = Math.max(1, Math.floor(tI * this.tMult));
    }

    this.cachedEpoch = this.lastEpoch;
    this.cachedTCur = tCur;
    this.cachedTI = tI;

    const cosVal = Math.cos((Math.PI * tCur) / tI);
    return this.baseLrs.map((baseLr) => {
      return this.etaMin + 0.5 * (baseLr - this.etaMin) * (1 + cosVal);
    });
  }
}

/**
 * Cyclic learning rate scheduler.
 *
 * Cycles the learning rate between baseLr and maxLr using a triangular or
 * triangular2 policy.
 *
 * @category Optimization
 */
export class CyclicLR extends LRScheduler {
  private readonly baseLr: number;
  private readonly maxLr: number;
  private readonly stepSizeUp: number;
  private readonly stepSizeDown: number;
  private readonly mode: "triangular" | "triangular2";

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      baseLr: number;
      maxLr: number;
      stepSizeUp?: number;
      stepSizeDown?: number;
      mode?: "triangular" | "triangular2";
      lastEpoch?: number;
    }
  ) {
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.baseLr = validateNonNegativeNumber(options.baseLr, "baseLr");
    this.maxLr = validatePositiveNumber(options.maxLr, "maxLr");
    this.stepSizeUp = validatePositiveInteger(options.stepSizeUp ?? 2000, "stepSizeUp");
    this.stepSizeDown = validatePositiveInteger(
      options.stepSizeDown ?? this.stepSizeUp,
      "stepSizeDown"
    );
    this.mode = options.mode ?? "triangular";
    if (this.mode !== "triangular" && this.mode !== "triangular2") {
      throw new InvalidParameterError(
        "mode must be 'triangular' or 'triangular2'",
        "mode",
        this.mode
      );
    }
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    const cycleLen = this.stepSizeUp + this.stepSizeDown;
    const cycle = Math.floor(this.lastEpoch / cycleLen);
    const x = this.lastEpoch - cycle * cycleLen;

    let scaleFn: number;
    if (x <= this.stepSizeUp) {
      scaleFn = x / this.stepSizeUp;
    } else {
      scaleFn = (this.stepSizeDown - (x - this.stepSizeUp)) / this.stepSizeDown;
    }

    let scaleMode = 1;
    if (this.mode === "triangular2") {
      scaleMode = 1 / 2 ** cycle;
    }

    const lr = this.baseLr + (this.maxLr - this.baseLr) * Math.max(0, scaleFn) * scaleMode;

    // Scale for each param group
    const baseRef = this.baseLrs[0] ?? 0;
    return this.baseLrs.map((baseLr) => {
      if (baseRef === 0) return baseLr === 0 ? 0 : lr;
      return lr * (baseLr / baseRef);
    });
  }
}

/**
 * Lambda learning rate scheduler.
 *
 * Sets the learning rate to baseLr * lrLambda(epoch) for each parameter group.
 * Each parameter group can have its own lambda function if an array is provided.
 *
 * @example
 * ```ts
 * // Halve the learning rate every 10 epochs
 * const scheduler = new LambdaLR(optimizer, {
 *   lrLambda: (epoch) => 0.5 ** Math.floor(epoch / 10)
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class LambdaLR extends LRScheduler {
  private lrLambdas: Array<(epoch: number) => number>;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      lrLambda: ((epoch: number) => number) | Array<(epoch: number) => number>;
      lastEpoch?: number;
    }
  ) {
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);

    if (typeof options.lrLambda === "function") {
      this.lrLambdas = this.baseLrs.map(() => options.lrLambda as (epoch: number) => number);
    } else if (Array.isArray(options.lrLambda)) {
      if (options.lrLambda.length !== this.baseLrs.length) {
        throw new InvalidParameterError(
          `lrLambda array length (${options.lrLambda.length}) must match number of param groups (${this.baseLrs.length})`,
          "lrLambda",
          options.lrLambda.length
        );
      }
      this.lrLambdas = [...options.lrLambda];
    } else {
      throw new InvalidParameterError(
        "lrLambda must be a function or array of functions",
        "lrLambda",
        options.lrLambda
      );
    }

    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    return this.baseLrs.map((baseLr, i) => {
      const lambda = this.lrLambdas[i];
      if (!lambda) return baseLr;
      const factor = lambda(this.lastEpoch);
      if (!Number.isFinite(factor)) {
        throw new InvalidParameterError(
          `lrLambda[${i}] returned non-finite value at epoch ${this.lastEpoch}`,
          "lrLambda",
          factor
        );
      }
      return baseLr * factor;
    });
  }
}

/**
 * Sequential learning rate scheduler.
 *
 * Chains multiple schedulers in sequence. Each scheduler is active for a
 * specified number of epochs defined by milestones.
 *
 * @example
 * ```ts
 * const scheduler1 = new LinearLR(optimizer, { startFactor: 0.1, totalIters: 5 });
 * const scheduler2 = new ExponentialLR(optimizer, { gamma: 0.9 });
 * const scheduler = new SequentialLR(optimizer, {
 *   schedulers: [scheduler1, scheduler2],
 *   milestones: [5]
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class SequentialLR extends LRScheduler {
  private schedulers: LRScheduler[];
  private milestoneEpochs: number[];

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      schedulers: LRScheduler[];
      milestones: number[];
      lastEpoch?: number;
    }
  ) {
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);

    if (!Array.isArray(options.schedulers) || options.schedulers.length < 2) {
      throw new InvalidParameterError(
        "schedulers must be an array of at least 2 schedulers",
        "schedulers",
        options.schedulers?.length
      );
    }
    if (
      !Array.isArray(options.milestones) ||
      options.milestones.length !== options.schedulers.length - 1
    ) {
      throw new InvalidParameterError(
        `milestones length must be schedulers.length - 1 (${options.schedulers.length - 1})`,
        "milestones",
        options.milestones?.length
      );
    }

    // Validate milestones are sorted and positive
    for (let i = 0; i < options.milestones.length; i++) {
      const m = options.milestones[i];
      if (m === undefined || !Number.isInteger(m) || m < 0) {
        throw new InvalidParameterError(
          "milestones must be non-negative integers",
          "milestones",
          m
        );
      }
      if (i > 0) {
        const prev = options.milestones[i - 1];
        if (prev !== undefined && m <= prev) {
          throw new InvalidParameterError(
            "milestones must be strictly increasing",
            "milestones",
            options.milestones
          );
        }
      }
    }

    this.schedulers = [...options.schedulers];
    this.milestoneEpochs = [...options.milestones];
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    // Find which scheduler is active
    let idx = 0;
    for (let i = 0; i < this.milestoneEpochs.length; i++) {
      if (this.lastEpoch >= (this.milestoneEpochs[i] ?? 0)) {
        idx = i + 1;
      }
    }
    const scheduler = this.schedulers[idx];
    if (!scheduler) return this.baseLrs;
    return scheduler.getLr();
  }

  override step(): void {
    // Find the active scheduler and step it
    let idx = 0;
    for (let i = 0; i < this.milestoneEpochs.length; i++) {
      if (this.lastEpoch >= (this.milestoneEpochs[i] ?? 0)) {
        idx = i + 1;
      }
    }
    this.lastEpoch++;
    const scheduler = this.schedulers[idx];
    if (scheduler) {
      scheduler.step();
    }

    const newLrs = this.getLr();
    for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
      const group = this.optimizer.paramGroups[i];
      if (group) {
        const next = newLrs[i];
        if (next !== undefined) {
          setGroupLr(group, next);
        }
      }
    }
  }
}

/**
 * Polynomial learning rate scheduler.
 *
 * Decays the learning rate using a polynomial function from the initial lr
 * to a target value over a given number of epochs.
 *
 * lr = (baseLr - endLr) * (1 - epoch/totalIters)^power + endLr
 *
 * After totalIters, lr stays at endLr.
 *
 * @example
 * ```ts
 * const scheduler = new PolynomialLR(optimizer, {
 *   totalIters: 100,
 *   power: 2.0
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/optim-schedulers | Deepbox LR Schedulers}
 */
export class PolynomialLR extends LRScheduler {
  private totalIters: number;
  private power: number;
  private endLr: number;

  constructor(
    optimizer: SchedulerOptimizer,
    options: {
      totalIters: number;
      power?: number;
      endLr?: number;
      lastEpoch?: number;
    }
  ) {
    const totalIters = validatePositiveInteger(options.totalIters, "totalIters");
    const power = validatePositiveNumber(options.power ?? 1.0, "power");
    const endLr = validateNonNegativeNumber(options.endLr ?? 0, "endLr");
    const lastEpoch = validateLastEpoch(options.lastEpoch ?? -1);
    super(optimizer, -1);
    this.totalIters = totalIters;
    this.power = power;
    this.endLr = endLr;
    this.initializeFromLastEpoch(lastEpoch);
  }

  getLr(): number[] {
    if (this.lastEpoch >= this.totalIters) {
      return this.baseLrs.map(() => this.endLr);
    }
    const decay = (1 - this.lastEpoch / this.totalIters) ** this.power;
    return this.baseLrs.map((baseLr) => (baseLr - this.endLr) * decay + this.endLr);
  }
}
