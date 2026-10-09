/**
 * High-level Trainer abstraction for neural network training loops.
 *
 * Provides a canonical training loop with built-in support for:
 * - epoch-based training
 * - validation
 * - early stopping
 * - model checkpointing
 * - gradient clipping
 * - gradient accumulation over several batches
 * - restoring the weights of the best epoch
 * - logging
 *
 * @module nn/Trainer
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../core";
import { type AnyTensor, GradTensor, mulScalar, noGrad, type Tensor, tensor } from "../ndarray";
import { clip_grad_norm_ } from "./clip";
import type { Module } from "./module/Module";
import { EarlyStopping, ModelCheckpoint } from "./training";

/**
 * A loss function that takes model output and targets, returning a scalar loss.
 * The output may be a Tensor or GradTensor (AnyTensor) depending on training mode.
 */
export type LossFn = (output: AnyTensor, target: Tensor) => AnyTensor;

/**
 * An optimizer-like object with step() and zeroGrad() methods.
 */
export interface TrainerOptimizer {
  step(): void;
  zeroGrad(): void;
}

/**
 * A callback invoked at the end of each epoch.
 */
export type TrainerCallback = (info: EpochInfo) => void;

/**
 * Information about a completed epoch.
 */
export type EpochInfo = {
  readonly epoch: number;
  /** Mean training loss over the epoch, weighted by the first dimension of each input batch. */
  readonly trainLoss: number;
  /** Mean validation loss, or `undefined` when no validation data was given. */
  readonly valLoss: number | undefined;
};

/**
 * Options for the {@link Trainer}.
 */
export type TrainerOptions = {
  /** Number of training epochs. Default: 10 */
  readonly epochs?: number;
  /**
   * Early stopping configuration (monitors the validation loss, or the
   * training loss when no validation data is given). Omit to disable.
   * `patience` defaults to 5 and `minDelta` to 0.
   */
  readonly earlyStopping?: {
    readonly patience?: number;
    readonly minDelta?: number;
  };
  /** Model checkpointing helper. Omit to disable. */
  readonly checkpoint?: ModelCheckpoint;
  /** Callbacks invoked after each epoch, including the one that triggers early stopping. */
  readonly callbacks?: readonly TrainerCallback[];
  /** Whether to log epoch progress to the console. Default: false */
  readonly verbose?: boolean;
  /**
   * Clip the global L2 norm of the gradients to this value before every
   * optimizer step. Must be a positive finite number. Omit to disable.
   */
  readonly maxGradNorm?: number;
  /**
   * Accumulate the gradients of this many batches before every optimizer step (default: 1).
   * Each batch loss is divided by this number, so the step uses the mean gradient, as if the
   * batches had been one larger batch. When an epoch ends in the middle of a group, the
   * remaining batches are stepped as a smaller group (their gradient is the mean over the
   * batches seen). Must be a positive integer.
   */
  readonly accumulationSteps?: number;
  /**
   * After training, load the weights of the epoch with the lowest monitored loss (the
   * validation loss, or the training loss when there is no validation data) back into the
   * model (default: false). The weights are those at the end of that epoch. Only an epoch
   * with a strictly lower loss than all earlier ones counts as the best, and an epoch whose
   * loss is NaN is never the best.
   */
  readonly restoreBestWeights?: boolean;
};

/**
 * Result returned by {@link Trainer.fit}.
 */
export type TrainerResult = {
  readonly history: EpochInfo[];
  readonly stoppedEarly: boolean;
  /**
   * 1-based epoch with the best monitored loss. Only set when `earlyStopping` or
   * `restoreBestWeights` is configured; with `restoreBestWeights` it is the epoch whose weights
   * were restored.
   */
  readonly bestEpoch: number | undefined;
};

/**
 * High-level training loop for neural network modules.
 *
 * Each epoch trains on every training batch (zero gradients, forward, loss,
 * backward, optional gradient clipping, optimizer step; with `accumulationSteps` the
 * gradients of several batches are summed before each step), then evaluates the
 * validation batches without recording an autograd graph. The loss function
 * must return a single-element loss. Input batches are wrapped in a
 * `GradTensor` during training so that the model's parameters receive
 * gradients. If the loss is a plain `Tensor` rather than a `GradTensor`, no
 * backward pass is run.
 *
 * `fit()` can be called again to continue training: the history restarts at
 * epoch 1 and early-stopping state is reset. The model is left in the mode it
 * was last in (evaluation mode when validation data was given).
 *
 * The loss value is read on the host after each step, so `Trainer` is meant for
 * CPU tensors. A loss that lives in device memory throws a `DeviceError`; train
 * on a device with an explicit loop instead.
 *
 * @example
 * ```ts
 * import { Linear, mseLoss, Trainer } from 'deepbox/nn';
 * import { Adam } from 'deepbox/optim';
 *
 * const model = new Linear(2, 1);
 * const optimizer = new Adam(model.parameters(), { lr: 0.001 });
 * const lossFn = (pred, target) => mseLoss(pred, target);
 *
 * const trainer = new Trainer(model, optimizer, lossFn, {
 *   epochs: 50,
 *   earlyStopping: { patience: 5 },
 *   verbose: true,
 * });
 *
 * const trainData = [[xBatch1, yBatch1], [xBatch2, yBatch2]];
 * const result = trainer.fit(trainData);
 *
 * // Step once per 4 batches and load the best epoch's weights at the end
 * const accumulating = new Trainer(model, optimizer, lossFn, {
 *   epochs: 50,
 *   accumulationSteps: 4,
 *   restoreBestWeights: true,
 * });
 * ```
 */
export class Trainer {
  private readonly model: Module;
  private readonly optimizer: TrainerOptimizer;
  private readonly lossFn: LossFn;
  private readonly epochs: number;
  private readonly earlyStopping: EarlyStopping | undefined;
  private readonly checkpoint: ModelCheckpoint | undefined;
  private readonly callbacks: readonly TrainerCallback[];
  private readonly verbose: boolean;
  private readonly maxGradNorm: number | undefined;
  private readonly accumulationSteps: number;
  private readonly restoreBest: ModelCheckpoint | undefined;
  /** Batches whose gradients are accumulated since the last optimizer step. */
  private pending = 0;

  /**
   * @param model - Module to train
   * @param optimizer - Object with `step()` and `zeroGrad()` (any Deepbox optimizer)
   * @param lossFn - Maps `(output, target)` to a single-element loss
   * @param options - See {@link TrainerOptions}
   * @throws {InvalidParameterError} If an argument or option is invalid
   */
  constructor(
    model: Module,
    optimizer: TrainerOptimizer,
    lossFn: LossFn,
    options: TrainerOptions = {}
  ) {
    if (
      typeof model !== "object" ||
      model === null ||
      typeof model.forward !== "function" ||
      typeof model.train !== "function" ||
      typeof model.eval !== "function"
    ) {
      throw new InvalidParameterError("model must be a Module", "model", model);
    }
    if (typeof lossFn !== "function") {
      throw new InvalidParameterError("lossFn must be a function", "lossFn", lossFn);
    }
    if (
      typeof optimizer !== "object" ||
      optimizer === null ||
      typeof optimizer.step !== "function" ||
      typeof optimizer.zeroGrad !== "function"
    ) {
      throw new InvalidParameterError(
        "optimizer must provide step() and zeroGrad() methods",
        "optimizer",
        optimizer
      );
    }
    this.model = model;
    this.optimizer = optimizer;
    this.lossFn = lossFn;
    this.epochs = options.epochs ?? 10;
    this.checkpoint = options.checkpoint;
    this.callbacks = options.callbacks ?? [];
    this.verbose = options.verbose ?? false;
    this.maxGradNorm = options.maxGradNorm;
    this.accumulationSteps = options.accumulationSteps ?? 1;
    this.restoreBest =
      options.restoreBestWeights === true ? new ModelCheckpoint({ mode: "min" }) : undefined;

    if (!Number.isInteger(this.epochs) || this.epochs < 1) {
      throw new InvalidParameterError("epochs must be an integer >= 1", "epochs", this.epochs);
    }
    for (const cb of this.callbacks) {
      if (typeof cb !== "function") {
        throw new InvalidParameterError("callbacks must be functions", "callbacks", cb);
      }
    }
    if (
      this.maxGradNorm !== undefined &&
      (!Number.isFinite(this.maxGradNorm) || this.maxGradNorm <= 0)
    ) {
      throw new InvalidParameterError(
        `maxGradNorm must be a positive finite number; received ${this.maxGradNorm}`,
        "maxGradNorm",
        this.maxGradNorm
      );
    }

    if (!Number.isInteger(this.accumulationSteps) || this.accumulationSteps < 1) {
      throw new InvalidParameterError(
        `accumulationSteps must be an integer >= 1; received ${this.accumulationSteps}`,
        "accumulationSteps",
        this.accumulationSteps
      );
    }

    if (options.earlyStopping) {
      this.earlyStopping = new EarlyStopping({
        patience: options.earlyStopping.patience ?? 5,
        minDelta: options.earlyStopping.minDelta ?? 0,
        mode: "min",
      });
    }
  }

  /**
   * Run the training loop.
   *
   * `trainData` and `valData` are any synchronous iterables of `[input, target]`
   * batches: a pre-built array, a {@link import('../datasets').DataLoader | DataLoader}
   * over in-memory tensors, or a synchronous streaming DataLoader. Batches are
   * pulled lazily, one at a time, so no full batch array need be materialized.
   * The iterables are traversed once per epoch, so they must be re-iterable
   * (an array or a DataLoader, not a one-shot generator object).
   *
   * For an asynchronous / prefetching streaming source, use
   * {@link Trainer.fitAsync | fitAsync} instead.
   *
   * @param trainData - Iterable of [input, target] tensor pairs (batches)
   * @param valData - Optional validation batches
   * @returns Training result with history and early stopping info
   * @throws {InvalidParameterError} If an epoch yields no batches
   * @throws {ShapeError} If the loss function does not return a single-element loss
   */
  fit(
    trainData: Iterable<readonly [Tensor, Tensor]>,
    valData?: Iterable<readonly [Tensor, Tensor]>
  ): TrainerResult {
    const run = this.beginRun();

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      this.model.train();
      const train = new LossMeter();
      for (const [x, y] of trainData) {
        train.add(this.trainStep(x, y), batchWeight(x));
      }
      this.flushAccumulation();
      const trainLoss = train.mean("trainData", epoch);

      let valLoss: number | undefined;
      if (valData) {
        this.model.eval();
        const val = new LossMeter();
        for (const [x, y] of valData) {
          val.add(this.evalStep(x, y), batchWeight(x));
        }
        valLoss = val.mean("valData", epoch);
      }

      if (this.endEpoch(run, epoch, trainLoss, valLoss)) break;
    }

    return this.finishRun(run);
  }

  /**
   * Run the training loop over an asynchronous (or synchronous) batch source.
   *
   * Identical in behavior to {@link Trainer.fit | fit}, but consumes the data
   * with `for await`, so it accepts an out-of-core streaming DataLoader whose
   * batches arrive asynchronously (e.g. with read-ahead prefetch, or a source
   * reading from disk/network). Batches are pulled lazily one at a time, so the
   * corpus is never fully materialized.
   *
   * @param trainData - Async or sync iterable of [input, target] batches.
   * @param valData - Optional validation batches (async or sync iterable).
   * @returns A promise of the training result with history and early-stopping info.
   * @throws {InvalidParameterError} If an epoch yields no batches
   * @throws {ShapeError} If the loss function does not return a single-element loss
   */
  async fitAsync(
    trainData: AsyncIterable<readonly [Tensor, Tensor]> | Iterable<readonly [Tensor, Tensor]>,
    valData?: AsyncIterable<readonly [Tensor, Tensor]> | Iterable<readonly [Tensor, Tensor]>
  ): Promise<TrainerResult> {
    const run = this.beginRun();

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      this.model.train();
      const train = new LossMeter();
      for await (const [x, y] of trainData) {
        train.add(this.trainStep(x, y), batchWeight(x));
      }
      this.flushAccumulation();
      const trainLoss = train.mean("trainData", epoch);

      let valLoss: number | undefined;
      if (valData) {
        this.model.eval();
        const val = new LossMeter();
        for await (const [x, y] of valData) {
          val.add(this.evalStep(x, y), batchWeight(x));
        }
        valLoss = val.mean("valData", epoch);
      }

      if (this.endEpoch(run, epoch, trainLoss, valLoss)) break;
    }

    return this.finishRun(run);
  }

  /** Reset per-run state: early stopping starts fresh on every fit. */
  private beginRun(): RunState {
    this.earlyStopping?.reset();
    this.restoreBest?.reset();
    this.pending = 0;
    return { history: [], stoppedEarly: false };
  }

  /**
   * One batch of training; returns the scalar loss. The optimizer steps once every
   * `accumulationSteps` batches.
   */
  private trainStep(x: Tensor, y: Tensor): number {
    // A new accumulation group starts from zero gradients.
    if (this.pending === 0) this.optimizer.zeroGrad();
    // Layers only record an autograd graph when they receive a GradTensor, so a
    // plain input batch is wrapped (without tracking its own gradient). Without
    // this the loss is a plain Tensor and the parameters never get gradients.
    const input = x.dtype === "string" ? x : GradTensor.fromTensor(x, { requiresGrad: false });
    const output = this.model.forward(input);
    const loss = this.lossFn(output, y);
    const lossVal = scalarLoss(loss);

    // Backward pass (if loss is a GradTensor). With accumulation each batch contributes
    // 1 / accumulationSteps of its gradient, so the sum over a group is the mean.
    if (GradTensor.isGradTensor(loss)) {
      if (this.accumulationSteps === 1) {
        loss.backward();
      } else {
        loss.backward(tensor([1 / this.accumulationSteps], { dtype: loss.dtype }));
      }
    }

    this.pending++;
    if (this.pending >= this.accumulationSteps) this.applyStep();
    return lossVal;
  }

  /** Clip the accumulated gradients (when asked to) and take one optimizer step. */
  private applyStep(): void {
    if (this.maxGradNorm !== undefined) {
      clip_grad_norm_(this.model.parameters(), this.maxGradNorm);
    }
    this.optimizer.step();
    this.pending = 0;
  }

  /**
   * At the end of an epoch, step on a group that is smaller than `accumulationSteps`. Its
   * gradients were each scaled by 1 / accumulationSteps, so they are rescaled to the mean over
   * the batches that were actually accumulated.
   */
  private flushAccumulation(): void {
    if (this.pending === 0) return;
    if (this.pending < this.accumulationSteps) {
      const factor = this.accumulationSteps / this.pending;
      for (const param of this.model.parameters()) {
        const grad = param.grad;
        if (grad !== null) param.setGrad(mulScalar(grad, factor));
      }
    }
    this.applyStep();
  }

  /**
   * Loss of a validation batch. Evaluation must not record an autograd graph:
   * model.eval() only toggles train-mode layers (dropout/batchnorm), it does
   * NOT disable gradient tracking, so without noGrad every validation batch
   * would build and discard a full backward graph.
   */
  private evalStep(x: Tensor, y: Tensor): number {
    return noGrad(() => scalarLoss(this.lossFn(this.model.forward(x), y)));
  }

  /**
   * Record the epoch, log, checkpoint, run callbacks, then check early
   * stopping. Returns true when training should stop.
   */
  private endEpoch(
    run: RunState,
    epoch: number,
    trainLoss: number,
    valLoss: number | undefined
  ): boolean {
    const epochInfo: EpochInfo = { epoch, trainLoss, valLoss };
    run.history.push(epochInfo);

    if (this.verbose) {
      let msg = `Epoch ${epoch}/${this.epochs}, train_loss: ${trainLoss.toFixed(6)}`;
      if (valLoss !== undefined) msg += `, val_loss: ${valLoss.toFixed(6)}`;
      console.log(msg);
    }

    const monitored = valLoss ?? trainLoss;
    if (this.checkpoint) {
      this.checkpoint.step(this.model, monitored);
    }
    this.restoreBest?.step(this.model, monitored);

    for (const cb of this.callbacks) {
      cb(epochInfo);
    }

    if (this.earlyStopping?.step(monitored)) {
      run.stoppedEarly = true;
      return true;
    }
    return false;
  }

  private finishRun(run: RunState): TrainerResult {
    let bestEpoch = this.earlyStopping ? this.earlyStopping.bestEpochNum : undefined;
    // No state is saved when no monitored loss was below infinity (for example all NaN); the
    // model is then left as it is and `bestEpoch` stays as early stopping reports it.
    if (this.restoreBest?.hasSavedState) {
      this.restoreBest.restore(this.model);
      bestEpoch = this.restoreBest.bestEpochNum;
    }
    return { history: run.history, stoppedEarly: run.stoppedEarly, bestEpoch };
  }
}

type RunState = {
  readonly history: EpochInfo[];
  stoppedEarly: boolean;
};

/** Read the value of a single-element loss tensor. */
function scalarLoss(loss: AnyTensor): number {
  if (loss.size !== 1) {
    throw new ShapeError(
      `lossFn must return a single-element loss; received shape [${loss.shape.join(", ")}]. ` +
        "Reduce the loss (for example with a mean) before returning it."
    );
  }
  return Number(loss.data[loss.offset]);
}

/** Weight of a batch in the epoch mean: its number of samples (first input dimension). */
function batchWeight(x: Tensor): number {
  return x.ndim >= 1 ? (x.shape[0] ?? 1) : 1;
}

/** Running sample-weighted mean of batch losses. */
class LossMeter {
  private weightedSum = 0;
  private plainSum = 0;
  private weight = 0;
  private batches = 0;

  add(loss: number, weight: number): void {
    this.weightedSum += loss * weight;
    this.plainSum += loss;
    this.weight += weight;
    this.batches++;
  }

  mean(source: string, epoch: number): number {
    if (this.batches === 0) {
      throw new InvalidParameterError(
        `${source} yielded no batches in epoch ${epoch}. Pass a non-empty, re-iterable ` +
          "source such as an array or a DataLoader; a one-shot generator is exhausted after its first epoch.",
        source,
        undefined
      );
    }
    // Batches without samples carry no weight; fall back to a plain mean.
    return this.weight > 0 ? this.weightedSum / this.weight : this.plainSum / this.batches;
  }
}
