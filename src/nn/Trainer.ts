/**
 * High-level Trainer abstraction for neural network training loops.
 *
 * Provides a canonical training loop with built-in support for:
 * - epoch-based training
 * - validation
 * - early stopping
 * - model checkpointing
 * - logging
 *
 * @module nn/Trainer
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { type AnyTensor, noGrad, type Tensor } from "../ndarray";
import type { Module } from "./module/Module";
import { EarlyStopping, type ModelCheckpoint } from "./training";

/**
 * A loss function that takes model output and targets, returning a scalar loss.
 * The output may be a Tensor or GradTensor (AnyTensor) depending on training mode.
 */
export type LossFn = (output: AnyTensor, target: Tensor) => AnyTensor;

/**
 * An optimizer-like object with step() and zeroGrad() methods.
 */
interface OptimizerLike {
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
  readonly trainLoss: number;
  readonly valLoss: number | undefined;
};

/**
 * Options for the {@link Trainer}.
 */
export type TrainerOptions = {
  /** Number of training epochs. Default: 10 */
  readonly epochs?: number;
  /** Early stopping configuration. Omit to disable. */
  readonly earlyStopping?: {
    readonly patience?: number;
    readonly minDelta?: number;
  };
  /** Model checkpointing helper. Omit to disable. */
  readonly checkpoint?: ModelCheckpoint;
  /** Callbacks invoked after each epoch. */
  readonly callbacks?: readonly TrainerCallback[];
  /** Whether to log epoch progress to the console. Default: false */
  readonly verbose?: boolean;
};

/**
 * Result returned by {@link Trainer.fit}.
 */
export type TrainerResult = {
  readonly history: EpochInfo[];
  readonly stoppedEarly: boolean;
  readonly bestEpoch: number | undefined;
};

/**
 * High-level training loop for neural network modules.
 *
 * @example
 * ```ts
 * import { Trainer } from 'deepbox/nn';
 * import { Adam } from 'deepbox/optim';
 *
 * const model = new MyModel();
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
 * ```
 */
export class Trainer {
  private readonly model: Module;
  private readonly optimizer: OptimizerLike;
  private readonly lossFn: LossFn;
  private readonly epochs: number;
  private readonly earlyStopping: EarlyStopping | undefined;
  private readonly checkpoint: ModelCheckpoint | undefined;
  private readonly callbacks: readonly TrainerCallback[];
  private readonly verbose: boolean;

  constructor(
    model: Module,
    optimizer: OptimizerLike,
    lossFn: LossFn,
    options: TrainerOptions = {}
  ) {
    this.model = model;
    this.optimizer = optimizer;
    this.lossFn = lossFn;
    this.epochs = options.epochs ?? 10;
    this.checkpoint = options.checkpoint;
    this.callbacks = options.callbacks ?? [];
    this.verbose = options.verbose ?? false;

    if (!Number.isInteger(this.epochs) || this.epochs < 1) {
      throw new InvalidParameterError("epochs must be an integer >= 1", "epochs", this.epochs);
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
   * batches — a pre-built array, a {@link import('../datasets').DataLoader | DataLoader}
   * over in-memory tensors, or a synchronous streaming DataLoader. Batches are
   * pulled lazily, one at a time, so no full batch array need be materialized.
   *
   * For an asynchronous / prefetching streaming source, use
   * {@link Trainer.fitAsync | fitAsync} instead.
   *
   * @param trainData - Iterable of [input, target] tensor pairs (batches)
   * @param valData - Optional validation batches
   * @returns Training result with history and early stopping info
   */
  fit(
    trainData: Iterable<readonly [Tensor, Tensor]>,
    valData?: Iterable<readonly [Tensor, Tensor]>
  ): TrainerResult {
    const history: EpochInfo[] = [];
    let stoppedEarly = false;
    let bestEpoch: number | undefined;

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      // Training phase
      this.model.train();
      let trainLossSum = 0;
      let trainBatches = 0;

      for (const [x, y] of trainData) {
        this.optimizer.zeroGrad();
        const output = this.model.forward(x);
        const loss = this.lossFn(output, y);

        // Get scalar loss value
        const lossVal = Number(loss.data[loss.offset]);
        trainLossSum += lossVal;
        trainBatches++;

        // Backward pass (if loss is a GradTensor)
        if ("backward" in loss && typeof loss.backward === "function") {
          (loss as { backward(): void }).backward();
        }

        this.optimizer.step();
      }

      const trainLoss = trainBatches > 0 ? trainLossSum / trainBatches : 0;

      // Validation phase
      let valLoss: number | undefined;
      if (valData) {
        this.model.eval();
        let valLossSum = 0;
        let valBatches = 0;

        // Evaluation must not record an autograd graph: model.eval() only
        // toggles train-mode layers (dropout/batchnorm), it does NOT disable
        // gradient tracking. Without noGrad, every validation batch builds and
        // immediately discards a full backward graph — doubling peak memory on
        // large validation sets. noGrad also models the correct inference
        // pattern for users copying this loop.
        for (const [x, y] of valData) {
          const lossVal = noGrad(() => {
            const output = this.model.forward(x);
            const loss = this.lossFn(output, y);
            return Number(loss.data[loss.offset]);
          });
          valLossSum += lossVal;
          valBatches++;
        }

        valLoss = valBatches > 0 ? valLossSum / valBatches : 0;
      }

      const epochInfo: EpochInfo = { epoch, trainLoss, valLoss };
      history.push(epochInfo);

      if (this.verbose) {
        let msg = `Epoch ${epoch}/${this.epochs} — train_loss: ${trainLoss.toFixed(6)}`;
        if (valLoss !== undefined) msg += ` — val_loss: ${valLoss.toFixed(6)}`;
        console.log(msg);
      }

      // Checkpointing
      const metricForCallbacks = valLoss ?? trainLoss;
      if (this.checkpoint) {
        this.checkpoint.step(this.model, metricForCallbacks);
      }

      // Early stopping
      if (this.earlyStopping) {
        if (this.earlyStopping.step(metricForCallbacks)) {
          stoppedEarly = true;
          bestEpoch = this.earlyStopping.bestEpochNum;
          break;
        }
      }

      // User callbacks
      for (const cb of this.callbacks) {
        cb(epochInfo);
      }
    }

    if (!stoppedEarly && this.earlyStopping) {
      bestEpoch = this.earlyStopping.bestEpochNum;
    }

    return { history, stoppedEarly, bestEpoch };
  }

  /**
   * Run the training loop over an asynchronous (or synchronous) batch source.
   *
   * Identical in behavior to {@link Trainer.fit | fit}, but consumes the data
   * with `for await`, so it accepts an out-of-core streaming DataLoader whose
   * batches arrive asynchronously (e.g. with read-ahead prefetch, or a source
   * reading from disk/network). Batches are pulled lazily one at a time — the
   * corpus is never fully materialized.
   *
   * @param trainData - Async or sync iterable of [input, target] batches.
   * @param valData - Optional validation batches (async or sync iterable).
   * @returns A promise of the training result with history and early-stopping info.
   */
  async fitAsync(
    trainData: AsyncIterable<readonly [Tensor, Tensor]> | Iterable<readonly [Tensor, Tensor]>,
    valData?: AsyncIterable<readonly [Tensor, Tensor]> | Iterable<readonly [Tensor, Tensor]>
  ): Promise<TrainerResult> {
    const history: EpochInfo[] = [];
    let stoppedEarly = false;
    let bestEpoch: number | undefined;

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      // Training phase
      this.model.train();
      let trainLossSum = 0;
      let trainBatches = 0;

      for await (const [x, y] of trainData) {
        this.optimizer.zeroGrad();
        const output = this.model.forward(x);
        const loss = this.lossFn(output, y);

        const lossVal = Number(loss.data[loss.offset]);
        trainLossSum += lossVal;
        trainBatches++;

        if ("backward" in loss && typeof loss.backward === "function") {
          (loss as { backward(): void }).backward();
        }

        this.optimizer.step();
      }

      const trainLoss = trainBatches > 0 ? trainLossSum / trainBatches : 0;

      // Validation phase
      let valLoss: number | undefined;
      if (valData) {
        this.model.eval();
        let valLossSum = 0;
        let valBatches = 0;

        for await (const [x, y] of valData) {
          // See fit(): eval() does not disable gradient tracking; noGrad avoids
          // building throwaway backward graphs per validation batch.
          const lossVal = noGrad(() => {
            const output = this.model.forward(x);
            const loss = this.lossFn(output, y);
            return Number(loss.data[loss.offset]);
          });
          valLossSum += lossVal;
          valBatches++;
        }

        valLoss = valBatches > 0 ? valLossSum / valBatches : 0;
      }

      const epochInfo: EpochInfo = { epoch, trainLoss, valLoss };
      history.push(epochInfo);

      if (this.verbose) {
        let msg = `Epoch ${epoch}/${this.epochs} — train_loss: ${trainLoss.toFixed(6)}`;
        if (valLoss !== undefined) msg += ` — val_loss: ${valLoss.toFixed(6)}`;
        console.log(msg);
      }

      const metricForCallbacks = valLoss ?? trainLoss;
      if (this.checkpoint) {
        this.checkpoint.step(this.model, metricForCallbacks);
      }

      if (this.earlyStopping) {
        if (this.earlyStopping.step(metricForCallbacks)) {
          stoppedEarly = true;
          bestEpoch = this.earlyStopping.bestEpochNum;
          break;
        }
      }

      for (const cb of this.callbacks) {
        cb(epochInfo);
      }
    }

    if (!stoppedEarly && this.earlyStopping) {
      bestEpoch = this.earlyStopping.bestEpochNum;
    }

    return { history, stoppedEarly, bestEpoch };
  }
}
