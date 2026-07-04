/**
 * Training utilities for neural networks.
 *
 * Provides callbacks and helpers for common training patterns:
 * - EarlyStopping: Stop training when a metric stops improving
 * - GradientAccumulator: Accumulate gradients over multiple mini-batches
 *
 * @module nn/training
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import type { Module } from "./module/Module";

/**
 * Early stopping callback to terminate training when a monitored metric
 * stops improving.
 *
 * Tracks a metric value across epochs and signals when training should
 * stop based on patience (number of epochs with no improvement).
 *
 * @example
 * ```ts
 * import { EarlyStopping } from 'deepbox/nn';
 *
 * const es = new EarlyStopping({ patience: 5, minDelta: 0.001 });
 *
 * for (let epoch = 0; epoch < maxEpochs; epoch++) {
 *   const valLoss = trainOneEpoch();
 *   if (es.step(valLoss)) {
 *     console.log(`Early stopping at epoch ${epoch}`);
 *     break;
 *   }
 * }
 * ```
 */
export class EarlyStopping {
  private readonly patience: number;
  private readonly minDelta: number;
  private readonly mode: "min" | "max";

  private bestScore: number;
  private counter: number;
  private stopped: boolean;
  private bestEpoch: number;
  private epochCount: number;

  /**
   * Create an EarlyStopping callback.
   *
   * @param options.patience - Number of epochs with no improvement before stopping. Default: 10
   * @param options.minDelta - Minimum change to qualify as an improvement. Default: 0
   * @param options.mode - "min" if lower is better (e.g. loss), "max" if higher is better (e.g. accuracy). Default: "min"
   */
  constructor(
    options: {
      readonly patience?: number;
      readonly minDelta?: number;
      readonly mode?: "min" | "max";
    } = {}
  ) {
    this.patience = options.patience ?? 10;
    this.minDelta = options.minDelta ?? 0;
    this.mode = options.mode ?? "min";

    if (!Number.isInteger(this.patience) || this.patience < 1) {
      throw new InvalidParameterError(
        `patience must be a positive integer; received ${this.patience}`,
        "patience",
        this.patience
      );
    }
    if (!Number.isFinite(this.minDelta) || this.minDelta < 0) {
      throw new InvalidParameterError(
        `minDelta must be a non-negative finite number; received ${this.minDelta}`,
        "minDelta",
        this.minDelta
      );
    }
    if (this.mode !== "min" && this.mode !== "max") {
      throw new InvalidParameterError(
        `mode must be "min" or "max"; received "${String(this.mode)}"`,
        "mode",
        this.mode
      );
    }

    this.bestScore = this.mode === "min" ? Infinity : -Infinity;
    this.counter = 0;
    this.stopped = false;
    this.bestEpoch = 0;
    this.epochCount = 0;
  }

  /**
   * Report the current metric value and check if training should stop.
   *
   * @param metric - Current metric value
   * @returns true if training should stop, false otherwise
   */
  step(metric: number): boolean {
    this.epochCount++;

    const improved =
      this.mode === "min"
        ? metric < this.bestScore - this.minDelta
        : metric > this.bestScore + this.minDelta;

    if (improved) {
      this.bestScore = metric;
      this.counter = 0;
      this.bestEpoch = this.epochCount;
    } else {
      this.counter++;
      if (this.counter >= this.patience) {
        this.stopped = true;
        return true;
      }
    }

    return false;
  }

  /** Whether early stopping has been triggered. */
  get isStopped(): boolean {
    return this.stopped;
  }

  /** The best metric value observed so far. */
  get best(): number {
    return this.bestScore;
  }

  /** The epoch at which the best metric was observed (1-indexed). */
  get bestEpochNum(): number {
    return this.bestEpoch;
  }

  /** Number of epochs since last improvement. */
  get waitCount(): number {
    return this.counter;
  }

  /** Reset the callback state to start fresh. */
  reset(): void {
    this.bestScore = this.mode === "min" ? Infinity : -Infinity;
    this.counter = 0;
    this.stopped = false;
    this.bestEpoch = 0;
    this.epochCount = 0;
  }
}

/**
 * Gradient accumulation helper for training with effective batch sizes
 * larger than what fits in memory.
 *
 * Accumulates gradients over multiple forward/backward passes before
 * performing a single optimizer step, then zeros the gradients.
 *
 * @example
 * ```ts
 * import { GradientAccumulator } from 'deepbox/nn';
 *
 * const accumulator = new GradientAccumulator(4); // accumulate over 4 steps
 *
 * for (const batch of dataLoader) {
 *   const loss = forward(batch);
 *   backward(loss);
 *
 *   if (accumulator.step()) {
 *     // Time to update: 4 mini-batches accumulated
 *     optimizer.step();
 *     optimizer.zeroGrad();
 *   }
 * }
 * ```
 */
export class GradientAccumulator {
  private readonly accumSteps: number;
  private currentStep: number;
  private totalSteps: number;

  /**
   * Create a GradientAccumulator.
   *
   * @param accumSteps - Number of mini-batches to accumulate before updating. Must be >= 1.
   */
  constructor(accumSteps: number) {
    if (!Number.isInteger(accumSteps) || accumSteps < 1) {
      throw new InvalidParameterError(
        `accumSteps must be a positive integer; received ${accumSteps}`,
        "accumSteps",
        accumSteps
      );
    }
    this.accumSteps = accumSteps;
    this.currentStep = 0;
    this.totalSteps = 0;
  }

  /**
   * Record one mini-batch and check if it's time to perform an optimizer step.
   *
   * @returns true if accumSteps mini-batches have been accumulated and
   *          the optimizer should step. false otherwise.
   */
  step(): boolean {
    this.currentStep++;
    this.totalSteps++;

    if (this.currentStep >= this.accumSteps) {
      this.currentStep = 0;
      return true;
    }
    return false;
  }

  /** Number of mini-batches accumulated since the last optimizer step. */
  get accumulated(): number {
    return this.currentStep;
  }

  /** Total number of mini-batches processed. */
  get totalProcessed(): number {
    return this.totalSteps;
  }

  /** Number of optimizer steps that have been triggered. */
  get optimizerSteps(): number {
    return Math.floor(this.totalSteps / this.accumSteps);
  }

  /** The configured number of accumulation steps. */
  get steps(): number {
    return this.accumSteps;
  }

  /**
   * The scaling factor for loss to account for accumulation.
   * Divide your loss by this value before backward() to get correct gradient magnitudes.
   */
  get scaleFactor(): number {
    return this.accumSteps;
  }

  /** Reset the accumulator state. */
  reset(): void {
    this.currentStep = 0;
    this.totalSteps = 0;
  }
}

/**
 * Model checkpoint helper to track and save the best model state
 * during training.
 *
 * Stores a copy of the model's state dictionary whenever the
 * monitored metric improves.
 *
 * @example
 * ```ts
 * import { ModelCheckpoint } from 'deepbox/nn';
 *
 * const checkpoint = new ModelCheckpoint({ mode: 'min' });
 *
 * for (let epoch = 0; epoch < maxEpochs; epoch++) {
 *   const valLoss = trainOneEpoch();
 *   checkpoint.step(model, valLoss);
 * }
 *
 * // Restore best model
 * checkpoint.restore(model);
 * ```
 */
export class ModelCheckpoint {
  private readonly mode: "min" | "max";
  private bestScore: number;
  private savedState: ReturnType<Module["stateDict"]> | undefined;
  private bestEpoch: number;
  private epochCount: number;

  /**
   * Create a ModelCheckpoint.
   *
   * @param options.mode - "min" if lower is better, "max" if higher is better. Default: "min"
   */
  constructor(
    options: {
      readonly mode?: "min" | "max";
    } = {}
  ) {
    this.mode = options.mode ?? "min";
    if (this.mode !== "min" && this.mode !== "max") {
      throw new InvalidParameterError(
        `mode must be "min" or "max"; received "${String(this.mode)}"`,
        "mode",
        this.mode
      );
    }
    this.bestScore = this.mode === "min" ? Infinity : -Infinity;
    this.savedState = undefined;
    this.bestEpoch = 0;
    this.epochCount = 0;
  }

  /**
   * Check the metric and save model state if improved.
   *
   * @param model - The model to checkpoint
   * @param metric - Current metric value
   * @returns true if the model state was saved (metric improved), false otherwise
   */
  step(model: Module, metric: number): boolean {
    this.epochCount++;

    const improved = this.mode === "min" ? metric < this.bestScore : metric > this.bestScore;

    if (improved) {
      this.bestScore = metric;
      this.savedState = model.stateDict();
      this.bestEpoch = this.epochCount;
      return true;
    }
    return false;
  }

  /**
   * Restore the best saved model state.
   *
   * @param model - The model to restore state into
   * @throws {InvalidParameterError} If no checkpoint has been saved
   */
  restore(model: Module): void {
    if (!this.savedState) {
      throw new InvalidParameterError(
        "No checkpoint saved yet. Call step() with a model first.",
        "savedState",
        undefined
      );
    }
    model.loadStateDict(this.savedState);
  }

  /** Whether a checkpoint has been saved. */
  get hasSavedState(): boolean {
    return this.savedState !== undefined;
  }

  /** The best metric value observed. */
  get best(): number {
    return this.bestScore;
  }

  /** The epoch at which the best metric was observed (1-indexed). */
  get bestEpochNum(): number {
    return this.bestEpoch;
  }

  /** Reset the checkpoint state. */
  reset(): void {
    this.bestScore = this.mode === "min" ? Infinity : -Infinity;
    this.savedState = undefined;
    this.bestEpoch = 0;
    this.epochCount = 0;
  }
}
