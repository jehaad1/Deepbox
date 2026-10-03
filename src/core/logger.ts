/**
 * Logging and verbosity system for Deepbox.
 *
 * Models and utilities can create a Logger with a verbosity level and
 * use it to emit progress messages during long-running operations
 * (e.g. iterative fitting, training loops).
 *
 * Verbosity levels:
 * - 0: silent (no output)
 * - 1: progress summary (e.g. final convergence info)
 * - 2: per-iteration progress
 * - 3+: debug-level detail
 *
 * @module core/logger
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { InvalidParameterError } from "./errors/index";

/** Verbosity level type (0 = silent, higher = more verbose). */
export type VerboseLevel = 0 | 1 | 2 | 3;

/** A log entry produced by the logger. */
export interface LogEntry {
  readonly level: VerboseLevel;
  readonly message: string;
  /** Milliseconds since the Unix epoch, from `Date.now()`. */
  readonly timestamp: number;
  /** Name of the component that produced the entry (the Logger's `source`). */
  readonly source?: string;
}

/** Maximum number of entries a Logger keeps for {@link Logger.getEntries}. */
const MAX_RECORDED_ENTRIES = 10_000;

type LogHandler = (entry: LogEntry) => void;

let globalHandler: LogHandler | undefined;

/**
 * Set a global log handler for all Logger instances.
 * The handler is called only for entries that pass the logger's verbosity level.
 * Pass undefined to revert to default (console.log).
 */
export function setLogHandler(handler: LogHandler | undefined): void {
  globalHandler = handler;
}

/**
 * Get the current global log handler (undefined = console.log default).
 */
export function getLogHandler(): LogHandler | undefined {
  return globalHandler;
}

/**
 * Logger for Deepbox models and utilities.
 *
 * @example
 * ```ts
 * import { Logger } from 'deepbox/core';
 *
 * const log = new Logger(1, 'KMeans');
 * log.info('Converged in 15 iterations');    // prints at verbosity >= 1
 * log.debug('Iteration 5: inertia=23.4');    // prints at verbosity >= 2
 * log.trace('Distance matrix computed');      // prints at verbosity >= 3
 * ```
 */
export class Logger {
  private readonly verbose: VerboseLevel;
  private readonly prefix: string;
  /** Ring buffer holding the most recent MAX_RECORDED_ENTRIES entries. */
  private readonly entries: LogEntry[];
  private head = 0;

  /**
   * @param verbose - Verbosity level (0=silent, 1=summary, 2=progress, 3=debug)
   * @param source - Name of the component (used as prefix in log messages)
   */
  constructor(verbose: VerboseLevel, source: string) {
    if (typeof verbose !== "number" || !Number.isInteger(verbose) || verbose < 0 || verbose > 3) {
      throw new InvalidParameterError(
        `verbose must be 0, 1, 2, or 3; received ${String(verbose)}`,
        "verbose",
        verbose
      );
    }
    if (typeof source !== "string") {
      throw new InvalidParameterError(
        `source must be a string; received ${typeof source}`,
        "source",
        source
      );
    }
    this.verbose = verbose;
    this.prefix = source;
    this.entries = [];
  }

  /**
   * Log at level 1 (summary). Visible when verbose >= 1.
   */
  info(message: string): void {
    this.log(1, message);
  }

  /**
   * Log at level 2 (per-iteration progress). Visible when verbose >= 2.
   */
  debug(message: string): void {
    this.log(2, message);
  }

  /**
   * Log at level 3 (trace/debug detail). Visible when verbose >= 3.
   */
  trace(message: string): void {
    this.log(3, message);
  }

  /**
   * Log a message at a specific level.
   *
   * @param level - Level of the message (1, 2 or 3). Level 0 never emits.
   * @param message - Message text
   * @throws {InvalidParameterError} If `level` is not an integer in [0, 3]
   */
  log(level: VerboseLevel, message: string): void {
    if (!Number.isInteger(level) || level < 0 || level > 3) {
      throw new InvalidParameterError(
        `level must be 0, 1, 2, or 3; received ${String(level)}`,
        "level",
        level
      );
    }
    if (level === 0) return; // level 0 never emits
    const entry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
      source: this.prefix,
    };
    if (this.entries.length < MAX_RECORDED_ENTRIES) {
      this.entries.push(entry);
    } else {
      this.entries[this.head] = entry;
      this.head = (this.head + 1) % MAX_RECORDED_ENTRIES;
    }

    if (this.verbose >= level) {
      if (globalHandler) {
        globalHandler(entry);
      } else {
        console.log(`[${this.prefix}] ${message}`);
      }
    }
  }

  /**
   * Get the recorded log entries in chronological order (regardless of verbosity).
   *
   * Only the most recent 10,000 entries are kept. The returned array is a copy.
   */
  getEntries(): readonly LogEntry[] {
    if (this.head === 0) return this.entries.slice();
    return [...this.entries.slice(this.head), ...this.entries.slice(0, this.head)];
  }

  /** Clear all recorded entries. */
  clear(): void {
    this.entries.length = 0;
    this.head = 0;
  }

  /** The configured verbosity level. */
  get level(): VerboseLevel {
    return this.verbose;
  }
}
