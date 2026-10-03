/**
 * Warning system for Deepbox.
 *
 * Provides structured warnings that can be filtered, silenced, or converted to errors.
 * Mirrors scikit-learn's warning categories.
 *
 * @module core/warnings
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { DeepboxError } from "./errors/base";
import { InvalidParameterError } from "./errors/invalid_parameter";

/** Category of a Deepbox warning, mirroring scikit-learn's warning categories. */
export type WarningCategory =
  | "ConvergenceWarning"
  | "DataConversionWarning"
  | "UndefinedMetricWarning"
  | "FitFailedWarning"
  | "UserWarning";

/** Structured warning emitted by Deepbox components. */
export interface DeepboxWarning {
  category: WarningCategory;
  message: string;
  source?: string | undefined;
}

/**
 * Action to take when a warning is emitted.
 *
 * - `"default"` and `"always"`: emit every time
 * - `"ignore"`: drop the warning
 * - `"error"`: throw a {@link DeepboxError} instead of emitting
 * - `"once"`: emit the first occurrence of each category and message only
 */
export type WarningAction = "default" | "error" | "ignore" | "always" | "once";

const WARNING_ACTIONS: readonly WarningAction[] = ["default", "error", "ignore", "always", "once"];

type WarningHandler = (warning: DeepboxWarning) => void;

/** Filter rule for warnings. */
interface WarningFilter {
  action: WarningAction;
  category?: WarningCategory | undefined;
  /** Compiled once; never carries the stateful `g`/`y` flags. */
  messagePattern?: RegExp | undefined;
}

const filters: WarningFilter[] = [];
const seenOnce = new Set<string>();
let customHandler: WarningHandler | undefined;

/**
 * Issue a warning. Behavior depends on current filters.
 *
 * The most recently added filter that matches the category and message wins.
 * Without a matching filter the warning goes to the custom handler (see
 * {@link setWarningHandler}) or to `console.warn`.
 *
 * @param message - Warning message
 * @param category - Warning category (default: "UserWarning")
 * @param source - Optional source identifier (e.g. function name)
 * @throws {DeepboxError} If a matching filter has the `"error"` action
 */
export function warn(
  message: string,
  category: WarningCategory = "UserWarning",
  source?: string
): void {
  const warning: DeepboxWarning = { category, message, source };

  // Find matching filter (first match wins)
  let action: WarningAction = "default";
  for (const f of filters) {
    if (f.category && f.category !== category) continue;
    if (f.messagePattern && !f.messagePattern.test(message)) continue;
    action = f.action;
    break;
  }

  if (action === "ignore") return;

  if (action === "error") {
    throw new DeepboxError(`[${category}] ${message}`);
  }

  if (action === "once") {
    const key = `${category}::${message}`;
    if (seenOnce.has(key)) return;
    seenOnce.add(key);
  }

  if (customHandler) {
    customHandler(warning);
  } else {
    const prefix = source ? `[${category} from ${source}]` : `[${category}]`;
    console.warn(`${prefix} ${message}`);
  }
}

/**
 * Add a warning filter. Filters added later take precedence over earlier ones.
 *
 * @param action - What to do: "default", "error", "ignore", "always", "once"
 * @param options - Optional category and message pattern to match. A string
 *   message is treated as a regular expression source and matched anywhere in
 *   the message.
 * @throws {InvalidParameterError} If `action` is unknown or the message pattern is not a valid regular expression
 */
export function filterWarnings(
  action: WarningAction,
  options?: { category?: WarningCategory; message?: string | RegExp }
): void {
  if (!WARNING_ACTIONS.includes(action)) {
    throw new InvalidParameterError(
      `action must be one of [${WARNING_ACTIONS.join(", ")}]; received ${String(action)}`,
      "action",
      action
    );
  }
  const raw = options?.message;
  let messagePattern: RegExp | undefined;
  if (raw !== undefined) {
    try {
      messagePattern =
        typeof raw === "string"
          ? new RegExp(raw)
          : new RegExp(raw.source, raw.flags.replace(/[gy]/g, ""));
    } catch (err) {
      throw new InvalidParameterError(
        `message must be a valid regular expression; received ${String(raw)}`,
        "message",
        raw,
        { cause: err }
      );
    }
  }
  filters.unshift({
    action,
    category: options?.category,
    messagePattern,
  });
}

/** Remove all warning filters and forget which "once" warnings were already emitted. */
export function resetWarnings(): void {
  filters.length = 0;
  seenOnce.clear();
}

/** Set a custom warning handler (replaces console.warn). Pass undefined to restore console.warn. */
export function setWarningHandler(handler: WarningHandler | undefined): void {
  customHandler = handler;
}

/**
 * Run `fn` and return the warnings that reached the handler while it ran.
 *
 * Filters still apply: ignored warnings are not collected and `"error"`
 * filters still throw. The previous handler is restored afterwards, also when
 * `fn` throws. `fn` must be synchronous; warnings emitted after it returns are
 * not collected.
 *
 * @param fn - Synchronous function to run
 * @returns Collected warnings in emission order
 */
export function catchWarnings(fn: () => void): DeepboxWarning[] {
  const collected: DeepboxWarning[] = [];
  const prev = customHandler;
  customHandler = (w) => collected.push(w);
  try {
    fn();
  } finally {
    customHandler = prev;
  }
  return collected;
}
