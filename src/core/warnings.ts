/**
 * Warning system for Deepbox.
 *
 * Provides structured warnings that can be filtered, silenced, or converted to errors.
 * Mirrors scikit-learn's warning categories.
 *
 * @module core/warnings
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

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

/** Action to take when a warning is emitted. */
export type WarningAction = "default" | "error" | "ignore" | "always" | "once";

type WarningHandler = (warning: DeepboxWarning) => void;

/** Filter rule for warnings. */
interface WarningFilter {
  action: WarningAction;
  category?: WarningCategory | undefined;
  messagePattern?: string | RegExp | undefined;
}

const filters: WarningFilter[] = [];
const seenOnce = new Set<string>();
let customHandler: WarningHandler | undefined;

/**
 * Issue a warning. Behavior depends on current filters.
 *
 * @param message - Warning message
 * @param category - Warning category (default: "UserWarning")
 * @param source - Optional source identifier (e.g. function name)
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
    if (f.messagePattern) {
      const pat =
        typeof f.messagePattern === "string" ? new RegExp(f.messagePattern) : f.messagePattern;
      if (!pat.test(message)) continue;
    }
    action = f.action;
    break;
  }

  if (action === "ignore") return;

  if (action === "error") {
    throw new Error(`[${category}] ${message}`);
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
 * Add a warning filter.
 *
 * @param action - What to do: "default", "error", "ignore", "always", "once"
 * @param options - Optional category and message pattern to match
 */
export function filterWarnings(
  action: WarningAction,
  options?: { category?: WarningCategory; message?: string | RegExp }
): void {
  filters.unshift({
    action,
    category: options?.category,
    messagePattern: options?.message,
  });
}

/** Remove all warning filters. */
export function resetWarnings(): void {
  filters.length = 0;
  seenOnce.clear();
}

/** Set a custom warning handler (replaces console.warn). */
export function setWarningHandler(handler: WarningHandler | undefined): void {
  customHandler = handler;
}

/** Get warnings captured by a collector. Useful for testing. */
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
