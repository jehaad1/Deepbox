/**
 * Remote dataset fetching utilities.
 *
 * Downloads a numeric CSV file over HTTP and parses it into tensors.
 * Uses the runtime `fetch` API (available in Node 18+ and all
 * modern browsers).
 *
 * @module datasets/remote
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DataValidationError, DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import { assertBoolean, defaultFloatDtype } from "./utils";

/**
 * Result of a remote dataset fetch.
 */
export type RemoteDataset = {
  readonly data: Tensor;
  readonly target: Tensor;
  readonly featureNames: string[];
  readonly targetName: string;
  readonly description: string;
};

/**
 * Options for {@link fetchCSVDataset}.
 */
export type FetchCSVDatasetOptions = {
  /** URL of the CSV file to fetch. */
  readonly url: string;
  /** 0-based column index to use as the target variable. Defaults to last column. */
  readonly targetColumn?: number;
  /** Whether the first row is a header. Defaults to `true`. */
  readonly header?: boolean;
  /** Single-character column separator. Defaults to `","`. */
  readonly separator?: string;
  /** Maximum bytes to download (streaming limit). Rejects oversized responses. Defaults to 100 MB. */
  readonly maxBytes?: number;
  /** Fetch timeout in milliseconds. Defaults to 30 000; `0` disables the timeout. */
  readonly timeout?: number;
  /** AbortSignal for cancellation. When combined with `timeout`, whichever fires first aborts the fetch. */
  readonly abortSignal?: AbortSignal;
};

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_BYTES = 100 * 1024 * 1024; // 100 MB safety cap
const MAX_TIMEOUT_MS = 2_147_483_647;

/**
 * Build the signal passed to `fetch`: the caller's signal, a timeout, or both.
 * `cleanup` clears the timer and detaches the listener once the request is done.
 */
function buildFetchSignal(opts: FetchCSVDatasetOptions): {
  signal: AbortSignal | undefined;
  cleanup: () => void;
} {
  const ms = opts.timeout === undefined ? DEFAULT_TIMEOUT_MS : opts.timeout;
  const userSignal = opts.abortSignal;
  if (ms <= 0) return { signal: userSignal, cleanup: () => {} };

  const controller = new AbortController();
  // Timers overflow above 2^31 - 1 ms and would fire immediately, so clamp the delay.
  const timer = setTimeout(
    () => controller.abort(new DOMException("Fetch timed out", "TimeoutError")),
    Math.min(ms, MAX_TIMEOUT_MS)
  );
  let onAbort: (() => void) | undefined;
  if (userSignal) {
    if (userSignal.aborted) {
      controller.abort(userSignal.reason);
    } else {
      onAbort = () => controller.abort(userSignal.reason);
      userSignal.addEventListener("abort", onAbort, { once: true });
    }
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      // Clear the timer so it does not keep the event loop alive after a fast response.
      clearTimeout(timer);
      if (userSignal && onAbort) userSignal.removeEventListener("abort", onAbort);
    },
  };
}

async function streamResponseText(response: Response, maxBytes: number): Promise<string> {
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null) {
    const parsed = Number(contentLength);
    if (Number.isFinite(parsed) && parsed > maxBytes) {
      throw new DeepboxError(
        `Response content-length ${parsed} exceeds maxBytes limit ${maxBytes}. ` +
          "Increase maxBytes or use a smaller dataset."
      );
    }
  }

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength > maxBytes) {
      throw new DeepboxError(
        `Response body ${buffer.byteLength} bytes exceeds maxBytes limit ${maxBytes}.`
      );
    }
    return new TextDecoder().decode(buffer);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let totalBytes = 0;
  const chunks: string[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        totalBytes += value.byteLength;
        if (totalBytes > maxBytes) {
          await reader.cancel().catch(() => undefined);
          throw new DeepboxError(
            `Response body exceeds maxBytes limit ${maxBytes}. Streamed ${totalBytes} bytes so far.`
          );
        }
        chunks.push(decoder.decode(value, { stream: true }));
      }
    }
  } finally {
    reader.releaseLock();
  }

  chunks.push(decoder.decode());
  return chunks.join("");
}

function validateCSVOptions(options: {
  readonly targetColumn?: number | undefined;
  readonly header?: boolean | undefined;
  readonly separator?: string | undefined;
}): void {
  if (options.header !== undefined) assertBoolean("header", options.header);
  const { separator } = options;
  if (separator !== undefined) {
    if (
      typeof separator !== "string" ||
      separator.length !== 1 ||
      separator === '"' ||
      separator === "\n" ||
      separator === "\r"
    ) {
      throw new InvalidParameterError(
        "separator must be a single character other than a double quote or a line break",
        "separator",
        separator
      );
    }
  }
  const { targetColumn } = options;
  if (targetColumn !== undefined && !Number.isSafeInteger(targetColumn)) {
    throw new InvalidParameterError(
      `targetColumn must be an integer; received ${targetColumn}`,
      "targetColumn",
      targetColumn
    );
  }
}

/**
 * Fetch a CSV dataset from a remote URL and parse it into tensors.
 *
 * The file must be numeric: every cell has to be a finite decimal number.
 * See {@link parseCSV} for the accepted format.
 *
 * @param options - Fetch configuration
 * @returns Parsed remote dataset
 * @throws {InvalidParameterError} If an option is invalid
 * @throws {DeepboxError} If the request fails, times out, is aborted or exceeds `maxBytes`
 * @throws {DataValidationError} If the CSV is malformed or contains non-numeric or missing values
 *
 * @example
 * ```ts
 * import { fetchCSVDataset } from 'deepbox/datasets';
 *
 * const ds = await fetchCSVDataset({
 *   url: 'https://example.com/iris.csv',
 *   targetColumn: 4,
 *   timeout: 5000,
 *   maxBytes: 10 * 1024 * 1024, // 10 MB
 * });
 * console.log(ds.data.shape, ds.target.shape);
 * ```
 */
export async function fetchCSVDataset(options: FetchCSVDatasetOptions): Promise<RemoteDataset> {
  const { url, header = true, separator = "," } = options;
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;

  if (!url || typeof url !== "string") {
    throw new InvalidParameterError("url must be a non-empty string", "url", url);
  }
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
    throw new InvalidParameterError(
      "maxBytes must be a positive safe integer",
      "maxBytes",
      maxBytes
    );
  }
  if (
    options.timeout !== undefined &&
    !(Number.isFinite(options.timeout) && options.timeout >= 0)
  ) {
    throw new InvalidParameterError(
      "timeout must be a non-negative number of milliseconds",
      "timeout",
      options.timeout
    );
  }
  validateCSVOptions(options);

  const { signal, cleanup } = buildFetchSignal(options);

  let text: string;
  try {
    const response = await fetch(url, signal ? { signal } : {});
    if (!response.ok) {
      throw new DeepboxError(
        `Failed to fetch dataset from ${url}: HTTP ${response.status} ${response.statusText}`
      );
    }
    text = await streamResponseText(response, maxBytes);
  } catch (err) {
    if (err instanceof DeepboxError) throw err;
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new DeepboxError(
        `Timed out fetching dataset from ${url}. Increase timeout or check network.`
      );
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new DeepboxError(`Fetch aborted for ${url}`);
    }
    throw new DeepboxError(`Failed to fetch dataset from ${url}: ${String(err)}`);
  } finally {
    cleanup();
  }

  return parseCSV(text, { ...options, header, separator });
}

type CSVRecord = { readonly fields: string[]; readonly line: number };

/**
 * Split CSV text into records of trimmed fields.
 *
 * Quoted fields may contain the separator, line breaks and doubled quotes
 * (`""`). Blank lines are skipped. `line` is the 1-based line on which each
 * record starts.
 */
function parseRecords(text: string, separator: string): CSVRecord[] {
  const records: CSVRecord[] = [];
  let fields: string[] = [];
  let current = "";
  let inQuotes = false;
  let line = 1;
  let recordLine = 1;

  const endRecord = (): void => {
    fields.push(current.trim());
    if (!(fields.length === 1 && fields[0] === "")) records.push({ fields, line: recordLine });
    fields = [];
    current = "";
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i] as string;
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        if (ch === "\n") line++;
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === separator) {
      fields.push(current.trim());
      current = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      endRecord();
      line++;
      recordLine = line;
    } else {
      current += ch;
    }
  }
  if (inQuotes) {
    throw new DataValidationError(`Unterminated quoted field starting at line ${recordLine}`);
  }
  endRecord();
  return records;
}

const DECIMAL_NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

function parseCell(cell: string, line: number, column: number): number {
  if (cell === "") {
    throw new DataValidationError(
      `Missing value at line ${line}, column ${column + 1}. ` +
        "Remove or impute missing values before loading."
    );
  }
  const val = DECIMAL_NUMBER.test(cell) ? Number(cell) : Number.NaN;
  if (Number.isNaN(val)) {
    throw new DataValidationError(
      `Non-numeric value "${cell}" found at line ${line}, column ${column + 1}. ` +
        "Only numeric CSV datasets are supported."
    );
  }
  if (!Number.isFinite(val)) {
    throw new DataValidationError(
      `Value "${cell}" at line ${line}, column ${column + 1} is outside the finite number range.`
    );
  }
  return val;
}

/**
 * Parse a CSV string into a RemoteDataset.
 *
 * Supports quoted fields (including embedded separators, line breaks and
 * doubled quotes) and `\n`, `\r\n` or `\r` line endings. Blank lines are
 * skipped. Every data cell must be a finite decimal number (`12`, `-0.5`,
 * `1e-3`); empty cells, `NaN`, `Infinity`, hexadecimal and text are rejected.
 * Every row must have as many cells as the header (or as the first row when
 * `header` is `false`).
 *
 * Exported for testing and for users who already have CSV text.
 *
 * @param text - CSV text
 * @param options.targetColumn - 0-based target column. Defaults to the last column.
 * @param options.header - Whether the first row holds column names. Defaults to `true`.
 * @param options.separator - Single-character separator. Defaults to `","`.
 * @returns The features, the target column and the column names
 * @throws {InvalidParameterError} If an option is invalid or `targetColumn` is out of range
 * @throws {DataValidationError} If the text is empty, a row has the wrong number of cells,
 *   or a cell is missing or non-numeric
 */
export function parseCSV(
  text: string,
  options: {
    readonly targetColumn?: number;
    readonly header?: boolean;
    readonly separator?: string;
  } = {}
): RemoteDataset {
  if (typeof text !== "string") {
    throw new InvalidParameterError("text must be a string", "text", text);
  }
  validateCSVOptions(options);
  const { header = true, separator = "," } = options;

  const records = parseRecords(text, separator);
  if (records.length === 0) {
    throw new DataValidationError("CSV is empty");
  }

  let columnNames: string[] = [];
  let firstData = 0;
  if (header) {
    columnNames = (records[0] as CSVRecord).fields;
    firstData = 1;
  }
  if (records.length === firstData) {
    throw new DataValidationError("CSV contains no data rows");
  }

  const nCols = header ? columnNames.length : (records[0] as CSVRecord).fields.length;
  const targetCol = options.targetColumn ?? nCols - 1;
  if (targetCol < 0 || targetCol >= nCols) {
    throw new InvalidParameterError(
      `targetColumn must be in [0, ${nCols - 1}]`,
      "targetColumn",
      targetCol
    );
  }

  const names: string[] = [];
  for (let c = 0; c < nCols; c++) {
    const name = columnNames[c];
    names.push(name === undefined || name === "" ? `feature_${c}` : name);
  }

  const nRows = records.length - firstData;
  const dataRows: number[][] = new Array(nRows);
  const targetValues: number[] = new Array(nRows);
  for (let r = 0; r < nRows; r++) {
    const { fields, line } = records[firstData + r] as CSVRecord;
    if (fields.length !== nCols) {
      throw new DataValidationError(`Line ${line} has ${fields.length} cells; expected ${nCols}.`);
    }
    const row: number[] = [];
    for (let c = 0; c < nCols; c++) {
      const val = parseCell(fields[c] as string, line, c);
      if (c === targetCol) targetValues[r] = val;
      else row.push(val);
    }
    dataRows[r] = row;
  }

  return {
    data: tensor(dataRows, { dtype: defaultFloatDtype() }),
    target: tensor(targetValues, { dtype: defaultFloatDtype() }),
    featureNames: names.filter((_, i) => i !== targetCol),
    targetName: names[targetCol] as string,
    description: `Remote CSV dataset with ${nRows} samples and ${nCols - 1} features.`,
  };
}
