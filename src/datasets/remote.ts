/**
 * Remote dataset fetching utilities.
 *
 * Provides `fetchOpenML`-style remote dataset loading over HTTP.
 * Uses the runtime `fetch` API (available in Node 18+ and all
 * modern browsers).
 *
 * @module datasets/remote
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";

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
  /** Column separator. Defaults to `","`. */
  readonly separator?: string;
  /** Maximum bytes to download (streaming limit). Rejects oversized responses. */
  readonly maxBytes?: number;
  /** Fetch timeout in milliseconds. */
  readonly timeout?: number;
  /** AbortSignal for cancellation. Takes precedence over timeout. */
  readonly abortSignal?: AbortSignal;
};

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_BYTES = 100 * 1024 * 1024; // 100 MB safety cap

function createAbortController(
  timeout?: number,
  signal?: AbortSignal
): { controller: AbortController; timer: ReturnType<typeof setTimeout> } | undefined {
  if (signal) return undefined; // caller's signal handles cancellation
  // A timeout of 0 disables the timeout; undefined falls back to the default.
  // (The old `timeout === undefined && !timeout` short-circuit returned before
  // DEFAULT_TIMEOUT_MS was ever applied, so no timeout was ever set.)
  const ms = timeout === undefined ? DEFAULT_TIMEOUT_MS : timeout;
  if (ms <= 0) return undefined;
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new DOMException("Fetch timed out", "TimeoutError")),
    ms
  );
  return { controller, timer };
}

function buildFetchOptions(opts: FetchCSVDatasetOptions): {
  fetchOpts: { signal?: AbortSignal };
  timer?: ReturnType<typeof setTimeout>;
} {
  if (opts.abortSignal) return { fetchOpts: { signal: opts.abortSignal } };
  const c = createAbortController(opts.timeout);
  return c ? { fetchOpts: { signal: c.controller.signal }, timer: c.timer } : { fetchOpts: {} };
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
          reader.cancel();
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

/**
 * Fetch a CSV dataset from a remote URL and parse it into tensors.
 *
 * @param options - Fetch configuration
 * @returns Parsed remote dataset
 * @throws {DeepboxError} If the fetch fails or the CSV is malformed
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

  if (maxBytes !== undefined) {
    if (!Number.isFinite(maxBytes) || maxBytes <= 0 || !Number.isSafeInteger(maxBytes)) {
      throw new InvalidParameterError(
        "maxBytes must be a positive safe integer",
        "maxBytes",
        maxBytes
      );
    }
  }

  const { fetchOpts, timer } = buildFetchOptions(options);

  let text: string;
  try {
    const response = await fetch(url, fetchOpts);
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
    // Clear the timeout timer so it doesn't keep the event loop alive after a
    // fast response.
    if (timer !== undefined) clearTimeout(timer);
  }

  return parseCSV(text, { ...options, header, separator });
}

/**
 * Split a CSV line into fields, respecting quoted fields.
 *
 * Handles double-quote escaping within quoted fields.
 */
function splitCSVLine(line: string, separator: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === separator) {
        fields.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
  }
  fields.push(current.trim());
  return fields;
}

/**
 * Parse a CSV string into a RemoteDataset.
 *
 * Supports quoted fields and double-quote escaping. Requires
 * all data cells to be numeric.
 *
 * Exported for testing and for users who already have CSV text.
 */
export function parseCSV(
  text: string,
  options: {
    readonly targetColumn?: number;
    readonly header?: boolean;
    readonly separator?: string;
  } = {}
): RemoteDataset {
  const { header = true, separator = "," } = options;

  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    throw new DeepboxError("CSV is empty");
  }

  let featureNames: string[] = [];
  let dataStart = 0;

  if (header) {
    const headerLine = lines[0];
    if (!headerLine) throw new DeepboxError("CSV header line is empty");
    featureNames = splitCSVLine(headerLine, separator);
    dataStart = 1;
  }

  const rows: number[][] = [];
  for (let i = dataStart; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const cells = splitCSVLine(line, separator);
    const numericRow: number[] = [];
    for (const cell of cells) {
      const val = Number(cell);
      if (!Number.isFinite(val)) {
        throw new DeepboxError(
          `Non-numeric value "${cell}" found at row ${i + 1}. Only numeric CSV datasets are supported.`
        );
      }
      numericRow.push(val);
    }
    rows.push(numericRow);
  }

  if (rows.length === 0) {
    throw new DeepboxError("CSV contains no data rows");
  }

  const nCols = rows[0]!.length;
  const targetCol = options.targetColumn ?? nCols - 1;

  if (targetCol < 0 || targetCol >= nCols) {
    throw new InvalidParameterError(
      `targetColumn must be in [0, ${nCols - 1}]`,
      "targetColumn",
      targetCol
    );
  }

  if (featureNames.length === 0) {
    for (let c = 0; c < nCols; c++) featureNames.push(`feature_${c}`);
  }

  const targetName = featureNames[targetCol] ?? `feature_${targetCol}`;
  const dataFeatureNames = featureNames.filter((_, i) => i !== targetCol);

  const dataRows: number[][] = [];
  const targetValues: number[] = [];

  for (const row of rows) {
    const dataRow: number[] = [];
    for (let c = 0; c < nCols; c++) {
      if (c === targetCol) {
        targetValues.push(row[c] ?? 0);
      } else {
        dataRow.push(row[c] ?? 0);
      }
    }
    dataRows.push(dataRow);
  }

  return {
    data: tensor(dataRows),
    target: tensor(targetValues),
    featureNames: dataFeatureNames,
    targetName,
    description: `Remote CSV dataset with ${rows.length} samples and ${nCols - 1} features.`,
  };
}
