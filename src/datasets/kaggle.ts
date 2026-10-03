/**
 * Kaggle dataset integration for Deepbox.
 *
 * Provides programmatic access to Kaggle datasets via the Kaggle API.
 * Requires a Kaggle API token (username + key) for authentication.
 *
 * The API token can be provided directly, read from the `KAGGLE_USERNAME` and
 * `KAGGLE_KEY` environment variables, or read from the standard
 * `~/.kaggle/kaggle.json` file (Node.js only).
 *
 * @module datasets/kaggle
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError, InvalidParameterError } from "../core/errors";

/**
 * Kaggle API credentials.
 */
export type KaggleCredentials = {
  /** Kaggle username. */
  readonly username: string;
  /** Kaggle API key. */
  readonly key: string;
};

/**
 * Options for fetching a Kaggle dataset.
 */
export type KaggleFetchOptions = {
  /** API credentials. If not provided, see {@link readKaggleCredentials}. */
  readonly credentials?: KaggleCredentials;
  /** Specific file within the dataset to download. Default: all files. */
  readonly file?: string;
  /**
   * Maximum number of bytes to keep (non-negative integer). Only
   * {@link fetchKaggleDataset} uses it: the download stops once this many
   * bytes have been read, so `totalBytes` is at most `maxBytes`.
   */
  readonly maxBytes?: number;
  /** API base URL override (mirrors, proxies, tests). Default: the Kaggle v1 API. */
  readonly apiBaseUrl?: string;
  /** Abort signal forwarded to `fetch`. */
  readonly signal?: AbortSignal;
};

/**
 * Metadata about a Kaggle dataset.
 */
export type KaggleDatasetInfo = {
  readonly id: string;
  readonly title: string;
  readonly subtitle: string;
  readonly totalBytes: number;
  readonly fileCount: number;
  readonly lastUpdated: string;
  readonly downloadCount: number;
  readonly usabilityRating: number;
};

/**
 * Result of a Kaggle dataset download.
 */
export type KaggleDownloadResult = {
  /** Raw file data as Uint8Array (may be a ZIP archive). */
  readonly data: Uint8Array;
  /** The filename. */
  readonly filename: string;
  /** Content type from the response. */
  readonly contentType: string;
  /** Number of bytes in `data` (at most `maxBytes` when that option is set). */
  readonly totalBytes: number;
};

const KAGGLE_API_BASE = "https://www.kaggle.com/api/v1";

function apiBase(options: KaggleFetchOptions): string {
  return (options.apiBaseUrl ?? KAGGLE_API_BASE).replace(/\/+$/, "");
}

function requestInit(creds: KaggleCredentials, options: KaggleFetchOptions): RequestInit {
  const init: RequestInit = { headers: { Authorization: authHeader(creds) } };
  if (options.signal) init.signal = options.signal;
  return init;
}

/**
 * Read Kaggle credentials.
 *
 * Looks, in order, at the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment
 * variables, then at `kaggle.json` in `$KAGGLE_CONFIG_DIR` (if set) or in
 * `~/.kaggle`. File access needs Node.js; in browsers the result is
 * `undefined`.
 *
 * @returns Credentials if found, undefined otherwise
 */
export async function readKaggleCredentials(): Promise<KaggleCredentials | undefined> {
  const env = typeof process !== "undefined" ? process.env : undefined;
  const envUser = env?.["KAGGLE_USERNAME"];
  const envKey = env?.["KAGGLE_KEY"];
  if (envUser && envKey) {
    return { username: envUser, key: envKey };
  }

  try {
    const [{ readFile }, { join }, { homedir }] = await Promise.all([
      import("node:fs/promises"),
      import("node:path"),
      import("node:os"),
    ]);
    const configDir = env?.["KAGGLE_CONFIG_DIR"] || join(homedir(), ".kaggle");
    const kagglePath = join(configDir, "kaggle.json");
    const content = await readFile(kagglePath, "utf-8");
    const parsed = JSON.parse(content) as Record<string, unknown>;
    if (typeof parsed["username"] === "string" && typeof parsed["key"] === "string") {
      return { username: parsed["username"], key: parsed["key"] };
    }
    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Get Basic auth header value from credentials.
 */
function authHeader(creds: KaggleCredentials): string {
  // Kaggle API uses Basic auth: base64(username:key), UTF-8 encoded.
  const bytes = new TextEncoder().encode(`${creds.username}:${creds.key}`);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  const encoded = typeof btoa === "function" ? btoa(binary) : Buffer.from(bytes).toString("base64");
  return `Basic ${encoded}`;
}

async function resolveCredentials(options: KaggleFetchOptions): Promise<KaggleCredentials> {
  const creds = options.credentials ?? (await readKaggleCredentials());
  if (
    !creds ||
    typeof creds.username !== "string" ||
    typeof creds.key !== "string" ||
    creds.username.length === 0 ||
    creds.key.length === 0
  ) {
    throw new InvalidParameterError(
      "Kaggle credentials required. Provide the credentials option, set KAGGLE_USERNAME and " +
        "KAGGLE_KEY, or create ~/.kaggle/kaggle.json",
      "credentials",
      undefined
    );
  }
  return creds;
}

/**
 * Turn a failed HTTP response into an error. Client errors (4xx: unknown
 * dataset, bad credentials) are reported as invalid parameters; anything else
 * (server errors, rate limiting) is a plain {@link DeepboxError}.
 */
function httpError(
  prefix: string,
  response: Response,
  param: string,
  value: unknown
): DeepboxError {
  const message = `${prefix}: ${response.status} ${response.statusText}`;
  if (response.status >= 400 && response.status < 500 && response.status !== 429) {
    return new InvalidParameterError(message, param, value);
  }
  return new DeepboxError(message);
}

async function readJson(response: Response, what: string): Promise<unknown> {
  try {
    return await response.json();
  } catch (e) {
    throw new DeepboxError(
      `Kaggle API returned invalid JSON for ${what}: ${e instanceof Error ? e.message : String(e)}`
    );
  }
}

function toNumber(value: unknown): number {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n : 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseDatasetInfo(json: Record<string, unknown>, fallbackId: string): KaggleDatasetInfo {
  return {
    id: String(json["ref"] ?? fallbackId),
    title: String(json["title"] ?? ""),
    subtitle: String(json["subtitle"] ?? ""),
    totalBytes: toNumber(json["totalBytes"]),
    fileCount: toNumber(json["fileCount"]),
    lastUpdated: String(json["lastUpdated"] ?? ""),
    downloadCount: toNumber(json["downloadCount"]),
    usabilityRating: toNumber(json["usabilityRating"]),
  };
}

/**
 * Fetch metadata for a Kaggle dataset.
 *
 * @param datasetId - Dataset identifier in "owner/dataset-name" format
 * @param options - Fetch options (credentials)
 * @returns Dataset metadata
 * @throws {@link InvalidParameterError} If the id is malformed, credentials are missing, or Kaggle answers with a 4xx status.
 * @throws {@link DeepboxError} On other HTTP failures or an unparseable response.
 *
 * @example
 * ```ts
 * import { fetchKaggleDatasetInfo } from 'deepbox/datasets';
 *
 * const info = await fetchKaggleDatasetInfo('zillow/zecon', {
 *   credentials: { username: 'myuser', key: 'mykey' }
 * });
 * console.log(info.title, info.totalBytes);
 * ```
 */
export async function fetchKaggleDatasetInfo(
  datasetId: string,
  options: KaggleFetchOptions = {}
): Promise<KaggleDatasetInfo> {
  validateDatasetId(datasetId);
  const creds = await resolveCredentials(options);

  const url = `${apiBase(options)}/datasets/view/${datasetId}`;
  const response = await fetch(url, requestInit(creds, options));

  if (!response.ok) {
    throw httpError("Kaggle API error", response, "datasetId", datasetId);
  }

  const json = await readJson(response, "dataset info");
  if (!isRecord(json)) {
    throw new DeepboxError("Kaggle API returned an unexpected response for dataset info");
  }
  return parseDatasetInfo(json, datasetId);
}

/**
 * Read a response body, stopping once `maxBytes` bytes have been collected.
 * Falls back to buffering the whole body when streaming is unavailable.
 */
async function readBody(response: Response, maxBytes: number | undefined): Promise<Uint8Array> {
  const body = response.body;
  if (maxBytes === undefined || !body || typeof body.getReader !== "function") {
    const all = new Uint8Array(await response.arrayBuffer());
    return maxBytes !== undefined && all.length > maxBytes ? all.slice(0, maxBytes) : all;
  }

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (received < maxBytes) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
  }
  if (received >= maxBytes) {
    await reader.cancel().catch(() => undefined);
  }

  const out = new Uint8Array(Math.min(received, maxBytes));
  let offset = 0;
  for (const chunk of chunks) {
    const take = Math.min(chunk.byteLength, out.length - offset);
    if (take <= 0) break;
    out.set(chunk.subarray(0, take), offset);
    offset += take;
  }
  return out;
}

/**
 * Download a Kaggle dataset or specific file.
 *
 * Without `file` the result is the whole dataset as a ZIP archive.
 *
 * @param datasetId - Dataset identifier in "owner/dataset-name" format
 * @param options - Fetch options
 * @returns Download result with raw data
 * @throws {@link InvalidParameterError} If the id, `file` or `maxBytes` is invalid, credentials are missing, or Kaggle answers with a 4xx status.
 * @throws {@link DeepboxError} On other HTTP failures.
 *
 * @example
 * ```ts
 * import { fetchKaggleDataset } from 'deepbox/datasets';
 *
 * const result = await fetchKaggleDataset('zillow/zecon', {
 *   credentials: { username: 'myuser', key: 'mykey' },
 *   file: 'Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
 * });
 * console.log(result.totalBytes);
 * ```
 */
export async function fetchKaggleDataset(
  datasetId: string,
  options: KaggleFetchOptions = {}
): Promise<KaggleDownloadResult> {
  validateDatasetId(datasetId);
  if (
    options.maxBytes !== undefined &&
    (!Number.isSafeInteger(options.maxBytes) || options.maxBytes < 0)
  ) {
    throw new InvalidParameterError(
      `maxBytes must be a non-negative safe integer; received ${options.maxBytes}`,
      "maxBytes",
      options.maxBytes
    );
  }
  if (options.file !== undefined && typeof options.file !== "string") {
    throw new InvalidParameterError("file must be a string", "file", options.file);
  }
  if (options.file === "." || options.file === "..") {
    throw new InvalidParameterError(
      `file must be a file name, not a path segment; received "${options.file}"`,
      "file",
      options.file
    );
  }
  const creds = await resolveCredentials(options);

  let url: string;
  let filename: string;

  if (options.file) {
    url = `${apiBase(options)}/datasets/download/${datasetId}/${encodeURIComponent(options.file)}`;
    filename = options.file;
  } else {
    url = `${apiBase(options)}/datasets/download/${datasetId}`;
    filename = `${datasetId.replace("/", "_")}.zip`;
  }

  const response = await fetch(url, requestInit(creds, options));

  if (!response.ok) {
    throw httpError("Kaggle download error", response, "datasetId", datasetId);
  }

  const data = await readBody(response, options.maxBytes);
  return {
    data,
    filename,
    contentType: response.headers.get("content-type") ?? "application/octet-stream",
    totalBytes: data.length,
  };
}

/**
 * List files in a Kaggle dataset.
 *
 * @param datasetId - Dataset identifier in "owner/dataset-name" format
 * @param options - Fetch options
 * @returns Array of file names and sizes
 * @throws {@link InvalidParameterError} If the id is malformed, credentials are missing, or Kaggle answers with a 4xx status.
 * @throws {@link DeepboxError} On other HTTP failures or an unparseable response.
 */
export async function listKaggleFiles(
  datasetId: string,
  options: KaggleFetchOptions = {}
): Promise<{ name: string; totalBytes: number }[]> {
  validateDatasetId(datasetId);
  const creds = await resolveCredentials(options);

  const url = `${apiBase(options)}/datasets/list/${datasetId}/files`;
  const response = await fetch(url, requestInit(creds, options));

  if (!response.ok) {
    throw httpError("Kaggle API error", response, "datasetId", datasetId);
  }

  const json = await readJson(response, "file list");
  const files = isRecord(json) ? (json["datasetFiles"] ?? []) : undefined;
  if (!Array.isArray(files)) {
    throw new DeepboxError("Kaggle API returned an unexpected response for the file list");
  }
  return files.filter(isRecord).map((f) => ({
    name: String(f["name"] ?? ""),
    totalBytes: toNumber(f["totalBytes"]),
  }));
}

/**
 * Search Kaggle datasets.
 *
 * @param query - Search query string
 * @param options - Fetch options (credentials)
 * @returns Array of dataset metadata
 * @throws {@link InvalidParameterError} If `query` is not a string, credentials are missing, or Kaggle answers with a 4xx status.
 * @throws {@link DeepboxError} On other HTTP failures or an unparseable response.
 */
export async function searchKaggleDatasets(
  query: string,
  options: KaggleFetchOptions = {}
): Promise<KaggleDatasetInfo[]> {
  if (typeof query !== "string") {
    throw new InvalidParameterError("query must be a string", "query", query);
  }
  const creds = await resolveCredentials(options);

  const url = `${apiBase(options)}/datasets/list?search=${encodeURIComponent(query)}`;
  const response = await fetch(url, requestInit(creds, options));

  if (!response.ok) {
    throw httpError("Kaggle API error", response, "query", query);
  }

  const json = await readJson(response, "search results");
  if (!Array.isArray(json)) {
    throw new DeepboxError("Kaggle API returned an unexpected response for the search results");
  }
  return json.filter(isRecord).map((d) => parseDatasetInfo(d, ""));
}

const DATASET_SEGMENT = /^[A-Za-z0-9_.-]+$/;

function validateDatasetId(id: string): void {
  const parts = typeof id === "string" ? id.split("/") : [];
  const valid =
    parts.length === 2 && parts.every((p) => DATASET_SEGMENT.test(p) && p !== "." && p !== "..");
  if (!valid) {
    throw new InvalidParameterError(
      'Dataset ID must be in "owner/dataset-name" format',
      "datasetId",
      id
    );
  }
}
