/**
 * Kaggle dataset integration for Deepbox.
 *
 * Provides programmatic access to Kaggle datasets via the Kaggle API.
 * Requires a Kaggle API token (username + key) for authentication.
 *
 * The API token can be provided directly or read from the standard
 * `~/.kaggle/kaggle.json` file (Node.js only).
 *
 * @module datasets/kaggle
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { InvalidParameterError } from "../core/errors/invalid_parameter";

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
  /** API credentials. If not provided, reads from ~/.kaggle/kaggle.json. */
  readonly credentials?: KaggleCredentials;
  /** Specific file within the dataset to download. Default: all files. */
  readonly file?: string;
  /** Maximum number of bytes to download (for large datasets). */
  readonly maxBytes?: number;
  /** API base URL override (mirrors, proxies, tests). Default: the Kaggle v1 API. */
  readonly apiBaseUrl?: string;
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
  /** Total bytes downloaded. */
  readonly totalBytes: number;
};

const KAGGLE_API_BASE = "https://www.kaggle.com/api/v1";

function apiBase(options: KaggleFetchOptions): string {
  return (options.apiBaseUrl ?? KAGGLE_API_BASE).replace(/\/$/, "");
}

/**
 * Read Kaggle credentials from the standard kaggle.json file.
 *
 * In Node.js, reads from `~/.kaggle/kaggle.json`.
 * In browsers, this is not available and will return undefined.
 *
 * @returns Credentials if found, undefined otherwise
 */
export async function readKaggleCredentials(): Promise<KaggleCredentials | undefined> {
  try {
    const [{ readFile }, { join }, { homedir }] = await Promise.all([
      import("node:fs/promises"),
      import("node:path"),
      import("node:os"),
    ]);
    const kagglePath = join(homedir(), ".kaggle", "kaggle.json");
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
  // Kaggle API uses Basic auth: base64(username:key)
  const raw = `${creds.username}:${creds.key}`;
  // Use btoa if available (browser), otherwise Buffer (Node)
  let encoded: string;
  if (typeof btoa === "function") {
    encoded = btoa(raw);
  } else {
    encoded = Buffer.from(raw).toString("base64");
  }
  return `Basic ${encoded}`;
}

/**
 * Fetch metadata for a Kaggle dataset.
 *
 * @param datasetId - Dataset identifier in "owner/dataset-name" format
 * @param options - Fetch options (credentials)
 * @returns Dataset metadata
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
  const creds = options.credentials ?? (await readKaggleCredentials());
  if (!creds) {
    throw new InvalidParameterError(
      "Kaggle credentials required. Provide credentials option or create ~/.kaggle/kaggle.json",
      "credentials",
      undefined
    );
  }

  validateDatasetId(datasetId);

  const url = `${apiBase(options)}/datasets/view/${datasetId}`;
  const response = await fetch(url, {
    headers: { Authorization: authHeader(creds) },
  });

  if (!response.ok) {
    throw new InvalidParameterError(
      `Kaggle API error: ${response.status} ${response.statusText}`,
      "datasetId",
      datasetId
    );
  }

  const json = (await response.json()) as Record<string, unknown>;
  return {
    id: String(json["ref"] ?? datasetId),
    title: String(json["title"] ?? ""),
    subtitle: String(json["subtitle"] ?? ""),
    totalBytes: Number(json["totalBytes"] ?? 0),
    fileCount: Number(json["fileCount"] ?? 0),
    lastUpdated: String(json["lastUpdated"] ?? ""),
    downloadCount: Number(json["downloadCount"] ?? 0),
    usabilityRating: Number(json["usabilityRating"] ?? 0),
  };
}

/**
 * Download a Kaggle dataset or specific file.
 *
 * @param datasetId - Dataset identifier in "owner/dataset-name" format
 * @param options - Fetch options
 * @returns Download result with raw data
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
  const creds = options.credentials ?? (await readKaggleCredentials());
  if (!creds) {
    throw new InvalidParameterError(
      "Kaggle credentials required. Provide credentials option or create ~/.kaggle/kaggle.json",
      "credentials",
      undefined
    );
  }

  validateDatasetId(datasetId);

  let url: string;
  let filename: string;

  if (options.file) {
    url = `${apiBase(options)}/datasets/download/${datasetId}/${encodeURIComponent(options.file)}`;
    filename = options.file;
  } else {
    url = `${apiBase(options)}/datasets/download/${datasetId}`;
    filename = `${datasetId.replace("/", "_")}.zip`;
  }

  const response = await fetch(url, {
    headers: { Authorization: authHeader(creds) },
  });

  if (!response.ok) {
    throw new InvalidParameterError(
      `Kaggle download error: ${response.status} ${response.statusText}`,
      "datasetId",
      datasetId
    );
  }

  const buffer = await response.arrayBuffer();
  const data = new Uint8Array(buffer);

  if (options.maxBytes !== undefined && data.length > options.maxBytes) {
    return {
      data: data.slice(0, options.maxBytes),
      filename,
      contentType: response.headers.get("content-type") ?? "application/octet-stream",
      totalBytes: options.maxBytes,
    };
  }

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
 */
export async function listKaggleFiles(
  datasetId: string,
  options: KaggleFetchOptions = {}
): Promise<{ name: string; totalBytes: number }[]> {
  const creds = options.credentials ?? (await readKaggleCredentials());
  if (!creds) {
    throw new InvalidParameterError(
      "Kaggle credentials required. Provide credentials option or create ~/.kaggle/kaggle.json",
      "credentials",
      undefined
    );
  }

  validateDatasetId(datasetId);

  const url = `${apiBase(options)}/datasets/list/${datasetId}/files`;
  const response = await fetch(url, {
    headers: { Authorization: authHeader(creds) },
  });

  if (!response.ok) {
    throw new InvalidParameterError(
      `Kaggle API error: ${response.status} ${response.statusText}`,
      "datasetId",
      datasetId
    );
  }

  const json = (await response.json()) as Record<string, unknown>;
  const files = (json["datasetFiles"] ?? []) as Record<string, unknown>[];
  return files.map((f) => ({
    name: String(f["name"] ?? ""),
    totalBytes: Number(f["totalBytes"] ?? 0),
  }));
}

/**
 * Search Kaggle datasets.
 *
 * @param query - Search query string
 * @param options - Fetch options (credentials)
 * @returns Array of dataset metadata
 */
export async function searchKaggleDatasets(
  query: string,
  options: KaggleFetchOptions = {}
): Promise<KaggleDatasetInfo[]> {
  const creds = options.credentials ?? (await readKaggleCredentials());
  if (!creds) {
    throw new InvalidParameterError("Kaggle credentials required", "credentials", undefined);
  }

  const url = `${apiBase(options)}/datasets/list?search=${encodeURIComponent(query)}`;
  const response = await fetch(url, {
    headers: { Authorization: authHeader(creds) },
  });

  if (!response.ok) {
    throw new InvalidParameterError(
      `Kaggle API error: ${response.status} ${response.statusText}`,
      "query",
      query
    );
  }

  const json = (await response.json()) as Record<string, unknown>[];
  return json.map((d) => ({
    id: String(d["ref"] ?? ""),
    title: String(d["title"] ?? ""),
    subtitle: String(d["subtitle"] ?? ""),
    totalBytes: Number(d["totalBytes"] ?? 0),
    fileCount: Number(d["fileCount"] ?? 0),
    lastUpdated: String(d["lastUpdated"] ?? ""),
    downloadCount: Number(d["downloadCount"] ?? 0),
    usabilityRating: Number(d["usabilityRating"] ?? 0),
  }));
}

function validateDatasetId(id: string): void {
  if (!id.includes("/") || id.split("/").length !== 2) {
    throw new InvalidParameterError(
      'Dataset ID must be in "owner/dataset-name" format',
      "datasetId",
      id
    );
  }
}
