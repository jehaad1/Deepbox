/**
 * Internal download helpers shared by the remote dataset loaders: timeout and cancellation
 * signals, a bounded body reader and an optional SHA-256 integrity check.
 *
 * @module datasets/_download
 * @internal
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DataValidationError, DeepboxError, InvalidParameterError } from "../core/errors";

/** Default fetch timeout for a small file, in milliseconds. */
export const DEFAULT_TIMEOUT_MS = 30_000;
/** Default fetch timeout for a large archive, in milliseconds. */
export const DEFAULT_ARCHIVE_TIMEOUT_MS = 600_000;
/** Default cap on the bytes downloaded for one archive (512 MB). */
export const DEFAULT_ARCHIVE_MAX_BYTES = 512 * 1024 * 1024;
/** Default cap on the bytes a gzip stream may expand to (1 GB). */
export const DEFAULT_MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024;
/** Largest delay (about 24.8 days) that timers accept without overflowing. */
const MAX_TIMEOUT_MS = 2_147_483_647;

/**
 * Build the signal passed to `fetch`: the caller's signal, a timeout, or both.
 * `cleanup` clears the timer and detaches the listener once the request is done.
 */
export function buildFetchSignal(opts: {
  readonly timeout?: number | undefined;
  readonly abortSignal?: AbortSignal | undefined;
  readonly defaultTimeout?: number | undefined;
}): {
  signal: AbortSignal | undefined;
  cleanup: () => void;
} {
  const ms =
    opts.timeout === undefined ? (opts.defaultTimeout ?? DEFAULT_TIMEOUT_MS) : opts.timeout;
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

/**
 * Read a response body into one array, rejecting it as soon as it is known to exceed `maxBytes`
 * (from the `content-length` header or from the bytes streamed so far).
 */
export async function readResponseBytes(response: Response, maxBytes: number): Promise<Uint8Array> {
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
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  let totalBytes = 0;
  const chunks: Uint8Array[] = [];

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
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }

  const out = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/** Download options shared by the archive loaders. */
export type DownloadLimits = {
  readonly timeout?: number | undefined;
  readonly maxBytes?: number | undefined;
  readonly maxDecompressedBytes?: number | undefined;
};

/** Validate `timeout`, `maxBytes` and `maxDecompressedBytes` when they are given. */
export function validateDownloadLimits(options: DownloadLimits): void {
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
  for (const name of ["maxBytes", "maxDecompressedBytes"] as const) {
    const value = options[name];
    if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
      throw new InvalidParameterError(
        `${name} must be a positive safe integer; received ${String(value)}`,
        name,
        value
      );
    }
  }
}

const SHA256_HEX = /^[0-9a-f]{64}$/i;

/** Validate an optional SHA-256 checksum option (64 hexadecimal characters). */
export function validateChecksum(name: string, value: unknown): void {
  if (value !== undefined && (typeof value !== "string" || !SHA256_HEX.test(value))) {
    throw new InvalidParameterError(
      `${name} must be a SHA-256 digest of 64 hexadecimal characters`,
      name,
      value
    );
  }
}

/**
 * Compare the SHA-256 digest of `bytes` with `expected` (hexadecimal, any case).
 *
 * @throws {DeepboxError} If `globalThis.crypto.subtle` is unavailable.
 * @throws {DataValidationError} If the digests differ.
 */
export async function verifySha256(
  bytes: Uint8Array,
  expected: string,
  label: string
): Promise<void> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) {
    throw new DeepboxError(
      `Cannot verify the checksum of ${label}: globalThis.crypto.subtle is not available in this runtime.`
    );
  }
  const digest = new Uint8Array(await subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
  let actual = "";
  for (const byte of digest) actual += byte.toString(16).padStart(2, "0");
  if (actual !== expected.toLowerCase()) {
    throw new DataValidationError(
      `SHA-256 checksum mismatch for ${label}: expected ${expected.toLowerCase()}, received ${actual}.`
    );
  }
}

/**
 * Download `url` into memory with a timeout, a size cap and an optional SHA-256 check.
 *
 * Errors that are not {@link DeepboxError}s and not a timeout or an abort are rethrown as they
 * are, so the caller can add its own context.
 *
 * @param url - Address to fetch
 * @param label - Name of the download used in timeout, abort and checksum messages
 * @param options - Limits, signal and expected digest
 * @param httpError - Builds the message for a non-2xx response
 * @throws {DeepboxError} If the response is not ok, times out, is aborted or exceeds `maxBytes`
 * @throws {DataValidationError} If `checksum` is given and does not match
 */
export async function downloadBytes(
  url: string,
  label: string,
  options: {
    readonly timeout?: number | undefined;
    readonly defaultTimeout: number;
    readonly maxBytes?: number | undefined;
    readonly checksum?: string | undefined;
    readonly abortSignal?: AbortSignal | undefined;
  },
  httpError: (status: number, statusText: string) => string
): Promise<Uint8Array> {
  const maxBytes = options.maxBytes ?? DEFAULT_ARCHIVE_MAX_BYTES;
  const { signal, cleanup } = buildFetchSignal({
    timeout: options.timeout,
    abortSignal: options.abortSignal,
    defaultTimeout: options.defaultTimeout,
  });
  let bytes: Uint8Array;
  try {
    const response = await fetch(url, signal ? { signal } : {});
    if (!response.ok) throw new DeepboxError(httpError(response.status, response.statusText));
    bytes = await readResponseBytes(response, maxBytes);
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new DeepboxError(`Timed out fetching ${label}. Increase timeout or check network.`);
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new DeepboxError(`Fetch aborted for ${label}`);
    }
    throw err;
  } finally {
    cleanup();
  }
  if (options.checksum !== undefined) await verifySha256(bytes, options.checksum, label);
  return bytes;
}
