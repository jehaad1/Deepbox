/**
 * Internal archive helpers shared by the dataset loaders: gzip decompression and a small tar
 * reader.
 *
 * @module datasets/_archive
 * @internal
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError } from "../core/errors";

/**
 * Decompress gzip data using DecompressionStream (available in Node 18+ and modern browsers).
 */
export async function decompressGzip(data: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new DeepboxError(
      "DecompressionStream is not available. Use Node.js 18+ or a modern browser."
    );
  }

  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();

  const writePromise = writer.write(data as Uint8Array<ArrayBuffer>).then(() => writer.close());
  // If the data is not valid gzip, the read below reports the error; without
  // this handler the failed write would surface as an unhandled rejection.
  writePromise.catch(() => undefined);

  const chunks: Uint8Array[] = [];
  let totalLength = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value as Uint8Array);
    totalLength += (value as Uint8Array).byteLength;
  }

  await writePromise;

  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function readTarString(tarData: Uint8Array, start: number, length: number): string {
  let out = "";
  for (let i = start; i < start + length; i++) {
    const ch = tarData[i] ?? 0;
    if (ch === 0) break;
    out += String.fromCharCode(ch);
  }
  return out;
}

/**
 * Minimal tar file extraction (ustar format). File contents are returned as
 * views into `tarData`, not copies.
 */
export function extractTarFiles(tarData: Uint8Array): { name: string; data: Uint8Array }[] {
  const files: { name: string; data: Uint8Array }[] = [];
  let offset = 0;
  // Path announced by a preceding GNU long-name ('L') or pax extended ('x') header.
  let pendingName: string | undefined;

  while (offset + 512 <= tarData.length) {
    // Check for end-of-archive (two zero blocks)
    let allZero = true;
    for (let i = 0; i < 512; i++) {
      if ((tarData[offset + i] ?? 0) !== 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) break;

    // Parse header. POSIX ustar archives (magic "ustar\0", version "00") may
    // split long paths into a prefix field (bytes 345-499) and the name field
    // (bytes 0-99). The old GNU format ("ustar  \0") reuses those bytes for
    // timestamps, so its prefix field must not be read.
    let name = readTarString(tarData, offset, 100);
    if (
      readTarString(tarData, offset + 257, 6) === "ustar" &&
      readTarString(tarData, offset + 263, 2) === "00"
    ) {
      const prefix = readTarString(tarData, offset + 345, 155);
      if (prefix.length > 0) name = `${prefix}/${name}`;
    }
    if (pendingName !== undefined) {
      name = pendingName;
      pendingName = undefined;
    }

    // File size (octal, bytes 124-135)
    let sizeStr = "";
    for (let i = 124; i < 136; i++) {
      const ch = tarData[offset + i] ?? 0;
      if (ch === 0 || ch === 32) break;
      sizeStr += String.fromCharCode(ch);
    }
    const fileSize = parseInt(sizeStr, 8) || 0;

    // Type flag (byte 156)
    const typeFlag = tarData[offset + 156] ?? 0;

    offset += 512; // Move past header

    if (typeFlag === 76 && offset + fileSize <= tarData.length) {
      // GNU long name: the data block holds the path of the next entry.
      pendingName = readTarString(tarData, offset, fileSize);
    } else if (typeFlag === 120 && offset + fileSize <= tarData.length) {
      // pax extended header: "<length> path=<value>\n" records.
      const text = readTarString(tarData, offset, fileSize);
      const match = /(?:^|\n)\d+ path=([^\n]*)/.exec(text);
      if (match?.[1] !== undefined) pendingName = match[1];
    } else if (typeFlag === 48 || typeFlag === 0) {
      // Regular file ('0' or null)
      if (fileSize > 0 && offset + fileSize <= tarData.length) {
        files.push({
          name,
          data: tarData.subarray(offset, offset + fileSize),
        });
      }
    }

    // Move past file data (padded to 512 bytes)
    offset += Math.ceil(fileSize / 512) * 512;
  }

  return files;
}
