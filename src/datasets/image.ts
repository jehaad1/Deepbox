/**
 * Image dataset loaders (MNIST, CIFAR-10) via fetch.
 *
 * Downloads and parses standard image classification datasets from
 * public mirrors. Returns data as flattened tensors ready for ML.
 *
 * @module datasets/image
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";

/**
 * Result of an image dataset fetch.
 */
export type ImageDataset = {
  /** Flattened image data tensor of shape (nSamples, nPixels) */
  readonly data: Tensor;
  /** Integer label tensor of shape (nSamples,) */
  readonly target: Tensor;
  /** Number of classes */
  readonly nClasses: number;
  /** Class label names */
  readonly classNames: string[];
  /** Image dimensions [height, width, channels] */
  readonly imageDims: readonly [number, number, number];
  /** Dataset description */
  readonly description: string;
};

const MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const CIFAR10_BASE = "https://www.cs.toronto.edu/~kriz/";

/**
 * Fetch the MNIST handwritten digits dataset.
 *
 * Downloads the dataset from a public mirror and returns
 * 60,000 training images (28×28 grayscale) with labels 0-9.
 *
 * @param options - Configuration options
 * @returns ImageDataset with MNIST data
 *
 * @example
 * ```ts
 * import { fetchMNIST } from 'deepbox/datasets';
 *
 * const mnist = await fetchMNIST();
 * console.log(mnist.data.shape);   // [60000, 784]
 * console.log(mnist.target.shape); // [60000]
 * ```
 */
export async function fetchMNIST(
  options: {
    /** Base URL override for custom mirrors */
    readonly baseUrl?: string;
    /** Maximum number of samples to load (default: all) */
    readonly maxSamples?: number;
    /** Which split to load: 'train' (60k) or 'test' (10k) */
    readonly split?: "train" | "test";
  } = {}
): Promise<ImageDataset> {
  const base = options.baseUrl ?? MNIST_BASE;
  const split = options.split ?? "train";

  const imageFile = split === "train" ? "train-images-idx3-ubyte.gz" : "t10k-images-idx3-ubyte.gz";
  const labelFile = split === "train" ? "train-labels-idx1-ubyte.gz" : "t10k-labels-idx1-ubyte.gz";

  let imageBytes: Uint8Array;
  let labelBytes: Uint8Array;

  try {
    const [imageResp, labelResp] = await Promise.all([
      fetch(`${base}${imageFile}`),
      fetch(`${base}${labelFile}`),
    ]);

    if (!imageResp.ok) {
      throw new DeepboxError(
        `Failed to fetch MNIST images: ${imageResp.status} ${imageResp.statusText}`
      );
    }
    if (!labelResp.ok) {
      throw new DeepboxError(
        `Failed to fetch MNIST labels: ${labelResp.status} ${labelResp.statusText}`
      );
    }

    const imageBuf = await imageResp.arrayBuffer();
    const labelBuf = await labelResp.arrayBuffer();

    imageBytes = await decompressGzip(new Uint8Array(imageBuf));
    labelBytes = await decompressGzip(new Uint8Array(labelBuf));
  } catch (e) {
    if (e instanceof DeepboxError) throw e;
    throw new DeepboxError(
      `Failed to fetch MNIST dataset: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  // Parse IDX format
  // Images: magic (4) | nImages (4) | rows (4) | cols (4) | pixels...
  const nImages = readUint32BE(imageBytes, 4);
  const rows = readUint32BE(imageBytes, 8);
  const cols = readUint32BE(imageBytes, 12);
  const nPixels = rows * cols;

  const nLabels = readUint32BE(labelBytes, 4);
  if (nImages !== nLabels) {
    throw new DeepboxError(
      `MNIST image/label count mismatch: ${nImages} images vs ${nLabels} labels`
    );
  }

  const n = options.maxSamples ? Math.min(options.maxSamples, nImages) : nImages;

  const data = new Float64Array(n * nPixels);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < nPixels; p++) {
      data[i * nPixels + p] = (imageBytes[16 + i * nPixels + p] ?? 0) / 255.0;
    }
  }

  const labels = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    labels[i] = labelBytes[8 + i] ?? 0;
  }

  return {
    data: tensor(Array.from(data)).reshape([n, nPixels]),
    target: tensor(Array.from(labels)),
    nClasses: 10,
    classNames: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    imageDims: [rows, cols, 1],
    description:
      "MNIST handwritten digits dataset. " +
      `${n} images of ${rows}x${cols} grayscale pixels, 10 classes (0-9). ` +
      "Split: " +
      split,
  };
}

/**
 * Fetch the CIFAR-10 image classification dataset.
 *
 * Downloads the binary version from a public mirror and returns
 * images (32×32×3 RGB) with labels across 10 categories.
 *
 * @param options - Configuration options
 * @returns ImageDataset with CIFAR-10 data
 *
 * @example
 * ```ts
 * import { fetchCIFAR10 } from 'deepbox/datasets';
 *
 * const cifar = await fetchCIFAR10();
 * console.log(cifar.data.shape);   // [50000, 3072]
 * console.log(cifar.target.shape); // [50000]
 * ```
 */
export async function fetchCIFAR10(
  options: {
    /** Base URL override for custom mirrors */
    readonly baseUrl?: string;
    /** Maximum number of samples to load (default: all) */
    readonly maxSamples?: number;
    /** Which split to load: 'train' (50k) or 'test' (10k) */
    readonly split?: "train" | "test";
  } = {}
): Promise<ImageDataset> {
  const base = options.baseUrl ?? CIFAR10_BASE;
  const split = options.split ?? "train";

  const url = `${base}cifar-10-binary.tar.gz`;

  let tarBytes: Uint8Array;
  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new DeepboxError(`Failed to fetch CIFAR-10: ${resp.status} ${resp.statusText}`);
    }
    const buf = await resp.arrayBuffer();
    tarBytes = await decompressGzip(new Uint8Array(buf));
  } catch (e) {
    if (e instanceof DeepboxError) throw e;
    throw new DeepboxError(
      `Failed to fetch CIFAR-10 dataset: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  // Extract binary batch files from tar
  const batchFiles = extractTarFiles(tarBytes);

  const CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
  ];

  const trainBatches = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
  ];
  const testBatches = ["test_batch.bin"];
  const targetBatches = split === "train" ? trainBatches : testBatches;

  const allData: number[] = [];
  const allLabels: number[] = [];
  const nPixels = 32 * 32 * 3;
  const recordSize = 1 + nPixels; // 1 byte label + 3072 bytes image

  for (const batchName of targetBatches) {
    const file = batchFiles.find((f) => f.name.endsWith(batchName));
    if (!file) continue;

    const batchData = file.data;
    const nRecords = Math.floor(batchData.length / recordSize);

    for (let i = 0; i < nRecords; i++) {
      const offset = i * recordSize;
      allLabels.push(batchData[offset] ?? 0);
      for (let p = 0; p < nPixels; p++) {
        allData.push((batchData[offset + 1 + p] ?? 0) / 255.0);
      }
    }
  }

  const n = options.maxSamples ? Math.min(options.maxSamples, allLabels.length) : allLabels.length;

  return {
    data: tensor(allData.slice(0, n * nPixels)).reshape([n, nPixels]),
    target: tensor(allLabels.slice(0, n)),
    nClasses: 10,
    classNames: CIFAR10_CLASSES,
    imageDims: [32, 32, 3],
    description:
      "CIFAR-10 image classification dataset. " +
      `${n} images of 32x32x3 RGB pixels, 10 classes. ` +
      "Split: " +
      split,
  };
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function readUint32BE(data: Uint8Array, offset: number): number {
  return (
    (((data[offset] ?? 0) << 24) |
      ((data[offset + 1] ?? 0) << 16) |
      ((data[offset + 2] ?? 0) << 8) |
      (data[offset + 3] ?? 0)) >>>
    0
  );
}

/**
 * Decompress gzip data using DecompressionStream (available in Node 18+ and modern browsers).
 */
async function decompressGzip(data: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new DeepboxError(
      "DecompressionStream is not available. Use Node.js 18+ or a modern browser."
    );
  }

  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();

  const writePromise = writer.write(data as Uint8Array<ArrayBuffer>).then(() => writer.close());

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

/**
 * Minimal tar file extraction (ustar format).
 */
function extractTarFiles(tarData: Uint8Array): { name: string; data: Uint8Array }[] {
  const files: { name: string; data: Uint8Array }[] = [];
  let offset = 0;

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

    // Parse header
    const nameBytes = tarData.slice(offset, offset + 100);
    let name = "";
    for (let i = 0; i < nameBytes.length; i++) {
      const ch = nameBytes[i] ?? 0;
      if (ch === 0) break;
      name += String.fromCharCode(ch);
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

    if (typeFlag === 48 || typeFlag === 0) {
      // Regular file ('0' or null)
      if (fileSize > 0 && offset + fileSize <= tarData.length) {
        files.push({
          name,
          data: tarData.slice(offset, offset + fileSize),
        });
      }
    }

    // Move past file data (padded to 512 bytes)
    offset += Math.ceil(fileSize / 512) * 512;
  }

  return files;
}
