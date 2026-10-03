/**
 * Image dataset loaders (MNIST, CIFAR-10) via fetch.
 *
 * Downloads and parses standard image classification datasets from
 * public mirrors. Returns data as flattened tensors ready for ML.
 *
 * @module datasets/image
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { decompressGzip, extractTarFiles } from "./_archive";
import { assertPositiveInt } from "./utils";

/**
 * Result of an image dataset fetch.
 */
export type ImageDataset = {
  /** Flattened image tensor of shape `(nSamples, nPixels)`, dtype `float32`, pixel values scaled to [0, 1]. */
  readonly data: Tensor;
  /** Class index tensor of shape `(nSamples,)`, dtype `int32`. */
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

/** Options shared by {@link fetchMNIST} and {@link fetchCIFAR10}. */
export type ImageFetchOptions = {
  /** Base URL override for custom mirrors. A trailing slash is optional. */
  readonly baseUrl?: string;
  /** Maximum number of samples to load (positive integer). Default: all. */
  readonly maxSamples?: number;
  /** Which split to load: 'train' or 'test'. Default: 'train'. */
  readonly split?: "train" | "test";
  /** Abort signal forwarded to `fetch`. */
  readonly signal?: AbortSignal;
};

const MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const CIFAR10_BASE = "https://www.cs.toronto.edu/~kriz/";

const MNIST_IMAGE_MAGIC = 0x00000803;
const MNIST_LABEL_MAGIC = 0x00000801;

function resolveImageOptions(options: ImageFetchOptions): {
  base: string;
  split: "train" | "test";
  maxSamples: number | undefined;
} {
  if (options === null || typeof options !== "object") {
    throw new InvalidParameterError("options must be an object", "options", options);
  }
  const rawBase = options.baseUrl ?? "";
  if (options.baseUrl !== undefined && (typeof rawBase !== "string" || rawBase.length === 0)) {
    throw new InvalidParameterError("baseUrl must be a non-empty string", "baseUrl", rawBase);
  }
  const split = options.split ?? "train";
  if (split !== "train" && split !== "test") {
    throw new InvalidParameterError(
      `split must be "train" or "test"; received ${String(split)}`,
      "split",
      split
    );
  }
  if (options.maxSamples !== undefined) assertPositiveInt("maxSamples", options.maxSamples);
  return { base: rawBase, split, maxSamples: options.maxSamples };
}

function withTrailingSlash(url: string): string {
  return url.endsWith("/") ? url : `${url}/`;
}

/**
 * Fetch the MNIST handwritten digits dataset.
 *
 * Downloads the dataset from a public mirror and returns
 * 60,000 training images (28×28 grayscale) with labels 0-9.
 *
 * @param options - Configuration options
 * @param options.baseUrl - Mirror URL that serves the four `*-idx*-ubyte.gz` files.
 * @param options.maxSamples - Keep only the first `maxSamples` images.
 * @param options.split - `"train"` (60,000 images, default) or `"test"` (10,000 images).
 * @returns ImageDataset with MNIST data (`float32` pixels in [0, 1], `int32` labels)
 * @throws {@link InvalidParameterError} If an option is invalid.
 * @throws {@link DeepboxError} If the download fails or the IDX files are malformed or truncated.
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
export async function fetchMNIST(options: ImageFetchOptions = {}): Promise<ImageDataset> {
  const { base: rawBase, split, maxSamples } = resolveImageOptions(options);
  const base = withTrailingSlash(rawBase === "" ? MNIST_BASE : rawBase);

  const imageFile = split === "train" ? "train-images-idx3-ubyte.gz" : "t10k-images-idx3-ubyte.gz";
  const labelFile = split === "train" ? "train-labels-idx1-ubyte.gz" : "t10k-labels-idx1-ubyte.gz";
  const init: RequestInit = options.signal ? { signal: options.signal } : {};

  let imageBytes: Uint8Array;
  let labelBytes: Uint8Array;

  try {
    const [imageResp, labelResp] = await Promise.all([
      fetch(`${base}${imageFile}`, init),
      fetch(`${base}${labelFile}`, init),
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
  // Labels: magic (4) | nLabels (4) | labels...
  if (imageBytes.length < 16 || readUint32BE(imageBytes, 0) !== MNIST_IMAGE_MAGIC) {
    throw new DeepboxError("MNIST image file is not a valid IDX3 (unsigned byte) file");
  }
  if (labelBytes.length < 8 || readUint32BE(labelBytes, 0) !== MNIST_LABEL_MAGIC) {
    throw new DeepboxError("MNIST label file is not a valid IDX1 (unsigned byte) file");
  }
  const nImages = readUint32BE(imageBytes, 4);
  const rows = readUint32BE(imageBytes, 8);
  const cols = readUint32BE(imageBytes, 12);
  const nPixels = rows * cols;
  if (nPixels === 0) {
    throw new DeepboxError(`MNIST image dimensions are invalid: ${rows}x${cols}`);
  }

  const nLabels = readUint32BE(labelBytes, 4);
  if (nImages !== nLabels) {
    throw new DeepboxError(
      `MNIST image/label count mismatch: ${nImages} images vs ${nLabels} labels`
    );
  }

  const n = maxSamples === undefined ? nImages : Math.min(maxSamples, nImages);
  if (imageBytes.length < 16 + n * nPixels) {
    throw new DeepboxError(
      `MNIST image file is truncated: expected ${16 + n * nPixels} bytes, got ${imageBytes.length}`
    );
  }
  if (labelBytes.length < 8 + n) {
    throw new DeepboxError(
      `MNIST label file is truncated: expected ${8 + n} bytes, got ${labelBytes.length}`
    );
  }

  const data = new Float32Array(n * nPixels);
  const total = n * nPixels;
  for (let i = 0; i < total; i++) {
    data[i] = (imageBytes[16 + i] as number) / 255.0;
  }

  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    labels[i] = labelBytes[8 + i] as number;
  }

  return {
    data: TensorClass.fromTypedArray({
      data,
      shape: [n, nPixels],
      dtype: "float32",
      device: "cpu",
    }),
    target: TensorClass.fromTypedArray({
      data: labels,
      shape: [n],
      dtype: "int32",
      device: "cpu",
    }),
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
 * images (32×32×3 RGB) with labels across 10 categories. Each row of `data`
 * keeps the CIFAR channel-major layout (1024 red values, then 1024 green,
 * then 1024 blue).
 *
 * @param options - Configuration options
 * @param options.baseUrl - Mirror URL that serves `cifar-10-binary.tar.gz`.
 * @param options.maxSamples - Keep only the first `maxSamples` images.
 * @param options.split - `"train"` (50,000 images, default) or `"test"` (10,000 images).
 * @returns ImageDataset with CIFAR-10 data (`float32` pixels in [0, 1], `int32` labels)
 * @throws {@link InvalidParameterError} If an option is invalid.
 * @throws {@link DeepboxError} If the download fails or the archive is missing batch files or is malformed.
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
export async function fetchCIFAR10(options: ImageFetchOptions = {}): Promise<ImageDataset> {
  const { base: rawBase, split, maxSamples } = resolveImageOptions(options);
  const base = withTrailingSlash(rawBase === "" ? CIFAR10_BASE : rawBase);

  const url = `${base}cifar-10-binary.tar.gz`;

  let tarBytes: Uint8Array;
  try {
    const resp = await fetch(url, options.signal ? { signal: options.signal } : {});
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

  const nPixels = 32 * 32 * 3;
  const recordSize = 1 + nPixels; // 1 byte label + 3072 bytes image

  const batches: Uint8Array[] = [];
  let available = 0;
  for (const batchName of targetBatches) {
    const file = batchFiles.find((f) => f.name === batchName || f.name.endsWith(`/${batchName}`));
    if (!file) {
      throw new DeepboxError(`CIFAR-10 archive is missing ${batchName}`);
    }
    if (file.data.length % recordSize !== 0) {
      throw new DeepboxError(
        `CIFAR-10 batch ${batchName} has ${file.data.length} bytes, ` +
          `which is not a multiple of the ${recordSize}-byte record size`
      );
    }
    batches.push(file.data);
    available += file.data.length / recordSize;
  }

  const n = maxSamples === undefined ? available : Math.min(maxSamples, available);
  const data = new Float32Array(n * nPixels);
  const labels = new Int32Array(n);

  let row = 0;
  for (const batch of batches) {
    const nRecords = batch.length / recordSize;
    for (let i = 0; i < nRecords && row < n; i++, row++) {
      const offset = i * recordSize;
      const label = batch[offset] as number;
      if (label >= CIFAR10_CLASSES.length) {
        throw new DeepboxError(`CIFAR-10 record has invalid label ${label}`);
      }
      labels[row] = label;
      const dst = row * nPixels;
      for (let p = 0; p < nPixels; p++) {
        data[dst + p] = (batch[offset + 1 + p] as number) / 255.0;
      }
    }
  }

  return {
    data: TensorClass.fromTypedArray({
      data,
      shape: [n, nPixels],
      dtype: "float32",
      device: "cpu",
    }),
    target: TensorClass.fromTypedArray({
      data: labels,
      shape: [n],
      dtype: "int32",
      device: "cpu",
    }),
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
