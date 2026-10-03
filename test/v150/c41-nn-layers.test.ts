/**
 * Regression tests for the convolution, pooling, dropout and embedding layers (v1.5.0 review).
 *
 * Reference values come from PyTorch 2.12 (`torch.nn.functional` pooling, `ConvTranspose1d/2d`,
 * `Conv1d/2d/3d`, `Embedding`, `EmbeddingBag`) evaluated on the dyadic inputs built by `vals`
 * below, so every sum is exact in float32.
 */
import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import { GradTensor, tensor, zeros } from "../../src/ndarray";
import {
  AdaptiveAvgPool1d,
  AdaptiveAvgPool2d,
  AdaptiveMaxPool1d,
  AdaptiveMaxPool2d,
  AlphaDropout,
  AvgPool1d,
  AvgPool2d,
  AvgPool3d,
  Conv1d,
  Conv2d,
  Conv3d,
  ConvTranspose1d,
  ConvTranspose2d,
  Dropout,
  Dropout2d,
  Embedding,
  EmbeddingBag,
  MaxPool1d,
  MaxPool2d,
  MaxPool3d,
} from "../../src/nn";
import { setSeed } from "../../src/random";

interface Case {
  readonly shape: number[];
  readonly y: number[];
  readonly gx: number[];
  readonly gw?: number[];
  readonly gb?: number[];
}

interface BagCase {
  readonly shape: number[];
  readonly y: number[];
  readonly g: number[];
}

const CONV: Record<string, Case> = {
  ct2d: {
    shape: [1, 2, 4, 6],
    y: [
      -1.0625, -3.125, -0.625, 0.5, -0.875, -1.5, -2.0, 1.4375, -0.4375, 0.25, -0.9375, -0.875,
      -0.4375, 0.875, -0.6875, 0.375, -0.9375, -1.875, 0.1875, -2.4375, -0.5625, -0.6875, -1.3125,
      0.1875, 1.875, 0.5625, -0.75, -0.375, 0.75, 0.125, 2.8125, -0.75, -0.1875, -0.5625, -0.4375,
      0.4375, -1.875, -0.1875, -0.375, -0.4375, 1.125, 0.1875, -1.25, 0.5625, -0.25, -0.6875, 0.75,
      -0.1875,
    ],
    gx: [0.375, -0.6875, -1.25, 2.125, 3.625, -0.375],
    gw: [
      -0.75, 0.5, -1.125, 0.1875, -0.3125, 0.625, -0.3125, 0.5, -0.3125, 0.3125, 0.875, 0.125,
      -0.0625, 0.75, 0.375, 0.3125, -1.9375, 0.75,
    ],
    gb: [0.0, -0.25],
  },
  ct2d_asym: {
    shape: [2, 1, 3, 5],
    y: [
      1.0625, -3.25, -0.125, 1.75, -0.625, 1.1875, -3.0625, -0.125, 3.5625, -2.125, 1.25, 1.0625,
      -0.25, 1.3125, -1.75, -0.125, 1.75, -0.625, -1.5, 0.25, -0.125, 3.5625, -2.125, -2.1875,
      1.375, -0.25, 1.3125, -1.75, -1.1875, 0.875,
    ],
    gx: [
      0.4375, -0.875, 0.4375, 1.0625, -0.4375, -0.625, -1.375, -0.0625, 1.4375, -1.375, -1.5,
      -0.375, -0.3125, 1.3125, -1.4375, -1.4375, 1.3125, -0.3125, -0.0625, -1.75, -0.9375, 2.125,
      0.3125, -0.5625,
    ],
    gw: [1.75, -1.0, -0.4375, 1.0, -0.25, 0.0625, 1.125, -2.6875, -1.0625, 1.125, -1.0, 0.875],
  },
  ct1d: {
    shape: [1, 2, 8],
    y: [
      -0.25, -2.6875, -1.4375, 0.9375, 0.125, -4.375, -1.0625, 0.5, 0.0, -2.5625, -0.8125, 1.8125,
      1.125, -2.75, 0.3125, 1.875,
    ],
    gx: [1.3125, -0.5, -0.5, -0.5, -1.125, 0.1875, 0.3125, 0.4375],
    gw: [
      -0.5, 1.375, -0.8125, 0.4375, -0.8125, -0.8125, -0.25, -0.25, -0.3125, -0.0625, -0.3125,
      0.0625,
    ],
    gb: [-0.75, 0.0],
  },
  c3d: {
    shape: [1, 2, 4, 1, 4],
    y: [
      -3.375, 2.125, -0.5, -2.6875, -0.9375, 1.6875, -4.0625, 1.25, -1.125, -4.0625, 0.5, -2.25,
      -1.5625, -0.6875, -2.4375, -0.625, -2.0625, -0.0625, 0.125, -0.1875, -2.3125, 1.8125, 3.0625,
      -1.25, 0.3125, 3.0625, -1.875, -0.5625, 1.75, -2.375, 0.0625, -0.4375,
    ],
    gx: [
      1.8125, -1.25, 1.8125, 0.75, -0.4375, -1.1875, 0.5, 0.375, 0.6875, 0.0, 0.0, 0.0, -0.8125,
      -0.8125, -0.375, 1.25, -1.25, 1.0625, 0.875, 0.75, -1.5625, 0.0, 0.0, 0.0, -1.25, 1.8125,
      -0.375, -0.4375, -1.1875, -0.1875, 0.375, 0.6875, -1.625, 0.0, 0.0, 0.0, -1.3125, 2.0, 0.0625,
      0.0625, 2.0, -1.3125, -1.0, -0.4375, 0.5625, 0.0, 0.0, 0.0, 1.0, -1.8125, -0.6875, -1.8125,
      1.0, 0.75, -1.375, 1.375, -0.25, 0.0, 0.0, 0.0, 2.0, 0.0625, 0.75, 2.0, -1.3125, -0.6875,
      -0.4375, 0.5625, 1.125, 0.0, 0.0, 0.0,
    ],
    gw: [
      1.1875, -1.3125, -0.0625, 0.125, 0.0625, 0.1875, -1.375, 2.25, -1.5, 0.0, -1.625, -0.1875,
      -0.0625, 0.125, 0.0625, 0.1875, 0.1875, 0.25, -1.5, 0.0, -1.625, -0.1875, -1.75, -0.375, 2.25,
      -1.125, 0.0, -0.25, -0.1875, -0.0625, 0.5625, -0.1875, -0.125, 1.8125, -0.125, 1.75, 0.0,
      -0.25, -0.1875, -0.0625, -0.375, 0.125, -0.125, 1.8125, -0.125, 1.75, -0.125, 1.6875,
    ],
    gb: [-0.75, 0.5],
  },
  c3d_nb: {
    shape: [1, 2, 2, 2, 2],
    y: [
      1.9375, -0.375, -2.25, 3.6875, -2.375, 0.8125, -0.375, -1.3125, 3.0625, 3.375, -0.125, -0.5,
      0.375, -2.0625, 3.375, -2.5,
    ],
    gx: [
      1.125, -0.75, 0.0, -1.5625, 1.25, -0.5, 0.4375, -0.1875, -0.125, -0.3125, -0.4375, -1.0,
      0.875, 1.6875, 0.125, -1.0, -0.125, 1.0625, -0.125, 1.5, -1.125, -0.4375, 1.375, -1.25,
      0.0625, -0.1875, -0.5625,
    ],
    gw: [
      2.8125, -0.5625, 0.9375, -1.75, 1.3125, -2.0625, -0.5625, -1.1875, -0.625, -0.625, -0.625,
      0.0625, -0.625, 1.4375, -0.625, 1.4375,
    ],
  },
  c2d: {
    shape: [1, 2, 4, 2],
    y: [
      0.6875, 1.875, -1.6875, 2.5, -1.375, -3.75, 1.125, -4.5, 0.75, 2.25, 1.9375, -4.75, 1.8125,
      0.5, 0.75, 1.6875,
    ],
    gx: [
      0.0, 1.0625, -1.4375, -0.1875, 1.0625, 3.3125, -0.8125, -0.8125, 1.6875, 2.9375, -0.1875,
      -1.4375, -0.25, -0.625, -0.4375, 1.625, 1.5625, -1.875, -0.8125, 1.625, 1.1875, -1.8125,
      -1.1875, 1.625,
    ],
    gw: [
      -0.3125, -1.25, 0.8125, -0.1875, 0.5, 0.0625, -0.1875, 0.8125, 0.125, -0.125, 0.0625, 1.0,
      0.0625, 0.9375, 0.3125, 0.1875, 0.5, 0.4375, 0.0, 0.3125, -0.3125, 0.0625, 0.4375, -0.3125,
    ],
    gb: [-0.75, 0.0],
  },
  c1d: {
    shape: [1, 2, 3],
    y: [1.375, -2.625, -1.5, -3.25, -1.4375, 0.5],
    gx: [0.0, -0.6875, 0.375, -1.5, -0.5625, -0.5, -0.9375, 0.625, -1.125, -0.4375],
    gw: [0.9375, 1.125, -0.375, -0.75, 1.125, -0.75, -0.375, -0.0625, 0.5, 1.0, -0.1875, -0.75],
    gb: [0.0, -0.25],
  },
};

const POOL: Record<string, Case> = {
  mp2d: {
    shape: [1, 2, 3, 3],
    y: [
      1.25, 1.75, 2.0, 1.5, 1.75, 2.0, 0.5, 2.0, 2.0, 1.75, 1.75, 1.25, 1.75, 2.0, 1.5, 1.75, 1.25,
      1.25,
    ],
    gx: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
    ],
  },
  ap2d_inc: {
    shape: [1, 2, 3, 3],
    y: [
      -0.3611111, 0.0, 0.25, 0.1111111, -0.0833333, 0.0555556, -0.2777778, 0.3611111, 0.1388889,
      0.25, -0.0277778, -0.25, 0.0833333, 0.1111111, -0.2222222, -0.1388889, -0.1388889, -0.3611111,
    ],
    gx: [
      -0.0833333, -0.0833333, 0.0, 0.0833333, 0.0833333, 0.0833333, -0.1111111, -0.0555556,
      0.0555556, 0.0833333, 0.0277778, 0.0277778, -0.0277778, 0.0277778, 0.0555556, 0.0, -0.0555556,
      -0.0555556, 0.0, -0.0277778, -0.0277778, -0.0833333, -0.0555556, -0.0555556, 0.0277778,
      -0.0555556, -0.0833333, -0.0833333, 0.0, 0.0, 0.0833333, 0.0555556, -0.0277778, 0.0277778,
      0.0555556, 0.0555556, 0.0277778, 0.0277778, 0.0, -0.0277778, -0.0277778, -0.0277778,
      -0.0555556, -0.0277778, 0.0277778, -0.0555556, -0.0833333, -0.0833333, -0.0555556, 0.0555556,
      0.1111111, -0.0, -0.1111111, -0.1111111, 0.0, 0.0833333, 0.0833333, 0.0555556, -0.0277778,
      -0.0277778,
    ],
  },
  ap2d_exc: {
    shape: [1, 2, 3, 3],
    y: [
      -0.8125, 0.0, 0.375, 0.1666667, -0.0833333, 0.0555556, -0.625, 0.5416667, 0.2083333, 0.5625,
      -0.0416667, -0.375, 0.125, 0.1111111, -0.2222222, -0.3125, -0.2083333, -0.5416667,
    ],
    gx: [
      -0.1875, -0.1875, 0.0, 0.125, 0.125, 0.125, -0.2291667, -0.1736111, 0.0555556, 0.125,
      0.0694444, 0.0694444, -0.0416667, 0.0138889, 0.0555556, 0.0, -0.0555556, -0.0555556,
      0.0208333, -0.0486111, -0.0694444, -0.125, -0.0555556, -0.0555556, 0.0625, -0.0625, -0.125,
      -0.125, 0.0, 0.0, 0.1875, 0.1458333, -0.0416667, 0.0416667, 0.0833333, 0.0833333, 0.1041667,
      0.0902778, -0.0138889, -0.0138889, 0.0, 0.0, -0.0833333, -0.0555556, 0.0277778, -0.0555556,
      -0.0833333, -0.0833333, -0.0833333, 0.0694444, 0.1527778, 0.0277778, -0.125, -0.125, 0.0,
      0.125, 0.125, 0.0833333, -0.0416667, -0.0416667,
    ],
  },
  mp1d: {
    shape: [1, 2, 4],
    y: [-0.75, 1.75, 1.75, 1.25, -0.5, 2.0, 2.0, 1.5],
    gx: [0.0, -0.75, 0.0, 0.75, 0.0, 0.0, -0.25, 0.0, 0.5, 0.0, -0.25, 0.0, 0.0, -0.75],
  },
  ap1d_inc: {
    shape: [1, 2, 4],
    y: [-0.9166667, 0.5, 0.1666667, 0.4166667, -0.75, 0.75, 0.4166667, 0.5833333],
    gx: [
      -0.25, -0.25, 0.0, 0.25, 0.25, 0.1666667, -0.0833333, 0.1666667, 0.0, -0.1666667, -0.0833333,
      0.0833333, -0.1666667, -0.25,
    ],
  },
  ap1d_exc: {
    shape: [1, 2, 4],
    y: [-1.375, 0.5, 0.1666667, 0.625, -1.125, 0.75, 0.4166667, 0.875],
    gx: [
      -0.375, -0.375, 0.0, 0.25, 0.25, 0.125, -0.125, 0.25, 0.0833333, -0.1666667, -0.0833333,
      0.0833333, -0.2916667, -0.375,
    ],
  },
  mp3d: {
    shape: [1, 1, 3, 3, 4],
    y: [
      -1.75, 0.0, 1.75, 1.75, -0.75, 1.0, 2.25, 2.25, -0.25, 1.5, 1.5, -1.5, 0.75, 2.0, 2.0, 2.0,
      1.75, 1.75, 2.0, 2.0, 2.25, 2.25, 1.75, 1.0, 1.0, 2.25, 2.25, -0.25, 2.0, 2.0, 0.75, 0.75,
      2.0, 2.0, 1.25, 1.25,
    ],
    gx: [
      0.0, 0.0, 0.0, -0.75, 0.0, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.0, -0.5, 0.0,
      -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.25, 0.0, 0.0, 0.0,
      -0.75, 0.0, 0.5, 0.0, 0.0, -0.5,
    ],
  },
  ap3d_inc: {
    shape: [1, 1, 3, 3, 4],
    y: [
      -0.3333333, -0.375, 0.2083333, 0.25, -0.3125, -0.1875, 0.2916667, 0.1666667, -0.0833333,
      0.125, -0.0833333, -0.2916667, -0.2083333, -0.2291667, 0.1458333, 0.1666667, 0.0625, -0.1875,
      -0.4166667, -0.1666667, 0.2916667, 0.375, -0.0416667, -0.125, 0.125, 0.1458333, -0.0625,
      -0.0833333, 0.375, 0.0, -0.3125, 0.0625, -0.0208333, -0.1458333, 0.0416667, 0.1666667,
    ],
    gx: [
      -0.0625, 0.0625, 0.0416667, -0.0625, 0.0416667, 0.0, 0.0, -0.0208333, -0.0416667, 0.0625,
      0.0208333, -0.0208333, 0.0625, 0.0416667, 0.0208333, -0.0208333, -0.0416667, -0.0625,
      0.0208333, -0.0208333, -0.0625, 0.0416667, 0.0208333, 0.0, 0.0, -0.0416667, 0.0625,
      -0.0416667, -0.0625, 0.0625, -0.0208333, -0.0416667, -0.0625, 0.0208333, -0.0208333, -0.0625,
      0.0416667, 0.0208333, 0.0, 0.0, -0.0416667, 0.0625, -0.0416667, -0.0625, 0.0625, 0.0208333,
      0.0, -0.0208333, -0.0416667, 0.0625, 0.0208333, -0.0625, 0.0625, 0.0416667, -0.0625,
      0.0416667, 0.0, 0.0, -0.0208333, -0.0416667,
    ],
  },
  ap3d_exc: {
    shape: [1, 1, 3, 3, 4],
    y: [
      -2.0, -1.125, 0.625, 1.5, -1.25, -0.375, 0.5833333, 0.6666667, -0.5, 0.375, -0.25, -1.75,
      -0.625, -0.34375, 0.21875, 0.5, 0.125, -0.1875, -0.4166667, -0.3333333, 0.875, 0.5625,
      -0.0625, -0.375, 0.75, 0.4375, -0.1875, -0.5, 1.5, 0.0, -0.625, 0.25, -0.125, -0.4375, 0.125,
      1.0,
    ],
    gx: [
      -0.375, 0.1875, 0.0625, -0.2916667, 0.1458333, -0.1458333, 0.0833333, -0.0416667, -0.2083333,
      0.2708333, 0.0833333, -0.0208333, 0.1875, 0.125, 0.1875, -0.09375, -0.0625, -0.09375,
      0.0104167, -0.0416667, -0.1354167, 0.1041667, 0.0208333, -0.0416667, 0.0729167, -0.0729167,
      0.1458333, -0.03125, -0.09375, 0.1875, -0.09375, -0.0625, -0.09375, 0.0104167, -0.0416667,
      -0.1354167, 0.1041667, 0.0208333, -0.0416667, 0.0729167, -0.0729167, 0.1458333, -0.03125,
      -0.09375, 0.1875, 0.0, 0.0, 0.0, -0.25, 0.125, 0.0416667, -0.25, 0.125, 0.0416667, -0.125,
      0.0625, -0.2708333, 0.125, -0.0625, -0.3125,
    ],
  },
  aap1d_5_3: {
    shape: [1, 1, 3],
    y: [-0.375, 0.4166667, 0.75],
    gx: [-0.375, -0.375, 0.0, 0.375, 0.375],
  },
  amp1d_5_3: { shape: [1, 1, 3], y: [0.5, 1.25, 1.25], gx: [0.0, -0.75, 0.0, 0.75, 0.0] },
  aap1d_2_4: { shape: [1, 1, 4], y: [-1.25, -1.25, 0.5, 0.5], gx: [-0.75, 0.5] },
  amp1d_2_4: { shape: [1, 1, 4], y: [-1.25, -1.25, 0.5, 0.5], gx: [-0.75, 0.5] },
  aap2d_5x7_2x3: {
    shape: [1, 2, 2, 3],
    y: [
      -0.1944444, 0.2222222, -0.0833333, 0.1944444, 0.25, -0.0555556, 0.0555556, -0.25, -0.1944444,
      0.0833333, -0.2222222, 0.1944444,
    ],
    gx: [
      -0.0833333, -0.0833333, -0.0833333, 0.0, 0.0833333, 0.0833333, 0.0833333, -0.0833333,
      -0.0833333, -0.0833333, 0.0, 0.0833333, 0.0833333, 0.0833333, -0.1111111, -0.1111111,
      -0.0555556, 0.0555556, 0.0833333, 0.0277778, 0.0277778, -0.0277778, -0.0277778, 0.0277778,
      0.0555556, 0.0, -0.0555556, -0.0555556, -0.0277778, -0.0277778, 0.0277778, 0.0555556, 0.0,
      -0.0555556, -0.0555556, 0.0277778, 0.0277778, -0.0555556, -0.0833333, -0.0833333, 0.0, 0.0,
      0.0277778, 0.0277778, -0.0555556, -0.0833333, -0.0833333, 0.0, 0.0, 0.1111111, 0.1111111,
      -0.0, -0.1111111, -0.0555556, 0.0555556, 0.0555556, 0.0833333, 0.0833333, 0.0555556,
      -0.0277778, 0.0277778, 0.0555556, 0.0555556, 0.0833333, 0.0833333, 0.0555556, -0.0277778,
      0.0277778, 0.0555556, 0.0555556,
    ],
  },
  amp2d_4x5_3x2: {
    shape: [1, 2, 3, 2],
    y: [1.5, 1.0, 1.5, 0.75, 1.25, 1.5, 0.75, 1.25, 1.5, 1.0, 1.5, 0.75],
    gx: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      -0.5, 0.0, 0.25, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, -0.25, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
    ],
  },
};

const BAG: Record<string, BagCase> = {
  bag_sum_None: {
    y: [-5.25, 0.0, 5.25, 0.0, 0.0, 0.0, -2.75, 2.5, -1.75, -0.75, 4.5, 0.25],
    g: [
      -0.75, 0.0, 0.75, -0.5, -0.75, 0.75, 0.0, -0.25, 1.25, 0.25, -0.75, 0.0, 0.25, -0.75, 0.0,
      1.5, -0.5, 1.0,
    ],
    shape: [4, 3],
  },
  bag_sum_1: {
    y: [-3.5, 0.0, 3.5, 0.0, 0.0, 0.0, -1.0, 2.5, -3.5, -0.75, 4.5, 0.25],
    g: [
      -0.75, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, -0.25, 1.25, 0.25, -0.75, 0.0, 0.25, -0.75, 0.0, 1.5,
      -0.5, 1.0,
    ],
    shape: [4, 3],
  },
  bag_mean_None: {
    y: [-1.75, 0.0, 1.75, 0.0, 0.0, 0.0, -0.9166667, 0.8333333, -0.5833333, -0.25, 1.5, 0.0833333],
    g: [
      -0.25, 0.0, 0.25, -0.1666667, -0.25, 0.25, 0.0, -0.0833333, 0.4166667, 0.0833333, -0.25, 0.0,
      0.0833333, -0.25, 0.0, 0.5, -0.1666667, 0.3333333,
    ],
    shape: [4, 3],
  },
  bag_mean_1: {
    y: [-1.75, 0.0, 1.75, 0.0, 0.0, 0.0, -0.5, 1.25, -1.75, -0.25, 1.5, 0.0833333],
    g: [
      -0.375, 0.0, 0.375, 0.0, 0.0, 0.0, -0.125, -0.0833333, 0.5416667, 0.125, -0.375, 0.0, 0.125,
      -0.375, 0.0, 0.5, -0.1666667, 0.3333333,
    ],
    shape: [4, 3],
  },
  bag_max_None: {
    y: [-1.25, 0.5, 2.25, 0.0, 0.0, 0.0, -0.25, 1.5, 1.75, 0.25, 2.0, 2.25],
    g: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, 0.0, 1.25, 0.0, 0.0, 0.0, 0.25, -0.75, 0.0, 0.75, -0.25,
      0.0,
    ],
    shape: [4, 3],
  },
  bag_max_1: {
    y: [-1.25, 0.5, 2.25, 0.0, 0.0, 0.0, -0.25, 1.5, -1.5, 0.25, 2.0, 2.25],
    g: [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, 0.0, 1.25, 0.0, 0.0, 0.0, 0.25, -0.75, 0.0, 0.75, -0.25,
      0.0,
    ],
    shape: [4, 3],
  },
  emb: {
    y: [
      0.0, 0.0, 0.0, -2.25, -0.5, 1.25, -1.25, 0.5, 2.25, -1.25, 0.5, 2.25, 0.0, 0.0, 0.0, -1.25,
      0.5, 2.25,
    ],
    g: [
      -0.25, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0, -0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0,
    ],
    shape: [2, 3, 3],
  },
};

const BAG_W: number[] = [
  -2.25, -0.5, 1.25, -1.75, 0.0, 1.75, -1.25, 0.5, 2.25, -0.75, 1.0, -2.0, -0.25, 1.5, -1.5, 0.25,
  2.0, -1.0,
];

const BAG_LAST: number[] = [-5.25, 0.0, 5.25, 0.0, 0.0, 0.0, -2.75, 2.5, -1.75, -0.75, 4.5, 0.25];

const BAG_2D: number[] = [-1.75, 0.0, 1.75, -0.9166667, 0.8333333, -0.5833333];

const POOL_MP2D_NAN: number[] = [
  Number.NaN,
  1.75,
  2.0,
  1.5,
  1.75,
  1.25,
  1.75,
  1.0,
  1.25,
  0.75,
  1.0,
  1.5,
];

const ALPHA = { a: 0.8609526162463561, b: 0.4540920681370629, aAlphaB: -1.0595481589864801 };

/** Deterministic dyadic test data: ((i * a) % m - c) / 4, shaped like the PyTorch fixtures. */
function vals(shape: number[], a = 7, m = 11, c = 5): Float32Array {
  const n = shape.reduce((x, y) => x * y, 1);
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = (((i * a) % m) - c) / 4;
  return d;
}

function mk(shape: number[], a = 7, m = 11, c = 5, requiresGrad = false): GradTensor {
  const t = tensor(Array.from(vals(shape, a, m, c)), { dtype: "float32" }).reshape(shape);
  return GradTensor.fromTensor(t, { requiresGrad });
}

function flat(t: GradTensor | { toArray(): unknown } | null | undefined): number[] {
  if (t === null || t === undefined) throw new Error("expected a tensor");
  const source = t instanceof GradTensor ? t.tensor : t;
  return (source.toArray() as unknown[]).flat(Number.POSITIVE_INFINITY) as number[];
}

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number;
    const b = expected[i] as number;
    expect(Math.abs(a - b), `index ${i}: ${a} vs ${b}`).toBeLessThanOrEqual(
      1e-5 * (1 + Math.abs(b))
    );
  }
}

/** Upstream gradient used by the fixtures: vals(shape, 3, 7, 3). */
function upstream(shape: readonly number[]): GradTensor {
  const s = [...shape];
  return GradTensor.fromTensor(
    tensor(Array.from(vals(s, 3, 7, 3)), { dtype: "float32" }).reshape(s)
  );
}

function setWeights(layer: { weight: GradTensor; bias?: GradTensor | undefined }): void {
  (layer.weight.tensor.data as Float32Array).set(vals([...layer.weight.shape], 5, 13, 6));
  if (layer.bias)
    (layer.bias.tensor.data as Float32Array).set(vals([...layer.bias.shape], 3, 7, 3));
}

interface Module1 {
  forward(x: GradTensor): GradTensor;
}

/** Forward + backward of `sum(y * upstream)` and comparison with a PyTorch fixture. */
function checkPool(layer: Module1, x: GradTensor, ref: Case): void {
  const y = layer.forward(x);
  expect(y.shape).toEqual(ref.shape);
  expectClose(flat(y), ref.y);
  y.mul(upstream(y.shape)).sum().backward();
  expectClose(flat(x.grad), ref.gx);
}

function checkConv(
  layer: Module1 & { weight: GradTensor; bias?: GradTensor | undefined },
  x: GradTensor,
  ref: Case
): void {
  setWeights(layer);
  const y = layer.forward(x);
  expect(y.shape).toEqual(ref.shape);
  expect(y.dtype).toBe("float32");
  expectClose(flat(y), ref.y);
  y.mul(upstream(y.shape)).sum().backward();
  expectClose(flat(x.grad), ref.gx);
  expect(x.grad?.dtype).toBe("float32");
  expectClose(flat(layer.weight.grad), ref.gw as number[]);
  expect(layer.weight.grad?.dtype).toBe("float32");
  if (ref.gb) expectClose(flat(layer.bias?.grad), ref.gb);
}

describe("adaptive pooling windows (PyTorch parity)", () => {
  it("AdaptiveAvgPool1d uses ceil for the window end (5 -> 3)", () => {
    // Windows are [0,2), [1,4), [3,5); the old floor/floor windows were [0,1), [1,3), [3,5).
    checkPool(new AdaptiveAvgPool1d(3), mk([1, 1, 5], 7, 11, 5, true), POOL.aap1d_5_3 as Case);
  });

  it("AdaptiveMaxPool1d uses ceil for the window end and routes the gradient", () => {
    checkPool(new AdaptiveMaxPool1d(3), mk([1, 1, 5], 7, 11, 5, true), POOL.amp1d_5_3 as Case);
  });

  it("windows are never empty when the input is shorter than the output (2 -> 4)", () => {
    checkPool(new AdaptiveAvgPool1d(4), mk([1, 1, 2], 7, 11, 5, true), POOL.aap1d_2_4 as Case);
    checkPool(new AdaptiveMaxPool1d(4), mk([1, 1, 2], 7, 11, 5, true), POOL.amp1d_2_4 as Case);
  });

  it("AdaptiveAvgPool2d matches PyTorch for 5x7 -> 2x3", () => {
    checkPool(
      new AdaptiveAvgPool2d([2, 3]),
      mk([1, 2, 5, 7], 3, 13, 6, true),
      POOL.aap2d_5x7_2x3 as Case
    );
  });

  it("AdaptiveMaxPool2d matches PyTorch for 4x5 -> 3x2", () => {
    checkPool(
      new AdaptiveMaxPool2d([3, 2]),
      mk([1, 2, 4, 5], 5, 13, 6, true),
      POOL.amp2d_4x5_3x2 as Case
    );
  });

  it("rejects an empty spatial size instead of returning zeros", () => {
    expect(() => new AdaptiveAvgPool2d(2).forward(zeros([1, 1, 0, 3]))).toThrow(ShapeError);
    expect(() => new AdaptiveMaxPool2d(2).forward(zeros([1, 1, 3, 0]))).toThrow(ShapeError);
    expect(() => new AdaptiveAvgPool1d(2).forward(zeros([1, 1, 0]))).toThrow(ShapeError);
    expect(() => new AdaptiveMaxPool1d(2).forward(zeros([1, 1, 0]))).toThrow(ShapeError);
  });

  it("rejects an output size tuple of the wrong length", () => {
    expect(() => new AdaptiveAvgPool2d([1, 2, 3] as unknown as [number, number])).toThrow(
      InvalidParameterError
    );
    expect(() => new AdaptiveMaxPool2d([1] as unknown as [number, number])).toThrow(
      InvalidParameterError
    );
  });
});

describe("max pooling", () => {
  it("MaxPool2d with padding matches PyTorch values and first-max gradients", () => {
    checkPool(
      new MaxPool2d(3, { stride: 2, padding: 1 }),
      mk([1, 2, 5, 6], 5, 17, 8, true),
      POOL.mp2d as Case
    );
  });

  it("MaxPool1d and MaxPool3d with padding match PyTorch", () => {
    checkPool(
      new MaxPool1d(3, { stride: 2, padding: 1 }),
      mk([1, 2, 7], 5, 17, 8, true),
      POOL.mp1d as Case
    );
    checkPool(
      new MaxPool3d([2, 3, 2], { stride: [2, 2, 1], padding: [1, 1, 1] }),
      mk([1, 1, 4, 5, 3], 7, 19, 9, true),
      POOL.mp3d as Case
    );
  });

  it("a NaN inside a window propagates to the output", () => {
    const data = vals([1, 2, 5, 6], 5, 17, 8);
    data[1 * 6 + 1] = Number.NaN;
    data[30 + 2 * 6 + 2] = Number.NEGATIVE_INFINITY;
    const x = GradTensor.fromTensor(
      tensor(Array.from(data), { dtype: "float32" }).reshape([1, 2, 5, 6])
    );
    const y = new MaxPool2d(2).forward(x);
    const out = flat(y);
    expect(Number.isNaN(out[0])).toBe(true);
    expect(out.slice(1)).toEqual(POOL_MP2D_NAN.slice(1));
  });

  it("a window of only -Infinity still sends its gradient to the first element", () => {
    const x = GradTensor.fromTensor(
      tensor([-Infinity, -Infinity, -Infinity, -Infinity], { dtype: "float32" }).reshape([
        1, 1, 2, 2,
      ]),
      { requiresGrad: true }
    );
    const y = new MaxPool2d(2).forward(x);
    expect(flat(y)).toEqual([-Infinity]);
    y.sum().backward();
    expect(flat(x.grad)).toEqual([1, 0, 0, 0]);
  });

  it("MaxPool2d throws a ShapeError when the kernel does not fit the input", () => {
    expect(() => new MaxPool2d(5).forward(zeros([1, 1, 3, 3]))).toThrow(ShapeError);
  });

  it("AvgPool2d throws a ShapeError when the kernel does not fit, with and without padding counts", () => {
    expect(() => new AvgPool2d(5).forward(zeros([1, 1, 3, 3]))).toThrow(ShapeError);
    expect(() => new AvgPool2d(5, { countIncludePad: false }).forward(zeros([1, 1, 3, 3]))).toThrow(
      ShapeError
    );
  });

  it("pooling padding larger than half the kernel is rejected like PyTorch", () => {
    expect(() => new MaxPool2d(2, { padding: 2 })).toThrow(InvalidParameterError);
    expect(() => new MaxPool1d(3, { padding: 2 })).toThrow(InvalidParameterError);
    expect(() => new AvgPool1d(2, { padding: 2 })).toThrow(InvalidParameterError);
    expect(() => new AvgPool2d([2, 4], { padding: [1, 3] })).toThrow(InvalidParameterError);
    expect(() => new MaxPool3d(2, { padding: [1, 1, 2] })).toThrow(InvalidParameterError);
    expect(() => new AvgPool3d(2, { padding: 2 })).toThrow(InvalidParameterError);
    expect(() => new MaxPool2d(3, { padding: 1 })).not.toThrow();
  });

  it("1-D pooling validates stride and padding", () => {
    expect(() => new MaxPool1d(2, { stride: 0 })).toThrow(InvalidParameterError);
    expect(() => new MaxPool1d(2, { stride: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new AvgPool1d(2, { stride: -1 })).toThrow(InvalidParameterError);
    expect(() => new AvgPool1d(2, { padding: -1 })).toThrow(InvalidParameterError);
    expect(() => new MaxPool1d(2, { padding: 0.5 })).toThrow(InvalidParameterError);
  });
});

describe("average pooling and count_include_pad", () => {
  it("AvgPool2d counts padding by default and can exclude it", () => {
    checkPool(
      new AvgPool2d(3, { stride: 2, padding: 1 }),
      mk([1, 2, 5, 6], 5, 17, 8, true),
      POOL.ap2d_inc as Case
    );
    checkPool(
      new AvgPool2d(3, { stride: 2, padding: 1, countIncludePad: false }),
      mk([1, 2, 5, 6], 5, 17, 8, true),
      POOL.ap2d_exc as Case
    );
  });

  it("AvgPool1d counts padding by default and can exclude it", () => {
    checkPool(
      new AvgPool1d(3, { stride: 2, padding: 1 }),
      mk([1, 2, 7], 5, 17, 8, true),
      POOL.ap1d_inc as Case
    );
    checkPool(
      new AvgPool1d(3, { stride: 2, padding: 1, countIncludePad: false }),
      mk([1, 2, 7], 5, 17, 8, true),
      POOL.ap1d_exc as Case
    );
  });

  it("AvgPool3d counts padding by default and can exclude it", () => {
    const opts = {
      stride: [2, 2, 1] as [number, number, number],
      padding: [1, 1, 1] as [number, number, number],
    };
    checkPool(
      new AvgPool3d([2, 3, 2], opts),
      mk([1, 1, 4, 5, 3], 7, 19, 9, true),
      POOL.ap3d_inc as Case
    );
    checkPool(
      new AvgPool3d([2, 3, 2], { ...opts, countIncludePad: false }),
      mk([1, 1, 4, 5, 3], 7, 19, 9, true),
      POOL.ap3d_exc as Case
    );
  });
});

describe("convolution kernels match PyTorch", () => {
  it("ConvTranspose2d (stride 2, padding 1, outputPadding 1) values and gradients", () => {
    checkConv(
      new ConvTranspose2d(1, 2, 3, { stride: 2, padding: 1, outputPadding: 1 }),
      mk([1, 1, 2, 3], 7, 11, 5, true),
      CONV.ct2d as Case
    );
  });

  it("ConvTranspose2d with an asymmetric kernel and no bias", () => {
    checkConv(
      new ConvTranspose2d(2, 1, [2, 3], { stride: [1, 2], padding: [0, 1], bias: false }),
      mk([2, 2, 2, 3], 7, 11, 5, true),
      CONV.ct2d_asym as Case
    );
  });

  it("ConvTranspose1d values and gradients", () => {
    checkConv(
      new ConvTranspose1d(2, 2, 3, { stride: 2, padding: 1, outputPadding: 1 }),
      mk([1, 2, 4], 7, 11, 5, true),
      CONV.ct1d as Case
    );
  });

  it("Conv3d with mixed stride and padding values and gradients", () => {
    checkConv(
      new Conv3d(2, 2, [2, 3, 2], { stride: [1, 2, 1], padding: [1, 0, 1] }),
      mk([1, 2, 3, 4, 3], 7, 11, 5, true),
      CONV.c3d as Case
    );
    checkConv(
      new Conv3d(1, 2, 2, { bias: false }),
      mk([1, 1, 3, 3, 3], 7, 11, 5, true),
      CONV.c3d_nb as Case
    );
  });

  it("Conv2d and Conv1d values and gradients", () => {
    checkConv(
      new Conv2d(2, 2, [2, 3], { stride: [1, 2], padding: [1, 1] }),
      mk([1, 2, 3, 4], 7, 11, 5, true),
      CONV.c2d as Case
    );
    checkConv(
      new Conv1d(2, 2, 3, { stride: 2, padding: 1 }),
      mk([1, 2, 5], 7, 11, 5, true),
      CONV.c1d as Case
    );
  });

  it("handles non-contiguous inputs the same way as contiguous ones", () => {
    const base = GradTensor.fromTensor(
      tensor(Array.from(vals([1, 4, 4, 2])), { dtype: "float32" }).reshape([1, 4, 4, 2])
    );
    const view = base.transpose([0, 3, 1, 2]);
    const dense = GradTensor.fromTensor(
      tensor(Array.from(flat(view)), { dtype: "float32" }).reshape([1, 2, 4, 4])
    );
    const layers: Module1[] = [
      new MaxPool2d(2),
      new AvgPool2d(2, { countIncludePad: false }),
      new AdaptiveAvgPool2d(3),
      new ConvTranspose2d(2, 2, 2),
    ];
    for (const layer of layers) {
      expect(flat(layer.forward(view))).toEqual(flat(layer.forward(dense)));
    }
  });

  it("empty batches give empty outputs", () => {
    expect(new ConvTranspose2d(1, 2, 1).forward(zeros([0, 1, 3, 3])).shape).toEqual([0, 2, 3, 3]);
    expect(new Conv3d(1, 2, 1).forward(zeros([0, 1, 3, 3, 3])).shape).toEqual([0, 2, 3, 3, 3]);
    expect(new MaxPool2d(2).forward(zeros([0, 1, 4, 4])).shape).toEqual([0, 1, 2, 2]);
    expect(new AdaptiveAvgPool2d(2).forward(zeros([0, 1, 4, 4])).shape).toEqual([0, 1, 2, 2]);
  });
});

describe("convolution validation", () => {
  it("outputPadding must be smaller than the stride", () => {
    expect(() => new ConvTranspose2d(1, 1, 3, { stride: 2, outputPadding: 2 })).toThrow(
      InvalidParameterError
    );
    expect(() => new ConvTranspose1d(1, 1, 3, { stride: 1, outputPadding: 1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new ConvTranspose1d(1, 1, 3, { stride: 2, outputPadding: 1 })).not.toThrow();
  });

  it("a non-positive computed output size is a ShapeError, not a RangeError", () => {
    expect(() => new ConvTranspose1d(1, 1, 1, { padding: 3 }).forward(zeros([1, 1, 2]))).toThrow(
      ShapeError
    );
    expect(() => new ConvTranspose2d(1, 1, 1, { padding: 3 }).forward(zeros([1, 1, 2, 2]))).toThrow(
      ShapeError
    );
    expect(() => new Conv3d(1, 1, 3).forward(zeros([1, 1, 2, 2, 2]))).toThrow(ShapeError);
  });

  it("rejects string tensors and wrong ranks", () => {
    expect(() => new Conv3d(1, 1, 1).forward(tensor([[[[["a"]]]]]))).toThrow(DTypeError);
    expect(() => new MaxPool2d(2).forward(tensor([["a"]]))).toThrow(DTypeError);
    expect(() => new ConvTranspose2d(1, 1, 1).forward(zeros([1, 1, 2]))).toThrow(ShapeError);
  });

  it("exposes the bias parameter", () => {
    expect(new Conv1d(1, 2, 3).bias?.shape).toEqual([2]);
    expect(new Conv2d(1, 2, 3).bias?.shape).toEqual([2]);
    expect(new Conv3d(1, 2, 3).bias?.shape).toEqual([2]);
    expect(new ConvTranspose1d(1, 2, 3).bias?.shape).toEqual([2]);
    expect(new ConvTranspose2d(1, 2, 3).bias?.shape).toEqual([2]);
    expect(new Conv2d(1, 2, 3, { bias: false }).bias).toBeUndefined();
    expect(new Conv3d(1, 2, 3, { bias: false }).bias).toBeUndefined();
  });
});

describe("dtype handling", () => {
  it("pooling and transposed convolution keep float32 so residual graphs accumulate", () => {
    const x = mk([1, 1, 4, 4], 7, 11, 5, true);
    const skip = x.mul(GradTensor.scalar(2, { dtype: "float32" }));
    for (const branch of [
      new MaxPool2d(1).forward(x),
      new AdaptiveAvgPool2d(4).forward(x),
      new ConvTranspose2d(1, 1, 1).forward(x),
    ]) {
      expect(branch.dtype).toBe("float32");
      x.zeroGrad();
      skip.add(branch).sum().backward();
      expect(x.grad?.dtype).toBe("float32");
    }
  });

  it("layers compute in their parameter dtype: float64 input is cast to float32", () => {
    const x64 = GradTensor.fromTensor(
      tensor(Array.from(vals([1, 1, 4, 4])), { dtype: "float64" }).reshape([1, 1, 4, 4]),
      { requiresGrad: true }
    );
    // Parameter-free layers keep the input dtype ...
    expect(new MaxPool2d(2).forward(x64).dtype).toBe("float64");
    // ... layers with parameters use the parameter dtype (a float64 layer keeps float64).
    expect(new ConvTranspose2d(1, 1, 2).forward(x64).dtype).toBe("float32");
    expect(new ConvTranspose2d(1, 1, 2, { dtype: "float64" }).forward(x64).dtype).toBe("float64");
    const y = new Conv2d(1, 1, 2).forward(x64);
    expect(y.dtype).toBe("float32");
    y.sum().backward();
    expect(x64.grad?.dtype).toBe("float64");
  });

  it("pooling an integer tensor produces float64 values", () => {
    const x = tensor(
      [
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
      ],
      { dtype: "int32" }
    );
    const y = new MaxPool2d(2).forward(x);
    expect(y.dtype).toBe("float64");
    expect(flat(y)).toEqual([4]);
  });
});

describe("Dropout", () => {
  it("converts integer inputs to float32 so the 1/(1-p) scale is exact", () => {
    setSeed(1);
    const d = new Dropout(0.3);
    const y = d.forward(tensor([1, 2, 3, 4, 5, 6, 7, 8], { dtype: "int32" }));
    expect(y.dtype).toBe("float32");
    const scale = 1 / 0.7;
    flat(y).forEach((v, i) => {
      expect(v === 0 || Math.abs(v - (i + 1) * scale) < 1e-5).toBe(true);
    });
    expect(flat(y).some((v) => v !== 0)).toBe(true);
  });

  it("evaluation mode returns integer inputs unchanged", () => {
    const d = new Dropout(0.3).eval();
    expect(d.forward(tensor([1, 2, 3], { dtype: "int32" })).dtype).toBe("int32");
  });
});

describe("Dropout2d", () => {
  const input = (requiresGrad = false): GradTensor => mk([3, 4, 2, 2], 7, 11, 5, requiresGrad);

  it("keeps the input dtype and scales kept channels by 1/(1-p)", () => {
    setSeed(7);
    const d = new Dropout2d(0.5);
    const x = input(true);
    const y = d.forward(x);
    expect(y.dtype).toBe("float32");
    const xs = flat(x);
    const ys = flat(y);
    let kept = 0;
    for (let plane = 0; plane < 12; plane++) {
      const slice = ys.slice(plane * 4, plane * 4 + 4);
      const src = xs.slice(plane * 4, plane * 4 + 4);
      if (slice.every((v) => v === 0)) continue;
      kept++;
      for (let i = 0; i < slice.length; i++) {
        expect(slice[i]).toBeCloseTo((src[i] as number) * 2, 6);
      }
    }
    expect(kept).toBeGreaterThan(0);
    expect(kept).toBeLessThan(12);
    // The gradient is the same per-channel multiplier.
    y.sum().backward();
    expect(x.grad?.dtype).toBe("float32");
    const grad = flat(x.grad);
    for (let plane = 0; plane < 12; plane++) {
      const zeroed = ys.slice(plane * 4, plane * 4 + 4).every((v) => v === 0);
      const expected = zeroed ? 0 : 2;
      expect(grad.slice(plane * 4, plane * 4 + 4)).toEqual([
        expected,
        expected,
        expected,
        expected,
      ]);
    }
  });

  it("is reproducible under a seed", () => {
    const d = new Dropout2d(0.5);
    setSeed(11);
    const a = flat(d.forward(input()));
    setSeed(11);
    const b = flat(d.forward(input()));
    expect(a).toEqual(b);
  });

  it("works inside a float32 residual graph", () => {
    const x = input(true);
    const y = new Dropout2d(0.5).forward(x).add(x);
    expect(() => y.sum().backward()).not.toThrow();
  });

  it("accepts a single (C, H, W) sample", () => {
    setSeed(3);
    const ones = GradTensor.fromTensor(
      tensor(new Array(16).fill(1), { dtype: "float32" }).reshape([4, 2, 2])
    );
    const y = new Dropout2d(0.5).forward(ones);
    expect(y.shape).toEqual([4, 2, 2]);
    const ys = flat(y);
    for (let c = 0; c < 4; c++) {
      const plane = ys.slice(c * 4, c * 4 + 4);
      expect(plane.every((v) => v === plane[0])).toBe(true);
      expect([0, 2]).toContain(plane[0]);
    }
  });

  it("reports a ShapeError for other ranks while training", () => {
    expect(() => new Dropout2d(0.5).forward(tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => new Dropout2d(0.5).forward(tensor([[1, 2, 3]]))).toThrow(/4D/);
  });

  it("converts integer inputs to float32", () => {
    setSeed(5);
    const y = new Dropout2d(0.5).forward(tensor([[[[1, 2]], [[3, 4]]]], { dtype: "int32" }));
    expect(y.dtype).toBe("float32");
    expect(flat(y).every((v, i) => v === 0 || v === (i + 1) * 2)).toBe(true);
  });
});

describe("AlphaDropout", () => {
  const p = 0.3;
  const { a, b, aAlphaB } = ALPHA;

  it("maps kept values to a*x + b and dropped values to a*alpha' + b", () => {
    setSeed(2);
    const x = mk([8, 8], 7, 11, 5, true);
    const y = new AlphaDropout(p).forward(x);
    expect(y.dtype).toBe("float32");
    const xs = flat(x);
    let dropped = 0;
    flat(y).forEach((v, i) => {
      const keptValue = a * (xs[i] as number) + b;
      if (Math.abs(v - keptValue) < 1e-5) return;
      dropped++;
      expect(v).toBeCloseTo(aAlphaB, 5);
    });
    expect(dropped).toBeGreaterThan(0);
    expect(dropped).toBeLessThan(64);
    y.sum().backward();
    for (const g of flat(x.grad)) expect(g === 0 || Math.abs(g - a) < 1e-6).toBe(true);
  });

  it("handles 0-d inputs and non-contiguous views", () => {
    setSeed(4);
    const layer = new AlphaDropout(0.5);
    expect(layer.forward(tensor(1.5)).shape).toEqual([]);
    const base = mk([3, 4]);
    const view = base.transpose();
    setSeed(9);
    const a1 = flat(layer.forward(view));
    const dense = GradTensor.fromTensor(tensor(flat(view), { dtype: "float32" }).reshape([4, 3]));
    setSeed(9);
    const a2 = flat(layer.forward(dense));
    expect(a1).toEqual(a2);
  });

  it("converts integer inputs to float32", () => {
    expect(new AlphaDropout(0.3).forward(tensor([1, 2, 3], { dtype: "int32" })).dtype).toBe(
      "float32"
    );
  });
});

describe("Embedding", () => {
  it("matches PyTorch with a negative paddingIdx, repeated indices and gradients", () => {
    const emb = new Embedding(6, 3, { paddingIdx: -1 });
    const w = Float32Array.from(BAG_W);
    for (let j = 0; j < 3; j++) w[15 + j] = 0;
    (emb.weight.tensor.data as Float32Array).set(w);
    const y = emb.forward(
      tensor(
        [
          [5, 0, 2],
          [2, 5, 2],
        ],
        { dtype: "int32" }
      )
    );
    expect(y.shape).toEqual(BAG.emb.shape);
    expect(y.dtype).toBe("float32");
    expectClose(flat(y), BAG.emb.y);
    (y as GradTensor).mul(upstream(y.shape)).sum().backward();
    expectClose(flat(emb.weight.grad), BAG.emb.g);
    expect(emb.weight.grad?.dtype).toBe("float32");
    expect(emb.toString()).toBe("Embedding(6, 3, padding_idx=5)");
  });

  it("exposes its configuration", () => {
    const emb = new Embedding(6, 3, { paddingIdx: -1 });
    expect([emb.numEmbeddings, emb.embeddingDim, emb.paddingIdx]).toEqual([6, 3, 5]);
    const bag = new EmbeddingBag(7, 2, { mode: "max", paddingIdx: 0, includeLastOffset: true });
    expect([bag.numEmbeddings, bag.embeddingDim, bag.mode, bag.paddingIdx]).toEqual([
      7,
      2,
      "max",
      0,
    ]);
    expect(bag.includeLastOffset).toBe(true);
    expect(new EmbeddingBag(7, 2).mode).toBe("mean");
  });

  it("rejects paddingIdx outside [-numEmbeddings, numEmbeddings)", () => {
    expect(() => new Embedding(6, 3, { paddingIdx: -7 })).toThrow(/paddingIdx/);
    expect(() => new Embedding(6, 3, { paddingIdx: 6 })).toThrow(/paddingIdx/);
    expect(() => new Embedding(6, 3, { paddingIdx: 1.5 })).toThrow(/paddingIdx/);
    expect(() => new Embedding(6, 3, { paddingIdx: -6 })).not.toThrow();
  });

  it("zeroes the padding row at construction", () => {
    const emb = new Embedding(4, 3, { paddingIdx: -2 });
    expect(flat(emb.weight.tensor).slice(6, 9)).toEqual([0, 0, 0]);
  });

  it("rejects NaN and fractional indices instead of rounding them", () => {
    const emb = new Embedding(5, 2);
    expect(() => emb.forward(tensor([Number.NaN]))).toThrow(InvalidParameterError);
    expect(() => emb.forward(tensor([1.5]))).toThrow(/integers/);
    expect(() => emb.forward(tensor([Number.POSITIVE_INFINITY]))).toThrow(InvalidParameterError);
    expect(() => emb.forward(tensor([-1]))).toThrow(/out of range/);
    expect(() => emb.forward(tensor([5]))).toThrow(/out of range/);
  });

  it("looks up 0-d, empty and int64 indices", () => {
    const emb = new Embedding(5, 2);
    expect(emb.forward(tensor(2)).shape).toEqual([2]);
    expect(emb.forward(tensor([], { dtype: "int32" })).shape).toEqual([0, 2]);
    expect(emb.forward(tensor([1, 2], { dtype: "int64" })).shape).toEqual([2, 2]);
  });

  it("fromPretrained copies the weights and freezes them by default", () => {
    const source = tensor(
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      { dtype: "float32" }
    );
    const frozen = Embedding.fromPretrained(source);
    expect(flat(frozen.weight.tensor)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(frozen.weight.requiresGrad).toBe(false);
    expect(flat(frozen.forward(tensor([2, 0], { dtype: "int32" })))).toEqual([5, 6, 1, 2]);

    const trainable = Embedding.fromPretrained(source, { freeze: false, paddingIdx: 1 });
    expect(trainable.weight.requiresGrad).toBe(true);
    // The lookup returns the stored row, which fromPretrained leaves as given (PyTorch).
    expect(flat(trainable.forward(tensor([1, 2], { dtype: "int32" })))).toEqual([3, 4, 5, 6]);
    // The source tensor is copied, not shared.
    (trainable.weight.tensor.data as Float32Array)[0] = 99;
    expect(flat(source)[0]).toBe(1);

    expect(() => Embedding.fromPretrained(tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => Embedding.fromPretrained(tensor([["a"]]))).toThrow(DTypeError);
  });

  it("freezing keeps lookups working", () => {
    const emb = new Embedding(4, 2);
    emb.freezeParameters(["weight"]);
    const y = emb.forward(tensor([1, 2], { dtype: "int32" }));
    // A frozen table gives a plain tensor: nothing is tracked.
    expect(GradTensor.isGradTensor(y)).toBe(false);
  });
});

describe("EmbeddingBag", () => {
  const indices = tensor([0, 1, 2, 4, 1, 3, 5, 5, 2], { dtype: "int32" });
  const offsets = tensor([0, 3, 3, 6], { dtype: "int32" });

  function makeBag(
    mode: "sum" | "mean" | "max",
    paddingIdx?: number,
    extra: { includeLastOffset?: boolean } = {}
  ): EmbeddingBag {
    const bag = new EmbeddingBag(6, 3, {
      mode,
      ...(paddingIdx === undefined ? {} : { paddingIdx }),
      ...extra,
    });
    (bag.weight.tensor.data as Float32Array).set(BAG_W);
    return bag;
  }

  for (const mode of ["sum", "mean", "max"] as const) {
    for (const pad of [undefined, 1] as const) {
      it(`${mode} mode${pad === undefined ? "" : " with paddingIdx"} matches PyTorch`, () => {
        const ref = BAG[`bag_${mode}_${pad ?? "None"}`] as BagCase;
        const bag = makeBag(mode, pad);
        const y = bag.forward(indices, offsets);
        expect(y.shape).toEqual(ref.shape);
        expect(y.dtype).toBe("float32");
        expectClose(flat(y), ref.y);
        (y as GradTensor).mul(upstream(y.shape)).sum().backward();
        expectClose(flat(bag.weight.grad), ref.g);
      });
    }
  }

  it("supports includeLastOffset", () => {
    const bag = makeBag("sum", undefined, { includeLastOffset: true });
    const y = bag.forward(indices, tensor([0, 3, 3, 6, 9], { dtype: "int32" }));
    expect(y.shape).toEqual([4, 3]);
    expectClose(flat(y), BAG_LAST);
  });

  it("accepts 2-D indices without offsets", () => {
    const bag = makeBag("mean");
    const y = bag.forward(
      tensor(
        [
          [0, 1, 2],
          [4, 1, 3],
        ],
        { dtype: "int32" }
      )
    );
    expect(y.shape).toEqual([2, 3]);
    expectClose(flat(y), BAG_2D);
    expect(() =>
      bag.forward(tensor([[0, 1]], { dtype: "int32" }), tensor([0], { dtype: "int32" }))
    ).toThrow(ShapeError);
  });

  it("max mode keeps -Infinity values and still routes the gradient", () => {
    const bag = new EmbeddingBag(6, 3, { mode: "max" });
    const w = Float32Array.from(BAG_W);
    for (let j = 0; j < 3; j++) w[6 + j] = Number.NEGATIVE_INFINITY;
    (bag.weight.tensor.data as Float32Array).set(w);
    const y = bag.forward(tensor([2, 2], { dtype: "int32" }), tensor([0], { dtype: "int32" }));
    expect(flat(y)).toEqual([-Infinity, -Infinity, -Infinity]);
    (y as GradTensor).sum().backward();
    expect(flat(bag.weight.grad).slice(6, 9)).toEqual([1, 1, 1]);
  });

  it("max mode propagates NaN weights", () => {
    const bag = new EmbeddingBag(3, 2, { mode: "max" });
    (bag.weight.tensor.data as Float32Array).set([1, 2, Number.NaN, 0, 5, 6]);
    const y = bag.forward(tensor([0, 1, 2], { dtype: "int32" }), tensor([0], { dtype: "int32" }));
    const out = flat(y);
    expect(Number.isNaN(out[0])).toBe(true);
    expect(out[1]).toBe(6);
  });

  it("zeroes the padding row at construction", () => {
    const bag = new EmbeddingBag(4, 3, { paddingIdx: 2 });
    expect(flat(bag.weight.tensor).slice(6, 9)).toEqual([0, 0, 0]);
    expect(bag.toString()).toBe("EmbeddingBag(4, 3, mode=mean, padding_idx=2)");
  });

  it("validates offsets instead of reading out of bounds", () => {
    const bag = new EmbeddingBag(5, 2);
    const idx = tensor([1, 2, 3], { dtype: "int32" });
    expect(() => bag.forward(idx, tensor([1, 2], { dtype: "int32" }))).toThrow(/offsets\[0\]/);
    expect(() => bag.forward(idx, tensor([0, 2, 1], { dtype: "int32" }))).toThrow(/non-decreasing/);
    expect(() => bag.forward(idx, tensor([0, 4], { dtype: "int32" }))).toThrow(/offsets/);
    expect(() => bag.forward(idx, tensor([0, -1], { dtype: "int32" }))).toThrow(/offsets/);
    expect(() => bag.forward(idx, tensor([[0]], { dtype: "int32" }))).toThrow(ShapeError);
    expect(bag.forward(idx, tensor([], { dtype: "int32" })).shape).toEqual([0, 2]);
  });

  it("rejects fractional indices, unknown modes and bad ranks", () => {
    const bag = new EmbeddingBag(5, 2);
    expect(() => bag.forward(tensor([1.5]), tensor([0], { dtype: "int32" }))).toThrow(/integers/);
    expect(() => new EmbeddingBag(5, 2, { mode: "median" as unknown as "sum" })).toThrow(
      InvalidParameterError
    );
    expect(() => bag.forward(zeros([1, 1, 1]), tensor([0], { dtype: "int32" }))).toThrow(
      ShapeError
    );
  });
});
