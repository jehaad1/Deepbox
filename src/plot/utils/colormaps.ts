/**
 * Perceptual colormaps used by heatmaps, images and filled contours.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { ColormapName } from "../types";

export type { ColormapName };

type RGB = readonly [number, number, number];

/**
 * 33 evenly spaced samples (t = 0, 1/32, ..., 1) of the matplotlib 3.10 colormaps, as 8-bit
 * RGB. Linear interpolation between neighbouring samples stays within 5 levels of the
 * original 256-entry lookup table.
 */
const SAMPLED: Readonly<Record<Exclude<ColormapName, "grayscale">, readonly RGB[]>> = {
  viridis: [
    [68, 1, 84],
    [71, 13, 96],
    [72, 24, 106],
    [72, 35, 116],
    [71, 45, 123],
    [69, 55, 129],
    [66, 64, 134],
    [62, 73, 137],
    [59, 82, 139],
    [55, 91, 141],
    [51, 99, 141],
    [47, 107, 142],
    [44, 114, 142],
    [41, 122, 142],
    [38, 130, 142],
    [35, 137, 142],
    [33, 145, 140],
    [31, 152, 139],
    [31, 160, 136],
    [34, 167, 133],
    [40, 174, 128],
    [50, 182, 122],
    [63, 188, 115],
    [78, 195, 107],
    [94, 201, 98],
    [112, 207, 87],
    [132, 212, 75],
    [152, 216, 62],
    [173, 220, 48],
    [194, 223, 35],
    [216, 226, 25],
    [236, 229, 27],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [34, 6, 144],
    [49, 5, 151],
    [63, 4, 156],
    [76, 2, 161],
    [89, 1, 165],
    [102, 0, 167],
    [114, 1, 168],
    [126, 3, 168],
    [138, 9, 165],
    [149, 17, 161],
    [160, 26, 156],
    [170, 35, 149],
    [179, 44, 142],
    [188, 53, 135],
    [196, 62, 127],
    [204, 71, 120],
    [211, 81, 113],
    [218, 90, 106],
    [224, 99, 99],
    [230, 108, 92],
    [235, 118, 85],
    [240, 128, 78],
    [245, 139, 71],
    [248, 149, 64],
    [251, 161, 57],
    [253, 172, 51],
    [254, 184, 44],
    [253, 197, 39],
    [252, 210, 37],
    [248, 223, 37],
    [244, 237, 39],
    [240, 249, 33],
  ],
  inferno: [
    [0, 0, 4],
    [4, 3, 18],
    [11, 7, 36],
    [21, 11, 55],
    [33, 12, 74],
    [47, 10, 91],
    [61, 9, 101],
    [74, 12, 107],
    [87, 16, 110],
    [100, 21, 110],
    [113, 25, 110],
    [125, 30, 109],
    [138, 34, 106],
    [151, 39, 102],
    [163, 44, 97],
    [176, 49, 91],
    [188, 55, 84],
    [199, 62, 76],
    [210, 70, 68],
    [219, 80, 59],
    [228, 90, 49],
    [235, 102, 40],
    [241, 115, 29],
    [246, 128, 19],
    [249, 142, 9],
    [251, 157, 7],
    [252, 172, 17],
    [251, 188, 33],
    [249, 203, 53],
    [245, 219, 76],
    [242, 234, 105],
    [243, 246, 138],
    [252, 255, 164],
  ],
  magma: [
    [0, 0, 4],
    [3, 3, 18],
    [10, 8, 34],
    [19, 13, 52],
    [29, 17, 71],
    [41, 17, 90],
    [54, 16, 107],
    [68, 15, 118],
    [81, 18, 124],
    [93, 23, 127],
    [106, 28, 129],
    [118, 33, 129],
    [131, 38, 129],
    [144, 42, 129],
    [156, 46, 127],
    [170, 51, 125],
    [183, 55, 121],
    [196, 60, 117],
    [208, 65, 111],
    [220, 72, 105],
    [231, 82, 99],
    [239, 93, 94],
    [245, 107, 92],
    [249, 121, 93],
    [252, 137, 97],
    [253, 152, 105],
    [254, 167, 114],
    [254, 182, 124],
    [254, 196, 136],
    [254, 211, 149],
    [253, 226, 163],
    [252, 240, 178],
    [252, 253, 191],
  ],
  cividis: [
    [0, 34, 78],
    [0, 40, 91],
    [0, 46, 106],
    [5, 51, 113],
    [26, 56, 111],
    [39, 62, 110],
    [50, 67, 109],
    [59, 73, 108],
    [67, 78, 108],
    [75, 84, 108],
    [83, 90, 109],
    [90, 95, 110],
    [97, 101, 111],
    [104, 106, 113],
    [111, 112, 115],
    [118, 118, 118],
    [125, 124, 120],
    [132, 130, 121],
    [140, 136, 120],
    [147, 142, 120],
    [155, 148, 118],
    [163, 154, 116],
    [171, 160, 114],
    [180, 167, 111],
    [188, 174, 108],
    [196, 180, 104],
    [205, 187, 99],
    [213, 194, 94],
    [222, 201, 88],
    [231, 209, 80],
    [240, 216, 70],
    [249, 224, 58],
    [254, 232, 56],
  ],
};

/** Every supported colormap name, in the order they are documented. */
export const COLORMAP_NAMES: readonly ColormapName[] = [
  "viridis",
  "plasma",
  "inferno",
  "magma",
  "cividis",
  "grayscale",
];

/** Whether `name` is a supported colormap name. @internal */
export function isColormapName(name: unknown): name is ColormapName {
  return typeof name === "string" && (name === "grayscale" || Object.hasOwn(SAMPLED, name));
}

/**
 * Checks a user-supplied colormap name and narrows its type. Every drawable that accepts a
 * `colormap` option validates it through this function, so the list of names has one source.
 * @throws {InvalidParameterError} If `name` is not a supported colormap name.
 * @internal
 */
export function assertColormapName(name: unknown): asserts name is ColormapName {
  if (!isColormapName(name)) {
    throw new InvalidParameterError(
      `colormap must be one of ${COLORMAP_NAMES.join(", ")}; received ${String(name)}`,
      "colormap",
      name
    );
  }
}

/**
 * Maps a normalized value to an RGB triple (0-255 integers) using a colormap.
 *
 * Values outside [0, 1] are clamped to the end colors. A NaN value maps to black so missing
 * data is visible.
 * @throws {InvalidParameterError} If `colormap` is not a supported colormap name.
 * @internal
 */
export function applyColormap(value: number, colormap: ColormapName): [number, number, number] {
  if (!isColormapName(colormap)) {
    throw new InvalidParameterError(
      `Unknown colormap '${String(colormap)}'. Available: ${COLORMAP_NAMES.join(", ")}`,
      "colormap",
      colormap
    );
  }
  if (Number.isNaN(value)) return [0, 0, 0];
  const clamped = Math.max(0, Math.min(1, value));

  if (colormap === "grayscale") {
    const g = Math.round(clamped * 255);
    return [g, g, g];
  }

  const cmap = SAMPLED[colormap];
  const n = cmap.length;
  const pos = clamped * (n - 1);
  const idx = Math.min(n - 2, Math.floor(pos));
  const t = pos - idx;

  const c1 = cmap[idx];
  const c2 = cmap[idx + 1];
  if (!c1 || !c2) return [0, 0, 0];

  return [
    Math.round(c1[0] + (c2[0] - c1[0]) * t),
    Math.round(c1[1] + (c2[1] - c1[1]) * t),
    Math.round(c1[2] + (c2[2] - c1[2]) * t),
  ];
}
