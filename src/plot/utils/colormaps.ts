type ColormapName = "viridis" | "plasma" | "inferno" | "magma" | "grayscale";

const colormaps: Record<ColormapName, readonly [number, number, number][]> = {
  viridis: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 73, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [181, 222, 43],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [75, 3, 161],
    [125, 3, 168],
    [168, 34, 150],
    [203, 70, 121],
    [229, 107, 93],
    [248, 148, 65],
    [253, 195, 40],
    [240, 249, 33],
    [240, 249, 33],
  ],
  inferno: [
    [0, 0, 4],
    [40, 11, 84],
    [101, 21, 110],
    [159, 42, 99],
    [212, 72, 66],
    [245, 125, 21],
    [250, 193, 39],
    [245, 251, 161],
    [252, 255, 164],
    [252, 255, 164],
  ],
  magma: [
    [0, 0, 4],
    [28, 16, 68],
    [79, 18, 123],
    [129, 37, 129],
    [181, 54, 122],
    [229, 80, 100],
    [251, 135, 97],
    [254, 194, 135],
    [252, 253, 191],
    [252, 253, 191],
  ],
  grayscale: [
    [0, 0, 0],
    [28, 28, 28],
    [57, 57, 57],
    [85, 85, 85],
    [113, 113, 113],
    [142, 142, 142],
    [170, 170, 170],
    [198, 198, 198],
    [227, 227, 227],
    [255, 255, 255],
  ],
};

/**
 * Maps a normalized value to RGB using colormap.
 * @internal
 */
export function applyColormap(value: number, colormap: ColormapName): [number, number, number] {
  const cmap = colormaps[colormap];
  const n = cmap.length;
  const clamped = Math.max(0, Math.min(1, value));
  const idx = Math.max(0, Math.min(n - 2, Math.floor(clamped * (n - 1))));
  const nextIdx = idx + 1;
  const t = Math.max(0, Math.min(1, clamped * (n - 1) - idx));

  const c1 = cmap[idx];
  const c2 = cmap[nextIdx];
  if (!c1 || !c2) return [0, 0, 0];

  return [
    Math.round(c1[0] + (c2[0] - c1[0]) * t),
    Math.round(c1[1] + (c2[1] - c1[1]) * t),
    Math.round(c1[2] + (c2[2] - c1[2]) * t),
  ];
}
