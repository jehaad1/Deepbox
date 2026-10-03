# Advanced Visualization

> **View online:** https://deepbox.dev/examples/44-advanced-visualization

Ten plots through the function-style plotting API: line, scatter, bar, histogram, heatmap, confusion matrix, ROC curve, feature importance, elbow curve and residual plot. Each plot is rendered with `show({ format: "svg" })`, and the example reports the size of the SVG text. Example 25 writes SVG files with the `Figure` class.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/plot`    | plot, scatter, bar, hist, heatmap, plotConfusionMatrix, plotRocCurve, plotFeatureImportance, plotElbowCurve, plotResiduals, show |
| `deepbox/ndarray` | tensor, linspace, sin, cos                                                                                                       |

## Usage

```bash
npm run example:44
```

## Output

- Console output only: the length of the SVG text for each plot. No files are written.

## Files

```
44-advanced-visualization/
├── index.ts     # Example script
└── README.md    # This file
```
