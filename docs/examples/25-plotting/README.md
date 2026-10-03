# Data Visualization

> **View online:** https://deepbox.dev/examples/25-plotting

Create line, scatter, bar, histogram and heatmap plots and save each one as an SVG file. `Figure.renderSVG()` returns the SVG text. `Figure.renderPNG()` renders PNG in Node.js.

## Deepbox Modules Used

| Module            | Features Used                                  |
| ----------------- | ---------------------------------------------- |
| `deepbox/ndarray` | tensor, linspace, sin, cos                     |
| `deepbox/plot`    | Figure, Axes.plot, scatter, bar, hist, heatmap |

## Usage

```bash
npm run example:25
```

## Output

- Five SVG files in `output/`:
  - `line-plot.svg`: sine and cosine curves
  - `scatter-plot.svg`: scatter plot of a noisy linear trend
  - `bar-chart.svg`: bar chart of five categories
  - `histogram.svg`: histogram of 25 values in 9 bins
  - `heatmap.svg`: heatmap of a 3 by 4 matrix
