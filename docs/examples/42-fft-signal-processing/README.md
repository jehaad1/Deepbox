# FFT & Signal Processing

> **View online:** https://deepbox.dev/examples/42-fft-signal-processing

Uses `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2` and `rfftfreq` to find the frequencies in a signal, rebuild it, and remove high-frequency noise with a low-pass filter. It also checks Parseval's theorem. Each transform returns an object with a `real` and an `imag` tensor.

## Deepbox Modules Used

| Module            | Features Used                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ndarray` | `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `rfftfreq`, `tensor`, fluent `square`, `sqrt`, `sub`, `mean`, `item` |

## Usage

```bash
npm run example:42
```

## Output

- Console output only: the peaks found in a two-tone signal, reconstruction errors, rfft sizes and bin frequencies, the strongest 2D frequency, the RMSE before and after low-pass filtering, and both sides of Parseval's identity.
- The signals are created as `float64`. `tensor()` defaults to `float32`, and float32 input gives float32 output, so reconstruction errors would be about 1e-7 instead of 1e-15.

## Files

```
42-fft-signal-processing/
├── index.ts     # Example script
└── README.md    # This file
```
