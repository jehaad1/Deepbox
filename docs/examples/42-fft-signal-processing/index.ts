/**
 * Example 42: FFT & Signal Processing
 *
 * Fast Fourier transforms (fft, ifft, rfft, irfft, fft2, rfftfreq) used for
 * spectral analysis and low-pass filtering. Each transform returns an object
 * with a real and an imaginary tensor.
 *
 * The signals are float64 so that reconstruction errors are close to machine
 * precision. tensor() defaults to float32, and float32 input gives float32 output.
 */

import {
  fft,
  fft2,
  ifft,
  ifft2,
  irfft,
  rfft,
  rfftfreq,
  type Tensor,
  tensor,
} from "deepbox/ndarray";

console.log("=".repeat(60));
console.log("Example 42: FFT & Signal Processing");
console.log("=".repeat(60));

// Magnitude of a complex result: sqrt(real^2 + imag^2)
const magnitude = (z: { real: Tensor; imag: Tensor }): Tensor =>
  z.real.square().add(z.imag.square()).sqrt();

// ============================================================================
// Part 1: Basic FFT (time domain to frequency domain)
// ============================================================================
console.log("\nPart 1: Basic FFT");
console.log("-".repeat(60));

// A signal made of two sine waves: 5 Hz with amplitude 1.0 and 12 Hz with amplitude 0.5
const sampleRate = 64;
const n = 64; // one second of samples
const signalData: number[] = [];

for (let i = 0; i < n; i++) {
  const t = i / sampleRate;
  signalData.push(Math.sin(2 * Math.PI * 5 * t) + 0.5 * Math.sin(2 * Math.PI * 12 * t));
}

const signal = tensor(signalData, { dtype: "float64" });
console.log(`Signal: ${n} samples at ${sampleRate} Hz (1 second)`);
console.log("  Components: 5 Hz (amplitude 1.0) + 12 Hz (amplitude 0.5)");
console.log(`  Signal shape: [${signal.shape.join(", ")}]`);

const spectrum = fft(signal);
console.log(
  `  FFT output: real shape [${spectrum.real.shape.join(", ")}], imag shape [${spectrum.imag.shape.join(", ")}]`
);

// Amplitude of each frequency bin. Dividing by n undoes the FFT scaling, and the
// factor 2 folds the mirrored negative frequencies into the positive ones.
const amplitudes = magnitude(spectrum).div(n).mul(2).toArray() as number[];

console.log("\n  Frequency spectrum (peaks above 0.1):");
const peaks: { freq: number; amplitude: number }[] = [];
for (let i = 1; i < n / 2; i++) {
  const amplitude = amplitudes[i] ?? 0;
  if (amplitude > 0.1) {
    peaks.push({ freq: (i * sampleRate) / n, amplitude });
  }
}
peaks.sort((a, b) => b.amplitude - a.amplitude);
for (const peak of peaks.slice(0, 5)) {
  console.log(`    ${peak.freq.toFixed(1)} Hz, amplitude: ${peak.amplitude.toFixed(4)}`);
}

// ============================================================================
// Part 2: Inverse FFT (frequency domain back to time domain)
// ============================================================================
console.log("\nPart 2: Inverse FFT (Reconstruction)");
console.log("-".repeat(60));

const reconstructed = ifft(spectrum.real, spectrum.imag);
console.log("Inverse FFT reconstruction:");
console.log(`  Reconstructed real shape: [${reconstructed.real.shape.join(", ")}]`);

const maxError = Number(signal.sub(reconstructed.real).abs().max().item());
console.log(`  Max reconstruction error: ${maxError.toExponential(4)}`);
console.log("  (about 1e-15 for float64, about 1e-7 for float32)");

// ============================================================================
// Part 3: Real FFT (rfft) for real-valued signals
// ============================================================================
console.log("\nPart 3: Real FFT (rfft)");
console.log("-".repeat(60));

// A real signal has a mirrored spectrum, so rfft returns only the n/2 + 1 non-negative frequencies
const rspec = rfft(signal);
console.log("rfft keeps only the non-negative frequencies:");
console.log(`  Input shape:  [${signal.shape.join(", ")}] (${n} samples)`);
console.log(`  Output real:  [${rspec.real.shape.join(", ")}] (${n / 2 + 1} frequencies)`);
console.log(`  Output imag:  [${rspec.imag.shape.join(", ")}]`);

// rfftfreq gives the frequency in Hz of each rfft bin, for a sample spacing of 1 / sampleRate
const freqs = rfftfreq(n, 1 / sampleRate).toArray() as number[];
console.log(
  `  Bin frequencies: ${freqs[0]} Hz to ${freqs[freqs.length - 1]} Hz in steps of ${freqs[1]} Hz`
);

// irfft goes back to the original length
const irecon = irfft(rspec.real, rspec.imag);
const irfftError = Number(signal.sub(irecon.real).abs().max().item());
console.log(
  `  irfft output shape: [${irecon.real.shape.join(", ")}], max error: ${irfftError.toExponential(4)}`
);

// ============================================================================
// Part 4: 2D FFT (image or matrix frequency analysis)
// ============================================================================
console.log("\nPart 4: 2D FFT");
console.log("-".repeat(60));

// A pattern that varies once per 8 rows (sine) and twice per 8 columns (cosine)
const size = 8;
const pattern2d: number[][] = [];
for (let i = 0; i < size; i++) {
  const row: number[] = [];
  for (let j = 0; j < size; j++) {
    row.push(Math.sin((2 * Math.PI * i) / size) + Math.cos((2 * Math.PI * 2 * j) / size));
  }
  pattern2d.push(row);
}

const matrix = tensor(pattern2d, { dtype: "float64" });
console.log(`2D signal shape: [${matrix.shape.join(", ")}]`);

const spec2d = fft2(matrix);
console.log(
  `2D FFT output: real [${spec2d.real.shape.join(", ")}], imag [${spec2d.imag.shape.join(", ")}]`
);

// Find the strongest non-constant frequency, skipping the (0, 0) bin
const mag2d = magnitude(spec2d).toArray() as number[][];
let maxMag2d = 0;
let maxI = 0;
let maxJ = 0;
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    const mag = mag2d[i]?.[j] ?? 0;
    if (i + j > 0 && mag > maxMag2d) {
      maxMag2d = mag;
      maxI = i;
      maxJ = j;
    }
  }
}
console.log(`  Strongest 2D frequency: (${maxI}, ${maxJ}) with magnitude ${maxMag2d.toFixed(4)}`);

const back2d = ifft2(spec2d.real, spec2d.imag);
console.log(
  `  ifft2 max error: ${Number(matrix.sub(back2d.real).abs().max().item()).toExponential(4)}`
);

// ============================================================================
// Part 5: Spectral filtering (removing high-frequency noise)
// ============================================================================
console.log("\nPart 5: Spectral Filtering");
console.log("-".repeat(60));

const cleanData: number[] = [];
const noisyData: number[] = [];
for (let i = 0; i < n; i++) {
  const t = i / sampleRate;
  const clean = Math.sin(2 * Math.PI * 3 * t); // 3 Hz signal
  const noise = 0.3 * Math.sin(2 * Math.PI * 25 * t); // 25 Hz noise
  cleanData.push(clean);
  noisyData.push(clean + noise);
}

const cleanSignal = tensor(cleanData, { dtype: "float64" });
const noisySignal = tensor(noisyData, { dtype: "float64" });
console.log("Noisy signal: 3 Hz clean + 25 Hz noise");

const rmse = (a: Tensor, b: Tensor): number => Number(a.sub(b).square().mean().sqrt().item());
console.log(`  RMSE of the noisy signal: ${rmse(cleanSignal, noisySignal).toFixed(6)}`);

// Low-pass filter: keep the bins up to 10 Hz and their mirrored negative-frequency bins
const noisySpectrum = fft(noisySignal);
const cutoffBin = Math.floor((10 * n) / sampleRate);
const keep = tensor(
  Array.from({ length: n }, (_, k) => (k <= cutoffBin || k >= n - cutoffBin ? 1 : 0)),
  { dtype: "float64" }
);

console.log(`  Low-pass filter: cutoff at 10 Hz (bin ${cutoffBin})`);
console.log(`  Zeroed ${n - (2 * cutoffBin + 1)} frequency bins`);

const filtered = ifft(noisySpectrum.real.mul(keep), noisySpectrum.imag.mul(keep));
console.log(
  `  RMSE after filtering: ${rmse(cleanSignal, filtered.real).toFixed(6)} (lower is better)`
);

// ============================================================================
// Part 6: Parseval's theorem (energy is conserved)
// ============================================================================
console.log("\nPart 6: Parseval's Theorem");
console.log("-".repeat(60));

// The energy of the signal equals the energy of its spectrum divided by n
const timeEnergy = Number(signal.square().sum().item());
const freqEnergy = Number(magnitude(spectrum).square().sum().item()) / n;

console.log("Parseval's theorem: sum |x[n]|^2 = (1/N) sum |X[k]|^2");
console.log(`  Time-domain energy:  ${timeEnergy.toFixed(6)}`);
console.log(`  Freq-domain energy:  ${freqEnergy.toFixed(6)}`);
console.log(`  Difference:          ${Math.abs(timeEnergy - freqEnergy).toExponential(4)}`);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• fft, ifft: convert between the time and frequency domains");
console.log("• rfft, irfft: for real signals, half the output size");
console.log("• fft2, ifft2: the same for images and matrices");
console.log("• rfftfreq: the frequency in Hz of each rfft bin");
console.log("• Filtering: zero some bins, then invert the transform");
console.log("• Parseval's theorem: the energy is the same in both domains");
console.log("• Use float64 input when you need results near machine precision");

console.log("\nFFT & Signal Processing Example Complete!");
console.log("=".repeat(60));
