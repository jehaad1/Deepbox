/**
 * Example 42: FFT & Signal Processing
 *
 * New in v1.0.0: Fast Fourier Transform (fft, ifft, rfft, irfft, fft2, ifft2, fftn)
 * for spectral analysis, filtering, and signal processing.
 */

import { fft, fft2, ifft, irfft, rfft, tensor } from "deepbox/ndarray";

console.log("=".repeat(60));
console.log("Example 42: FFT & Signal Processing");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Basic FFT — Time Domain to Frequency Domain
// ============================================================================
console.log("\n📊 Part 1: Basic FFT");
console.log("-".repeat(60));

// A simple signal: sum of two sine waves at 5 Hz and 12 Hz
const sampleRate = 64;
const duration = 1; // 1 second
const n = sampleRate * duration;
const signalData: number[] = [];

for (let i = 0; i < n; i++) {
  const t = i / sampleRate;
  // 5 Hz component (amplitude 1.0) + 12 Hz component (amplitude 0.5)
  signalData.push(Math.sin(2 * Math.PI * 5 * t) + 0.5 * Math.sin(2 * Math.PI * 12 * t));
}

const signal = tensor(signalData);
console.log(`Signal: ${n} samples at ${sampleRate} Hz (1 second)`);
console.log(`  Components: 5 Hz (amplitude 1.0) + 12 Hz (amplitude 0.5)`);
console.log(`  Signal shape: ${signal.shape}`);

// Compute FFT
const spectrum = fft(signal);
console.log(
  `  FFT output — real shape: ${spectrum.real.shape}, imag shape: ${spectrum.imag.shape}`
);

// Compute magnitude spectrum
const realData = spectrum.real.data as Float64Array;
const imagData = spectrum.imag.data as Float64Array;
const magnitudes: number[] = [];
for (let i = 0; i < n; i++) {
  const re = Number(realData[spectrum.real.offset + i]);
  const im = Number(imagData[spectrum.imag.offset + i]);
  magnitudes.push(Math.sqrt(re * re + im * im) / n);
}

// Show dominant frequencies (first half of spectrum)
console.log("\n  Frequency spectrum (top peaks):");
const halfN = Math.floor(n / 2);
const peaks: { freq: number; mag: number }[] = [];
for (let i = 1; i < halfN; i++) {
  const freq = (i * sampleRate) / n;
  const mag = magnitudes[i]! * 2; // multiply by 2 for single-sided spectrum
  if (mag > 0.1) {
    peaks.push({ freq, mag });
  }
}
peaks.sort((a, b) => b.mag - a.mag);
for (const peak of peaks.slice(0, 5)) {
  console.log(`    ${peak.freq.toFixed(1)} Hz — magnitude: ${peak.mag.toFixed(4)}`);
}

// ============================================================================
// Part 2: Inverse FFT — Frequency Domain back to Time Domain
// ============================================================================
console.log("\n🔄 Part 2: Inverse FFT (Reconstruction)");
console.log("-".repeat(60));

// Reconstruct the signal from its FFT
const reconstructed = ifft(spectrum.real, spectrum.imag);
console.log("Inverse FFT reconstruction:");
console.log(`  Reconstructed real shape: ${reconstructed.real.shape}`);

// Check reconstruction error
const reconData = reconstructed.real.data as Float64Array;
let maxError = 0;
for (let i = 0; i < n; i++) {
  const original = Number((signal.data as Float64Array)[signal.offset + i]);
  const recon = Number(reconData[reconstructed.real.offset + i]);
  maxError = Math.max(maxError, Math.abs(original - recon));
}
console.log(`  Max reconstruction error: ${maxError.toExponential(4)}`);
console.log("  (Should be near machine epsilon ≈ 1e-15)");

// ============================================================================
// Part 3: Real FFT (rfft) — Optimized for Real Signals
// ============================================================================
console.log("\n⚡ Part 3: Real FFT (rfft)");
console.log("-".repeat(60));

// rfft is optimized for real-valued signals — returns only positive frequencies
const rspec = rfft(signal);
console.log("rfft — optimized for real signals:");
console.log(`  Input shape:  ${signal.shape} (${n} samples)`);
console.log(`  Output real:  ${rspec.real.shape} (${Math.floor(n / 2) + 1} frequencies)`);
console.log(`  Output imag:  ${rspec.imag.shape}`);
console.log("  Only positive frequencies (Nyquist symmetry exploited)");

// Inverse real FFT
const irecon = irfft(rspec.real, rspec.imag);
console.log(`\n  irfft reconstruction shape: ${irecon.real.shape}`);

// ============================================================================
// Part 4: 2D FFT — Image/Matrix Frequency Analysis
// ============================================================================
console.log("\n🖼️  Part 4: 2D FFT");
console.log("-".repeat(60));

// Create a simple 2D pattern (checkerboard-like)
const size = 8;
const pattern2d: number[][] = [];
for (let i = 0; i < size; i++) {
  const row: number[] = [];
  for (let j = 0; j < size; j++) {
    row.push(Math.sin((2 * Math.PI * i) / size) + Math.cos((2 * Math.PI * 2 * j) / size));
  }
  pattern2d.push(row);
}

const matrix = tensor(pattern2d);
console.log(`2D signal shape: ${matrix.shape}`);

const spec2d = fft2(matrix);
console.log(`2D FFT output — real: ${spec2d.real.shape}, imag: ${spec2d.imag.shape}`);

// Compute 2D magnitude
const real2d = spec2d.real.data as Float64Array;
const imag2d = spec2d.imag.data as Float64Array;
let maxMag2d = 0;
let maxI = 0;
let maxJ = 0;
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    const idx = spec2d.real.offset + i * size + j;
    const re = Number(real2d[idx]);
    const im = Number(imag2d[idx]);
    const mag = Math.sqrt(re * re + im * im);
    if (i + j > 0 && mag > maxMag2d) {
      maxMag2d = mag;
      maxI = i;
      maxJ = j;
    }
  }
}
console.log(`  Dominant 2D frequency: (${maxI}, ${maxJ}) with magnitude ${maxMag2d.toFixed(4)}`);

// ============================================================================
// Part 5: Spectral Filtering — Removing High-Frequency Noise
// ============================================================================
console.log("\n🔇 Part 5: Spectral Filtering");
console.log("-".repeat(60));

// Create a noisy signal
const cleanSignalData: number[] = [];
const noisySignalData: number[] = [];
for (let i = 0; i < n; i++) {
  const t = i / sampleRate;
  const clean = Math.sin(2 * Math.PI * 3 * t); // 3 Hz pure signal
  const noise = 0.3 * Math.sin(2 * Math.PI * 25 * t); // 25 Hz noise
  cleanSignalData.push(clean);
  noisySignalData.push(clean + noise);
}

const noisySignal = tensor(noisySignalData);
console.log("Noisy signal: 3 Hz clean + 25 Hz noise");

// FFT the noisy signal
const noisySpectrum = fft(noisySignal);
const noisyReal = new Float64Array(noisySpectrum.real.data as Float64Array);
const noisyImag = new Float64Array(noisySpectrum.imag.data as Float64Array);

// Low-pass filter: zero out frequencies above 10 Hz
const cutoffBin = Math.floor((10 * n) / sampleRate);
for (let i = cutoffBin; i < n - cutoffBin; i++) {
  noisyReal[noisySpectrum.real.offset + i] = 0;
  noisyImag[noisySpectrum.imag.offset + i] = 0;
}

console.log(`  Low-pass filter: cutoff at 10 Hz (bin ${cutoffBin})`);
console.log(`  Zeroed ${n - 2 * cutoffBin} frequency bins`);

// Reconstruct filtered signal
const filteredReal = tensor(Array.from(noisyReal));
const filteredImag = tensor(Array.from(noisyImag));
const filtered = ifft(filteredReal, filteredImag);

// Measure filtering quality
const filteredData = filtered.real.data as Float64Array;
let filterError = 0;
for (let i = 0; i < n; i++) {
  const diff = cleanSignalData[i]! - Number(filteredData[filtered.real.offset + i]);
  filterError += diff * diff;
}
const rmse = Math.sqrt(filterError / n);
console.log(`  RMSE after filtering: ${rmse.toFixed(6)} (lower is better)`);

// ============================================================================
// Part 6: Parseval's Theorem — Energy Conservation
// ============================================================================
console.log("\n⚖️  Part 6: Parseval's Theorem");
console.log("-".repeat(60));

// Parseval's theorem: energy in time domain equals energy in frequency domain
let timeEnergy = 0;
const sigData = signal.data as Float64Array;
for (let i = 0; i < n; i++) {
  const val = Number(sigData[signal.offset + i]);
  timeEnergy += val * val;
}

let freqEnergy = 0;
for (let i = 0; i < n; i++) {
  const re = Number(realData[spectrum.real.offset + i]);
  const im = Number(imagData[spectrum.imag.offset + i]);
  freqEnergy += re * re + im * im;
}
freqEnergy /= n; // normalization

console.log("Parseval's theorem: Σ|x[n]|² = (1/N) Σ|X[k]|²");
console.log(`  Time-domain energy:  ${timeEnergy.toFixed(6)}`);
console.log(`  Freq-domain energy:  ${freqEnergy.toFixed(6)}`);
console.log(`  Difference:          ${Math.abs(timeEnergy - freqEnergy).toExponential(4)}`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• fft/ifft: convert between time and frequency domains");
console.log("• rfft/irfft: optimized for real-valued signals (half the output)");
console.log("• fft2/ifft2: 2D FFT for images and matrices");
console.log("• Spectral filtering: modify frequency components then reconstruct");
console.log("• Parseval's theorem: energy is conserved between domains");
console.log("• FFT complexity: O(N log N) vs O(N²) for naive DFT");

console.log("\n✅ FFT & Signal Processing Example Complete!");
console.log("=".repeat(60));
