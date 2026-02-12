/**
 * Example 21: Random Sampling & Distributions
 *
 * Generate random numbers from various probability distributions.
 * Useful for simulations, Monte Carlo methods, and data generation.
 */

import {
  beta,
  binomial,
  exponential,
  gamma,
  normal,
  poisson,
  rand,
  randint,
  randn,
  setSeed,
  uniform,
} from "deepbox/random";

console.log("=== Random Sampling & Distributions ===\n");

// Set seed for reproducibility
setSeed(42);
console.log("Random seed set to 42 for reproducibility\n");

// Uniform distribution [0, 1)
console.log("1. Uniform Distribution [0, 1):");
const uniformSamples = rand([5]);
console.log(`${uniformSamples.toString()}\n`);

// Standard normal distribution
console.log("2. Standard Normal Distribution (mean=0, std=1):");
const normalSamples = randn([5]);
console.log(`${normalSamples.toString()}\n`);

// Random integers
console.log("3. Random Integers [0, 10):");
const intSamples = randint(0, 10, [8]);
console.log(`${intSamples.toString()}\n`);

// Custom uniform distribution
console.log("4. Uniform Distribution [-5, 5]:");
const customUniform = uniform(-5, 5, [6]);
console.log(`${customUniform.toString()}\n`);

// Custom normal distribution
console.log("5. Normal Distribution (mean=100, std=15):");
const customNormal = normal(100, 15, [6]);
console.log(`${customNormal.toString()}\n`);

// Binomial distribution (coin flips)
console.log("6. Binomial Distribution (n=10, p=0.5):");
const binomialSamples = binomial(10, 0.5, [8]);
console.log(binomialSamples.toString());
console.log("(Number of heads in 10 coin flips)\n");

// Poisson distribution
console.log("7. Poisson Distribution (λ=3):");
const poissonSamples = poisson(3, [8]);
console.log(poissonSamples.toString());
console.log("(Number of events with rate λ=3)\n");

// Exponential distribution
console.log("8. Exponential Distribution (scale=2):");
const expSamples = exponential(2, [6]);
console.log(expSamples.toString());
console.log("(Time between events)\n");

// Gamma distribution
console.log("9. Gamma Distribution (shape=2, scale=2):");
const gammaSamples = gamma(2, 2, [6]);
console.log(`${gammaSamples.toString()}\n`);

// Beta distribution
console.log("10. Beta Distribution (α=2, β=5):");
const betaSamples = beta(2, 5, [6]);
console.log(betaSamples.toString());
console.log("(Values between 0 and 1)\n");

console.log("✓ Random sampling complete!");
