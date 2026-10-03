/**
 * Example 21: Random Sampling & Distributions
 *
 * Draw random numbers from common probability distributions. Calling
 * setSeed() makes every later draw reproducible, which is what you want for
 * simulations, tests and repeatable experiments.
 */

import { tensor } from "deepbox/ndarray";
import {
  beta,
  binomial,
  choice,
  exponential,
  Generator,
  gamma,
  multivariateNormal,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  randn,
  setSeed,
  uniform,
} from "deepbox/random";

console.log("=== Random Sampling & Distributions ===\n");

// Set the seed for reproducibility
setSeed(42);
console.log("Random seed set to 42 for reproducibility\n");

// Seeding again with the same value repeats the same numbers.
const firstRun = rand([3]);
setSeed(42);
const secondRun = rand([3]);
console.log(`Same seed, same draws: ${Number(firstRun.eq(secondRun).all().item()) === 1}\n`);

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
console.log("7. Poisson Distribution (lambda = 3):");
const poissonSamples = poisson(3, [8]);
console.log(poissonSamples.toString());
console.log("(Number of events with rate lambda = 3)\n");

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
console.log("10. Beta Distribution (alpha = 2, beta = 5):");
const betaSamples = beta(2, 5, [6]);
console.log(betaSamples.toString());
console.log("(Values between 0 and 1)\n");

// Correlated samples from a multivariate normal. Each row is one draw.
console.log("11. Multivariate Normal (mean [0, 0], covariance [[1, 0.8], [0.8, 1]]):");
const mvn = multivariateNormal(
  tensor([0, 0]),
  tensor([
    [1, 0.8],
    [0.8, 1],
  ]),
  3
);
console.log(`${mvn.toString()}\n`);

// Sample from a list, with and without replacement, and shuffle.
console.log("12. Choice and Permutation:");
const items = tensor([10, 20, 30, 40, 50]);
console.log(`With replacement:    ${choice(items, 4).toString()}`);
console.log(`Without replacement: ${choice(items, 4, false).toString()}`);
console.log(`Shuffled copy:       ${permutation(items).toString()}`);
console.log(`Shuffled 0..4:       ${permutation(5).toString()}\n`);

// A Generator has its own state, so it does not disturb the global seed.
console.log("13. Independent Generator:");
const rngA = new Generator(7);
const rngB = new Generator(7);
console.log(`Two generators with seed 7 agree: ${rngA.random() === rngB.random()}`);
