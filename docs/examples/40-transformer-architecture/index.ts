/**
 * Example 40: Transformer Architecture
 *
 * The building blocks of a Transformer: MultiheadAttention, encoder and decoder
 * layers and stacks, PositionalEncoding, and a small encoder-decoder pipeline
 * from token ids to vocabulary logits.
 *
 * Every forward pass below is inference only, so it runs inside noGrad() and
 * returns a plain tensor. Without noGrad(), layers with trainable weights return
 * a GradTensor that tracks the weights, ready for backward().
 */

import { noGrad, randn, tensor } from "deepbox/ndarray";
import {
  causalMask,
  Embedding,
  Linear,
  MultiheadAttention,
  PositionalEncoding,
  TransformerDecoder,
  TransformerDecoderLayer,
  TransformerEncoder,
  TransformerEncoderLayer,
} from "deepbox/nn";

console.log("=".repeat(60));
console.log("Example 40: Transformer Architecture");
console.log("=".repeat(60));

// ============================================================================
// Part 1: MultiheadAttention
// ============================================================================
console.log("\nPart 1: MultiheadAttention");
console.log("-".repeat(60));

// MultiheadAttention splits the embedding into heads. Each head attends in its own subspace.
const embedDim = 16;
const numHeads = 4;

const mha = new MultiheadAttention(embedDim, numHeads);

// Input: (batch=2, seqLen=5, embedDim=16)
const query = randn([2, 5, embedDim]);
const key = randn([2, 5, embedDim]);
const value = randn([2, 5, embedDim]);

console.log(`MultiheadAttention(embedDim=${embedDim}, numHeads=${numHeads}):`);
console.log(`  Query shape:  [${query.shape.join(", ")}]`);
console.log(`  Key shape:    [${key.shape.join(", ")}]`);
console.log(`  Value shape:  [${value.shape.join(", ")}]`);

const attnOutput = noGrad(() => mha.forward(query, key, value));
console.log(`  Output shape: [${attnOutput.shape.join(", ")}]`);
console.log(`  Each head works on ${embedDim / numHeads} of the ${embedDim} dimensions`);

// Self-attention: query = key = value
console.log("\n  Self-attention (Q=K=V):");
const selfAttnOut = noGrad(() => mha.forward(query, query, query));
console.log(`  Output shape: [${selfAttnOut.shape.join(", ")}]`);

// ============================================================================
// Part 2: TransformerEncoderLayer
// ============================================================================
console.log("\nPart 2: TransformerEncoderLayer");
console.log("-".repeat(60));

// One encoder layer: self-attention, then a feed-forward network, each with a residual connection and LayerNorm
const dModel = 16;
const nHead = 4;
const dFF = 64;

const encoderLayer = new TransformerEncoderLayer({
  dModel,
  nHead,
  dFF,
  dropout: 0.1,
});

console.log(`TransformerEncoderLayer(dModel=${dModel}, nHead=${nHead}, dFF=${dFF}):`);

// Input: (batch=2, seqLen=8, dModel=16)
const encoderInput = randn([2, 8, dModel]);
console.log(`  Input shape:  [${encoderInput.shape.join(", ")}]`);

const encoderLayerOutput = noGrad(() => encoderLayer.forward(encoderInput));
console.log(`  Output shape: [${encoderLayerOutput.shape.join(", ")}]`);
console.log("  Components: self-attention, add and norm, feed-forward, add and norm");

// ============================================================================
// Part 3: TransformerEncoder (Stacked Layers)
// ============================================================================
console.log("\nPart 3: TransformerEncoder");
console.log("-".repeat(60));

// TransformerEncoder stacks copies of one encoder layer
const numEncoderLayers = 3;
const encoderLayerTemplate = new TransformerEncoderLayer({
  dModel,
  nHead,
  dFF,
  dropout: 0.1,
});

const encoder = new TransformerEncoder(encoderLayerTemplate, numEncoderLayers);

console.log(`TransformerEncoder(numLayers=${numEncoderLayers}):`);
const encSrc = randn([2, 10, dModel]);
console.log(`  Input shape:  [${encSrc.shape.join(", ")}]`);

const encoderOutput = noGrad(() => encoder.forward(encSrc));
console.log(`  Output shape: [${encoderOutput.shape.join(", ")}]`);

// Count parameters
let encParams = 0;
for (const [, p] of encoder.namedParameters()) {
  encParams += p.size;
}
console.log(`  Total parameters: ${encParams}`);

// ============================================================================
// Part 4: TransformerDecoderLayer
// ============================================================================
console.log("\nPart 4: TransformerDecoderLayer");
console.log("-".repeat(60));

// A decoder layer adds cross-attention over the encoder output (the memory)
const decoderLayer = new TransformerDecoderLayer(dModel, nHead, dFF, {
  dropout: 0.1,
});

console.log(`TransformerDecoderLayer(dModel=${dModel}, nHead=${nHead}, dFF=${dFF}):`);

// Target input: (batch=2, tgtLen=6, dModel=16)
const tgt = randn([2, 6, dModel]);
// Memory from encoder: (batch=2, srcLen=10, dModel=16)
const memory = encoderOutput;

console.log(`  Target shape: [${tgt.shape.join(", ")}]`);
console.log(`  Memory shape: [${memory.shape.join(", ")}]`);

const decoderLayerOutput = noGrad(() => decoderLayer.forward(tgt, memory));
console.log(`  Output shape: [${decoderLayerOutput.shape.join(", ")}]`);
console.log(
  "  Components: self-attention, cross-attention and feed-forward, each followed by add and norm"
);

// ============================================================================
// Part 5: TransformerDecoder (Stacked Layers)
// ============================================================================
console.log("\nPart 5: TransformerDecoder");
console.log("-".repeat(60));

const numDecoderLayers = 3;
const decoderLayerTemplate = new TransformerDecoderLayer(dModel, nHead, dFF, {
  dropout: 0.1,
});

const decoder = new TransformerDecoder(decoderLayerTemplate, numDecoderLayers);

console.log(`TransformerDecoder(numLayers=${numDecoderLayers}):`);
const decTgt = randn([2, 6, dModel]);
console.log(`  Target shape: [${decTgt.shape.join(", ")}]`);
console.log(`  Memory shape: [${memory.shape.join(", ")}]`);

// causalMask(n) stops position i from attending to later target positions
const decoderOutput = noGrad(() => decoder.forward(decTgt, memory, causalMask(6)));
console.log(`  Output shape: [${decoderOutput.shape.join(", ")}]`);

let decParams = 0;
for (const [, p] of decoder.namedParameters()) {
  decParams += p.size;
}
console.log(`  Total parameters: ${decParams}`);

// ============================================================================
// Part 6: PositionalEncoding
// ============================================================================
console.log("\nPart 6: PositionalEncoding");
console.log("-".repeat(60));

// PositionalEncoding adds a fixed sinusoidal pattern that tells the model where each token sits
const pe = new PositionalEncoding(dModel, { dropout: 0, maxLen: 100 });

console.log(`PositionalEncoding(dModel=${dModel}, maxLen=100):`);

const seqInput = randn([2, 8, dModel]);
console.log(`  Input shape:  [${seqInput.shape.join(", ")}]`);

const peOutput = pe.forward(seqInput);
console.log(`  Output shape: [${peOutput.shape.join(", ")}]`);
console.log("  Even dimensions: sin(pos / 10000^(2i/dModel))");
console.log("  Odd dimensions:  cos(pos / 10000^(2i/dModel))");

// ============================================================================
// Part 7: Complete Transformer Pipeline
// ============================================================================
console.log("\nPart 7: Complete Transformer Pipeline");
console.log("-".repeat(60));

// A small sequence-to-sequence transformer, from token ids to vocabulary logits
const vocabSize = 100;
const seqLen = 12;

// Token embeddings: one learned vector per vocabulary entry
const srcEmbedding = new Embedding(vocabSize, dModel);
const tgtEmbedding = new Embedding(vocabSize, dModel);

// Positional encodings, one per side
const srcPE = new PositionalEncoding(dModel, { dropout: 0.1, maxLen: seqLen });
const tgtPE = new PositionalEncoding(dModel, { dropout: 0.1, maxLen: seqLen });

// Encoder
const fullEncoderLayer = new TransformerEncoderLayer({
  dModel,
  nHead,
  dFF,
  dropout: 0.1,
});
const fullEncoder = new TransformerEncoder(fullEncoderLayer, 2);

// Decoder
const fullDecoderLayer = new TransformerDecoderLayer(dModel, nHead, dFF, {
  dropout: 0.1,
});
const fullDecoder = new TransformerDecoder(fullDecoderLayer, 2);

// Output projection
const outputProj = new Linear(dModel, vocabSize);

console.log("Complete Transformer Architecture:");
console.log(`  Vocabulary: ${vocabSize} tokens`);
console.log(`  Model dim: ${dModel}, Heads: ${nHead}, FF dim: ${dFF}`);
console.log("  Encoder layers: 2, Decoder layers: 2");

// Forward pass
// Source tokens: (batch=2, srcLen=12)
const srcTokens = tensor(
  [
    [1, 5, 23, 42, 7, 15, 33, 8, 19, 2, 0, 0],
    [1, 12, 45, 3, 28, 9, 2, 0, 0, 0, 0, 0],
  ],
  { dtype: "int32" }
);

// Target tokens: (batch=2, tgtLen=8)
const tgtTokens = tensor(
  [
    [1, 10, 25, 37, 14, 6, 2, 0],
    [1, 18, 44, 22, 2, 0, 0, 0],
  ],
  { dtype: "int32" }
);

console.log(`\n  Source tokens shape: [${srcTokens.shape.join(", ")}]`);
console.log(`  Target tokens shape: [${tgtTokens.shape.join(", ")}]`);

// Step 1: Embed the source tokens (scaled by sqrt(dModel)) and add positions
const srcEmbedded = srcPE.forward(
  noGrad(() => srcEmbedding.forward(srcTokens)).mul(Math.sqrt(dModel))
);
console.log(`  Source embedded shape: [${srcEmbedded.shape.join(", ")}]`);

// Step 2: Encode
const fullEncoderOutput = noGrad(() => fullEncoder.forward(srcEmbedded));
console.log(`  Encoder output shape: [${fullEncoderOutput.shape.join(", ")}]`);

// Step 3: Embed the target tokens and add positions
const tgtEmbedded = tgtPE.forward(
  noGrad(() => tgtEmbedding.forward(tgtTokens)).mul(Math.sqrt(dModel))
);
console.log(`  Target embedded shape: [${tgtEmbedded.shape.join(", ")}]`);

// Step 4: Decode with a causal mask and cross-attention to the encoder output
const fullDecoderOutput = noGrad(() =>
  fullDecoder.forward(tgtEmbedded, fullEncoderOutput, causalMask(8))
);
console.log(`  Decoder output shape: [${fullDecoderOutput.shape.join(", ")}]`);

// Step 5: Project to vocabulary logits
const logits = noGrad(() => outputProj.forward(fullDecoderOutput));
console.log(`  Logits shape: [${logits.shape.join(", ")}] (batch, tgtLen, vocab)`);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• MultiheadAttention: attention in several subspaces at once");
console.log("• TransformerEncoderLayer: self-attention and feed-forward, with residuals");
console.log("• TransformerDecoderLayer: self-attention, cross-attention and feed-forward");
console.log("• TransformerEncoder, TransformerDecoder: a stack of N copies of one layer");
console.log("• PositionalEncoding: fixed sinusoidal position information");
console.log("• causalMask(n): keeps the decoder from looking at later positions");
console.log("• Pipeline: embed, add positions, encode, decode, project to logits");
console.log("• Outside noGrad(), the same calls return GradTensors that support backward()");

console.log("\nTransformer Architecture Example Complete!");
console.log("=".repeat(60));
