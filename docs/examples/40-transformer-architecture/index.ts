/**
 * Example 40: Transformer Architecture
 *
 * New in v1.0.0: Full Transformer implementation including MultiheadAttention,
 * TransformerEncoder/Decoder layers, PositionalEncoding, and complete
 * encoder-decoder models for sequence-to-sequence tasks.
 */

import { randn, tensor } from "deepbox/ndarray";
import {
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

function toFloat32PositionalEncoding(
  dModel: number,
  maxLen: number,
  dropout: number
): PositionalEncoding {
  const pe = new PositionalEncoding(dModel, { dropout, maxLen });
  const table: number[][] = [];

  for (let pos = 0; pos < maxLen; pos++) {
    const row: number[] = [];
    for (let i = 0; i < dModel; i++) {
      const angle = pos / 10000 ** ((2 * Math.floor(i / 2)) / dModel);
      row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    table.push(row);
  }

  // Align the internal buffer dtype with the rest of the example's float32 tensors.
  (pe as unknown as { peBuffer: ReturnType<typeof tensor> }).peBuffer = tensor(table, {
    dtype: "float32",
  });

  return pe;
}

// ============================================================================
// Part 1: MultiheadAttention
// ============================================================================
console.log("\n🔍 Part 1: MultiheadAttention");
console.log("-".repeat(60));

// MultiheadAttention splits the embedding into multiple heads for parallel attention
const embedDim = 16;
const numHeads = 4;

const mha = new MultiheadAttention(embedDim, numHeads);

// Input: (batch=2, seqLen=5, embedDim=16)
const query = randn([2, 5, embedDim]);
const key = randn([2, 5, embedDim]);
const value = randn([2, 5, embedDim]);

console.log(`MultiheadAttention(embedDim=${embedDim}, numHeads=${numHeads}):`);
console.log(`  Query shape:  ${query.shape}`);
console.log(`  Key shape:    ${key.shape}`);
console.log(`  Value shape:  ${value.shape}`);

const attnOutput = mha.forward(query, key, value);
console.log(`  Output shape: ${attnOutput.shape}`);
console.log(`  Each head attends to ${embedDim / numHeads}D subspace`);

// Self-attention: query = key = value
console.log("\n  Self-attention (Q=K=V):");
const selfAttnOut = mha.forward(query, query, query);
console.log(`  Output shape: ${selfAttnOut.shape}`);

// ============================================================================
// Part 2: TransformerEncoderLayer
// ============================================================================
console.log("\n📦 Part 2: TransformerEncoderLayer");
console.log("-".repeat(60));

// A single encoder layer: self-attention + feedforward with residual connections
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
console.log(`  Input shape:  ${encoderInput.shape}`);

const encoderLayerOutput = encoderLayer.forward(encoderInput);
console.log(`  Output shape: ${encoderLayerOutput.shape}`);
console.log("  Components: SelfAttention → Add&Norm → FeedForward → Add&Norm");

// ============================================================================
// Part 3: TransformerEncoder (Stacked Layers)
// ============================================================================
console.log("\n🏗️  Part 3: TransformerEncoder");
console.log("-".repeat(60));

// Stack multiple encoder layers for deeper representations
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
console.log(`  Input shape:  ${encSrc.shape}`);

const encoderOutput = encoder.forward(encSrc);
console.log(`  Output shape: ${encoderOutput.shape}`);

// Count parameters
let encParams = 0;
for (const [, p] of encoder.namedParameters()) {
  encParams += p.size;
}
console.log(`  Total parameters: ${encParams}`);

// ============================================================================
// Part 4: TransformerDecoderLayer
// ============================================================================
console.log("\n📦 Part 4: TransformerDecoderLayer");
console.log("-".repeat(60));

// Decoder layer: self-attention + cross-attention + feedforward
const decoderLayer = new TransformerDecoderLayer(dModel, nHead, dFF, {
  dropout: 0.1,
});

console.log(`TransformerDecoderLayer(dModel=${dModel}, nHead=${nHead}, dFF=${dFF}):`);

// Target input: (batch=2, tgtLen=6, dModel=16)
const tgt = randn([2, 6, dModel]);
// Memory from encoder: (batch=2, srcLen=10, dModel=16)
const memory = encoderOutput;

console.log(`  Target shape: ${tgt.shape}`);
console.log(`  Memory shape: ${memory.shape}`);

const decoderLayerOutput = decoderLayer.forward(tgt, memory);
console.log(`  Output shape: ${decoderLayerOutput.shape}`);
console.log("  Components: SelfAttention → Add&Norm → CrossAttention → Add&Norm → FF → Add&Norm");

// ============================================================================
// Part 5: TransformerDecoder (Stacked Layers)
// ============================================================================
console.log("\n🏗️  Part 5: TransformerDecoder");
console.log("-".repeat(60));

const numDecoderLayers = 3;
const decoderLayerTemplate = new TransformerDecoderLayer(dModel, nHead, dFF, {
  dropout: 0.1,
});

const decoder = new TransformerDecoder(decoderLayerTemplate, numDecoderLayers);

console.log(`TransformerDecoder(numLayers=${numDecoderLayers}):`);
const decTgt = randn([2, 6, dModel]);
console.log(`  Target shape: ${decTgt.shape}`);
console.log(`  Memory shape: ${memory.shape}`);

const decoderOutput = decoder.forward(decTgt, memory);
console.log(`  Output shape: ${decoderOutput.shape}`);

let decParams = 0;
for (const [, p] of decoder.namedParameters()) {
  decParams += p.size;
}
console.log(`  Total parameters: ${decParams}`);

// ============================================================================
// Part 6: PositionalEncoding
// ============================================================================
console.log("\n🌊 Part 6: PositionalEncoding");
console.log("-".repeat(60));

// PositionalEncoding adds sinusoidal position information to embeddings
const pe = toFloat32PositionalEncoding(dModel, 100, 0.0);

console.log(`PositionalEncoding(dModel=${dModel}, maxLen=100):`);

const seqInput = randn([2, 8, dModel]);
console.log(`  Input shape:  ${seqInput.shape}`);

const peOutput = pe.forward(seqInput);
console.log(`  Output shape: ${peOutput.shape}`);
console.log("  Adds sin/cos positional information to token embeddings");
console.log("  Even dimensions: sin(pos / 10000^(2i/d_model))");
console.log("  Odd dimensions:  cos(pos / 10000^(2i/d_model))");

// ============================================================================
// Part 7: Complete Transformer Pipeline
// ============================================================================
console.log("\n🔄 Part 7: Complete Transformer Pipeline");
console.log("-".repeat(60));

// Build a complete sequence-to-sequence transformer
const vocabSize = 100;
const seqLen = 12;
const batchSize = 2;

// Source positional encoding for embedded token representations
const srcPE = toFloat32PositionalEncoding(dModel, seqLen, 0.1);

// Target positional encoding for decoder token representations
const tgtPE = toFloat32PositionalEncoding(dModel, seqLen, 0.1);

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
console.log(`  Encoder layers: 2, Decoder layers: 2`);

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

console.log(`\n  Source tokens shape: ${srcTokens.shape}`);
console.log(`  Target tokens shape: ${tgtTokens.shape}`);

// Step 1: Start from embedded source token representations + positional encoding
const srcTokenEmbeddings = randn([batchSize, seqLen, dModel]);
const srcEmbedded = srcPE.forward(srcTokenEmbeddings);
console.log(`  Source embedded shape: ${srcEmbedded.shape}`);

// Step 2: Encode
const fullEncoderOutput = fullEncoder.forward(srcEmbedded);
console.log(`  Encoder output shape: ${fullEncoderOutput.shape}`);

// Step 3: Start from embedded target token representations + positional encoding
const tgtTokenEmbeddings = randn([batchSize, 8, dModel]);
const tgtEmbedded = tgtPE.forward(tgtTokenEmbeddings);
console.log(`  Target embedded shape: ${tgtEmbedded.shape}`);

// Step 4: Decode with cross-attention to encoder output
const fullDecoderOutput = fullDecoder.forward(tgtEmbedded, fullEncoderOutput);
console.log(`  Decoder output shape: ${fullDecoderOutput.shape}`);

// Step 5: Project to vocabulary logits
const logits = outputProj.forward(fullDecoderOutput);
console.log(`  Logits shape: ${logits.shape} (batch, tgtLen, vocab)`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• MultiheadAttention: parallel attention across multiple subspaces");
console.log("• TransformerEncoderLayer: self-attention + feedforward with residuals");
console.log("• TransformerDecoderLayer: self-attn + cross-attn + feedforward");
console.log("• TransformerEncoder/Decoder: stack of N identical layers");
console.log("• PositionalEncoding: sinusoidal position information for sequences");
console.log("• Full pipeline: Embed → PosEncode → Encode → Decode → Project");
console.log("• All components support batched inputs and gradient computation");

console.log("\n✅ Transformer Architecture Example Complete!");
console.log("=".repeat(60));
