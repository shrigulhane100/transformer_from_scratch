# Transformer from Scratch with PyTorch

A from-scratch implementation of the **encoder–decoder Transformer** architecture using **PyTorch**, based on the landmark paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

This project explores how a Transformer works internally by building its individual components, assembling them into a sequence-to-sequence model, and training it for **neural machine translation**. The reference implementation uses the Hugging Face `opus_books` dataset and defaults to **English → Italian** translation.

> **Learning focus:** Understand and implement the Transformer architecture instead of relying on a prebuilt `torch.nn.Transformer` model.

## Architecture Overview

```text
Source sentence                             Target sentence (shifted right)
       │                                                   │
       ▼                                                   ▼
Source token embeddings                       Target token embeddings
       │                                                   │
Positional encoding                            Positional encoding
       │                                                   │
       ▼                                                   ▼
┌────────────────────────┐                 ┌───────────────────────────────┐
│       Encoder × N      │                 │          Decoder × N          │
│                        │                 │                               │
│ Multi-head self-attn   │                 │ Masked multi-head self-attn   │
│ Add & LayerNorm        │                 │ Add & LayerNorm               │
│ Feed-forward network   │──────┐          │ Encoder–decoder cross-attn    │
│ Add & LayerNorm        │      └─────────►│ Add & LayerNorm               │
└────────────────────────┘                 │ Feed-forward network          │
                                           │ Add & LayerNorm               │
                                           └──────────────┬────────────────┘
                                                          │
                                                  Linear projection
                                                          │
                                                 Vocabulary logits
                                                          │
                                                Next-token prediction
```

The **encoder** builds contextual representations of the source sequence. The **decoder** generates the output autoregressively, attending to previously generated target tokens and the encoder output.

## Concepts Implemented

### 1. Transformer fundamentals

| Concept | What it does | 
| --- | --- |
| **Token embeddings** | Convert token IDs into dense vectors of size `d_model`; embeddings are scaled by `√d_model`. |
| **Sinusoidal positional encoding** | Inject token-order information using sine and cosine functions, since attention alone does not encode sequence position. |
| **Layer normalization** | Normalizes hidden activations to stabilize optimization. |
| **Residual (skip) connections** | Add the sublayer input back to its output to improve gradient flow. |
| **Position-wise feed-forward network** | Applies two learned linear transformations with a ReLU activation independently at every sequence position. |
| **Scaled dot-product attention** | Computes query–key similarity, scales by `√d_k`, applies masking and softmax, then takes a weighted sum of values. |
| **Multi-head attention** | Learns several attention patterns in parallel through separate query, key and value projections. |
| **Encoder self-attention** | Lets each source token attend to other non-padding source tokens. |
| **Masked decoder self-attention** | Prevents target positions from attending to future tokens (causal masking). |
| **Cross-attention** | Uses decoder queries and encoder keys/values to condition translation on the source sentence. |
| **Encoder and decoder stacks** | Compose attention, feed-forward, normalization and residual modules into the complete architecture. |
| **Output projection** | Maps decoder hidden states to target-vocabulary logits. |




**Scaled dot-product attention:**

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

Here, `Q`, `K`, and `V` are query, key and value matrices, and `M` is an attention mask that blocks disallowed positions before softmax.

### 2. Data preparation and tokenization

- **Neural machine translation (sequence-to-sequence learning):** Translate a source-language sentence into a target-language sentence.
- **Word-level tokenization:** Train separate source and target tokenizers with Hugging Face `tokenizers`.
- **Special tokens:** Use `[SOS]`, `[EOS]`, `[PAD]`, and `[UNK]` to mark sequence boundaries, pad batches, and handle unknown words.
- **Fixed-length sequences:** Truncate or pad inputs according to the configured sequence length, with dataset constraints enforced by the implementation.
- **Padding masks:** Prevent the attention mechanism from using padding positions as meaningful tokens.
- **Causal masks:** Restrict decoder attention to the current and earlier target positions.
- **PyTorch datasets and data loaders:** Build a bilingual `Dataset`, train/validation splits, and mini-batches.

### 3. Training and inference

- **Teacher forcing:** Feed the decoder the ground-truth target sequence shifted by one position during training.
- **Cross-entropy loss:** Compare target vocabulary logits against the next expected token; ignore padding positions.
- **Label smoothing:** Reduce overconfidence in target-token predictions.
- **Adam optimizer and backpropagation:** Optimize trainable model parameters through the training loop.
- **Training and validation loops:** Switch between `model.train()` and `model.eval()`, using `torch.no_grad()` for validation.
- **Greedy decoding:** Autoregressively select the highest-scoring token until `[EOS]` or the length limit is reached.
- **Checkpoints:** Save model and optimizer states, epoch and global step; resume training from a saved checkpoint.
- **TensorBoard logging:** Monitor training loss and validation metrics.
- **Attention visualization:** Inspect how different attention heads distribute attention over tokens.
- **Beam search:** Explore an alternative decoding strategy in the accompanying notebook.

## Project Structure

```text
.
├── model.py                 # Embeddings, attention, encoder, decoder, Transformer
├── dataset.py               # Bilingual dataset and attention masks
├── train.py                 # Tokenizers, data loaders, training and validation
├── train_wb.py              # Training variant with Weights & Biases
├── translate.py             # Translation/inference script
├── config.py                # Hyperparameters, paths, checkpoint configuration
├── requirements.txt         # Python dependencies
├── Local_Train.ipynb        # Local training walkthrough
├── Colab_Train.ipynb        # Google Colab training walkthrough
├── Inference.ipynb          # Inference walkthrough
├── Beam_Search.ipynb        # Beam-search exploration
└── attention_visual.ipynb   # Attention-head visualization
```

## Getting Started

### Requirements

The original project specifies **Python 3.9** and pinned PyTorch/Hugging Face dependencies. A compatible environment, preferably with a CUDA-capable GPU, is recommended for training.

### Installation

```bash
git clone https://github.com/hkproj/pytorch-transformer.git
cd pytorch-transformer

python3.9 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

> The pinned dependencies reflect the original tutorial environment and may require version-specific setup on newer Python installations.

### Configuration

Edit `config.py` to change the model and training settings. Default values in the reference repository:

| Setting | Default |
| --- | --- |
| Dataset | `opus_books` |
| Source → target | `en` → `it` |
| Model dimension (`d_model`) | `512` |
| Sequence length | `350` |
| Batch size | `8` |
| Epochs | `20` |
| Learning rate | `1e-4` |
| Checkpoint preload | `latest` |

The model builder also configures the number of layers, attention heads, dropout and feed-forward width.

### Train

```bash
python train.py
```

Training downloads/loads the configured dataset, builds tokenizers as needed, and saves model checkpoints. To inspect training logs:

```bash
tensorboard --logdir runs
```

### Explore inference and attention

Open `Inference.ipynb`, `Beam_Search.ipynb`, or `attention_visual.ipynb` in Jupyter. The reference repository also includes `translate.py` for scripted inference. **Train the model or supply a compatible checkpoint first.**

## Training Pipeline

```text
OPUS Books parallel sentences
           │
           ▼
Train source & target tokenizers
           │
           ▼
Build token IDs, shifted labels, padding and causal masks
           │
           ▼
Encoder → Decoder → Vocabulary projection
           │
           ▼
Cross-entropy loss (with label smoothing)
           │
           ▼
Backward pass → Adam optimizer step
           │
           ▼
Validation → Save checkpoint → Repeat
           │
           ▼
Autoregressive decoding for translation
```


## References and Credits

- **Paper:** Vaswani et al., [*Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762).
- **Attention visualization reference:** [The Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/). The tutorial author credits an externally sourced attention-plotting function to this resource.

This repository is intended for **learning and experimentation**, not as a production-ready translation system. Translation quality depends on the available dataset, training time, configuration, and checkpoint.
