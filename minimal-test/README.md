# MoCo (Momentum Contrast) - Simplified CPU Implementation Plan

**Original Paper**: [Momentum Contrast for Unsupervised Visual Representation Learning (CVPR 2020)](https://github.com/facebookresearch/moco)

## Project Overview

MoCo is a self-supervised learning method for visual representation learning that uses **contrastive learning** to train encoders without labeled data.

### Core Problem
- Learn meaningful visual representations without labels
- Build a large, consistent dictionary for contrastive learning
- Balance dictionary size with computational efficiency

### Key Innovation
Two mechanisms that enable large-scale contrastive learning:
1. **Queue-based dictionary** - decouples dictionary size from batch size
2. **Momentum encoder** - maintains consistency across mini-batches

## Architecture

```
Input Image → Two Augmentations → [Query View, Key View]
                                         ↓           ↓
                                    Query Encoder  Momentum Encoder
                                         ↓           ↓
                                    Normalize    Normalize
                                         ↓           ↓
                                         q           k
                                         ↓           ↓
                                    Positive Pair (q·k)
                                         ↓
                                    Negative Pairs (q·queue)
                                         ↓
                                    Contrastive Loss
```

### Components
1. **Query Encoder** - trainable encoder (e.g., ResNet-18)
2. **Momentum Encoder** - slowly updated copy of query encoder
3. **Queue** - stores previous key embeddings as negative samples
4. **Contrastive Loss** - InfoNCE loss to learn representations

## Simplified Implementation Plan

### Simplifications for CPU Training
1. **Small backbone**: ResNet-18 instead of ResNet-50
2. **Small queue**: K=1024 instead of 65,536
3. **Small dataset**: CIFAR-10 (32×32 images)
4. **Small batch**: 32 instead of 256
5. **Fewer epochs**: 10-20 for demonstration
6. **CPU-friendly**: No distributed training, no shuffling BN

### Key Hyperparameters
- **Feature dimension**: 128
- **Queue size (K)**: 1024
- **Momentum (m)**: 0.999
- **Temperature (τ)**: 0.07
- **Batch size**: 32
- **Learning rate**: 0.03
- **Optimizer**: SGD with momentum 0.9

### Training Steps
1. Load two augmented views of each image
2. Encode query view with query encoder → q
3. Encode key view with momentum encoder → k (no gradient)
4. Compute positive similarity: q·k
5. Compute negative similarities: q·queue
6. Calculate contrastive loss
7. Update query encoder via backprop
8. Update momentum encoder: θ_k ← m·θ_k + (1-m)·θ_q
9. Update queue: enqueue k, dequeue oldest

### Evaluation
After pre-training, evaluate learned representations:
1. **Linear evaluation**: Freeze encoder, train linear classifier
2. **Measure accuracy** on CIFAR-10 test set

## References
- Official implementation: https://github.com/facebookresearch/moco
- Paper: [He et al., CVPR 2020](https://arxiv.org/abs/1911.05722)
