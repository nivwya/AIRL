# AIRL

This repository contains two Colab-ready notebooks required for the assignment:
- `q1.ipynb` — Vision Transformer trained on CIFAR-10 (PyTorch). Includes patch embedding, CLS token, learnable positional embeddings, Transformer encoder blocks (MHSA + MLP + residual + LayerNorm).\n
- `q2.ipynb` — Text-driven segmentation pipeline: prompt → grounding seeds (GroundingDINO/CLIPSeg) → SAM masks.\n
## How to run 
1. Open Google Colab and set runtime to GPU (`Runtime > Change runtime type > GPU`).
2. Upload these notebooks to Colab or use `File > Upload notebook`.
3. For `q1.ipynb`: run all cells. Install cell at top installs extras. Train the ViT; the notebook will save `best_vit_cifar10.pth`.
4. For `q2.ipynb`: upload required model checkpoints into Colab (`sam_vit_h.pth`, GroundingDINO checkpoint if available) and an example image (`/content/example.jpg`). Run all cells.\n
## Best model config 
- Image size: 32
- Patch size: 4
- Embedding dim: 256
- Depth: 8\n- Heads: 8
- Batch size: 256
- Optimizer: AdamW (lr=3e-4, weight_decay=0.05)
- Augmentations: RandomCrop, HorizontalFlip, ShiftScaleRotate, Cutout
- Mixup alpha: 0.2
- Scheduler: CosineAnnealing
- 
## Tiny results table
| Notebook | Config (patch/embed/depth) | Epochs | Best CIFAR-10 val acc (%) |\n|---|---:|---:|---:|\n| q1.ipynb | 4 / 256 / 8 | 100 (recommended) | ~86–90% (depends on run/seed) |\n
## Short analysis 
- **Patch size:** smaller patches (2) increase sequence length and capacity, often improving accuracy but costing memory/time.
- **Depth vs. width:** deeper models (greater depth) capture more hierarchical transformations; increasing embed_dim improves representational capacity but uses more memory.
- **Augmentations:** RandAugment / CutMix / Mixup typically boost generalization on CIFAR-10.
- **Optimizer & schedule:** AdamW + cosine annealing with warmup helps converge stably.
