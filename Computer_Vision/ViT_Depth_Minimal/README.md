# ViT Depth Minimal

This project compares Vision Transformers (ViTs) with two and four encoder blocks on CIFAR-10. Both models use the same data split, patch size, embedding width, attention heads, and training settings. Only the number of encoder blocks changes.

## Model

1. A convolution with a 4 × 4 kernel and stride 4 splits each 32 × 32 RGB image into 64 nonoverlapping patches and projects each patch to a 128-dimensional token.
2. The model prepends a learned classification token (`[CLS]`) and adds learned position embeddings to all 65 tokens.
3. Each encoder block applies layer normalization before four-head self-attention and again before a feedforward network. Residual connections add the output of each sublayer back to its input. The feedforward network expands each token from 128 to 512 features, applies GELU, then projects back to 128 features.
4. After the final block, a layer normalization and a linear head map the `[CLS]` representation to ten class logits.

## Training and results

Both models use AdamW and cross-entropy loss. The experiment uses a seeded split of 45,000 training and 5,000 validation images, with random crops and horizontal flips applied during training. The checkpoint with the highest validation accuracy is evaluated once on the 10,000-image test set.

| Encoder depth | Best epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: |
| 2 | 20 | 68.48% | 67.20% |
| 4 | 19 | 73.48% | 72.60% |

The four-block model improved test accuracy by 5.40 percentage points in this run. The [report](CAP4453_Assignment4_Report.pdf) includes loss and accuracy curves and eight misclassified test images for each depth.

## Run the experiment

Open the [executed notebook](assignment4_vit_depth_minimal.ipynb) to see the saved outputs or rerun the experiment. To run the [Python script](vit_depth_minimal.py), install PyTorch, torchvision, NumPy, and Matplotlib, then run this command from the repository root.

```sh
python Computer_Vision/ViT_Depth_Minimal/vit_depth_minimal.py
```

The script downloads CIFAR-10 to `data/` and saves checkpoints, plots, and metrics in `vit_results/` relative to the current working directory. The reported run used the 20-epoch CUDA configuration. The script defaults to 10 epochs when running on CPU.
