# COMPUTER VISION Coding

This Folder holds the Computer Vision Practice and small projects on datasets using PyTorch.

## Softmax Linear Classifier

Uses Python with PyTorch for classifying CiFAR 10

## Two Layer Neural network

Uses Python with PyTorch for classifying CiFAR 10 with 2-layer Neural Network.

## ViT Depth Minimal

A minimal Vision Transformer for CIFAR-10 that compares two and four encoder blocks using the same training and validation split. Each model is tested once from the checkpoint with the highest validation accuracy. Depth 2 reached 67.20% test accuracy, and depth 4 reached 72.60%.

- [Executed Jupyter notebook](ViT_Depth_Minimal/assignment4_vit_depth_minimal.ipynb)
- [Standalone Python script](ViT_Depth_Minimal/vit_depth_minimal.py)
- [Report with learning curves and misclassified images](ViT_Depth_Minimal/CAP4453_Assignment4_Report.pdf)

From the repository root, run `python Computer_Vision/ViT_Depth_Minimal/vit_depth_minimal.py` with PyTorch, torchvision, NumPy, and Matplotlib installed. The script downloads CIFAR-10 to `data/` and saves plots, checkpoints, and metrics in `vit_results/` under the current working directory.
