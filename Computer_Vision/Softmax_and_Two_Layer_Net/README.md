# Softmax and Two-Layer Network

These PyTorch examples train image classifiers on CIFAR-10 by default. Both can switch to MNIST by setting `dataset_name = 'mnist'`. They flatten the input image into a vector and use stochastic gradient descent with cross-entropy loss.

## Softmax linear classifier

A single linear layer maps the input pixels directly to ten class logits. Cross-entropy computes the log-softmax internally. The example plots training and validation metrics, compares learning rates, weight decay, and batch sizes, then retrains with the selected settings and reports test results with examples of incorrect predictions.

[Jupyter notebook](softmax_xavier_soto.ipynb) · [Python script](softmax.py)

## Two-layer neural network

The first linear layer maps the pixels to 100 hidden units. ReLU gives the model a nonlinear transformation before the second linear layer produces ten class logits. The example reports training loss, training accuracy, and validation accuracy for each epoch.

[Jupyter notebook](two_layer_net-1.ipynb) · [Python script](two_layer_net.py)

## Run the examples

Install PyTorch, torchvision, NumPy, and Matplotlib. Run the cells in either notebook in order, or run the corresponding Python script from this folder. The dataset downloads to `data/` relative to the working directory.

```sh
python softmax.py
python two_layer_net.py
```

On Windows, set the `DataLoader` calls to `num_workers=0` before running these scripts or notebooks so data loading stays in the main process.
