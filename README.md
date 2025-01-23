<p>
  <img src="symbol.png" alt="Example Image" align="left" width="55" style="vertical-align: middle; margin-right: 10px;">
  <h1>PyTorch-CNN-MNIST</h1>
</p>

This repository is a quick personal project of learning to use PyTorch and learning background of Convolutional Neural Networks(CNN). In this repository is my first project which trains a simple CNN on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It includes scripts to train the CNN and to use it for classifying user-drawn images of digits.

## Overview

The repository contains two main scripts:

1. **`main.py`** – Trains a CNN on the MNIST dataset and saves the trained model to a file (`trained_net.pth`).
2. **`test.py`** – Loads the saved model (`trained_net.pth`) and performs inference on a user-supplied image of a digit (`image.png`). It outputs the predicted digit and the model's confidence (posterior probability).

### Repository Structure

```
PyTorch-CNN-MNIST/
├── trained_net.pth   # Saved model weights (generated after training)
├── MNIST/
│   └── image.png     # Example user-provided image for testing
│   └── main.py           # Training script
│   └── test.py           # Testing/inference script
├── README.md         # This README file
└── data/             # Auto-generated MNIST data downloads
```

## Notes

- By default, trains on 50% of the MNIST training set for faster experimentation. If computational complexity is not an issue increase the subset of the training set used to train or there are several parameters of the neural network configuration you can change.
- The 'image.png' file is a 28x28 as that is the standard size of the MNIST images. Larger images can be used as rescaling is built into 'test.py' but I have kept the test file as that to keep it simple.
- The `test.py` script inverts the colors of the input image. If your image is already black-on-white, remove the line `test_image = 1 - test_image`.

