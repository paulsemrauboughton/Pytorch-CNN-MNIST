<p>
  <img src="symbol.png" alt="Example Image" align="left" width="55" style="vertical-align: middle; margin-right: 10px;">
  <h1>PyTorch-CNN-MNIST</h1>
</p>

This repository demonstrates the use of **PyTorch** to train a Convolutional Neural Network (CNN) on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It includes scripts to train the CNN and to use it for classifying user-drawn images of digits. By default, trains on 50% of the MNIST training set for faster experimentation.

## Overview

The repository contains two main scripts:

1. **`main.py`** – Trains a CNN on the MNIST dataset and saves the trained model to a file (`trained_net.pth`).
2. **`test.py`** – Loads the saved model (`trained_net.pth`) and performs inference on a user-supplied image of a digit (`image.png`). It outputs the predicted digit and the model's confidence (posterior probability).

### Repository Structure

```
PyTorch-CNN-MNIST/
├── main.py           # Training script
├── test.py           # Testing/inference script
├── trained_net.pth   # Saved model weights (generated after training)
├── MNIST/
│   └── image.png     # Example user-provided image for testing
├── README.md         # This README file
└── data/             # Auto-generated MNIST data downloads
```

## Notes

- By default, trains on 50% of the MNIST training set for faster experimentation.
- The `test.py` script inverts the colors of the input image. If your image is already black-on-white, remove the line `test_image = 1 - test_image`.
- If you encounter GPU/CPU device issues, ensure `map_location=torch.device('cpu')` is used when loading the model for CPU-based inference.

