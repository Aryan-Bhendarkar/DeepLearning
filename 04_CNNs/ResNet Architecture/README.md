# ResNet-18 on CIFAR-10 (TensorFlow/Keras)

A clean TensorFlow/Keras implementation of the ResNet-18 architecture from scratch, customized specifically for the CIFAR-10 dataset.

## Why Customize ResNet-18 for CIFAR-10?
The standard ResNet-18 architecture is designed for ImageNet images (224x224). It starts with:
1. A 7x7 Convolution (stride 2)
2. A 3x3 Max Pooling (stride 2)

If applied directly to CIFAR-10's 32x32 images, these two operations downsample the input resolution to 8x8 before it even reaches the first residual stage. This aggressive downsampling causes the model to lose fine-grained spatial details too early.

### Our modifications:
* **Initial Conv Layer**: Replaced the 7x7 conv (stride 2) with a 3x3 conv (stride 1).
* **MaxPooling**: Removed the initial max pooling layer entirely.

This keeps the input resolution at 32x32 for the first stage, improving feature extraction for smaller images.

---

## File Structure

* **`resnet_block.py`**: Defines the `ResidualBlock` class. Handles residual skip connections and uses a 1x1 projection conv when stride > 1 to match feature map shapes.
* **`resnet18.py`**: Builds the ResNet-18 network using the custom residual blocks, organized in four stages (filters: 64, 128, 256, 512) followed by Global Average Pooling and a Dense classifier.
* **`dataset.py`**: Downloads the CIFAR-10 dataset and normalizes pixel values to a `[0, 1]` range.
* **`train.py`**: Integrates the model and dataset, compiles it with the Adam optimizer, and trains it for 20 epochs.

---

## Requirements

* Python
* TensorFlow 

---

## How to Run

To start training the model, run:
```bash
python train.py
```

---

## Performance
* Trained with the Adam optimizer (learning rate: 0.001) and a batch size of 64.
* Achieved **87% validation accuracy** at the 18th epoch.
