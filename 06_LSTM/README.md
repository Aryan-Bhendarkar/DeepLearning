# LSTM Next Word Predictor

A simple implementation of a next-word prediction model using Long Short-Term Memory (LSTM) networks in TensorFlow and Keras.

## Project Overview

The model is trained on custom FAQ data. It processes raw text, creates training sequences using a sliding window (n-grams), and trains an LSTM network to predict the most likely next word in a sequence.

## Model Architecture

* **Embedding Layer**: Learns 100-dimensional word embeddings for the vocabulary.
* **LSTM Layer**: A single LSTM layer with 100 memory units to capture sequential context.
* **Dense Layer**: A softmax output layer predicting probability distributions across the 282 unique words in the vocabulary.

## Preprocessing Pipeline

1. **Tokenization**: Standardizing and index-mapping words using Keras `Tokenizer`.
2. **N-gram Generation**: Creating incremental word sequences (e.g., "what is", "what is the", "what is the course").
3. **Padding**: Pre-padding sequences to ensure uniform input length (`maxlen=56`).
4. **Target Encoding**: One-hot encoding the output labels for categorical cross-entropy loss.

## Performance & Training

* **Loss**: Categorical Cross-Entropy
* **Optimizer**: Adam
* **Training Accuracy**: ~95% achieved at epoch 100.
