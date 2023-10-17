# Emoji Prediction LSTM

## Overview

Emoji Prediction LSTM is a deep learning model that predicts emojis based on text input. This project utilizes a Long Short-Term Memory (LSTM) neural network and pre-trained word embeddings for emoji classification. It can be used to add emojis to text or messages automatically.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving and Loading Model](#saving-and-loading-model)
- [Contributing](#contributing)
- [License](#license)

## About

The Emoji Prediction LSTM project is designed to predict emojis for text input. It utilizes pre-trained GloVe word embeddings and a deep learning model to map text to appropriate emojis. The model is trained on a dataset containing text samples and their corresponding emojis.

## Features

- Text to Emoji Prediction
- Deep Learning Model (LSTM)
- Pre-trained Word Embeddings
- Training and Evaluation

## Demo

To see the model in action, you can run the following code:

```python
from emoji_predictor import predict_emoji

prompt = 'Machine learning is fascinating'
predicted_emoji = predict_emoji(prompt)

print(f"Predicted Emoji: {predicted_emoji}")

"""':disappointed_face:'"""
```
