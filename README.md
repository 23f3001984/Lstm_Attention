# LSTM Attention Model

This repository contains a trained LSTM Attention model.

## Files

- lstm_attn_best_fold4.pt
- config.yaml
- model.py

## Installation

pip install -r requirements.txt

## Loading

```python
from model import LSTMAttention
import torch

model = LSTMAttention(...)
model.load_state_dict(torch.load("lstm_attn_best_fold4.pt"))
model.eval()