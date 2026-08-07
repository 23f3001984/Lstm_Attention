import torch
import yaml

from model import LSTMAttention

with open("config.yaml") as f:
    config = yaml.safe_load(f)

model = LSTMAttention(**config["model"])
model.load_state_dict(torch.load("lstm_attn_best_fold4.pt", map_location="cpu"))

model.eval()
print("Model loaded successfully!")