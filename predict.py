import json
import re
import torch
import yaml

from model import BiLSTMAttentionMCQ

CONFIG_PATH = "config.yaml"
VOCAB_PATH = "vocab.json"
WEIGHTS_PATH = "lstm_attn_best_fold4.pt"


def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class MCQSolver:
    def __init__(self, config_path=CONFIG_PATH, vocab_path=VOCAB_PATH,
                 weights_path=WEIGHTS_PATH, device="cpu"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.choices = config["choices"]
        self.max_len = config["max_len"]
        self.device = torch.device(device)

        with open(vocab_path) as f:
            self.word2idx = json.load(f)

        self.model = BiLSTMAttentionMCQ(**config["model"]).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def encode(self, text: str):
        tokens = simple_tokenize(text)
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    @torch.no_grad()
    def predict(self, prompt: str, options: dict):
        encoded = []
        for c in self.choices:
            text = str(prompt) + " [SEP] " + str(options.get(c, ""))
            ids = self.encode(text)
            encoded.append(torch.tensor(ids, dtype=torch.long))

        input_ids = torch.stack(encoded).unsqueeze(0).to(self.device)
        logits = self.model(input_ids).squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        ranked_idx = torch.argsort(probs, descending=True).tolist()
        ranked = [(self.choices[i], probs[i].item()) for i in ranked_idx]
        return ranked


if __name__ == "__main__":
    solver = MCQSolver()
    print("Model loaded successfully!")
