# Smart MCQ Solver — BiLSTM + Attention

A from-scratch BiLSTM with additive attention, trained to pick the correct
answer (A–E) for multiple-choice questions. Deployed as a Streamlit app.

## Files

- `app.py` — Streamlit UI (prompt + 5 options → ranked predictions)
- `predict.py` — inference logic (tokenizer, encoder, `MCQSolver` class)
- `model.py` — `BiLSTMAttentionMCQ` architecture
- `config.yaml` — model hyperparameters (embed_dim, hidden_dim, etc.)
- `vocab.json` — word→index mapping built from `train.csv` (frequency ≥ 2),
  **must match exactly** what the checkpoint was trained on (vocab size 2981)
- `lstm_attn_best_fold4.pt` — trained weights (fold 4, best validation MAP@3)
- `requirements.txt` — minimal CPU-only dependencies for Streamlit Cloud

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Push all files above to this repo's `main` branch.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect this repo,
   and set the main file to `app.py`.
3. Deploy — no GPU or extra setup needed, the model runs on CPU.

## Notes

- The vocabulary is **not** a standard tokenizer (e.g. BERT's) — it's a custom
  word→index mapping built directly from the competition's `train.csv`. If you
  ever retrain the model, you must regenerate and commit a new `vocab.json`
  alongside the new weights, or predictions will silently be wrong (the model
  will run without error but map words to the wrong embedding indices).
