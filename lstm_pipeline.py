"""
Shared LSTM/RNN inference utilities — mirrors the cleaning + model classes
in eda_corrected.ipynb, so the Streamlit app processes new text exactly the
way training data was processed.
"""
import re

import torch
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


def get_clean_text(text: str) -> str:
    """Same cleaning function as eda_corrected.ipynb (minus the debug print bug)."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    if not cleaned_tokens:
        return "emptycomment"
    return " ".join(cleaned_tokens)


def encode_text(text: str, word2idx: dict) -> list:
    unk = word2idx["<UNK>"]
    return [word2idx.get(w, unk) for w in text.split()]


def pad_sequence(seq: list, max_len: int) -> list:
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))


class ToxicityLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=96, hidden_dim=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)


class ToxicityRNN(nn.Module):
    """Vanilla RNN — included so the checkpoint can hold either architecture."""
    def __init__(self, vocab_size, embedding_dim=96, hidden_dim=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, nonlinearity="tanh",
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)


def load_checkpoint(path: str, device="cpu"):
    """Load a checkpoint saved by train_deep_model.py / eda_corrected.ipynb and
    reconstruct the right model class (LSTM or RNN) with its trained weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_cls = ToxicityRNN if checkpoint.get("model_type") == "RNN" else ToxicityLSTM
    model = model_cls(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_proba(model, checkpoint, texts, device="cpu"):
    """texts: list[str] of raw comment text -> list[float] toxicity probabilities."""
    word2idx = checkpoint["word2idx"]
    max_len = checkpoint["max_len"]
    cleaned = [get_clean_text(t) for t in texts]
    encoded = [pad_sequence(encode_text(t, word2idx), max_len) for t in cleaned]
    batch = torch.tensor(encoded, dtype=torch.long).to(device)
    logits = model(batch)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs.tolist()
