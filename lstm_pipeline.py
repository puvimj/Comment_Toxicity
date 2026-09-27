import re
import torch
from torch import nn

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ============================================================
# NLTK SETUP
# ============================================================

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

_lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def get_clean_text(text: str):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove # but keep hashtag word
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize
    cleaned_tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
    ]

    # Handle empty comments
    if not cleaned_tokens:
        return "emptycomment"

    return " ".join(cleaned_tokens)


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(
    text: str,
    word2idx: dict
):

    unk = word2idx["<UNK>"]

    return [
        word2idx.get(
            word,
            unk
        )
        for word in text.split()
    ]


# ============================================================
# PAD SEQUENCE
# ============================================================

def pad_sequence(
    seq: list,
    max_len: int
):

    if len(seq) > max_len:
        return seq[:max_len]

    return seq + [0] * (
        max_len - len(seq)
    )


# ============================================================
# RNN MODEL
# ============================================================

class ToxicityRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        num_layers=1,
        dropout=0.5
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=(
                dropout
                if num_layers > 1
                else 0
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x,
        lengths
    ):

        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, hidden = self.rnn(
            packed
        )

        last_hidden = self.dropout(
            hidden[-1]
        )

        logits = self.fc(
            last_hidden
        )

        return logits.squeeze(1)


# ============================================================
# LSTM + ATTENTION MODEL
#
# IMPORTANT:
# The class name remains ToxicityLSTM.
# Internally it is now:
#
# Embedding
#     ↓
# 2-layer LSTM
#     ↓
# Attention
#     ↓
# Context Vector
#     ↓
# Dropout
#     ↓
# Fully Connected
# ============================================================

class ToxicityLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        hidden_dim=128,
        num_layers=2,
        dropout=0.4
    ):

        super().__init__()

        # ----------------------------------------------------
        # Embedding layer
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # ----------------------------------------------------
        # 2-layer LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Attention layer
        # ----------------------------------------------------

        self.attention = nn.Linear(
            hidden_dim,
            1
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Final classification layer
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x,
        lengths
    ):

        # ----------------------------------------------------
        # 1. Embedding
        # ----------------------------------------------------

        embedded = self.embedding(x)

        # ----------------------------------------------------
        # 2. Pack sequences
        # ----------------------------------------------------

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # ----------------------------------------------------
        # 3. LSTM
        # ----------------------------------------------------

        packed_output, (hidden, cell) = self.lstm(
            packed
        )

        # ----------------------------------------------------
        # 4. Unpack sequences
        # ----------------------------------------------------

        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        # ----------------------------------------------------
        # 5. Attention scores
        # ----------------------------------------------------

        attention_scores = self.attention(
            lstm_output
        ).squeeze(-1)

        # ----------------------------------------------------
        # 6. Mask padding positions
        # ----------------------------------------------------

        max_len = lstm_output.size(1)

        mask = (
            torch.arange(
                max_len,
                device=x.device
            ).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

        attention_scores = attention_scores.masked_fill(
            mask,
            -1e9
        )

        # ----------------------------------------------------
        # 7. Attention weights
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # ----------------------------------------------------
        # 8. Weighted context vector
        # ----------------------------------------------------

        context = torch.sum(
            lstm_output
            * attention_weights.unsqueeze(-1),
            dim=1
        )

        # ----------------------------------------------------
        # 9. Dropout
        # ----------------------------------------------------

        context = self.dropout(
            context
        )

        # ----------------------------------------------------
        # 10. Final classification
        # ----------------------------------------------------

        logits = self.fc(
            context
        )

        return logits.squeeze(1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path: str,
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model_type = checkpoint.get(
        "model_type",
        "LSTM"
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if model_type == "RNN":

        model_cls = ToxicityRNN

    elif model_type == "LSTM":

        # This is now the LSTM + Attention architecture
        model_cls = ToxicityLSTM

    else:

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    # --------------------------------------------------------
    # Create model using checkpoint parameters
    # --------------------------------------------------------

    model = model_cls(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get(
            "num_layers",
            1
        ),
        dropout=checkpoint.get(
            "dropout",
            0.5
        )
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# PREDICT PROBABILITY
# ============================================================

@torch.no_grad()
def predict_proba(
    model,
    checkpoint,
    texts,
    device="cpu"
):

    word2idx = checkpoint["word2idx"]

    max_len = checkpoint["max_len"]

    # --------------------------------------------------------
    # 1. Clean text
    # --------------------------------------------------------

    cleaned = [
        get_clean_text(text)
        for text in texts
    ]

    # --------------------------------------------------------
    # 2. Convert words → integer IDs
    # --------------------------------------------------------

    encoded = [
        encode_text(
            text,
            word2idx
        )
        for text in cleaned
    ]

    # --------------------------------------------------------
    # 3. Calculate actual lengths
    # --------------------------------------------------------

    lengths = torch.tensor(
        [
            min(
                len(seq),
                max_len
            )
            for seq in encoded
        ],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # 4. Pad sequences
    # --------------------------------------------------------

    padded = [
        pad_sequence(
            seq,
            max_len
        )
        for seq in encoded
    ]

    # --------------------------------------------------------
    # 5. Convert to tensor
    # --------------------------------------------------------

    batch = torch.tensor(
        padded,
        dtype=torch.long
    )

    batch = batch.to(device)

    lengths = lengths.to(device)

    # --------------------------------------------------------
    # 6. Model prediction
    # --------------------------------------------------------

    logits = model(
        batch,
        lengths
    )

    # --------------------------------------------------------
    # 7. Logits → probability
    # --------------------------------------------------------

    probabilities = torch.sigmoid(
        logits
    )

    return probabilities.cpu().numpy().tolist()