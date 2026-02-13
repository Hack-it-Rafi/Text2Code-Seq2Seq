import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return h, c

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb, (h, c))
        logits = self.fc(out)
        return logits, h, c

class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderLSTM(src_vocab, embed_dim, hidden_dim)
        self.decoder = DecoderLSTM(tgt_vocab, embed_dim, hidden_dim)

    def forward(self, src, tgt):
        h, c = self.encoder(src)
        logits, _, _ = self.decoder(tgt[:, :-1], h, c)
        return logits
