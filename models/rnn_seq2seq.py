import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.rnn(emb)
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

class Seq2SeqRNN(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderRNN(src_vocab, embed_dim, hidden_dim)
        self.decoder = DecoderRNN(tgt_vocab, embed_dim, hidden_dim)

    def forward(self, src, tgt):
        hidden = self.encoder(src)
        logits, _ = self.decoder(tgt[:, :-1], hidden)
        return logits
