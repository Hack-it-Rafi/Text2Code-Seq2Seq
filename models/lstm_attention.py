import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, h, c

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, dec_dim)
        self.W2 = nn.Linear(dec_dim, dec_dim)
        self.V = nn.Linear(dec_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        hidden = decoder_hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attn_weights = F.softmax(score, dim=1)
        context = attn_weights * encoder_outputs
        context = context.sum(dim=1)
        return context, attn_weights.squeeze(-1)

class DecoderAttnLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)
        self.lstm = nn.LSTM(embed_dim + enc_hidden_dim, dec_hidden_dim, batch_first=True)
        self.fc = nn.Linear(dec_hidden_dim, vocab_size)

    def forward(self, x, encoder_outputs, h, c):
        emb = self.embedding(x)
        context, attn_weights = self.attention(encoder_outputs, h[-1])
        context = context.unsqueeze(1).repeat(1, emb.size(1), 1)
        lstm_input = torch.cat([emb, context], dim=-1)

        out, (h, c) = self.lstm(lstm_input, (h, c))
        logits = self.fc(out)
        return logits, h, c, attn_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderBiLSTM(src_vocab, embed_dim, hidden_dim)
        self.decoder = DecoderAttnLSTM(
            tgt_vocab, embed_dim,
            enc_hidden_dim=hidden_dim * 2,
            dec_hidden_dim=hidden_dim
        )
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, tgt):
        encoder_outputs, h, c = self.encoder(src)

        h_cat = torch.cat([h[0], h[1]], dim=-1)
        c_cat = torch.cat([c[0], c[1]], dim=-1)

        h_dec = self.bridge_h(h_cat).unsqueeze(0)
        c_dec = self.bridge_c(c_cat).unsqueeze(0)

        logits, _, _, _ = self.decoder(tgt[:, :-1], encoder_outputs, h_dec, c_dec)
        return logits

    def forward_with_attention(self, src, tgt):
        encoder_outputs, h, c = self.encoder(src)

        h_cat = torch.cat([h[0], h[1]], dim=-1)
        c_cat = torch.cat([c[0], c[1]], dim=-1)

        h_dec = self.bridge_h(h_cat).unsqueeze(0)
        c_dec = self.bridge_c(c_cat).unsqueeze(0)

        logits, _, _, attn_weights = self.decoder(tgt[:, :-1], encoder_outputs, h_dec, c_dec)
        return logits, attn_weights
