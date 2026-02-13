import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.embedding_dropout(emb)
        outputs, (h, c) = self.lstm(emb)
        outputs = self.layer_norm(outputs)  # Normalize encoder outputs
        return outputs, h, c

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.V = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, src_len, enc_dim)
        # decoder_hidden: (batch, dec_dim)
        
        hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, dec_dim)
        
        # Calculate attention scores
        score = self.V(torch.tanh(
            self.W1(encoder_outputs) + self.W2(hidden)
        ))  # (batch, src_len, 1)
        
        attn_weights = F.softmax(score.squeeze(-1), dim=1)  # (batch, src_len)
        
        # Calculate context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1), 
            encoder_outputs
        ).squeeze(1)  # (batch, enc_dim)
        
        return context, attn_weights

class DecoderAttnLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dec_hidden_dim = dec_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim=dec_hidden_dim)
        
        # LSTM input is embedding + context
        self.lstm = nn.LSTM(
            embed_dim + enc_hidden_dim, 
            dec_hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection with residual connection
        self.fc = nn.Linear(dec_hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dec_hidden_dim)

    def forward(self, x, encoder_outputs, h, c):
        # x: (batch, seq_len)
        # encoder_outputs: (batch, src_len, enc_dim)
        # h, c: (num_layers, batch, dec_hidden_dim)
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.embedding_dropout(emb)
        
        outputs = []
        attn_weights_list = []
        
        # Process each time step
        for t in range(seq_len):
            # Get current embedding
            curr_emb = emb[:, t:t+1, :]  # (batch, 1, embed_dim)
            
            # Calculate attention using last layer hidden state
            context, attn_w = self.attention(encoder_outputs, h[-1])  # context: (batch, enc_dim)
            attn_weights_list.append(attn_w)
            
            # Concatenate embedding and context
            context = context.unsqueeze(1)  # (batch, 1, enc_dim)
            lstm_input = torch.cat([curr_emb, context], dim=-1)  # (batch, 1, embed_dim + enc_dim)
            
            # LSTM step
            out, (h, c) = self.lstm(lstm_input, (h, c))  # out: (batch, 1, dec_hidden_dim)
            out = self.layer_norm(out)
            
            # Combine output, context, and embedding for final prediction (residual connection)
            combined = torch.cat([
                out.squeeze(1),
                context.squeeze(1),
                curr_emb.squeeze(1)
            ], dim=-1)  # (batch, dec_hidden_dim + enc_dim + embed_dim)
            
            outputs.append(combined)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, dec_hidden_dim + enc_dim + embed_dim)
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
        
        attn_weights = torch.stack(attn_weights_list, dim=1)  # (batch, seq_len, src_len)
        
        return logits, h, c, attn_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = EncoderBiLSTM(src_vocab, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = DecoderAttnLSTM(
            tgt_vocab, embed_dim,
            enc_hidden_dim=hidden_dim * 2,
            dec_hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Bridge layers to transform bidirectional encoder states to decoder states
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_layers = num_layers
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Other weight matrices
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1 for LSTM
                if 'lstm' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def _bridge_hidden_states(self, h, c):
        """Transform bidirectional encoder hidden states to decoder hidden states"""
        # h, c: (num_layers * 2, batch, hidden_dim)
        batch_size = h.size(1)
        
        # Combine forward and backward for each layer
        h_list = []
        c_list = []
        for i in range(self.num_layers):
            # Get forward and backward hidden states for layer i
            h_fwd = h[i * 2]  # (batch, hidden_dim)
            h_bwd = h[i * 2 + 1]  # (batch, hidden_dim)
            c_fwd = c[i * 2]
            c_bwd = c[i * 2 + 1]
            
            # Concatenate and project
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, hidden_dim * 2)
            c_cat = torch.cat([c_fwd, c_bwd], dim=-1)
            
            h_dec = self.bridge_h(h_cat)  # (batch, hidden_dim)
            c_dec = self.bridge_c(c_cat)
            
            h_list.append(h_dec)
            c_list.append(c_dec)
        
        h_dec = torch.stack(h_list, dim=0)  # (num_layers, batch, hidden_dim)
        c_dec = torch.stack(c_list, dim=0)
        
        return h_dec, c_dec

    def forward(self, src, tgt):
        encoder_outputs, h, c = self.encoder(src)
        h_dec, c_dec = self._bridge_hidden_states(h, c)
        logits, _, _, _ = self.decoder(tgt[:, :-1], encoder_outputs, h_dec, c_dec)
        return logits

    def forward_with_attention(self, src, tgt):
        encoder_outputs, h, c = self.encoder(src)
        h_dec, c_dec = self._bridge_hidden_states(h, c)
        logits, _, _, attn_weights = self.decoder(tgt[:, :-1], encoder_outputs, h_dec, c_dec)
        return logits, attn_weights
