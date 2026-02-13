import torch
from config import *
from tokenizer import Tokenizer
from models.rnn_seq2seq import Seq2SeqRNN
from models.lstm_seq2seq import Seq2SeqLSTM
from models.lstm_attention import Seq2SeqAttention

# Allow safe loading of Tokenizer class
torch.serialization.add_safe_globals([Tokenizer])

def generate_code(model_type, checkpoint_path, docstring):
    src_tok = torch.load(f"{model_type}_src_tokenizer.pt")
    tgt_tok = torch.load(f"{model_type}_tgt_tokenizer.pt")

    src_vocab = len(src_tok.word2idx)
    tgt_vocab = len(tgt_tok.word2idx)

    if model_type == "rnn":
        model = Seq2SeqRNN(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    elif model_type == "lstm":
        model = Seq2SeqLSTM(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    else:
        model = Seq2SeqAttention(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_ids = src_tok.encode(docstring, MAX_SRC_LEN)
    src_ids = src_tok.pad_sequence(src_ids, MAX_SRC_LEN)

    src_tensor = torch.tensor([src_ids]).to(DEVICE)

    sos = tgt_tok.word2idx["<SOS>"]
    eos = tgt_tok.word2idx["<EOS>"]

    # Encode source sequence once
    with torch.no_grad():
        if model_type == "attn":
            encoder_outputs, h, c = model.encoder(src_tensor)
            # Bridge hidden states from bidirectional encoder to decoder
            h_cat = torch.cat([h[0], h[1]], dim=-1)
            c_cat = torch.cat([c[0], c[1]], dim=-1)
            h = model.bridge_h(h_cat).unsqueeze(0)
            c = model.bridge_c(c_cat).unsqueeze(0)
        elif model_type == "lstm":
            h, c = model.encoder(src_tensor)
            encoder_outputs = None
        else:  # rnn
            h = model.encoder(src_tensor)
            c = None
            encoder_outputs = None
        
        # Start with SOS token
        outputs = torch.full((1, 1), sos).to(DEVICE)

        for _ in range(MAX_TGT_LEN):
            # Decode one step at a time
            if model_type == "attn":
                logits, h, c, _ = model.decoder(outputs[:, -1:], encoder_outputs, h, c)
            elif model_type == "lstm":
                logits, h, c = model.decoder(outputs[:, -1:], h, c)
            else:  # rnn
                logits, h = model.decoder(outputs[:, -1:], h)
            
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            outputs = torch.cat([outputs, next_token], dim=1)

            if next_token.item() == eos:
                break

    return tgt_tok.decode(outputs[0].cpu().tolist())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["rnn", "lstm", "attn"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    print(generate_code(args.model, args.ckpt, args.text))
