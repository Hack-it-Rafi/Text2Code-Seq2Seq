import torch
import matplotlib.pyplot as plt
import numpy as np
from config import *
from tokenizer import Tokenizer
from models.lstm_attention import Seq2SeqAttention

# Allow safe loading of Tokenizer class
torch.serialization.add_safe_globals([Tokenizer])

def plot_attention(model, src_tokens, tgt_tokens, attn_weights, save_path):
    plt.figure(figsize=(12, 6))
    plt.imshow(attn_weights, aspect="auto")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=90)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_attention_plot(checkpoint_path, docstring, save_path="outputs/plots/attention.png"):
    src_tok = torch.load("attn_src_tokenizer.pt")
    tgt_tok = torch.load("attn_tgt_tokenizer.pt")

    src_vocab = len(src_tok.word2idx)
    tgt_vocab = len(tgt_tok.word2idx)

    model = Seq2SeqAttention(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_tokens = src_tok.tokenize(docstring)
    src_ids = src_tok.encode(docstring, MAX_SRC_LEN)
    src_ids = src_tok.pad_sequence(src_ids, MAX_SRC_LEN)

    src_tensor = torch.tensor([src_ids]).to(DEVICE)

    dummy_tgt = torch.tensor([[tgt_tok.word2idx["<SOS>"]] * (MAX_TGT_LEN + 2)]).to(DEVICE)

    with torch.no_grad():
        logits, attn = model.forward_with_attention(src_tensor, dummy_tgt)

    pred_ids = logits.argmax(dim=-1).cpu().numpy()[0]
    tgt_tokens = [tgt_tok.idx2word.get(i, "<UNK>") for i in pred_ids[:30]]

    attn = attn.cpu().numpy()
    attn = attn[:len(tgt_tokens), :len(src_tokens)]

    plot_attention(model, src_tokens, tgt_tokens, attn, save_path)
    print("Saved attention plot:", save_path)

if __name__ == "__main__":
    generate_attention_plot(
        checkpoint_path="outputs/checkpoints/attn_epoch10.pt",
        docstring="returns the maximum value in a list of integers",
        save_path="outputs/plots/attention_example.png"
    )
