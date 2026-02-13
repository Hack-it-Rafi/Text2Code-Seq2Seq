import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import load_data, CodeSearchNetDataset
from tokenizer import Tokenizer
from utils import set_seed, save_checkpoint

from models.rnn_seq2seq import Seq2SeqRNN
from models.lstm_seq2seq import Seq2SeqLSTM

def train_model(model_type="rnn"):
    set_seed()

    train_data, val_data, test_data = load_data()

    src_tok = Tokenizer()
    tgt_tok = Tokenizer()

    src_tok.build_vocab([x["docstring"] for x in train_data])
    tgt_tok.build_vocab([x["code"] for x in train_data])

    train_ds = CodeSearchNetDataset(train_data, src_tok, tgt_tok)
    val_ds = CodeSearchNetDataset(val_data, src_tok, tgt_tok)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    src_vocab = len(src_tok.word2idx)
    tgt_vocab = len(tgt_tok.word2idx)

    if model_type == "rnn":
        model = Seq2SeqRNN(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    elif model_type == "lstm":
        model = Seq2SeqLSTM(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    else:
        raise ValueError("model_type must be rnn/lstm/attn")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.word2idx["<PAD>"])

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            optimizer.zero_grad()
            logits = model(src, tgt)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_epoch{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch+1, ckpt_path)

    torch.save(src_tok, f"{model_type}_src_tokenizer.pt")
    torch.save(tgt_tok, f"{model_type}_tgt_tokenizer.pt")
    print("Training complete. Tokenizers saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "attn"])
    args = parser.parse_args()

    train_model(args.model)
