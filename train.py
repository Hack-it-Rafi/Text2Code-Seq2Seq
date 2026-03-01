import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from dataset import load_data, CodeSearchNetDataset
from tokenizer import Tokenizer
from utils import set_seed, save_checkpoint

from models.rnn_seq2seq import Seq2SeqRNN
from models.lstm_seq2seq import Seq2SeqLSTM
from models.lstm_attention import Seq2SeqAttention

def plot_loss_curves(train_losses, val_losses, model_type):
    """Plot and save training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss - {model_type.upper()}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(PLOT_DIR, f"{model_type}_loss_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to: {plot_path}")
    plt.close()

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
    elif model_type == "attn":
        model = Seq2SeqAttention(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    else:
        raise ValueError("model_type must be rnn/lstm/attn")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.word2idx["<PAD>"])

    # Track best and losses for plotting
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

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
        train_losses.append(avg_train_loss)

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
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save only if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_best.pt")
            save_checkpoint(model, optimizer, epoch+1, ckpt_path)
            print(f"✓ New best model saved! (Val Loss: {best_val_loss:.4f})")

    print(f"\nTraining complete. Best model from epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, model_type)
    
    torch.save(src_tok, f"{model_type}_src_tokenizer.pt")
    torch.save(tgt_tok, f"{model_type}_tgt_tokenizer.pt")
    print("Tokenizers saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "attn"])
    args = parser.parse_args()

    train_model(args.model)
