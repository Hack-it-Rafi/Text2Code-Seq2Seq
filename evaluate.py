import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import load_data, CodeSearchNetDataset
from metrics import token_accuracy, bleu_score, exact_match
from tokenizer import Tokenizer

from models.rnn_seq2seq import Seq2SeqRNN
from models.lstm_seq2seq import Seq2SeqLSTM


# Allow safe loading of Tokenizer class
torch.serialization.add_safe_globals([Tokenizer])

def greedy_decode(model, src, tgt_tok, model_type="rnn", max_len=MAX_TGT_LEN):
    model.eval()
    batch_size = src.size(0)

    sos = tgt_tok.word2idx["<SOS>"]
    eos = tgt_tok.word2idx["<EOS>"]

    # Encode source sequence once
    encoder_output = model.encoder(src)
    
    if model_type == "lstm":
        h, c = encoder_output
    else:
        h = encoder_output
        c = None
    
    # Start with SOS token
    outputs = torch.full((batch_size, 1), sos).to(DEVICE)

    for _ in range(max_len):
        # Decode one step at a time
        if model_type == "lstm":
            logits, h, c = model.decoder(outputs[:, -1:], h, c)
        else:
            logits, h = model.decoder(outputs[:, -1:], h)
        
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        outputs = torch.cat([outputs, next_token], dim=1)

        if (next_token == eos).all():
            break

    return outputs

def evaluate_model(model_type="rnn", checkpoint_path=None):
    _, _, test_data = load_data()

    src_tok = torch.load(f"{model_type}_src_tokenizer.pt")
    tgt_tok = torch.load(f"{model_type}_tgt_tokenizer.pt")

    test_ds = CodeSearchNetDataset(test_data, src_tok, tgt_tok)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    src_vocab = len(src_tok.word2idx)
    tgt_vocab = len(tgt_tok.word2idx)

    if model_type == "rnn":
        model = Seq2SeqRNN(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    elif model_type == "lstm":
        model = Seq2SeqLSTM(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    # else:
    #     model = Seq2SeqAttention(src_vocab, tgt_vocab, EMBED_DIM, HIDDEN_DIM).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    preds_text = []
    refs_text = []
    preds_ids_all = []
    refs_ids_all = []

    for src, tgt in tqdm(test_loader, desc="Evaluating"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        pred_ids = greedy_decode(model, src, tgt_tok, model_type)
        preds_ids_all.extend(pred_ids.cpu().tolist())
        refs_ids_all.extend(tgt.cpu().tolist())

        for p, r in zip(pred_ids.cpu().tolist(), tgt.cpu().tolist()):
            preds_text.append(tgt_tok.decode(p))
            refs_text.append(tgt_tok.decode(r))

    acc = token_accuracy(preds_ids_all, refs_ids_all, tgt_tok.word2idx["<PAD>"])
    bleu = bleu_score(preds_text, refs_text)
    em = exact_match(preds_text, refs_text)

    print(f"\nRESULTS ({model_type})")
    print(f"Token Accuracy: {acc:.4f}")
    print(f"BLEU Score: {bleu:.2f}")
    print(f"Exact Match: {em:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "attn"])
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.model, args.ckpt)
