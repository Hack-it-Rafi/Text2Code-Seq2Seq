# Text-to-Python Code Generation using Seq2Seq Models

This project implements and compares three Seq2Seq architectures:

1. Vanilla RNN Seq2Seq
2. LSTM Seq2Seq
3. BiLSTM Encoder + LSTM Decoder with Bahdanau Attention

Dataset: CodeSearchNet Python (HuggingFace)

---

# Setup

```bash
pip install -r requirements.txt
```
# Train Models

## Train Vanilla RNN
```bash
python train.py --model rnn
```
## Train LSTM
```bash
python train.py --model lstm
```
## Train Attention
```bash
python train.py --model attn
```
# Evaluate Models
```bash
python evaluate.py --model rnn --ckpt outputs/checkpoints/rnn_best.pt
python evaluate.py --model lstm --ckpt outputs/checkpoints/lstm_best.pt
python evaluate.py --model attn --ckpt outputs/checkpoints/attn_best.pt
```

# Outputs include:

- Token Accuracy

- BLEU Score

- Exact Match

# Inference
```bash
python infer.py --model attn --ckpt outputs/checkpoints/attn_best.pt --text "returns the maximum value in a list of integers"
```

# Attention Visualization
```bash
python outputs_attention_plot.py
```

This generates attention heatmaps in outputs/plots/.