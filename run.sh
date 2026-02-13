#!/bin/bash
mkdir -p outputs/checkpoints outputs/plots outputs/logs

echo "Training RNN..."
python train.py --model rnn

echo "Training LSTM..."
python train.py --model lstm

echo "Training Attention..."
python train.py --model attn

echo "Evaluating models..."
python evaluate.py --model rnn --ckpt outputs/checkpoints/rnn_best.pt
python evaluate.py --model lstm --ckpt outputs/checkpoints/lstm_best.pt
python evaluate.py --model attn --ckpt outputs/checkpoints/attn_best.pt

echo "Generating attention plot..."
python outputs_attention_plot.py
