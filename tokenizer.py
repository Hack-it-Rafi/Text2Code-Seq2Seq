import re
from collections import Counter

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

class Tokenizer:
    def __init__(self, max_vocab=30000):
        self.max_vocab = max_vocab
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text):
        return self.clean_text(text).split()

    def build_vocab(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))

        vocab = SPECIAL_TOKENS + [w for w, _ in counter.most_common(self.max_vocab - len(SPECIAL_TOKENS))]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        tokens = tokens[:max_len]
        ids = [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.idx2word.get(i, "<UNK>")
            if w in ["<PAD>", "<SOS>"]:
                continue
            if w == "<EOS>":
                break
            words.append(w)
        return " ".join(words)

    def pad_sequence(self, ids, max_len):
        ids = ids[:max_len]
        ids += [self.word2idx["<PAD>"]] * (max_len - len(ids))
        return ids
