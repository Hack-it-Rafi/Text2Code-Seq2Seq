import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from config import DATASET_NAME, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MAX_SRC_LEN, MAX_TGT_LEN

class CodeSearchNetDataset(Dataset):
    def __init__(self, split_data, src_tokenizer, tgt_tokenizer):
        self.data = split_data
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]["docstring"]
        code = self.data[idx]["code"]

        src_ids = self.src_tok.encode(doc, MAX_SRC_LEN)
        tgt_ids = self.tgt_tok.encode(code, MAX_TGT_LEN)

        src_ids = self.src_tok.pad_sequence(src_ids, MAX_SRC_LEN)

        tgt_ids = [self.tgt_tok.word2idx["<SOS>"]] + tgt_ids + [self.tgt_tok.word2idx["<EOS>"]]
        tgt_ids = self.tgt_tok.pad_sequence(tgt_ids, MAX_TGT_LEN + 2)

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def load_data():
    dataset = load_dataset(DATASET_NAME)
    
    if "validation" in dataset:
        train_data = dataset["train"].select(range(TRAIN_SIZE))
        val_data = dataset["validation"].select(range(VAL_SIZE))
    else:
        train_data = dataset["train"].select(range(TRAIN_SIZE))
        val_data = dataset["train"].select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))
    
    if "test" in dataset:
        test_data = dataset["test"].select(range(TEST_SIZE))
    else:
        test_data = dataset["train"].select(range(TRAIN_SIZE + VAL_SIZE, TRAIN_SIZE + VAL_SIZE + TEST_SIZE))

    return train_data, val_data, test_data
