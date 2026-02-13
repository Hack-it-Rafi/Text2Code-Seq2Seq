import sacrebleu

def token_accuracy(preds, targets, pad_idx):
    correct = 0
    total = 0
    for p, t in zip(preds, targets):
        for pi, ti in zip(p, t):
            if ti == pad_idx:
                continue
            if pi == ti:
                correct += 1
            total += 1
    return correct / max(total, 1)

def exact_match(pred_texts, ref_texts):
    correct = 0
    for p, r in zip(pred_texts, ref_texts):
        if p.strip() == r.strip():
            correct += 1
    return correct / len(pred_texts)

def bleu_score(pred_texts, ref_texts):
    refs = [[r] for r in ref_texts]
    return sacrebleu.corpus_bleu(pred_texts, refs).score
