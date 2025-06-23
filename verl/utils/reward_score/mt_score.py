import sacrebleu

def compute_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result