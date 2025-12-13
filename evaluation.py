import json
from nltk import word_tokenize, pos_tag
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# =============================
# Ground truth loader
# =============================
def load_ground_truth(path="data.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# =============================
# SPICE-like
# =============================
CONTENT_TAGS = {
    "NN", "NNS", "NNP", "NNPS",
    "JJ", "JJR", "JJS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"
}

def sentence_to_concept_set(sentence: str):
    tokens = word_tokenize(sentence.lower())
    tagged = pos_tag(tokens)
    return {word for (word, tag) in tagged if tag in CONTENT_TAGS}

def spice_like_for_one(refs, hypo):
    h_set = sentence_to_concept_set(hypo)
    if not h_set:
        return 0.0

    best_f1 = 0.0
    for r in refs:
        r_set = sentence_to_concept_set(r)
        inter = len(h_set & r_set)
        if inter == 0:
            continue

        prec = inter / len(h_set)
        rec = inter / len(r_set)
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)

    return best_f1

# =============================
# BLEU / METEOR / ROUGE-L
# =============================
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_bleu_scores(refs, hypo):
    refs_tok = [r.lower().split() for r in refs]
    hypo_tok = hypo.lower().split()

    bleu1 = sentence_bleu(refs_tok, hypo_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = sentence_bleu(refs_tok, hypo_tok, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    bleu3 = sentence_bleu(refs_tok, hypo_tok, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    bleu4 = sentence_bleu(refs_tok, hypo_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    return bleu1, bleu2, bleu3, bleu4

def compute_meteor(refs, hypo):
    refs_tok = [r.lower().split() for r in refs]
    hypo_tok = hypo.lower().split()
    return meteor_score(refs_tok, hypo_tok)

def compute_rouge_l(refs, hypo):
    best = 0
    for ref in refs:
        score = rouge.score(ref, hypo)['rougeL'].fmeasure
        best = max(best, score)
    return best

# =============================
# Evaluation API
# =============================
def evaluate_caption(ground_truth, img_name, predicted_caption):
    refs = [r["caption"] for r in ground_truth[img_name]]

    spice_like = spice_like_for_one(refs, predicted_caption)
    bleu1, bleu2, bleu3, bleu4 = compute_bleu_scores(refs, predicted_caption)
    meteor_val = compute_meteor(refs, predicted_caption)
    rouge_l = compute_rouge_l(refs, predicted_caption)

    return {
        "SPICE": spice_like,
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor_val,
        "ROUGE-L": rouge_l
    }

def evaluate_all(ground_truth, img_name, captions_dict):
    """
    captions_dict: {"BLIP Base": "...", "ViT-GPT2": "...", "BLIP-2": "...", "Gemini-Fusion": "..."}
    """
    return {k: evaluate_caption(ground_truth, img_name, v) for k, v in captions_dict.items()}
