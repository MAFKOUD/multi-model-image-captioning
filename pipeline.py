from caption_models import generate_all_captions
from consensus import semantic_consensus
from gemini_fusion import (
    fuse_captions_with_gemini,
    self_correct_caption,
    fuse_with_tree_of_thoughts
)
from evaluation import load_ground_truth, evaluate_all
from tot_selector import pick_best_tot_candidate
from agent_explanation import generate_agent_explanation


_ground_truth = None

def get_ground_truth():
    global _ground_truth
    if _ground_truth is None:
        _ground_truth = load_ground_truth()
    return _ground_truth


def run_captioning_pipeline(
    image_path: str,
    image_name: str,
    enable_self_correction: bool = True,
    enable_tot: bool = True
) -> dict:

    # 1) Multi-model captions
    captions = generate_all_captions(image_path)

    # 2) Consensus sémantique
    consensus = semantic_consensus(captions)

    # 3) Gemini: ToT ou fusion simple
    tot_debug = None
    if enable_tot:
        candidates = fuse_with_tree_of_thoughts(
            captions=captions,
            consensus_caption=consensus["best_caption"]
        )
        tot_debug = pick_best_tot_candidate(consensus["best_caption"], candidates)
        final_caption = tot_debug["picked_caption"]
    else:
        final_caption = fuse_captions_with_gemini(
            captions=captions,
            consensus_caption=consensus["best_caption"]
        )

    # 4) Self-correction
    if enable_self_correction:
        final_caption = self_correct_caption(final_caption)

    # 5) Evaluation (si image dans data.json)
    ground_truth = get_ground_truth()
    all_captions = captions.copy()
    all_captions["Gemini-Fusion"] = final_caption

    scores = None
    if image_name in ground_truth:
        scores = evaluate_all(ground_truth, image_name, all_captions)

    # 6) Explication dynamique
    explanation = generate_agent_explanation(
        captions=captions,
        consensus=consensus,
        final_caption=final_caption,
        tot_debug=tot_debug
    )

    return {
        "captions": captions,
        "consensus": consensus,
        "final_caption": final_caption,
        "evaluation": scores,
        "tot_debug": tot_debug,
        "agent_explanation": explanation
    }
