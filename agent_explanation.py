def generate_agent_explanation(captions: dict, consensus: dict, final_caption: str, tot_debug=None) -> str:
    lines = []

    # 1) Résumé des sorties
    lines.append("### 🔍 What each vision model detected")
    for model, caption in captions.items():
        lines.append(f"- **{model}**: {caption}")

    # 2) Similarité sémantique
    lines.append("\n### 📊 Semantic similarity (agreement between models)")
    scores = consensus["scores"]
    best_model = consensus["best_model"]
    best_caption = consensus["best_caption"]

    # tri décroissant pour expliquer “qui est le plus cohérent”
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for model, score in sorted_scores:
        lines.append(f"- **{model}** similarity score: `{score:.2f}`")

    lines.append(
        f"\n✅ **Consensus choice:** we selected **{best_model}** because its caption is the most "
        "semantically consistent with the others (highest average similarity)."
    )
    lines.append(f"- Selected caption: _{best_caption}_")

    # 3) ToT si présent
    if tot_debug:
        lines.append("\n### 🌳 Tree of Thoughts (Gemini candidates)")
        for i, cand in enumerate(tot_debug["candidates"], start=1):
            lines.append(f"- **cand_{i}**: {cand}")

        lines.append("\n**Selection among candidates** (closer to consensus + less verbose):")
        for k, v in tot_debug["final_score"].items():
            lines.append(f"- {k} final score: `{v:.2f}`")

        lines.append(f"\n✅ Picked: **{tot_debug['picked']}**")

    # 4) Résultat final
    lines.append("\n### 🤖 Final caption (after refinement)")
    lines.append(f"➡️ **{final_caption}**")

    return "\n".join(lines)
