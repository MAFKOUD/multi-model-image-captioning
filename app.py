import streamlit as st
from PIL import Image
import tempfile
import os

from pipeline import run_captioning_pipeline

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Multi-Model Consensus for Image Captioning",
    layout="wide"
)

st.title("🧠 Multi-Model Consensus for Image Captioning")

st.markdown("""
This application generates **robust and reliable image captions** by combining
multiple vision–language models and applying an **explicit reasoning pipeline**
(semantic consensus, Tree of Thoughts, refinement, and self-correction).
""")

# =========================================================
# SIDEBAR (CLEAN & UNIQUE)
# =========================================================
st.sidebar.header("⚙️ Settings")

enable_tot = st.sidebar.checkbox(
    "Enable Tree of Thoughts (explore multiple Gemini captions)",
    value=True
)

enable_self_correction = st.sidebar.checkbox(
    "Enable self-correction",
    value=True
)

show_reasoning = st.sidebar.checkbox(
    "Show agent reasoning (dynamic explanation)",
    value=True
)

# =========================================================
# IMAGE UPLOAD
# =========================================================
st.header("📸 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=700)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    image_name = uploaded_file.name

    if st.button("🚀 Generate Caption"):
        with st.spinner("Running intelligent captioning pipeline..."):

            result = run_captioning_pipeline(
                image_path=image_path,
                image_name=image_name,
                enable_self_correction=enable_self_correction,
                enable_tot=enable_tot
            )

        # =================================================
        # RESULTS
        # =================================================
        st.success("Captioning completed successfully!")

        # -----------------------------
        # RAW CAPTIONS
        # -----------------------------
        st.subheader("📝 Captions from Individual Models")
        for model, caption in result["captions"].items():
            st.write(f"**{model}**: {caption}")

        # -----------------------------
        # CONSENSUS
        # -----------------------------
        st.subheader("🧩 Semantic Consensus")
        st.info(
            f"**Selected model:** {result['consensus']['best_model']}\n\n"
            f"**Consensus caption:** {result['consensus']['best_caption']}"
        )

        # -----------------------------
        # FINAL CAPTION
        # -----------------------------
        st.subheader("🤖 Final Caption (after reasoning & refinement)")
        st.success(result["final_caption"])

        # -----------------------------
        # EVALUATION (OPTIONAL)
        # -----------------------------
        if result["evaluation"] is not None:
            st.subheader("📊 Evaluation Metrics (against ground truth)")
            for model, scores in result["evaluation"].items():
                with st.expander(model):
                    st.json(scores)
        else:
            st.info("No ground-truth available for this image → evaluation skipped.")

        # -----------------------------
        # AGENT REASONING (DYNAMIC)
        # -----------------------------
        if show_reasoning:
            st.subheader("🧠 Agent Reasoning (Dynamic & Data-Driven)")
            st.markdown(result["agent_explanation"])

            if result.get("tot_debug"):
                st.subheader("🌳 Tree of Thoughts – Candidate Analysis")
                st.json(result["tot_debug"])

    # Clean temp file
    os.remove(image_path)
