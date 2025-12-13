
# 🧠 Multi-Model Consensus for Image Captioning

## Project Description

This project implements an **intelligent and explainable image captioning system**
based on **multi-model consensus and advanced reasoning techniques**.

Instead of relying on a single vision–language model, the system generates captions
from multiple pretrained models and applies a structured reasoning pipeline to
select, refine, and justify the final caption.

The application is deployed using **Streamlit** and emphasizes:
- Reliability
- Transparency
- Traceable reasoning
- Human-like final descriptions

---

## 🎯 Objectives

- Improve robustness of image captioning through **model agreement**
- Avoid hallucinations by relying only on **shared semantic content**
- Apply **explicit reasoning techniques** (not a simple chat interface)
- Provide **clear explanations** of decisions made by the system
- Demonstrate **Agent-based reasoning** using modern LLMs

---

## 🧩 System Architecture

```

Image
↓
[ BLIP Base | ViT-GPT2 | GIT ]
↓
Semantic Similarity Analysis
↓
Consensus Caption Selection
↓
Tree of Thoughts (optional)
↓
Gemini Reasoning & Refinement
↓
Self-Correction
↓
Final Caption + Explanation

```

---

## 🤖 Models Used

### Vision–Language Models
- **BLIP Base** (Salesforce)
- **ViT-GPT2** (nlpconnect)
- **GIT** (Microsoft)

Each model independently generates a caption from the same image.

---

### Reasoning Model
- **Gemini (Google Generative AI)**

Gemini is **not used as an oracle** and does **not analyze the image directly**.
Instead, it acts as a **reasoning agent** that:
- Aggregates information from multiple captions
- Refines wording without adding new content
- Generates Tree of Thoughts candidates
- Performs self-correction

---

## 🧠 Reasoning Techniques Implemented

### 1️⃣ Semantic Consensus
- Captions are embedded using **Sentence-BERT**
- Pairwise **cosine similarity** is computed
- The caption with the highest average agreement is selected

This ensures that the chosen caption reflects the **most consistent interpretation**
among all models.

---

### 2️⃣ Tree of Thoughts (ToT)
When enabled:
- Gemini generates multiple candidate captions
- Each candidate is evaluated against the consensus caption
- The most semantically aligned and concise candidate is selected

This reasoning process is **explicit, inspectable, and traceable**.

---

### 3️⃣ Self-Correction
A final verification step ensures that the selected caption:
- Is coherent
- Is concise
- Does not introduce contradictions or hallucinations

---

## 📊 Evaluation Metrics

When ground truth captions are available, the system computes:

- **SPICE**
- **BLEU-1 to BLEU-4**
- **METEOR**
- **ROUGE-L**

All metrics are displayed transparently in the Streamlit interface.

---

## 🖥️ Streamlit Application Features

- Image upload
- Toggle Tree of Thoughts reasoning
- Toggle self-correction
- Display of all model captions
- Consensus decision visualization
- Final refined caption
- Evaluation scores
- **Dynamic, data-driven agent explanation**

---

## 📂 Project Structure

```

.
├── app.py                 # Streamlit interface
├── pipeline.py            # Main reasoning pipeline
├── caption_models.py      # Caption generation models
├── consensus.py           # Semantic similarity & consensus
├── tot_selector.py        # Tree of Thoughts selection
├── gemini_fusion.py       # Gemini reasoning logic
├── agent_explanation.py   # Dynamic explanation generator
├── evaluation.py          # Metrics computation
├── data.json              # Ground truth captions (optional)
├── requirements.txt
└── test_images_eval/

````

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
````

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## ⭐ Key Contributions

* Multi-model caption consensus
* Semantic similarity-based decision making
* Tree of Thoughts reasoning
* Gemini used as a **reasoning agent**, not an oracle
* Self-correction mechanism
* Explainable AI with transparent decisions
* Interactive Streamlit interface

---

## 👥 Team Members

* **Bellouch Ayoub**
* **Mafkoud Khaoula**
* **Hamid Hiba**
* **Berkani Mohammed Adam**

---

## 📌 Notes

* The Gemini API key is integrated internally for demonstration purposes
* No user configuration is required
* This project is intended for **academic and educational use**

---

## 📜 License

This project is provided for academic purposes only.

````

---

👉 dis-le-moi, on termine ça comme des pros 🚀
