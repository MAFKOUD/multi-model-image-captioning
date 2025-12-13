Parfait ✅
On va faire **un README académique, clair, crédible et bien noté**.
Tu pourras **copier-coller tel quel** dans `README.md`.

Je te donne **une version EN ANGLAIS** (fortement recommandée pour un projet IA).
Si tu veux ensuite une version FR, je pourrai te la traduire.

---

# 📘 README.md — **Multi-Model Consensus for Image Captioning**

```md
# 🧠 Multi-Model Consensus for Image Captioning

## Project Overview
This project presents an **intelligent image captioning system** that improves reliability by
combining multiple vision–language models and applying an **explicit reasoning pipeline**.

Instead of relying on a single model, the system:
- Generates captions using **multiple pretrained models**
- Measures **semantic agreement** between them
- Selects a **consensus caption**
- Applies **Tree of Thoughts (ToT)** reasoning
- Uses **Gemini as a reasoning and refinement agent**, not as an oracle
- Produces a final, coherent, and explainable caption via a **Streamlit interface**

This approach is particularly suited for **high-reliability AI systems** where transparency,
consistency, and reasoning traceability are required.

---

## 🧩 Architecture Overview

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

These models are lightweight and efficient, but may produce inconsistent outputs individually.

### Reasoning Model
- **Gemini (Google Generative AI)**  
Used strictly as a **reasoning and refinement agent**:
- It does **not** analyze the image directly
- It only reasons over captions already generated
- It is constrained to avoid hallucinations

---

## 🧠 Reasoning Techniques Implemented

### 1️⃣ Semantic Consensus
Captions are embedded using **Sentence-BERT**.
The system computes **pairwise cosine similarity** and selects the caption with the
highest average agreement.

This ensures the selected caption represents the **most consistent interpretation**.

---

### 2️⃣ Tree of Thoughts (ToT)
When enabled, Gemini generates multiple candidate refinements.
Each candidate is evaluated based on:
- Semantic similarity to the consensus caption
- Conciseness and clarity

The best candidate is selected programmatically, making the reasoning **explicit and traceable**.

---

### 3️⃣ Self-Correction
A final verification step checks for:
- Redundancy
- Inconsistencies
- Over-specification

This improves robustness without introducing new information.

---

## 📊 Evaluation Metrics

When ground truth captions are available (`data.json`), the system computes:
- **SPICE**
- **BLEU-1 to BLEU-4**
- **METEOR**
- **ROUGE-L**

These metrics are shown transparently in the Streamlit interface.

---

## 🖥️ Streamlit Interface

The application provides:
- Image upload
- Toggle options for Tree of Thoughts and self-correction
- Display of all model captions
- Consensus decision
- Final caption
- Evaluation scores
- **Dynamic, data-driven agent explanation**

---

## 📂 Project Structure

```

.
├── app.py                 # Streamlit UI
├── pipeline.py            # Main reasoning pipeline
├── caption_models.py      # Caption generation models
├── consensus.py           # Semantic similarity & consensus
├── tot_selector.py        # Tree of Thoughts selection logic
├── gemini_fusion.py       # Gemini reasoning & refinement
├── agent_explanation.py   # Dynamic reasoning explanation
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

## 🔍 Key Contributions

* Multi-model caption consensus
* Explicit semantic reasoning
* Tree of Thoughts integration
* Gemini used as a reasoning agent (not an oracle)
* Explainable AI with traceable decisions
* Interactive Streamlit interface

---

## 👥 Team Members

* Bellouch Ayoub
* Mafkoud Khaoula
* Hamid Hiba
* Berkani Mohammed Adam

---

## 📌 Notes

* The Gemini API key is embedded internally for demonstration purposes.
* No user input is required for API configuration.
* The system is designed for educational and research use.

---

## 📜 License

This project is released for academic and educational purposes.

````

---

érifie le README”** ou **“On prépare la soutenance”** 💪
