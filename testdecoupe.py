import os
import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
from difflib import SequenceMatcher

# === CONFIGURATION ===
image_path = "images_test/image10.jpeg"  # Remplace ce chemin si nécessaire
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip2_model_name = "Salesforce/blip2-opt-2.7b"
yolo_model_path = "yolov8n.pt"

# === CHARGEMENT DES MODÈLES ===
processor = Blip2Processor.from_pretrained(blip2_model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    blip2_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
yolo_model = YOLO(yolo_model_path)

# === CHARGER L’IMAGE ET DÉTECTER LES OBJETS ===
image = Image.open(image_path).convert("RGB")
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
results = yolo_model(image_cv)[0]

# === PARAMÈTRES DE FILTRAGE ===
relevant_objects = ["car", "person", "truck", "bus", "backpack", "handbag"]
grouped_captions = defaultdict(list)

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

# === GÉNÉRER CAPTIONS UNIQUES PAR OBJET ===
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    label = results.names[int(box.cls[0])]
    if label not in relevant_objects:
        continue

    cropped = image.crop((x1, y1, x2, y2))
    inputs = processor(images=cropped, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)
    caption = processor.decode(output[0], skip_special_tokens=True)

    if "skateboard" in caption and label not in ["skateboard", "person"]:
        continue

    if not any(is_similar(caption, existing) for existing in grouped_captions[label]):
        grouped_captions[label].append(caption)

# === GÉNÉRER UNE DESCRIPTION GLOBALE PROPRE ===
global_description = "The image shows "
segments = []

if "car" in grouped_captions:
    count = len(grouped_captions["car"])
    segments.append(f"{count} car{'s' if count > 1 else ''} parked in a lot or along the street")

if "person" in grouped_captions:
    count = len(grouped_captions["person"])
    segments.append(f"{count} person{'s' if count > 1 else ''} walking or standing nearby")

if "backpack" in grouped_captions:
    segments.append("a backpack visible on someone or nearby")

if "handbag" in grouped_captions:
    segments.append("a handbag present in the scene")

if segments:
    global_description += ", ".join(segments[:-1])
    if len(segments) > 1:
        global_description += ", and " + segments[-1]
    else:
        global_description += segments[0]
    global_description += "."
else:
    global_description += "no clearly recognizable elements based on object detection."

# === AFFICHAGE FINAL ===
print("\n🧾 DESCRIPTION GLOBALE :")
print(global_description)
