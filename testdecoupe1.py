import os
import cv2
import torch
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = "images_test"
model_blip = "Salesforce/blip2-opt-2.7b"
model_yolo = "yolov8n.pt"

# Load models
processor = Blip2Processor.from_pretrained(model_blip)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_blip,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
yolo = YOLO(model_yolo)

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def describe_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo(image_cv)[0]
    captions = []
    label_map = results.names

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = label_map[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = image.crop((x1, y1, x2, y2))
        prompt = f"Describe the {label} in the image."
        inputs = processor(images=cropped, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        caption = processor.decode(out[0], skip_special_tokens=True)
        if not any(is_similar(caption, c) for c in captions):
            captions.append(caption)

    if not captions:
        return "The image contains no clearly describable objects."
    
    if len(captions) == 1:
        return f"In the image, {captions[0]}."
    else:
        return "In the image, " + ", ".join(captions[:-1]) + ", and " + captions[-1] + "."

# Traitement
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_folder, filename)
        print(f"\n🖼️ {filename}")
        print("🧾", describe_image(path))
