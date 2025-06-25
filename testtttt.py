import os
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "Salesforce/blip-image-captioning-large"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

root_dir = r"C:\Users\User\BLIP\MA_Culture_Images"
output_json = "annotation/generated_captions.json"

data = []

def generate_caption(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except:
        return None

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for img_file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder_path, img_file)
            caption = generate_caption(full_path)
            if caption:
                data.append({"image": os.path.relpath(full_path, root_dir), "caption": caption})

# Enregistre dans un fichier JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ {len(data)} légendes générées et sauvegardées dans {output_json}")
