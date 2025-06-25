import os
from PIL import Image
import torch
from torchvision import transforms
from models.blip import blip_decoder

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_folder = 'C:/archive/coco2014/images/test2014'

# Charger le modèle BLIP
model = blip_decoder(pretrained='checkpoints/model_base_capfilt_large.pth', image_size=384, vit='base')
model.eval()
model = model.to(device)

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

# Récupérer toutes les images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Traiter chaque image
for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f" Erreur de lecture de l'image {img_name}: {e}")
        continue

    image = transform(image).unsqueeze(0).to(device)

    # Générer la légende
    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)[0]

    print(f" {img_name} → {caption}")
