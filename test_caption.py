import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from models.blip import blip_decoder

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 384
model_path = 'checkpoints/model_base_capfilt_large.pth'

# Chargement du modèle
model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# Prétraitement de l'image
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

# Fonction de génération de la légende
def generate_caption(path):
    try:
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        caption = model.generate(image_tensor, sample=False, num_beams=3, max_length=30, min_length=10)[0]
        print(f" {os.path.basename(path)} → {caption}\n")
    except Exception as e:
        print(f" Erreur : {e}\n")

# Lecture de l'argument ou prompt interactif
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help="Chemin d'une image pour générer une légende")
    args = parser.parse_args()

    if args.image_path:
        if os.path.exists(args.image_path):
            generate_caption(args.image_path)
        else:
            print(" Le chemin fourni est invalide.")
    else:
        while True:
            path = input(" Entrez le chemin complet d'une image (ou 'exit') : ").strip()
            if path.lower() == 'exit':
                break
            if not os.path.exists(path):
                print(" Fichier introuvable. Réessaie.")
                continue
            generate_caption(path)
