import torch
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM
)

# =========================================================
# 🔹 BLIP BASE
# =========================================================
class BlipBaseCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(output[0], skip_special_tokens=True)


# =========================================================
# 🔹 ViT-GPT2
# =========================================================
class VitGpt2Captioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        ).to(self.device)

        self.processor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

    def caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, max_length=30)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# =========================================================
# 🔹 GIT (microsoft/git-base)
# =========================================================
class GitCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-base"
        ).to(self.device)

    def caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(output[0], skip_special_tokens=True)


# =========================================================
# 🔹 SINGLETONS (MODELS LOADED ONCE)
# =========================================================
_blip_instance = None
_vit_gpt2_instance = None
_git_instance = None


def get_blip_base():
    global _blip_instance
    if _blip_instance is None:
        _blip_instance = BlipBaseCaptioner()
    return _blip_instance


def get_vit_gpt2():
    global _vit_gpt2_instance
    if _vit_gpt2_instance is None:
        _vit_gpt2_instance = VitGpt2Captioner()
    return _vit_gpt2_instance


def get_git():
    global _git_instance
    if _git_instance is None:
        _git_instance = GitCaptioner()
    return _git_instance


# =========================================================
# 🔹 MAIN MULTI-MODEL API
# =========================================================
def generate_all_captions(image_path: str) -> dict:
    """
    Generate captions using all available vision–language models.

    Returns:
        dict: {
            "BLIP Base": "...",
            "ViT-GPT2": "...",
            "GIT": "..."
        }
    """
    return {
        "BLIP Base": get_blip_base().caption(image_path),
        "ViT-GPT2": get_vit_gpt2().caption(image_path),
        "GIT": get_git().caption(image_path)
    }
