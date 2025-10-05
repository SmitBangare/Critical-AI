import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


def main():
    local_dir = Path(r"D:/New folder/Critical-AI/models/biovil_t")
    image_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        str(local_dir), trust_remote_code=True, local_files_only=True
    ).to(device).eval()

    print("Loaded BioViL-T from:", local_dir)
    print("Device:", device)

    if image_arg and image_arg.exists():
        # Minimal transforms compatible with BioViL-T (224x224, ImageNet stats)
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_arg).convert("RGB")
        pixel_values = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.get_projected_image_embeddings(
                pixel_values=pixel_values
            )
        print("Embedding shape:", tuple(emb.shape))
    else:
        print("No image path provided. Pass a path to a CXR image to compute an embedding.")


if __name__ == "__main__":
    main()


