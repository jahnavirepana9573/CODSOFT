import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os

def generate_caption(image_path):
    if not os.path.exists(image_path):
        print("Image file not found!")
        return

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    output = model.generate(pixel_values, max_length=20)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nGenerated Caption:")
    print("-------------------")
    print(caption)

if __name__ == "__main__":
    path = input("Enter image path: ")
    generate_caption(path)
