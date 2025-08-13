"""
File: caption_gen.py
Description: Generate captions for images using BLIP-2 and update IC9600 train/val text files.
Date: 2025-08-13
"""
import os
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Paths
dataset_dir = "IC9600"
images_dir = os.path.join(dataset_dir, "images")

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

def generate_caption(image_path):
	image = Image.open(image_path).convert("RGB")
	inputs = processor(images=image, return_tensors="pt").to(device)
	out = model.generate(**inputs, max_new_tokens=30)
	caption = processor.decode(out[0], skip_special_tokens=True)
	return caption


def process_txt_file(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.readlines()

	new_lines = []
	for line in lines:
		parts = line.strip().split()
		if len(parts) < 2:
			print(f"Skip invalid line: {line.strip()}")
			continue
		image_name, score = parts[0], parts[1]
		img_path = os.path.join(images_dir, image_name)

		if not os.path.exists(img_path):
			print(f"Image not found: {img_path}")
			caption = "Image not found"
		else:
			try:
				caption = generate_caption(img_path)
			except Exception as e:
				print(f"Failed to generate caption: {e}")
				caption = "Caption Error"

		new_lines.append(f"{image_name}  {score}  {caption}\n")

	with open(txt_path, "w", encoding="utf-8") as f:
		f.writelines(new_lines)

# Process both train.txt and val.txt
process_txt_file(os.path.join(dataset_dir, "train.txt"))
process_txt_file(os.path.join(dataset_dir, "val.txt"))

print("Processing complete!")
