"""
File: caption_gen.py
Description: Generate captions for images using Florence-2 and update IC9600 train/val text files.
Date: 2025-08-13
"""
import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Paths
dataset_dir = "IC9600"
images_dir = os.path.join(dataset_dir, "images")

# Florence-2 model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
	"microsoft/Florence-2-large",
	torch_dtype=torch_dtype,
	trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def generate_caption(image_path, task_prompt: str = "<MORE_DETAILED_CAPTION>", text_input: str | None = None) -> str:
	image = Image.open(image_path).convert("RGB")
	prompt = task_prompt if text_input is None else task_prompt + text_input
	inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
	generated_ids = model.generate(
		input_ids=inputs["input_ids"],
		pixel_values=inputs["pixel_values"],
		max_new_tokens=256,
		num_beams=3
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	try:
		parsed = processor.post_process_generation(
			generated_text,
			task=task_prompt,
			image_size=(image.width, image.height)
		)
		# Try to extract a string caption from parsed output
		if isinstance(parsed, dict):
			for _, v in parsed.items():
				if isinstance(v, str):
					return v
		if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
			return parsed[0]
	except Exception:
		pass
	# Fallback to raw decoded text
	return generated_text


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
