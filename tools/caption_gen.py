import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

prompt1 = "This image shows"
prompt2 = "The main objects are"
prompt3 = "The background looks"
prompt4 = "The overall visual complexity is"
prompts = [prompt1, prompt2, prompt3, prompt4]

def generate_caption(image_path: str) -> str:
	raw_image = Image.open(image_path).convert("RGB")

	caption = ""

	for prompt in prompts:
		inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
		output_ids = model.generate(**inputs, max_new_tokens=64)
		cur_caption = processor.decode(output_ids[0], skip_special_tokens=True)
		caption += (cur_caption + ". ")

	caption = caption.replace("\n", "").replace("\r", "")

	return caption

def process_txt_file(txt_path, images_dir, output_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.readlines()

	new_lines = []
	for line in tqdm(lines, desc=f"Captions for {os.path.basename(txt_path)}"):
		parts = line.strip().split("  ")
		if len(parts) < 2:
			continue
		image_name, score = parts[0], parts[1]
		img_path = os.path.join(images_dir, image_name)

		if not os.path.exists(img_path):
			caption = "Image not found"
		else:
			try:
				caption = generate_caption(img_path)

			except Exception as e:
				print(f"Error with {img_path}: {e}")
				caption = "Caption Error"


		new_lines.append(f"{image_name}  {score}  {caption}\n")

	with open(output_path, "w", encoding="utf-8") as f:
		f.writelines(new_lines)


if __name__ == "__main__":
	dataset_dir = "../data/IC9600"
	images_dir = os.path.join(dataset_dir, "images")
	process_txt_file(
		txt_path=os.path.join(dataset_dir, "train.txt"),
		images_dir=images_dir,
		output_path=os.path.join(dataset_dir, "train_caption.txt")
	)

	process_txt_file(
		txt_path=os.path.join(dataset_dir, "test.txt"),
		images_dir=images_dir,
		output_path=os.path.join(dataset_dir, "test_caption.txt")
	)
