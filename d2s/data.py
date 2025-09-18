import os

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_transforms(image_size: int, is_training: bool = True):
	base_transforms = [
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]

	if is_training:
		base_transforms.insert(1, transforms.RandomHorizontalFlip())

	return transforms.Compose(base_transforms)


class IC9600Caption(Dataset):
	def __init__(self, ann_file, img_dir, transform, tokenizer):
		super().__init__()
		self.img_dir = img_dir
		self.transform = transform
		self.tokenizer = tokenizer

		self.samples = []
		with open(ann_file, 'r', encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split("  ", 2)
				if len(parts) < 3:
					continue
				name, score, caption = parts[0], float(parts[1]), parts[2]
				self.samples.append((name, caption, score))

		if self.tokenizer:
			captions = [cap for (_, cap, _) in self.samples]
			all_tokens = self.tokenizer(captions)  # [N, L], dtype long
			self.all_tokens = all_tokens

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		name, caption, score = self.samples[idx]
		path = os.path.join(self.img_dir, name)
		with Image.open(path) as img:
			img = img.convert("RGB")
			img = self.transform(img)
		if self.tokenizer:
			tokens = self.all_tokens[idx]  # [L]
			return img, tokens, torch.tensor(score, dtype=torch.float32), name
		else:
			return img, torch.tensor(0), torch.tensor(score, dtype=torch.float32), name


class IC9600(Dataset):
	def __init__(self, txt_path, img_path, transform=None):
		super(IC9600, self).__init__()
		self.txt_lines = self.readlines(txt_path)
		self.img_path = img_path
		self.transform = transform
		self.img_info_list = self.parse_lines(self.txt_lines)

	def parse_lines(self, lines):
		image_info_list = []
		for line in lines:
			line_split = line.strip().split("  ")
			img_name = line_split[0]
			img_label = line_split[1]
			image_info_list.append((img_name, img_label))
		return image_info_list

	def readlines(self, txt_path):
		f = open(txt_path, 'r')
		lines = f.readlines()
		f.close()
		return lines

	def __getitem__(self, index):
		imgName, imgLabel = self.img_info_list[index]
		oriImgPath = os.path.join(self.img_path, imgName)
		try:
			img = Image.open(oriImgPath).convert("RGB")
		except:
			img = Image.open(oriImgPath.replace(".jpg", ".png")).convert("RGB")
		img = self.transform(img)
		label = torch.tensor(float(imgLabel))
		return img, label, imgName

	def __len__(self):
		return len(self.img_info_list)
