"""
File: d2s/data.py
Description: Dataset definitions for IC9600 and IC9600Caption with robust file handling.
Date: 2025-08-13
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IC9600(Dataset):
	def __init__(self, txt_path, img_path, transform=None):
		super(IC9600, self).__init__()
		if not os.path.exists(txt_path):
			raise FileNotFoundError(f"Annotation file not found: {txt_path}")
		if not os.path.exists(img_path):
			raise FileNotFoundError(f"Image directory not found: {img_path}")
			
		self.txt_lines = self.readlines(txt_path)
		self.img_path = img_path
		self.transform = transform
		self.img_info_list = self.parse_lines(self.txt_lines)

	def parse_lines(self, lines):
		image_info_list = []
		for i, line in enumerate(lines):
			try:
				line_split = line.strip().split("  ")
				if len(line_split) < 2:
					print(f"Warning: Line {i+1} has invalid format, skip: {line.strip()}")
					continue
				img_name = line_split[0]
				img_label = line_split[1]
				
				# Check image file existence
				img_path = os.path.join(self.img_path, img_name)
				if not os.path.exists(img_path):
					print(f"Warning: Image file not found, skip: {img_path}")
					continue
					
				image_info_list.append((img_name, img_label))
			except Exception as e:
				print(f"Warning: Failed to parse line {i+1}: {e}, content: {line.strip()}")
				continue
		return image_info_list

	def readlines(self, txt_path):
		try:
			with open(txt_path, 'r', encoding='utf-8') as f:
				lines = f.readlines()
			return lines
		except Exception as e:
			raise RuntimeError(f"Failed to read file {txt_path}: {e}")

	def __getitem__(self, index):
		try:
			imgName, imgLabel = self.img_info_list[index]
			oriImgPath = os.path.join(self.img_path, imgName)
			
			if not os.path.exists(oriImgPath):
				raise FileNotFoundError(f"Image file not found: {oriImgPath}")
				
			img = Image.open(oriImgPath).convert("RGB")
			if self.transform:
				img = self.transform(img)
				
			try:
				label = torch.tensor(float(imgLabel), dtype=torch.float32)
			except ValueError:
				print(f"Warning: Failed to parse label as float: {imgLabel}")
				label = torch.tensor(0.0, dtype=torch.float32)
				
			return img, label, imgName
		except Exception as e:
			print(f"Error loading item {index}: {e}")
			# Return a default value
			if self.transform:
				dummy_img = torch.zeros(3, 224, 224)
			else:
				dummy_img = Image.new('RGB', (224, 224))
			return dummy_img, torch.tensor(0.0), "error"

	def __len__(self):
		return len(self.img_info_list)


class IC9600Caption(Dataset):
	def __init__(self, txt_path, img_path, tokenizer, transform=None, max_length=32):
		"""
		Text file format: <image_name><double-space><score><double-space><caption>
		img_path: directory with images
		tokenizer: HuggingFace tokenizer
		transform: image preprocessing
		max_length: max token length for caption
		"""
		super(IC9600Caption, self).__init__()
		
		if not os.path.exists(txt_path):
			raise FileNotFoundError(f"Annotation file not found: {txt_path}")
		if not os.path.exists(img_path):
			raise FileNotFoundError(f"Image directory not found: {img_path}")
			
		self.img_path = img_path
		self.transform = transform
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.samples = self._read_file(txt_path)

	def _read_file(self, path):
		samples = []
		try:
			with open(path, 'r', encoding='utf-8') as f:
				for i, line in enumerate(f):
					try:
						parts = line.strip().split("  ", 2)  # split into 3 fields
						if len(parts) != 3:
							print(f"Warning: Line {i+1} has invalid format, skip: {line.strip()}")
							continue
						img_name, score, caption = parts
						
						# Check image file existence
						img_path = os.path.join(self.img_path, img_name)
						if not os.path.exists(img_path):
							print(f"Warning: Image file not found, skip: {img_path}")
							continue
							
						try:
							score = float(score)
						except ValueError:
							print(f"Warning: Failed to parse score as float: {score}")
							continue
							
						samples.append((img_name, score, caption))
					except Exception as e:
						print(f"Warning: Failed to parse line {i+1}: {e}, content: {line.strip()}")
						continue
		except Exception as e:
			raise RuntimeError(f"Failed to read file {path}: {e}")
		return samples

	def __getitem__(self, index):
		try:
			img_name, score, caption = self.samples[index]

			# Load image
			img_path = os.path.join(self.img_path, img_name)
			if not os.path.exists(img_path):
				raise FileNotFoundError(f"Image file not found: {img_path}")
				
			img = Image.open(img_path).convert("RGB")
			if self.transform:
				img = self.transform(img)

			# Tokenizer on caption
			try:
				encoding = self.tokenizer(
					caption,
					padding="max_length",
					truncation=True,
					max_length=self.max_length,
					return_tensors="pt"
				)
			except Exception as e:
				print(f"Warning: Tokenizer failed: {e}")
				# fallback encoding
				encoding = {
					"input_ids": torch.zeros(self.max_length, dtype=torch.long),
					"attention_mask": torch.zeros(self.max_length, dtype=torch.long)
				}

			return {
				"image": img,
				"input_ids": encoding["input_ids"].squeeze(0),
				"attention_mask": encoding["attention_mask"].squeeze(0),
				"score": torch.tensor(score, dtype=torch.float32),
				"image_name": img_name
			}
		except Exception as e:
			print(f"Error loading item {index}: {e}")
			# Return a default value
			if self.transform:
				dummy_img = torch.zeros(3, 224, 224)
			else:
				dummy_img = Image.new('RGB', (224, 224))
			return {
				"image": dummy_img,
				"input_ids": torch.zeros(self.max_length, dtype=torch.long),
				"attention_mask": torch.zeros(self.max_length, dtype=torch.long),
				"score": torch.tensor(0.0, dtype=torch.float32),
				"image_name": "error"
			}

	def __len__(self):
		return len(self.samples)