"""
File: val.py
Description: Validation and single-image inference entrypoint for D2S; loads checkpoints and reports metrics.
Date: 2025-08-13
"""
import os
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from d2s.model import FusionRegressor
from d2s.data import IC9600Caption
from d2s.loss import MSE_IB_Loss
from d2s.utils import load_checkpoint, compute_srcc_plcc
from d2s.config import load_config


def create_transforms(image_size: int):
	"""Create image transforms"""
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


def validate(model, dataloader, loss_fn, device):
	model.eval()
	total_loss, total_mse, total_ib = 0, 0, 0
	all_preds, all_scores = [], []
	
	with torch.no_grad():
		for batch in dataloader:
			images = batch["image"].to(device)
			scores = batch["score"].to(device)

			# Forward pass: get predictions and features
			if hasattr(model, 'use_text') and model.use_text:
				# Multimodal mode: include text inputs
				input_ids = batch["input_ids"].to(device)
				attn_mask = batch["attention_mask"].to(device)
				preds, feats = model(images, input_ids, attn_mask)
			else:
				# Text-free mode: only vision inputs
				preds, feats = model(images)

			loss, mse_val, ib_val = loss_fn(preds, scores, feats)

			total_loss += loss.item()
			total_mse += mse_val
			total_ib += ib_val
			
			# Collect predictions and labels to compute correlations
			all_preds.extend(preds.cpu().numpy())
			all_scores.extend(scores.cpu().numpy())

	n = len(dataloader)
	# Compute SRCC and PLCC correlations
	srcc, plcc = compute_srcc_plcc(all_preds, all_scores)
	
	return total_loss / n, total_mse / n, total_ib / n, srcc, plcc


def predict_single_image(model, tokenizer, img_path, caption, transform, max_length, device):
	model.eval()
	with torch.no_grad():
		img = Image.open(img_path).convert("RGB")
		img = transform(img).unsqueeze(0).to(device)

		if hasattr(model, 'use_text') and model.use_text:
			# Multimodal mode: include text inputs
			if tokenizer is None:
				raise ValueError("Tokenizer required in multimodal mode")
			encoding = tokenizer(
				caption,
				padding="max_length",
				truncation=True,
				max_length=max_length,
				return_tensors="pt"
			)
			input_ids = encoding["input_ids"].to(device)
			attn_mask = encoding["attention_mask"].to(device)
			pred, _ = model(img, input_ids, attn_mask)
		else:
			# Text-free mode: only vision inputs
			pred, _ = model(img)

	return pred.item()


def main():
	parser = argparse.ArgumentParser(description="D2S Validation")
	parser.add_argument("--config", type=str, default="config/base.yaml", help="Path to config file")
	parser.add_argument("--image_size", type=int, help="Override image size from config")
	args = parser.parse_args()

	# Load config
	try:
		config = load_config(args.config)
		
		# Override image_size from CLI if provided
		if args.image_size:
			config.dataset.image_size = args.image_size
		
		# Validate config
		if not config.validate():
			print("Config validation failed. Exit.")
			return
		
		# Print config summary
		config.print_summary()
		
	except Exception as e:
		print(f"Failed to load config: {e}")
		return

	# Get device
	device = config.get_device()
	print(f"Device: {device}")

	# Create transforms
	transform = create_transforms(config.dataset.image_size)

	# Data loading
	tokenizer = None
	if config.model.text_model_name and config.model.text_model_name.strip() != "":
		try:
			tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
			print(f"Loaded tokenizer: {config.model.text_model_name}")
		except Exception as e:
			print(f"Failed to load tokenizer: {e}")
			return
	else:
		print("Text-free mode: no tokenizer needed")

	try:
		val_dataset = IC9600Caption(
			config.dataset.val_file, 
			config.dataset.img_dir, 
			tokenizer, 
			transform, 
			config.dataset.max_length
		)
		val_loader = DataLoader(val_dataset, batch_size=config.val.batch_size, shuffle=False)
		print(f"Dataset loaded: val={len(val_dataset)} samples")
		
	except Exception as e:
		print(f"Failed to load dataset: {e}")
		return

	# Create model
	try:
		model = FusionRegressor(
			config.model.vision_model_name,
			config.model.text_model_name,
			config.model.hidden_dim,
			config.model.use_icnet_head
		).to(device)

		if hasattr(model, 'use_text') and model.use_text:
			print(f"Model created: {config.model.vision_model_name} + {config.model.text_model_name}")
			print(f"  Use ICNetHead: {config.model.use_icnet_head}")
		else:
			print(f"Model created: {config.model.vision_model_name} (text-free mode)")
			print(f"  Use ICNetHead: {config.model.use_icnet_head}")

	except Exception as e:
		print(f"Failed to create model: {e}")
		return

	# Load checkpoint
	checkpoint_path = config.val.checkpoint
	if os.path.exists(checkpoint_path):
		try:
			epoch = load_checkpoint(checkpoint_path, model, device=device)
			print(f"✓ Checkpoint loaded. Epoch: {epoch}")
		except Exception as e:
			print(f"Failed to load checkpoint: {e}")
			print("Use randomly initialized model")
	else:
		print(f"Checkpoint not found: {checkpoint_path}. Use randomly initialized model")

	# Loss
	loss_fn = MSE_IB_Loss(beta=config.train.beta)

	# Validation
	print(f"\nStart validation...")
	val_loss, val_mse, val_ib, val_srcc, val_plcc = validate(model, val_loader, loss_fn, device)

	print(f"\n[Validation]")
	print(f"Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | IB: {val_ib:.6f} | SRCC: {val_srcc:.4f} | PLCC: {val_plcc:.4f}")

	# Single image inference example
	print(f"\n[Single-image inference example]")
	test_img_name = "example.jpg"
	test_caption = "a cat sitting on a table"
	test_img_path = os.path.join(config.dataset.img_dir, test_img_name)
	
	if os.path.exists(test_img_path):
		try:
			pred_score = predict_single_image(
				model, tokenizer, test_img_path, test_caption, 
				transform, config.dataset.max_length, device
			)
			print(f"Image: {test_img_name}")
			print(f"Caption: {test_caption}")
			print(f"Predicted complexity score: {pred_score:.4f}")
		except Exception as e:
			print(f"Inference failed: {e}")
	else:
		print(f"Example image not found: {test_img_path}")
		print("Please put the test image into the image directory or change test_img_name")


if __name__ == "__main__":
	main()
