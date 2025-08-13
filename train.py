"""
File: train.py
Description: Training entrypoint for D2S; training/validation loops, checkpointing and correlation metrics.
Date: 2025-08-13
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from d2s.model import FusionRegressor
from d2s.data import IC9600Caption
from d2s.loss import MSE_IB_Loss
from d2s.utils import save_checkpoint, compute_srcc_plcc
from d2s.config import load_config


def create_transforms(image_size: int, is_training: bool = True):
	"""Create image transforms"""
	base_transforms = [
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
	
	if is_training:
		base_transforms.insert(1, transforms.RandomHorizontalFlip())
	
	return transforms.Compose(base_transforms)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
	model.train()
	total_loss, total_mse, total_ib = 0, 0, 0
	all_preds, all_scores = [], []
	
	for batch in dataloader:
		images = batch["image"].to(device)
		input_ids = batch["input_ids"].to(device)
		attn_mask = batch["attention_mask"].to(device)
		scores = batch["score"].to(device)

		preds, feats = model(images, input_ids, attn_mask)
		loss, mse_val, ib_val = loss_fn(preds, scores, feats)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

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


def validate(model, dataloader, loss_fn, device):
	model.eval()
	total_loss, total_mse, total_ib = 0, 0, 0
	all_preds, all_scores = [], []
	
	with torch.no_grad():
		for batch in dataloader:
			images = batch["image"].to(device)
			input_ids = batch["input_ids"].to(device)
			attn_mask = batch["attention_mask"].to(device)
			scores = batch["score"].to(device)

			preds, feats = model(images, input_ids, attn_mask)
			loss, reg_loss, ib_loss = loss_fn(preds, scores, feats)

			total_loss += loss.item()
			total_mse += reg_loss
			total_ib += ib_loss
			
			# Collect predictions and labels to compute correlations
			all_preds.extend(preds.cpu().numpy())
			all_scores.extend(scores.cpu().numpy())

	n = len(dataloader)
	# Compute SRCC and PLCC correlations
	srcc, plcc = compute_srcc_plcc(all_preds, all_scores)
	
	return total_loss / n, total_mse / n, total_ib / n, srcc, plcc


def main():
	parser = argparse.ArgumentParser(description="D2S Training")
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
	train_transform = create_transforms(config.dataset.image_size, is_training=True)
	val_transform = create_transforms(config.dataset.image_size, is_training=False)

	# Data loading
	try:
		tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
		print(f"Loaded tokenizer: {config.model.text_model_name}")
	except Exception as e:
		print(f"Failed to load tokenizer: {e}")
		return

	try:
		train_dataset = IC9600Caption(
			config.dataset.train_file, 
			config.dataset.img_dir, 
			tokenizer, 
			train_transform, 
			config.dataset.max_length
		)
		val_dataset = IC9600Caption(
			config.dataset.val_file, 
			config.dataset.img_dir, 
			tokenizer, 
			val_transform, 
			config.dataset.max_length
		)
		
		train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
		
		print(f"Datasets loaded: train={len(train_dataset)} samples, val={len(val_dataset)} samples")
		
	except Exception as e:
		print(f"Failed to load datasets: {e}")
		return

	# Create model
	try:
		model = FusionRegressor(
			config.model.vision_model_name,
			config.model.text_model_name,
			config.model.hidden_dim
		).to(device)
		print(f"Model created: {config.model.vision_model_name} + {config.model.text_model_name}")
	except Exception as e:
		print(f"Failed to create model: {e}")
		return

	# Optimizer and loss
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
	loss_fn = MSE_IB_Loss(beta=config.train.beta)

	# Training loop
	print(f"\nStart training for {config.train.epochs} epochs...")
	best_val_loss = float('inf')
	
	for epoch in range(config.train.epochs):
		# Train
		train_loss, train_mse, train_ib, train_srcc, train_plcc = train_one_epoch(
			model, train_loader, optimizer, loss_fn, device
		)
		
		# Validate
		val_loss, val_mse, val_ib, val_srcc, val_plcc = validate(
			model, val_loader, loss_fn, device
		)

		# Logs
		print(f"Epoch [{epoch+1}/{config.train.epochs}]")
		print(f"  Train Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | IB: {train_ib:.6f} | SRCC: {train_srcc:.4f} | PLCC: {train_plcc:.4f}")
		print(f"  Val   Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | IB: {val_ib:.6f} | SRCC: {val_srcc:.4f} | PLCC: {val_plcc:.4f}")
		
		# Save best checkpoint
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=True)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			print(f"  ✓ Saved best model: {checkpoint_path}")
		
		# Periodic checkpoint
		if (epoch + 1) % config.train.save_interval == 0:
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=False)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			print(f"  ✓ Saved checkpoint: {checkpoint_path}")

	# Save final model
	final_checkpoint_path = config.get_final_checkpoint_path()
	save_checkpoint(model, optimizer, config.train.epochs, final_checkpoint_path)
	print(f"\n✓ Training complete. Final model saved to: {final_checkpoint_path}")


if __name__ == "__main__":
	main()
