"""
File: train.py
Description: Training entrypoint for D2S; training/validation loops, checkpointing and correlation metrics.
Date: 2025-08-13
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from d2s.model import FusionRegressor
from d2s.data import IC9600Caption
from d2s.loss import MSE_IB_Loss
from d2s.utils import save_checkpoint, compute_srcc_plcc, WarmUpLR
from d2s.config import load_config


def create_transforms(image_size: int, is_training: bool = True):
	"""
	Create image transforms for data preprocessing
	
	Args:
		image_size (int): Target image size for resizing
		is_training (bool): Whether this is for training (adds augmentation)
	
	Returns:
		transforms.Compose: Composed transform pipeline
	"""
	base_transforms = [
		transforms.Resize((image_size, image_size)),  # Resize to target dimensions
		transforms.ToTensor(),                        # Convert PIL to tensor
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
	]
	
	if is_training:
		base_transforms.insert(1, transforms.RandomHorizontalFlip())  # Add horizontal flip for training
	
	return transforms.Compose(base_transforms)


def train_one_epoch(model, dataloader, optimizer, score_loss, map_loss, device, use_ic_map=False):
	"""
	Train the model for one epoch
	
	Args:
		model: The neural network model
		dataloader: Training data loader
		optimizer: Optimizer for updating model parameters
		score_loss: Score Loss function
		map_loss: Map Loss function
		device: Device to run training on (CPU/GPU)
		use_ic_map (bool): Whether to use image complexity map

	Returns:
		tuple: (avg_loss, avg_mse, avg_ib, srcc, plcc) - Average losses and correlation metrics
	"""
	model.train()  # Set model to training mode
	total_loss, total_mse, total_ib = 0, 0, 0
	all_preds, all_scores = [], []
	
	# Record start time for this epoch
	epoch_start_time = time.time()
	
	for iter_idx, batch in enumerate(dataloader):
		# Record start time for this iteration
		iter_start_time = time.time()
		
		# Move batch data to device
		images = batch["image"].to(device)           # Image tensors
		scores = batch["score"].to(device)           # Ground truth quality scores

		# Forward pass: get predictions and features
		if hasattr(model, 'use_text') and model.use_text:
			# Multimodal mode: include text inputs
			input_ids = batch["input_ids"].to(device)    # Tokenized text input IDs
			attn_mask = batch["attention_mask"].to(device)  # Attention mask for text
			preds, feats, ic_map = model(images, input_ids, attn_mask)
		else:
			# Text-free mode: only vision inputs
			preds, feats, ic_map = model(images)
		
		# Compute loss components (total loss, MSE loss, Information Bottleneck loss)
		loss1, mse, ib = score_loss(preds, scores, feats)
		if use_ic_map:
			score_map = ic_map.mean(axis=(1, 2, 3))  # Average IC map across spatial dimensions
			loss2 = map_loss(score_map, scores, feats)
			loss = 0.9 * loss1 + 0.1 * loss2
		else:
			loss = loss1

		# Backward pass and optimization
		optimizer.zero_grad()  # Clear previous gradients
		loss.backward()         # Compute gradients
		optimizer.step()        # Update model parameters

		# Accumulate losses for epoch average
		total_loss += loss.item()
		total_mse += mse
		total_ib += ib

		# Collect predictions and labels for correlation computation
		all_preds.extend(preds.detach().cpu().numpy())
		all_scores.extend(scores.detach().cpu().numpy())
		
		# Calculate iteration time and estimated total time
		iter_time = time.time() - iter_start_time
		elapsed_time = time.time() - epoch_start_time
		avg_iter_time = elapsed_time / (iter_idx + 1)
		remaining_iters = len(dataloader) - (iter_idx + 1)
		estimated_remaining_time = remaining_iters * avg_iter_time
		total_estimated_time = elapsed_time + estimated_remaining_time
		
		# Print training progress for each iteration
		print(f"  Iter [{iter_idx+1:3d}/{len(dataloader):3d}] | "
			  f"Loss: {loss.item():.4f} | MSE: {mse:.4f} | IB: {ib:.6f} | "
			  f"Time: {iter_time:.3f}s | ETA: {estimated_remaining_time:.1f}s | "
			  f"Total ETA: {total_estimated_time:.1f}s")

	n = len(dataloader)
	# Compute correlation metrics (Spearman's Rank Correlation Coefficient and Pearson's Linear Correlation Coefficient)
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)
	
	return total_loss / n, total_mse / n, total_ib / n, srcc, plcc, rmse, rmae


def validate(model, dataloader, score_loss, map_loss, device, use_ic_map=False):
	"""
	Validate the model on validation dataset
	
	Args:
		model: The neural network model
		dataloader: Validation data loader
		score_loss: Score Loss function
		map_loss: Map Loss function
		device: Device to run validation on (CPU/GPU)
		use_ic_map (bool): Whether to use image complexity map
	
	Returns:
		tuple: (avg_loss, avg_mse, avg_ib, srcc, plcc) - Average losses and correlation metrics
	"""
	model.eval()  # Set model to evaluation mode
	total_loss, total_mse, total_ib = 0, 0, 0
	all_preds, all_scores = [], []
	
	# Record start time for validation
	val_start_time = time.time()
	
	with torch.no_grad():  # Disable gradient computation for validation
		for batch in dataloader:
			# Move batch data to device
			images = batch["image"].to(device)
			scores = batch["score"].to(device)

			# Forward pass: get predictions and features
			if hasattr(model, 'use_text') and model.use_text:
				# Multimodal mode: include text inputs
				input_ids = batch["input_ids"].to(device)
				attn_mask = batch["attention_mask"].to(device)
				preds, feats, ic_map = model(images, input_ids, attn_mask)
			else:
				# Text-free mode: only vision inputs
				preds, feats = model(images)

			# Compute loss components (total loss, MSE loss, Information Bottleneck loss)
			loss1, mse, ib = score_loss(preds, scores, feats)
			if use_ic_map:
				score_map = ic_map.mean(axis=(1, 2, 3))  # Average IC map across spatial dimensions
				loss2 = map_loss(score_map, scores, feats)
				loss = 0.9 * loss1 + 0.1 * loss2
			else:
				loss = loss1

			# Accumulate losses for epoch average
			total_loss += loss.item()
			total_mse += mse
			total_ib += ib
			
			# Collect predictions and labels for correlation computation
			all_preds.extend(preds.detach().cpu().numpy())
			all_scores.extend(scores.detach().cpu().numpy())

	n = len(dataloader)
	# Compute correlation metrics
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)
	
	# Calculate validation time
	val_time = time.time() - val_start_time
	
	return total_loss / n, total_mse / n, total_ib / n, srcc, plcc, rmse, rmae, val_time


def main():
	"""
	Main training function that orchestrates the entire training process
	"""
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="D2S Training - Image Quality Assessment Model")
	parser.add_argument("--config", type=str, default="config/base.yaml", 
					   help="Path to configuration file")
	parser.add_argument("--image_size", type=int, 
					   help="Override image size from configuration file")
	args = parser.parse_args()

	# Load and validate configuration
	try:
		config = load_config(args.config)
		
		# Override image_size from command line if provided
		if args.image_size:
			config.dataset.image_size = args.image_size
			print(f"Overriding image size to: {args.image_size}")
		
		# Validate configuration parameters
		if not config.validate():
			print("Configuration validation failed. Exiting.")
			return
		
		# Print configuration summary
		config.print_summary()
		
	except Exception as e:
		print(f"Failed to load configuration: {e}")
		return

	# Determine device (GPU if available, otherwise CPU)
	device = config.get_device()
	print(f"Using device: {device}")

	# Create image transforms for training and validation
	train_transform = create_transforms(config.dataset.image_size, is_training=True)
	val_transform = create_transforms(config.dataset.image_size, is_training=False)
	print(f"Image transforms created for size: {config.dataset.image_size}x{config.dataset.image_size}")

	# Load tokenizer for text processing
	tokenizer = None
	if config.model.text_model_name and config.model.text_model_name.strip() != "":
		try:
			tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
			print(f"Successfully loaded tokenizer: {config.model.text_model_name}")
		except Exception as e:
			print(f"Failed to load tokenizer: {e}")
			return
	else:
		print("Text-free mode: no tokenizer needed")

	# Load training and validation datasets
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
		
		# Create data loaders with specified batch size
		train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
		
		print(f"Datasets loaded successfully:")
		print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
		print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
		
	except Exception as e:
		print(f"Failed to load datasets: {e}")
		return

	# Create and initialize the model
	try:
		model = FusionRegressor(
			config.model.vision_model_name,    # Vision encoder (e.g., ViT, ResNet)
			config.model.text_model_name,      # Text encoder (e.g., BERT, RoBERTa)
			config.model.hidden_dim,           # Hidden dimension for fusion
			config.model.use_icnet_head        # Whether to use ICNetHead
		).to(device)

		if hasattr(model, 'use_text') and model.use_text:
			print(f"Model created successfully:")
			print(f"  Vision encoder: {config.model.vision_model_name}")
			print(f"  Text encoder: {config.model.text_model_name}")
			print(f"  Hidden dimension: {config.model.hidden_dim}")
			print(f"  Use ICNetHead: {config.model.use_icnet_head}")
		else:
			print(f"Model created successfully (text-free mode):")
			print(f"  Vision encoder: {config.model.vision_model_name}")
			print(f"  Hidden dimension: {config.model.hidden_dim}")
			print(f"  Use ICNetHead: {config.model.use_icnet_head}")

	except Exception as e:
		print(f"Failed to create model: {e}")
		return

	# Initialize optimizer and loss function
	optimizer = optim.AdamW(
		model.parameters(),
		lr=config.train.lr,
		weight_decay=config.train.weight_decay
	)
	scheduler = optim.lr_scheduler.MultiStepLR(
		optimizer,
		milestones=config.train.milestone,
		gamma=config.train.lr_decay_rate)
	iter_per_epoch = len(train_loader)
	if config.train.warm > 0:
		Warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.train.warm)

	score_loss_fn = MSE_IB_Loss(Lambda=config.train.Lambda)
	map_loss_fn = MSE_IB_Loss(Lambda=0.)

	print(f"Optimizer: AdamW with learning rate {config.train.lr}")
	print(f"Loss function: MSE + Information Bottleneck (λ={config.train.Lambda})")

	# Main training loop
	print(f"\n{'='*60}")
	print(f"Starting training for {config.train.epochs} epochs...")
	print(f"{'='*60}")
	
	best_val_loss = float('inf')  # Track best validation loss for model saving
	total_training_start_time = time.time()  # Record overall training start time
	
	for epoch in range(config.train.epochs):
		epoch_start_time = time.time()
		print(f"\nEpoch [{epoch+1:2d}/{config.train.epochs:2d}] {'='*50}")
		
		# Training phase
		print(f"Training phase...")
		t_loss, t_mse, t_ib, t_srcc, t_plcc, t_rmse, t_rmae = train_one_epoch(
			model, train_loader, optimizer, score_loss_fn, map_loss_fn, device, config.model.use_icnet_head
		)

		# Calculate training time for this epoch
		train_time = time.time() - epoch_start_time

		if epoch >= config.train.warm:
			scheduler.step(epoch)

		# Validation phase
		print(f"Validation phase...")
		v_loss, v_mse, v_ib, v_srcc, v_plcc, v_rmse, v_rmae, v_time = validate(
			model, val_loader, score_loss_fn, map_loss_fn, device, config.model.use_icnet_head
		)
		
		# Calculate total epoch time
		epoch_total_time = time.time() - epoch_start_time
		
		# Calculate estimated total training time
		elapsed_total_time = time.time() - total_training_start_time
		avg_epoch_time = elapsed_total_time / (epoch + 1)
		remaining_epochs = config.train.epochs - (epoch + 1)
		estimated_remaining_total_time = remaining_epochs * avg_epoch_time
		total_estimated_time = elapsed_total_time + estimated_remaining_total_time

		# Print comprehensive epoch results
		print(f"\nEpoch [{epoch+1:2d}/{config.train.epochs:2d}] Results:")
		print(f"  Training:")
		print(f"    Loss: {t_loss:.4f} | MSE: {t_mse:.4f} | IB: {t_ib:.6f}")
		print(f"    SRCC: {t_srcc:.4f} | PLCC: {t_plcc:.4f} | RMSE: {t_rmse:.4f} | RMAE: {t_rmae:.4f}")
		print(f"    Time: {train_time:.2f}s")
		print(f"  Validation:")
		print(f"    Loss: {v_loss:.4f} | MSE: {v_mse:.4f} | IB: {v_ib:.6f}")
		print(f"    SRCC: {v_srcc:.4f} | PLCC: {v_plcc:.4f} | RMSE: {v_rmse:.4f} | RMAE: {v_rmae:.4f}")
		print(f"    Time: {v_time:.2f}s")
		print(f"  Epoch Total Time: {epoch_total_time:.2f}s")
		print(f"  Overall Progress: {elapsed_total_time/3600:.1f}h elapsed, "
			  f"ETA: {estimated_remaining_total_time/3600:.1f}h remaining, "
			  f"Total ETA: {total_estimated_time/3600:.1f}h")
		
		# Save best model based on validation loss
		if v_loss < best_val_loss:
			best_val_loss = v_loss
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=True)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			print(f"  ✓ New best model saved: {checkpoint_path}")
		
		# Periodic checkpoint saving
		if (epoch + 1) % config.train.save_interval == 0:
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=False)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			print(f"  ✓ Periodic checkpoint saved: {checkpoint_path}")

	# Training completed - save final model
	final_checkpoint_path = config.get_final_checkpoint_path()
	save_checkpoint(model, optimizer, config.train.epochs, final_checkpoint_path)
	
	# Calculate total training time
	total_training_time = time.time() - total_training_start_time
	
	print(f"\n{'='*60}")
	print(f"Training completed successfully!")
	print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.0f} seconds)")
	print(f"Final model saved to: {final_checkpoint_path}")
	print(f"{'='*60}")


if __name__ == "__main__":
	main()
