"""
File: train.py
Description: Training entrypoint for D2S; training/validation loops, checkpointing and correlation metrics.
Date: 2025-08-13
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import copy
import time
import argparse
import datetime

from PIL import Image
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from d2s.model import FusionRegressor
from d2s.data import IC9600Caption
from d2s.loss import mse_loss, w1D_loss
from d2s.utils import save_checkpoint, compute_srcc_plcc, WarmUpLR
from d2s.logger import create_work_dir, setup_logging
from d2s.entropy_buffer import FeatureBuffer
from config.train_config import get_default_config


def create_transforms(image_size: int, is_training: bool = True):
	base_transforms = [
		transforms.Resize((image_size, image_size)),  # Resize to target dimensions
		transforms.ToTensor(),                        # Convert PIL to tensor
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
	]
	
	if is_training:
		base_transforms.insert(1, transforms.RandomHorizontalFlip())  # Add horizontal flip for training
	
	return transforms.Compose(base_transforms)


def train_one_epoch(model, key_model, dataloader, optimizer, device,
                    align_buffer, logger, momentum=0.999, use_amp=True):
	model.train()  # Set model to training mode
	total_loss, total_mse, total_align = 0, 0, 0
	all_preds, all_scores = [], []
	
	# Record start time for this epoch
	epoch_start_time = time.time()
	
	for iter_idx, batch in enumerate(dataloader):
		# Record start time for this iteration
		iter_start_time = time.time()
		
		# Move batch data to device
		images = batch["image"].to(device)  # Image tensors
		scores = batch["score"].to(device)  # Ground truth scores

		# Forward pass: get predictions and features (query network for training/backward)
		if hasattr(model, 'use_text') and model.use_text:
			# Multimodal mode: include text inputs
			input_ids = batch["input_ids"].to(device)    # Tokenized text input IDs
			attn_mask = batch["attention_mask"].to(device)  # Attention mask for text
			preds, hv, hs = model(images, input_ids, attn_mask)
		else:
			# Text-free mode: only vision inputs
			preds, hv, hs = model(images)

		# 使用 key_encoder 计算队列用的熵（只前向，不反向）
		with torch.no_grad():
			key_model.eval()
			if hasattr(model, 'use_text') and model.use_text:
				if use_amp and device.type == 'cuda':
					with torch.amp.autocast('cuda'):
						_, hv_k, hs_k = key_model(images, input_ids, attn_mask)
				else:
					_, hv_k, hs_k = key_model(images, input_ids, attn_mask)
			else:
				if use_amp and device.type == 'cuda':
					with torch.amp.autocast('cuda'):
						_, hv_k, hs_k = key_model(images)
				else:
					_, hv_k, hs_k = key_model(images)
		# 将样本级信息压入对齐缓冲区
		if hasattr(model, 'use_text') and model.use_text:
			align_buffer.push_samples(batch["image_name"], batch["input_ids"], batch["attention_mask"], hv_k, hs_k)
		else:
			# 纯视觉：占位 input_ids/attention_mask（不会用于前向，仅为接口一致）
			dummy_ids = torch.zeros((hv_k.shape[0], 1), dtype=torch.long)
			dummy_mask = torch.zeros((hv_k.shape[0], 1), dtype=torch.long)
			align_buffer.push_samples(batch["image_name"], dummy_ids, dummy_mask, hv_k, hs_k)

		mse = mse_loss(preds, scores)

		# Align loss: 使用缓冲区的 Hv/Hs 估计
		align = 0.0
		if len(align_buffer) > 512:  # 至少累计够一定数量再算
			Hv_all = align_buffer.get(device, which='hv')
			Hs_all = align_buffer.get(device, which='hs')
			if Hv_all is not None and Hs_all is not None and len(Hv_all) > 1 and len(Hs_all) > 1:
				align = w1D_loss(Hv_all, Hs_all)
		loss = mse + (align if isinstance(align, torch.Tensor) else torch.tensor(align, device=device)) * 0.2

		# Backward pass and optimization
		optimizer.zero_grad()  # Clear previous gradients
		loss.backward()         # Compute gradients
		optimizer.step()        # Update model parameters

		# 影子网络动量更新（EMA）
		with torch.no_grad():
			for p_k, p_q in zip(key_model.parameters(), model.parameters()):
				p_k.data.mul_(momentum).add_(p_q.data, alpha=(1.0 - momentum))

		# 周期性刷新缓冲区的一段（每步刷 target_size/refresh_steps 个）
		try:
			align_buffer.refresh(key_model, device, batch_size=32, use_amp=(use_amp and device.type == 'cuda'))
		except Exception as e:
			print(f"Warning: Align buffer refresh failed: {e}")

		# Accumulate losses for epoch average
		total_loss += loss.item()
		total_mse += mse.item()
		align_val = align.item() if isinstance(align, torch.Tensor) else float(align)
		total_align += align_val

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
		logger.print(f"  Iter [{iter_idx+1:3d}/{len(dataloader):3d}] | "
			  f"Loss: {loss.item():.4f} | MSE: {mse:.4f} | Align: {align_val:.6f} | "
			  f"Time: {iter_time:.3f}s | ETA: {estimated_remaining_time:.1f}s | "
			  f"Total ETA: {total_estimated_time:.1f}s")

	n = len(dataloader)
	# Compute correlation metrics (Spearman's Rank Correlation Coefficient and Pearson's Linear Correlation Coefficient)
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)
	
	return total_loss / n, total_mse / n, total_align / n, srcc, plcc, rmse, rmae


def validate(model, dataloader, device):
	model.eval()  # Set model to evaluation mode
	total_loss, total_mse, total_align = 0, 0, 0
	all_preds, all_scores = [], []
	hvs, hss = [], []
	
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
				preds, hv, hs = model(images, input_ids, attn_mask)
			else:
				# Text-free mode: only vision inputs
				preds, hv, hs = model(images)

			hvs.append(hv.detach().cpu())
			hss.append(hs.detach().cpu())

			mse = mse_loss(preds, scores)
			loss = mse

			# Accumulate losses for epoch average
			total_loss += loss.item()
			total_mse += mse.item()
			total_align += 0
			
			# Collect predictions and labels for correlation computation
			all_preds.extend(preds.detach().cpu().numpy())
			all_scores.extend(scores.detach().cpu().numpy())

	n = len(dataloader)
	# Compute correlation metrics
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)

	total_align = w1D_loss(torch.cat(hvs, dim=0), torch.cat(hss, dim=0))
	
	# Calculate validation time
	val_time = time.time() - val_start_time
	
	return total_loss / n, total_mse / n, total_align / n, srcc, plcc, rmse, rmae, val_time


def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="D2S Training - Image Complexity Assessment Framework")
	parser.add_argument("--work_dir", type=str, default="work_dir", help="Base work directory")
	args = parser.parse_args()

	# Create work directory with date format
	work_dir = create_work_dir(args.work_dir)
	print(f"Created work directory: {work_dir}")
	
	# Setup logging
	logger = setup_logging(work_dir)
	logger.print(f"Starting D2S training at {datetime.datetime.now()}")
	logger.print(f"Work directory: {work_dir}")
	
	# Load configuration
	try:
		config = get_default_config()
		
		# Update work directory paths
		config.train.checkpoint_dir = os.path.join(work_dir, "checkpoints")
		config.log.log_dir = os.path.join(work_dir, "logs")

		config.init()
		config.print_summary()
		
	except Exception as e:
		logger.error(f"Failed to load configuration: {e}")
		return

	# Determine device (GPU if available, otherwise CPU)
	device = config.get_device()
	logger.print(f"Using device: {device}")

	# Create image transforms for training and validation
	train_transform = create_transforms(config.dataset.image_size, is_training=True)
	val_transform = create_transforms(config.dataset.image_size, is_training=False)
	logger.print(f"Image transforms created for size: {config.dataset.image_size}x{config.dataset.image_size}")

	# Load tokenizer for text processing
	tokenizer = None
	if config.model.text_model_name and config.model.text_model_name.strip() != "":
		try:
			tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
			logger.print(f"Successfully loaded tokenizer: {config.model.text_model_name}")
		except Exception as e:
			logger.error(f"Failed to load tokenizer: {e}")
			return
	else:
		logger.print("Text-free mode: no tokenizer needed")

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
		
		logger.print(f"Datasets loaded successfully:")
		logger.print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
		logger.print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
		
	except Exception as e:
		logger.error(f"Failed to load datasets: {e}")
		return

	# Create and initialize the model
	try:
		model = FusionRegressor(
			config.model.vision_model_name,    # Vision encoder (e.g., ViT, ResNet)
			config.model.text_model_name,      # Text encoder (e.g., BERT, RoBERTa)
		).to(device)

		if hasattr(model, 'use_text') and model.use_text:
			logger.print(f"Model created successfully:")
			logger.print(f"  Vision encoder: {config.model.vision_model_name}")
			logger.print(f"  Text encoder: {config.model.text_model_name}")
		else:
			logger.print(f"Model created successfully (text-free mode):")
			logger.print(f"  Vision encoder: {config.model.vision_model_name}")

	except Exception as e:
		logger.error(f"Failed to create model: {e}")
		return

	# Freeze text encoder parameters
	for param in model.text_encoder.parameters():
		param.requires_grad = False

	# Load pre-trained weights if available
	vl_align_path = "./vlalign_best.pt"
	if hasattr(model, 'fusion') and model.fusion is not None and os.path.exists(vl_align_path):
		try:
			vl_align_dict = torch.load(vl_align_path, map_location=device, weights_only=True)
			model.fusion.load_state_dict(vl_align_dict, strict=False)
			logger.print(f"Loaded fusion alignment weights from {vl_align_path} (strict=False)")
		except Exception as e:
			logger.warning(f"Failed to load fusion weights from {vl_align_path}: {e}")
	else:
		logger.print("Skip loading fusion alignment weights (file missing or fusion is None)")

	# build shadow key encoder (EMA)
	key_model = copy.deepcopy(model).to(device)
	for p in key_model.parameters():
		p.requires_grad = False
	key_model.eval()
	logger.print("Created shadow key encoder (EMA)")

	# Build align buffer: 使用与训练一致的transform进行单图重载
	def image_loader_fn(name: str):
		path = os.path.join(config.dataset.img_dir, name)
		img = Image.open(path).convert("RGB")
		return create_transforms(config.dataset.image_size, is_training=True)(img)

	AlignBuffer = FeatureBuffer(
		max_size=config.train.buffer_max_size, 
		refresh_steps=config.train.buffer_refresh_steps, 
		target_size=config.train.buffer_target_size, 
		image_loader=image_loader_fn
	)
	logger.print(f"Created align buffer: max_size={config.train.buffer_max_size}, refresh_steps={config.train.buffer_refresh_steps}")

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

	logger.print(f"Optimizer: AdamW with learning rate {config.train.lr}")
	logger.print(f"Loss: MSE + {config.train.align_weight}*Align (w1D on Hv/Hs from EMA key encoder)")

	# Main training loop
	logger.print(f"\n{'='*60}")
	logger.print(f"Starting training for {config.train.epochs} epochs...")
	logger.print(f"{'='*60}")
	
	best_val_loss = float('inf')  # Track best validation loss for model saving
	total_training_start_time = time.time()  # Record overall training start time
	
	for epoch in range(config.train.epochs):
		epoch_start_time = time.time()
		logger.print(f"\nEpoch [{epoch+1:2d}/{config.train.epochs:2d}] {'='*50}")

		if epoch + 1 == 20:
			for param in model.text_encoder.parameters():
				param.requires_grad = True
		
		# Training phase
		logger.print(f"Training phase...")
		t_loss, t_mse, t_align, t_srcc, t_plcc, t_rmse, t_rmae = train_one_epoch(
			model, key_model, train_loader, optimizer, device, AlignBuffer, logger, config.train.ema_momentum
		)

		# Calculate training time for this epoch
		train_time = time.time() - epoch_start_time

		if epoch > config.train.warm:
			scheduler.step()

		# Validation phase
		logger.print(f"Validation phase...")
		v_loss, v_mse, v_align, v_srcc, v_plcc, v_rmse, v_rmae, v_time = validate(
			model, val_loader, device
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
		logger.print(f"\nEpoch [{epoch+1:2d}/{config.train.epochs:2d}] Results:")
		logger.print(f"  Training:")
		logger.print(f"    Loss: {t_loss:.4f} | MSE: {t_mse:.4f} | Align: {t_align:.6f}")
		logger.print(f"    SRCC: {t_srcc:.4f} | PLCC: {t_plcc:.4f} | RMSE: {t_rmse:.4f} | RMAE: {t_rmae:.4f}")
		logger.print(f"    Time: {train_time:.2f}s")
		logger.print(f"  Validation:")
		logger.print(f"    Loss: {v_loss:.4f} | MSE: {v_mse:.4f} | Align: {v_align:.6f}")
		logger.print(f"    SRCC: {v_srcc:.4f} | PLCC: {v_plcc:.4f} | RMSE: {v_rmse:.4f} | RMAE: {v_rmae:.4f}")
		logger.print(f"    Time: {v_time:.2f}s | lr: {optimizer.param_groups[0]['lr']:.6f}")
		logger.print(f"  Epoch Total Time: {epoch_total_time:.2f}s")
		logger.print(f"  Overall Progress: {elapsed_total_time/3600:.1f}h elapsed, "
			  f"ETA: {estimated_remaining_total_time/3600:.1f}h remaining, "
			  f"Total ETA: {total_estimated_time/3600:.1f}h")
		
		# Save best model based on validation loss
		if v_loss < best_val_loss:
			best_val_loss = v_loss
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=True, checkpoint_dir=config.train.checkpoint_dir)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			logger.print(f"  ✓ New best model saved: {checkpoint_path}")
		
		# Periodic checkpoint saving
		if (epoch + 1) % config.train.save_interval == 0:
			checkpoint_path = config.get_checkpoint_path(epoch+1, is_best=False, checkpoint_dir=config.train.checkpoint_dir)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			logger.print(f"  ✓ Periodic checkpoint saved: {checkpoint_path}")

	# Training completed - save final model
	final_checkpoint_path = config.get_final_checkpoint_path(config.train.checkpoint_dir)
	save_checkpoint(model, optimizer, config.train.epochs, final_checkpoint_path)
	
	# Calculate total training time
	total_training_time = time.time() - total_training_start_time
	
	logger.print(f"\n{'='*60}")
	logger.print(f"Training completed successfully!")
	logger.print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.0f} seconds)")
	logger.print(f"Final model saved to: {final_checkpoint_path}")
	logger.print(f"{'='*60}")
	
	logger.print(f"All logs and checkpoints saved to: {work_dir}")


if __name__ == "__main__":
	main()
