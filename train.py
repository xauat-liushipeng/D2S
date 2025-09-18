import os
import copy
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import time
import argparse
import datetime
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
import torch

from d2s.utils import *
from d2s.buffer import FeatureBuffer
from d2s.loss import *
from d2s.model import DescribeScore, FeatureAlignLoss
from d2s.data import IC9600Caption, create_transforms, IC9600
from d2s.logger import create_work_dir, setup_logging


def get_parser():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="D2S Training - Image Complexity Assessment Framework")
	parser.add_argument("--work_dir", type=str, default="work_dir", help="Base work directory")
	parser.add_argument('--device', type=str, default='cuda', help='device for training (e.g., cuda or cpu)')
	parser.add_argument('--seed', type=int, default=826, help='random seed for reproducibility')
	# dataset
	parser.add_argument('--img_dir', type=str, default='../data/IC9600/images', help='path to image directory')
	parser.add_argument('--train_file', type=str, default='../data/IC9600/train_blip_caption.txt', help='path to training file')
	parser.add_argument('--val_file', type=str, default='../data/IC9600/test.txt', help='path to validation file')
	parser.add_argument('--image_size', type=int, default=512, help='input image size for model')
	parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading (set to 0 for Windows compatibility)')
	# model
	parser.add_argument('--vision_encoder', type=str, default='resnet18', help='vision model name (e.g., resnet18, vit)')
	parser.add_argument('--text_encoder', type=str, default='clip', help='text model name (e.g., bert, roberta); empty for text-free mode')
	parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for regression head')
	parser.add_argument('--num_heads', type=int, default=16, help='number of attention heads in vision encoder')
	parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained weights for encoders')
	# training
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
	parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
	parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
	parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')
	parser.add_argument('--save_freq', type=int, default=1, help='interval (in epochs) to save checkpoints')
	# Entropy Align
	parser.add_argument('--momentum', type=float, default=0.995, help='momentum for EMA update of key encoder')
	parser.add_argument('--buffer_size', type=int, default=2048, help='maximum size of the align buffer')
	parser.add_argument('--buffer_refresh_steps', type=int, default=50, help='steps interval to refresh align buffer')
	parser.add_argument('--eal_weight', type=float, default=5, help='weight for entropy align loss')
	parser.add_argument('--use_amp', type=bool, default=True, help='use automatic mixed precision (AMP) for training')
	# Feature Align
	parser.add_argument('--fal_weight', type=float, default=0.01, help='weight for feature alignment loss')
	parser.add_argument('--temperature', type=float, default=0.07, help='temperature for feature alignment loss')
	return parser.parse_args()


def train_one_epoch(epoch, model, key_model, dataloader, optimizer, device,
					align_buffer, logger, args, fal_loss):
	model.train()
	total_loss, total_mse, total_eal, total_fal = 0, 0, 0, 0
	all_preds, all_scores = [], []

	for iter_idx, (images, tokens, scores, img_names) in enumerate(dataloader):
		iter_start_time = time.time()

		images = images.to(device, non_blocking=True)
		scores = scores.to(device, non_blocking=True)
		tokens = tokens.to(device, non_blocking=True)

		if hasattr(model, 'use_text') and model.use_text:
			preds, vf, tf, _, _ = model(images, tokens)
		else:
			preds, vf, tf, _, _ = model(images, None)

		# Calculate the entropy of the queue
		with torch.no_grad():
			key_model.eval()
			if args.use_amp and device.type == 'cuda':
				with torch.amp.autocast('cuda'):
					_, _, _, hv_k, hs_k = key_model(images, tokens)
			else:
				_, _, _, hv_k, hs_k = key_model(images, tokens)
		align_buffer.push_samples(img_names, tokens, hv_k, hs_k)

		mse = mse_loss(preds, scores)
		fal = fal_loss(vf, tf)

		# entropy align loss: eal
		eal = torch.tensor(0.0, device=images.device)
		if len(align_buffer) >= args.buffer_size / 2:
			Hv_all = align_buffer.get(device, which='hv')
			Hs_all = align_buffer.get(device, which='hs')
			if Hv_all is not None and Hs_all is not None and len(Hv_all) > 1 and len(Hs_all) > 1:
				eal = w1D_loss(Hv_all, Hs_all)

		loss = mse + (eal * args.eal_weight) + (fal * args.fal_weight)

		# backward, optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# key model update with EMA
		with torch.no_grad():
			for p_k, p_q in zip(key_model.parameters(), model.parameters()):
				p_k.data.mul_(args.momentum).add_(p_q.data, alpha=(1.0 - args.momentum))
		# refresh align buffer
		align_buffer.refresh(key_model, device, batch_size=args.batch_size, use_amp=args.use_amp)

		# accumulate losses
		total_loss += loss.item()
		total_mse += mse.item()
		total_eal += eal.item()
		total_fal += fal.item()

		# collect
		all_preds.extend(preds.detach().cpu().numpy())
		all_scores.extend(scores.detach().cpu().numpy())

		# time
		iter_time = time.time() - iter_start_time

		logger.info(f"  Iter [{iter_idx+1:3d}/{len(dataloader):3d}] | "
			  f"Loss: {loss.item():.6f} | MSE: {mse:.6f} | EAL: {eal:.6f} _w = {args.eal_weight:.6f} | "
			  f"FAL: {fal:.6f} _w = {args.fal_weight:.6f} | Time: {iter_time:.3f}s")

	n = len(dataloader)
	# metrics
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)

	return total_loss / n, srcc, plcc, rmse, rmae


def validate(model, dataloader, device):
	model.eval()
	all_preds, all_scores = [], []
	val_start_time = time.time()

	with torch.no_grad():
		for images, scores, _ in dataloader:
			images = images.to(device, non_blocking=True)
			scores = scores.to(device, non_blocking=True)

			preds, _, _, _, _ = model(images, None)

			all_preds.extend(preds.detach().cpu().numpy())
			all_scores.extend(scores.detach().cpu().numpy())

	# metrics
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)
	val_time = time.time() - val_start_time

	return srcc, plcc, rmse, rmae, val_time


def main():
	args = get_parser()

	# logging
	args.work_dir = create_work_dir(args.work_dir)
	logger = setup_logging(args.work_dir)
	logger.info(f"Starting D2S training at {datetime.datetime.now()}")
	logger.info(f"Work directory: {args.work_dir}")

	# random seed
	set_global_seed(logger, args.seed)

	init_work_dir(args)
	print_args_summary(args, logger)
	device = get_device(args)
	logger.info(f"Using device: {device}")

	# transform
	train_transform = create_transforms(args.image_size, is_training=True)
	val_transform = create_transforms(args.image_size, is_training=False)
	logger.info(f"Image transforms created for size: {args.image_size}x{args.image_size}")

	# model
	model = DescribeScore(
		args.vision_encoder, args.text_encoder,
		args.hidden_dim, args.num_heads, args.pretrained
	).to(device)
	tokenizer = model.text_encoder.tokenizer

	# dataset and dataloader
	try:
		train_dataset = IC9600Caption(args.train_file, args.img_dir, train_transform, tokenizer)
		val_dataset = IC9600(args.val_file, args.img_dir, val_transform)

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

		logger.info(f"Datasets loaded successfully: "
			f"Training: {len(train_dataset)} samples, {len(train_loader)} batches; "
			f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
		if args.num_workers == 0:
			logger.info("Using single-threaded data loading for Windows compatibility")
	except Exception as e:
		logger.error(f"Failed to load datasets: {e}")
		return

	# freeze text encoder
	if hasattr(model, 'text_encoder') and model.text_encoder is not None:
		for param in model.text_encoder.parameters():
			param.requires_grad = False

	# build key model (EMA)
	key_model = copy.deepcopy(model).to(device)
	for p in key_model.parameters():
		p.requires_grad = False
	key_model.eval()
	logger.info("Build key encoder (EMA)")

	# image loader for buffer
	def image_loader_fn(name: str):
		path = os.path.join(args.img_dir, name)
		with Image.open(path) as img:
			img = img.convert("RGB")
			return train_transform(img)

	# align buffer
	EntropyBuffer = FeatureBuffer(
		buffer_size=args.buffer_size,
		refresh_steps=args.buffer_refresh_steps,
		image_loader=image_loader_fn
	)
	logger.info(f"Created align buffer: size={args.buffer_size}, refresh_steps={args.buffer_refresh_steps}")

	# optimizer
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=25e-6)
	logger.info(f"Optimizer: AdamW with learning rate {args.lr}")

	# loss
	fal_loss = FeatureAlignLoss(args.temperature)
	logger.info(f"Loss: MSE + {args.fal_weight} * FAL + {args.eal_weight} * EAL\n")

	# training loop
	logger.info(f"Starting training for {args.epochs} epochs...")
	best_val_pcc = 0.0
	total_training_start_time = time.time()

	for epoch in range(args.epochs):
		epoch_start_time = time.time()
		logger.info(f"Epoch [{epoch+1:2d}/{args.epochs:2d}] {'='*50}")

		# Training phase
		t_loss, t_srcc, t_plcc, t_rmse, t_rmae = train_one_epoch(
			epoch + 1, model, key_model, train_loader, optimizer, device, EntropyBuffer, logger, args, fal_loss
		)
		train_time = time.time() - epoch_start_time

		if epoch < args.epochs:
			scheduler.step()

		# Validation phase
		v_srcc, v_plcc, v_rmse, v_rmae, v_time = validate(model, val_loader, device)
		epoch_total_time = time.time() - epoch_start_time

		# time
		elapsed_total_time = time.time() - total_training_start_time
		avg_epoch_time = elapsed_total_time / (epoch + 1)
		remaining_epochs = args.epochs - (epoch + 1)
		estimated_remaining_total_time = remaining_epochs * avg_epoch_time
		total_estimated_time = elapsed_total_time + estimated_remaining_total_time

		# Print results
		logger.info(f"Epoch [{epoch+1:2d}/{args.epochs:2d}] Results:")
		logger.info(f"  Training: "
		f"SRCC: {t_srcc:.4f} | PLCC: {t_plcc:.4f} | RMSE: {t_rmse:.4f} | RMAE: {t_rmae:.4f} | "
		f"Loss: {t_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | "
		f"Time: {train_time:.2f}s")
		logger.info(f"  Validat : "
		f"SRCC: {v_srcc:.4f} | PLCC: {v_plcc:.4f} | RMSE: {v_rmse:.4f} | RMAE: {v_rmae:.4f} | Time: {v_time:.2f}s")
		logger.info(f"  Epoch Total Time: {epoch_total_time:.2f}s, "
		f"Overall Progress: {elapsed_total_time/3600:.1f}h elapsed, "
			  f"ETA: {estimated_remaining_total_time/3600:.1f}h remaining, "
			  f"Total ETA: {total_estimated_time/3600:.1f}h")

		# Save best model
		if v_plcc > best_val_pcc:
			best_val_pcc = v_plcc
			checkpoint_path = get_checkpoint_path(args, epoch+1, is_best=True)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			logger.info(f"  * New best model saved: {checkpoint_path}")

		# Periodic checkpoint saving
		if (epoch + 1) % args.save_freq == 0:
			checkpoint_path = get_checkpoint_path(args, epoch+1, is_best=False)
			save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
			logger.info(f"  Periodic checkpoint saved: {checkpoint_path}")

	# save final model
	final_checkpoint_path = get_final_checkpoint_path(args)
	save_checkpoint(model, optimizer, args.epochs, final_checkpoint_path)

	# total training time
	total_training_time = time.time() - total_training_start_time

	logger.info(f"{'='*60}")
	logger.info(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.0f} seconds)")
	logger.info(f"Final model saved to: {final_checkpoint_path}")
	logger.info(f"{'='*60}")

	logger.info(f"All logs and checkpoints saved to: {args.work_dir}")


if __name__ == "__main__":
	main()
