import os
import random

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr


def set_global_seed(logger, seed=42, strict_determinism=True):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if torch.cuda.is_available():
		if strict_determinism:
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			logger.info("Using strict determinism (slower but fully reproducible)")
		else:
			torch.backends.cudnn.deterministic = False
			torch.backends.cudnn.benchmark = True
			logger.info("Using optimized settings (faster but less reproducible)")

	os.environ['PYTHONHASHSEED'] = str(seed)

	try:
		from transformers import set_seed
		set_seed(seed)
	except ImportError:
		pass

	logger.info(f"Global random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, path):
	ckpt = {
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"epoch": epoch
	}
	torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
	ckpt = torch.load(path, map_location=device)
	model.load_state_dict(ckpt["model_state"])
	if optimizer and "optimizer_state" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer_state"])
	print(f"Checkpoint loaded: {path}, epoch {ckpt.get('epoch', 'N/A')}")
	return ckpt.get("epoch", None)


def compute_srcc_plcc(preds, labels):
	preds = np.array(preds)
	labels = np.array(labels)

	plcc = pearsonr(preds, labels)[0]
	srcc = spearmanr(preds, labels)[0]
	rmse = np.sqrt(np.mean(np.abs(preds - labels) ** 2))
	rmae = np.sqrt(np.abs(preds - labels).mean())

	return srcc, plcc, rmse, rmae


def _is_cuda_available() -> bool:
	try:
		import torch
		return torch.cuda.is_available()
	except ImportError:
		return False


def init_work_dir(args):
	args.checkpoint_dir = os.path.join(args.work_dir, args.checkpoint_dir)
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	if args.device == "cuda" and not _is_cuda_available():
		print("Warning: CUDA is not available, fallback to CPU")
		args.device = "cpu"


def get_device(args):
	import torch
	if args.device == "cuda" and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def get_checkpoint_path(args, epoch: int, is_best: bool = False) -> str:
	if is_best:
		filename = f"best_model_epoch_{epoch}.pth"
	else:
		filename = f"model_epoch_{epoch}.pth"

	return os.path.join(args.checkpoint_dir, filename)


def get_final_checkpoint_path(args) -> str:
	return os.path.join(args.checkpoint_dir, "final_model.pth")


def print_args_summary(args, logger):
	logger.info("=" * 50)
	logger.info("Config Summary")
	logger.info("=" * 50)

	logger.info(f"Dataset:")
	logger.info(f"  Image directory: {args.img_dir}")
	logger.info(f"  Train file: {args.train_file}")
	logger.info(f"  Validation file: {args.val_file}")
	logger.info(f"  Image size: {args.image_size}")
	logger.info(f"  Num workers: {args.num_workers}")

	logger.info(f"Model:")
	logger.info(f"  Vision encoder: {args.vision_encoder}")
	logger.info(f"  Text encoder: {args.text_encoder}")
	logger.info(f"  Hidden dimension: {args.hidden_dim}")
	logger.info(f"  Pretrained: {args.pretrained}")

	logger.info(f"Training:")
	logger.info(f"  Checkpoint dir: {args.checkpoint_dir}")
	logger.info(f"  Batch size: {args.batch_size}")
	logger.info(f"  Epochs: {args.epochs}")
	logger.info(f"  Learning rate: {args.lr}")
	logger.info(f"  Weight decay: {args.weight_decay}")
	logger.info(f"  Save frequency: {args.save_freq}")

	logger.info(f"Entropy Align: ")
	logger.info(f"  EMA momentum: {args.momentum}")
	logger.info(f"  Buffer size: {args.buffer_size}")
	logger.info(f"  Buffer refresh steps: {args.buffer_refresh_steps}")
	logger.info(f"  Entropy align weight: {args.eal_weight}")

	logger.info(f"Feature Align: ")
	logger.info(f"  Feature align weight: {args.fal_weight}")
	logger.info(f"  Feature align temperature: {args.temperature}")

	logger.info(f"Utils: ")
	logger.info(f"  Work dir: {args.work_dir}")
	logger.info(f"  Device: {args.device}")
	logger.info(f"  Seed: {args.seed}")

	logger.info("=" * 50 + "\n")
