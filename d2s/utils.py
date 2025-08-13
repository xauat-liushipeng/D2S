"""
File: d2s/utils.py
Description: Utility helpers for checkpoint I/O and correlation metrics.
Date: 2025-08-13
"""
import os
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

def save_checkpoint(model, optimizer, epoch, path):
	ckpt = {
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"epoch": epoch
	}
	torch.save(ckpt, path)
	print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None, device="cpu"):
	ckpt = torch.load(path, map_location=device)
	model.load_state_dict(ckpt["model_state"])
	if optimizer and "optimizer_state" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer_state"])
	print(f"Checkpoint loaded: {path}, epoch {ckpt.get('epoch', 'N/A')}")
	return ckpt.get("epoch", None)

def compute_srcc_plcc(preds, labels):
	"""
	preds, labels: list or numpy array
	Returns: (srcc, plcc) or (0.0, 0.0) if computation fails
	"""
	try:
		preds = np.array(preds)
		labels = np.array(labels)
		
		# Check for empty arrays
		if len(preds) == 0 or len(labels) == 0:
			print("Warning: Empty predictions or labels array")
			return 0.0, 0.0
		
		# Check for NaNs
		if np.isnan(preds).any() or np.isnan(labels).any():
			print("Warning: NaN values detected in predictions or labels")
			return 0.0, 0.0
		
		# Zero variance guard (correlations would be undefined)
		if np.std(preds) == 0 or np.std(labels) == 0:
			print("Warning: Zero variance in predictions or labels")
			return 0.0, 0.0
		
		srcc = spearmanr(preds, labels)[0]
		plcc = pearsonr(preds, labels)[0]
		
		# Check for NaN correlations
		if np.isnan(srcc) or np.isnan(plcc):
			print("Warning: NaN correlation coefficients computed")
			return 0.0, 0.0
			
		return srcc, plcc
		
	except Exception as e:
		print(f"Error computing correlation coefficients: {e}")
		return 0.0, 0.0
