"""
File: d2s/loss.py
Description: Loss definition for D2S training (MSE + information bottleneck regularization).
Date: 2025-08-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import overload


class MSE_IB_Loss(nn.Module):
	def __init__(self, Lambda=1e-4):
		"""
		beta: Information bottleneck regularization weight
		"""
		super().__init__()
		self.Lambda = Lambda
		self.mse = nn.MSELoss()

	def forward(self, preds, labels, fused_feats):
		"""
		preds: [B] predicted complexity scores
		labels: [B] ground-truth complexity scores
		fused_feats: [B, D] fused feature vectors (image_feat + text_feat)
		"""
		mse = self.mse(preds, labels)
		ib = self.Lambda * torch.mean(torch.sum(fused_feats ** 2, dim=1))

		total = mse + ib
		return total, mse.item(), ib.item()


# =========================
# Loss functions
# =========================

def mse_loss(pred, target):
	return F.mse_loss(pred, target)


def ib_loss(mu, logvar):
	"""KL divergence between N(mu, sigma^2) and N(0,1)"""
	return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def w1D_loss(x, y):
	"""
	Energy distance between 1D distributions x and y
	x, y: [B]
	"""
	x = x.view(-1, 1)
	y = y.view(-1, 1)
	a = torch.cdist(x, y, p=1).mean()  # E|X-Y|
	b = torch.cdist(x, x, p=1).mean()  # E|X-X'|
	c = torch.cdist(y, y, p=1).mean()  # E|Y-Y'|
	return 2 * a - b - c


def huber_loss(pred, target, delta=0.05):
	err = pred - target
	abs_err = err.abs()
	quad = torch.clamp(abs_err, max=delta)
	lin = abs_err - quad
	return torch.mean(0.5 * quad ** 2 + delta * lin)


def compute_loss(score, label, H_v, H_s,
                 lambda_mse=0.1, lambda_align=0.1):
	loss_main = huber_loss(score, label)
	loss_align = w1D_loss(H_v, H_s)
	return lambda_mse * loss_main + lambda_align * loss_align, {
		"mse": loss_main.item(),
		"align": loss_align.item()
	}

# @overload
# def compute_loss(score, label, mu, logvar, H_v, H_s,
# 			   lambda_ib=0.1, lambda_align=0.1):
# 	loss_main = mse_loss(score, label)
# 	loss_ib = ib_loss(mu, logvar)
# 	loss_align = w1D_loss(H_v, H_s)
# 	return loss_main + lambda_ib * loss_ib + lambda_align * loss_align, {
# 		"mse": loss_main.item(),
# 		"ib": loss_ib.item(),
# 		"align": loss_align.item()
# 	}
