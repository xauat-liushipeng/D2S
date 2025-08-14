"""
File: d2s/loss.py
Description: Loss definition for D2S training (MSE + information bottleneck regularization).
Date: 2025-08-13
"""
import torch
import torch.nn as nn

class MSE_IB_Loss(nn.Module):
	def __init__(self, Lambda=1e-4):
		"""
		beta: Information bottleneck regularization weight
		"""
		super().__init__()
		self.beta = Lambda
		self.mse = nn.MSELoss()

	def forward(self, preds, labels, fused_feats):
		"""
		preds: [B] predicted complexity scores
		labels: [B] ground-truth complexity scores
		fused_feats: [B, D] fused feature vectors (image_feat + text_feat)
		"""
		reg_loss = self.mse(preds, labels)
		ib_loss = self.Lambda * torch.mean(torch.sum(fused_feats ** 2, dim=1))

		total_loss = reg_loss + ib_loss
		return total_loss, reg_loss.item(), ib_loss.item()
