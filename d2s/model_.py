import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Fom OpenAI CLIP file
class AttentionPool2d(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
		super().__init__()
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
		self.num_heads = num_heads

	def forward(self, x):
		x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
		x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
		x, _ = F.multi_head_attention_forward(
			query=x[:1], key=x, value=x,
			embed_dim_to_check=x.shape[-1],
			num_heads=self.num_heads,
			q_proj_weight=self.q_proj.weight,
			k_proj_weight=self.k_proj.weight,
			v_proj_weight=self.v_proj.weight,
			in_proj_weight=None,
			in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
			bias_k=None,
			bias_v=None,
			add_zero_attn=False,
			dropout_p=0,
			out_proj_weight=self.c_proj.weight,
			out_proj_bias=self.c_proj.bias,
			use_separate_proj_weight=True,
			training=self.training,
			need_weights=False
		)
		return x.squeeze(0)


class VisionEncoder(nn.Module):
	def __init__(self, model_name="resnet50", pretrained=True, num_heads=16):
		super().__init__()
		if model_name == "resnet50":
			self.feat_dim = 2048
		elif model_name == "resnet18":
			self.feat_dim = 512
		self.out_dim = 1024

		self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
		self.attn_pool = AttentionPool2d(self.feat_dim, num_heads, self.out_dim)

	def forward(self, x):
		f4 = self.model(x)[4]
		f4p = self.attn_pool(f4).flatten(1)  # [B, out_dim]
		return f4, f4p


class D2S(nn.Module):
	def __init__(self, vision_model_name,  hidden_dim=512, num_heads=16, pretrained=False):
		super().__init__()
		self.vision_encoder = VisionEncoder(vision_model_name, pretrained, num_heads)

		self.fc = nn.Sequential(
			nn.Linear(self.vision_encoder.out_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, images):
		_, img_feats = self.vision_encoder(images)
		score = self.fc(img_feats).squeeze(-1)
		return score
