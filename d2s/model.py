import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer


class FeatureAlignLoss(nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = temperature
		self.cross_entropy_loss = nn.CrossEntropyLoss()

	def forward(self, image_features, text_features):
		image_features = F.normalize(image_features, dim=-1)
		text_features = F.normalize(text_features, dim=-1)

		logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
		logits_per_text = logits_per_image.T

		batch_size = image_features.shape[0]
		labels = torch.arange(batch_size, device=image_features.device)

		loss_images = self.cross_entropy_loss(logits_per_image, labels)
		loss_texts = self.cross_entropy_loss(logits_per_text, labels)

		total_loss = (loss_images + loss_texts) / 2

		return total_loss


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
		return f4p


class CLIPTextEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		model_name = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
		self.model, _, _ = create_model_and_transforms(model_name)
		self.tokenizer = get_tokenizer(model_name)
		self.out_dim = 512

	def forward(self, tokens):
		tokens = tokens.to(next(self.model.parameters()).device)
		features = self.model.encode_text(tokens)  # [B, out_dim]
		return features


class DescribeScore(nn.Module):
	def __init__(self, vision_model_name, text_model_name, hidden_dim=512, num_heads=16, pretrained=True):
		super().__init__()
		# vision encoder
		self.vision_encoder = VisionEncoder(vision_model_name, pretrained, num_heads)

		# if text branch
		self.use_text = text_model_name and text_model_name.strip() != ""

		if self.use_text:
			# text encoder
			self.text_encoder = CLIPTextEncoder()
			self.vision_proj = nn.Linear(self.vision_encoder.out_dim, self.text_encoder.out_dim)
		else:
			# Text-free mode
			self.text_encoder = None
			self.vision_proj = None

		self.fc = nn.Sequential(
			nn.Linear(self.vision_encoder.out_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

	def compute_entropy(self, feats, tau=1.0, eps=1e-8):
		probs = F.softmax(feats / tau, dim=-1)  # [B, D]
		log_probs = torch.log(probs + eps)
		entropy = -torch.sum(probs * log_probs, dim=-1)  # [B]
		return entropy

	def forward(self, images, texts):
		img_feats = self.vision_encoder(images)

		if self.use_text and texts is not None:
			txt_feats = self.text_encoder(texts)
			img_proj = self.vision_proj(img_feats)

			v_e = self.compute_entropy(img_proj)
			t_e = self.compute_entropy(txt_feats)
		else:
			# Text-free mode
			txt_feats = torch.zeros((img_feats.shape[0], 512), device=img_feats.device)
			img_proj = img_feats  # No projection needed in text-free mode

			v_e = self.compute_entropy(img_feats)
			t_e = torch.zeros_like(v_e)

		score = self.fc(img_feats).squeeze(-1)
		return score, img_proj, txt_feats, v_e, t_e
