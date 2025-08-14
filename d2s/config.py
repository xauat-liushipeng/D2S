#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: d2s/config.py
Description: Configuration management module for D2S with typed dataclasses, validation and helpers.
Date: 2025-08-13
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
	"""Dataset configuration"""
	train_file: str = "train.txt"
	val_file: str = "val.txt"
	img_dir: str = "./images"
	max_length: int = 32
	image_size: int = 224


@dataclass
class ModelConfig:
	"""Model configuration"""
	vision_model_name: str = "google/vit-base-patch16-224"
	text_model_name: str = "bert-base-uncased"
	hidden_dim: int = 512
	pretrained: bool = True
	use_icnet_head: bool = False  # Whether to use ICNetHead instead of simple FC head


@dataclass
class TrainConfig:
	"""Training configuration"""
	batch_size: int = 16
	epochs: int = 10
	warm: int = 1
	lr: float = 0.0001
	lr_decay_rate: float = 0.2
	weight_decay: float = 1e-3
	milestone = [10, 20]  #
	Lambda: float = 0.0001
	device: str = "cuda"
	save_interval: int = 10
	checkpoint_dir: str = "checkpoints"


@dataclass
class ValConfig:
	"""Validation configuration"""
	batch_size: int = 16
	checkpoint: str = "fusion_regressor.pth"


@dataclass
class Config:
	"""Root configuration class"""
	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	model: ModelConfig = field(default_factory=ModelConfig)
	train: TrainConfig = field(default_factory=TrainConfig)
	val: ValConfig = field(default_factory=ValConfig)
	
	def __post_init__(self):
		"""Post-initialization adjustments"""
		# Ensure checkpoint directory exists
		os.makedirs(self.train.checkpoint_dir, exist_ok=True)
		
		# Device availability check
		if self.train.device == "cuda" and not self._is_cuda_available():
			print("Warning: CUDA is not available, fallback to CPU")
			self.train.device = "cpu"
	
	def _is_cuda_available(self) -> bool:
		"""Check if CUDA is available"""
		try:
			import torch
			return torch.cuda.is_available()
		except ImportError:
			return False
	
	@classmethod
	def from_yaml(cls, config_path: str) -> 'Config':
		"""Load configuration from a YAML file"""
		config_path = Path(config_path)
		
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")
		
		try:
			with open(config_path, 'r', encoding='utf-8') as f:
				config_dict = yaml.safe_load(f)
			
			return cls._from_dict(config_dict)
			
		except yaml.YAMLError as e:
			raise ValueError(f"Failed to parse YAML: {e}")
		except Exception as e:
			raise RuntimeError(f"Failed to load config: {e}")
	
	@classmethod
	def _from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
		"""Create Config instance from a dictionary"""
		# Create default config
		config = cls()
		
		# Dataset overrides
		if 'dataset' in config_dict:
			dataset_dict = config_dict['dataset']
			for key, value in dataset_dict.items():
				if hasattr(config.dataset, key):
					setattr(config.dataset, key, value)
		
		# Model overrides
		if 'model' in config_dict:
			model_dict = config_dict['model']
			for key, value in model_dict.items():
				if hasattr(config.model, key):
					setattr(config.model, key, value)
		
		# Training overrides
		if 'train' in config_dict:
			train_dict = config_dict['train']
			for key, value in train_dict.items():
				if hasattr(config.train, key):
					setattr(config.train, key, value)
		
		# Validation overrides
		if 'val' in config_dict:
			val_dict = config_dict['val']
			for key, value in val_dict.items():
				if hasattr(config.val, key):
					setattr(config.val, key, value)
		
		return config
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to a plain dictionary"""
		return {
			'dataset': {
				'train_file': self.dataset.train_file,
				'val_file': self.dataset.val_file,
				'img_dir': self.dataset.img_dir,
				'max_length': self.dataset.max_length,
				'image_size': self.dataset.image_size,
			},
			'model': {
				'vision_model_name': self.model.vision_model_name,
				'text_model_name': self.model.text_model_name,
				'hidden_dim': self.model.hidden_dim,
				'pretrained': self.model.pretrained,
				'use_icnet_head': self.model.use_icnet_head,
			},
			'train': {
				'batch_size': self.train.batch_size,
				'epochs': self.train.epochs,
				'lr': self.train.lr,
				'lr_decay_rate': self.train.lr_decay_rate,
				'weight_decay': self.train.weight_decay,
				'milestone': self.train.milestone,
				'warm': self.train.warm,
				'Lambda': self.train.Lambda,
				'device': self.train.device,
				'save_interval': self.train.save_interval,
				'checkpoint_dir': self.train.checkpoint_dir,
			},
			'val': {
				'batch_size': self.val.batch_size,
				'checkpoint': self.val.checkpoint,
			}
		}
	
	def save_yaml(self, config_path: str):
		"""Save configuration to a YAML file"""
		config_path = Path(config_path)
		config_path.parent.mkdir(parents=True, exist_ok=True)
		
		try:
			with open(config_path, 'w', encoding='utf-8') as f:
				yaml.dump(self.to_dict(), f, default_flow_style=False, 
						 allow_unicode=True, indent=2)
			print(f"Config saved to: {config_path}")
		except Exception as e:
			raise RuntimeError(f"Failed to save config: {e}")
	
	def validate(self) -> bool:
		"""Validate configuration values"""
		errors = []
		
		# Dataset checks
		if not os.path.exists(self.dataset.img_dir):
			errors.append(f"Image directory not found: {self.dataset.img_dir}")
		
		if self.dataset.image_size <= 0:
			errors.append(f"Image size must be > 0: {self.dataset.image_size}")
		
		if self.dataset.max_length <= 0:
			errors.append(f"Max length must be > 0: {self.dataset.max_length}")
		
		# Training checks
		if self.train.batch_size <= 0:
			errors.append(f"Batch size must be > 0: {self.train.batch_size}")
		
		if self.train.epochs <= 0:
			errors.append(f"Epochs must be > 0: {self.train.epochs}")
		
		if self.train.lr <= 0:
			errors.append(f"Learning rate must be > 0: {self.train.lr}")
		
		if self.train.Lambda < 0:
			errors.append(f"Lambda must not be negative: {self.train.Lambda}")
		
		# Model checks
		if self.model.hidden_dim <= 0:
			errors.append(f"Hidden dimension must be > 0: {self.model.hidden_dim}")
		
		if errors:
			print("Config validation failed:")
			for error in errors:
				print(f"  - {error}")
			return False
		
		return True
	
	def get_device(self):
		"""Get torch.device based on availability and config"""
		import torch
		if self.train.device == "cuda" and torch.cuda.is_available():
			return torch.device("cuda")
		return torch.device("cpu")
	
	def get_checkpoint_path(self, epoch: int, is_best: bool = False) -> str:
		"""Build checkpoint path for a given epoch"""
		if is_best:
			filename = f"best_model_epoch_{epoch}.pth"
		else:
			filename = f"model_epoch_{epoch}.pth"
		
		return os.path.join(self.train.checkpoint_dir, filename)
	
	def get_final_checkpoint_path(self) -> str:
		"""Path for the final model checkpoint"""
		return os.path.join(self.train.checkpoint_dir, "final_model.pth")
	
	def print_summary(self):
		"""Print a human-readable summary of configuration"""
		print("=" * 50)
		print("Config Summary")
		print("=" * 50)
		
		print(f"Dataset:")
		print(f"  Train file: {self.dataset.train_file}")
		print(f"  Val file: {self.dataset.val_file}")
		print(f"  Image dir: {self.dataset.img_dir}")
		print(f"  Image size: {self.dataset.image_size}")
		print(f"  Max length: {self.dataset.max_length}")
		
		print(f"\nModel:")
		print(f"  Vision: {self.model.vision_model_name}")
		print(f"  Text: {self.model.text_model_name}")
		print(f"  Hidden dim: {self.model.hidden_dim}")
		print(f"  Pretrained: {self.model.pretrained}")
		print(f"  Use ICNetHead: {self.model.use_icnet_head}")

		print(f"\nTrain:")
		print(f"  Device: {self.train.device}")
		print(f"  Batch size: {self.train.batch_size}")
		print(f"  Epochs: {self.train.epochs}")
		print(f"  Warm epochs: {self.train.warm}")
		print(f"  LR: {self.train.lr}")
		print(f"  LR decay rate: {self.train.lr_decay_rate}")
		print(f"  Weight decay: {self.train.weight_decay}")
		print(f"  Milestone: {self.train.milestone}")
		print(f"  Lambda: {self.train.Lambda}")
		print(f"  Save interval: {self.train.save_interval}")
		print(f"  Checkpoint dir: {self.train.checkpoint_dir}")
		
		print(f"\nVal:")
		print(f"  Batch size: {self.val.batch_size}")
		print(f"  Checkpoint: {self.val.checkpoint}")
		
		print("=" * 50)


def load_config(config_path: str = "config/base.yaml") -> Config:
	"""Convenience function to load config from YAML"""
	return Config.from_yaml(config_path)


def create_default_config(config_path: str = "config/default.yaml"):
	"""Create and save a default configuration YAML"""
	config = Config()
	config.save_yaml(config_path)
	return config
