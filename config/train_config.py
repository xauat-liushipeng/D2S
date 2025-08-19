"""
Training configuration class for D2S
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    train_file: str = "../data/IC9600/train_more_detailed_caption.txt"
    val_file: str = "../data/IC9600/test_more_detailed_caption.txt"
    img_dir: str = "../data/IC9600/images"
    max_length: int = 128
    image_size: int = 512


@dataclass
class ModelConfig:
    """Model configuration"""
    vision_model_name: str = "resnet18"
    text_model_name: str = "bert-base-uncased"
    hidden_dim: int = 512
    pretrained: bool = True


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int = 32
    epochs: int = 30
    warm: int = 0
    lr: float = 0.001
    lr_decay_rate: float = 0.1
    weight_decay: float = 1e-3
    milestone: List[int] = field(default_factory=lambda: [10, 20])
    device: str = "cuda"
    save_interval: int = 1
    checkpoint_dir: str = "checkpoints"
    # EMA and buffer settings
    ema_momentum: float = 0.995
    buffer_max_size: int = 2048
    buffer_refresh_steps: int = 50
    buffer_target_size: int = 2048
    align_weight: float = 0.2


@dataclass
class ValConfig:
    """Validation configuration"""
    batch_size: int = 32
    checkpoint: str = "fusion_regressor.pth"


@dataclass
class LogConfig:
    """Logging configuration"""
    log_dir: str = "logs"
    save_stdout: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    """Root configuration class"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    def init(self):
        """Post-initialization adjustments"""
        # Ensure checkpoint directory exists
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log.log_dir, exist_ok=True)

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
    
    def get_device(self):
        """Get torch.device based on availability and config"""
        import torch
        if self.train.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def get_checkpoint_path(self, epoch: int, is_best: bool = False, checkpoint_dir=None) -> str:
        """Build checkpoint path for a given epoch"""
        if is_best:
            filename = f"best_model_epoch_{epoch}.pth"
        else:
            filename = f"model_epoch_{epoch}.pth"
        
        return os.path.join(checkpoint_dir, filename)
    
    def get_final_checkpoint_path(self, checkpoint_dir) -> str:
        """Path for the final model checkpoint"""
        return os.path.join(checkpoint_dir, "final_model.pth")
    
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

        print(f"\nTrain:")
        print(f"  Device: {self.train.device}")
        print(f"  Batch size: {self.train.batch_size}")
        print(f"  Epochs: {self.train.epochs}")
        print(f"  Warm epochs: {self.train.warm}")
        print(f"  LR: {self.train.lr}")
        print(f"  LR decay rate: {self.train.lr_decay_rate}")
        print(f"  Weight decay: {self.train.weight_decay}")
        print(f"  Milestone: {self.train.milestone}")
        print(f"  Save interval: {self.train.save_interval}")
        print(f"  Checkpoint dir: {self.train.checkpoint_dir}")
        print(f"  EMA momentum: {self.train.ema_momentum}")
        print(f"  Buffer size: {self.train.buffer_max_size}")
        print(f"  Align weight: {self.train.align_weight}")
        
        print(f"\nVal:")
        print(f"  Batch size: {self.val.batch_size}")
        print(f"  Checkpoint: {self.val.checkpoint}")
        
        print(f"\nLog:")
        print(f"  Log dir: {self.log.log_dir}")
        print(f"  Save stdout: {self.log.save_stdout}")
        print(f"  Log level: {self.log.log_level}")
        
        print("=" * 50)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
