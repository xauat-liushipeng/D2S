import time
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from d2s.model_ import D2S
from d2s.data import IC9600
from d2s.utils import *


def get_parser():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="D2S Validation - Image Complexity Assessment")
	parser.add_argument('--device', type=str, default='cuda', help='device for training (e.g., cuda or cpu)')
	parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
	# dataset
	parser.add_argument('--img_dir', type=str, default='../data/IC9600/images', help='path to image directory')
	parser.add_argument('--val_file', type=str, default='../data/IC9600/test.txt', help='path to validation file')
	parser.add_argument('--image_size', type=int, default=512, help='input image size for model')
	parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
	# model
	parser.add_argument('--vision_encoder', type=str, default='resnet18', help='vision model name (e.g., resnet18, vit)')
	parser.add_argument('--text_encoder', type=str, default='clip', help='text model name (e.g., bert, roberta); empty for text-free mode')
	parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for regression head')
	parser.add_argument('--num_heads', type=int, default=16, help='number of attention heads in vision encoder')
	parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained weights for encoders')
	# training
	parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
	parser.add_argument('--ckpts', type=str,
	                    default="./D2S_R18.pth",
	                    help='path to model checkpoint for val')

	return parser.parse_args()


def create_transforms(image_size: int):
	base_transforms = [
		transforms.Resize((image_size, image_size)),  # Resize to target dimensions
		transforms.ToTensor(),                        # Convert PIL to tensor
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
	]
	return transforms.Compose(base_transforms)


def validate(model, dataloader, device):
	model.eval()
	all_preds, all_scores = [], []
	val_start_time = time.time()

	with torch.no_grad():
		for images, scores, _ in tqdm(dataloader):
			images = images.to(device, non_blocking=True)
			scores = scores.to(device, non_blocking=True)

			preds = model(images)

			all_preds.extend(preds.detach().cpu().numpy())
			all_scores.extend(scores.detach().cpu().numpy())

	# Compute correlation metrics
	srcc, plcc, rmse, rmae = compute_srcc_plcc(all_preds, all_scores)
	val_time = time.time() - val_start_time

	return srcc, plcc, rmse, rmae, val_time


def main():
	args = get_parser()

	# Determine device (GPU if available, otherwise CPU)
	device = get_device(args)
	print(f"Using device: {device}")

	# Create image transforms for training and validation
	val_transform = create_transforms(args.image_size)
	print(f"Image transforms created for size: {args.image_size}x{args.image_size}")

	# Load training and validation datasets
	val_dataset = IC9600(
		args.val_file,
		args.img_dir,
		val_transform
	)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	# Create and initialize the model

	model = D2S(
		args.vision_encoder,
		args.hidden_dim,
		args.num_heads,
		args.pretrained
	).to(device)

	model.load_state_dict(torch.load(args.ckpts, weights_only=True)["model_state"], strict=False)

	# Validation phase
	v_srcc, v_plcc, v_rmse, v_rmae, v_time = validate(
		model, val_loader, device
	)

	print(f"SRCC: {v_srcc:.4f} | PLCC: {v_plcc:.4f} | RMSE: {v_rmse:.4f} | RMAE: {v_rmae:.4f} | Time: {v_time:.2f}s")


if __name__ == "__main__":
	main()
