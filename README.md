D2S - Multimodal Image Complexity (IC9600)
Files:
 - data.py    : Dataset and dataloader builder (CSV input: image_path, caption, score)
 - model.py   : Pretrained backbone loaders (timm/transformers lazy) + FusionRegressor
 - train.py   : Training loop (uses dataloaders, model)
 - val.py     : Validation helper computing PLCC / SRCC
 - utils.py   : helpers for saving/loading and metrics
 - config.py  : default config

Usage (example):
  python -m D2S.train --data_csv /path/to/ic9600.csv --image_root /path/to/images --epochs 30 --batch_size 4 --grad_accum 4 --fp16

Notes:
 - The code prefers timm and transformers but will fall back to lightweight modules if those packages are not available.
 - For 12GB GPU, use small batch_size (4) + grad_accum (4) + --fp16.
 - Offline-generate captions and store in CSV to avoid running a captioning model online during training.