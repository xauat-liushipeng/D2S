# D2S - Multimodal Image Complexity

Describe, Then Score: Language-Guided Image Complexity Assessment

D2S is a multimodal regression project that estimates image complexity by fusing visual and textual features. It includes training/validation entrypoints, a typed configuration system, and common correlation metrics (SRCC/PLCC).

## Project Structure
- `d2s/`
  - `data.py`: Dataset definitions (IC9600, IC9600Caption)
  - `model.py`: Vision/Text encoders and Fusion regressor
  - `loss.py`: Loss (MSE + information bottleneck regularization)
  - `utils.py`: Checkpoint save/load and correlation metrics
  - `config.py`: Typed configuration (dataclasses), validation, YAML I/O
- `train.py`: Training entrypoint
- `val.py`: Validation and single-image inference
- `config/base.yaml`: Base configuration
- `florence2.py`: Example script that uses Microsoft Florence-2 to generate more detailed captions

Suggested data layout (if using a local folder):
- `IC9600/images/` — images
- `IC9600/train.txt`, `IC9600/val.txt` — annotation files

## Environment & Dependencies
- Python 3.9+
- Core packages: `torch`, `torchvision`, `timm`, `transformers`, `numpy`, `scipy`, `Pillow`

Install:
```bash
pip install -r requirements.txt
```

## Data Format
Text annotations use double-space separators per line:
```
<image_name>  <score>  <caption>
```
Example:
```
0001.jpg  3.72  a cat sitting on a wooden table
```
- Image directory is set by `dataset.img_dir` (e.g., `./images` or `IC9600/images`).
- Training/validation text files are set by `dataset.train_file` and `dataset.val_file`.

## Configuration
Edit `config/base.yaml`. Key sections:
- `dataset`: `train_file`, `val_file`, `img_dir`, `max_length`, `image_size`
- `model`: `vision_model_name`, `text_model_name`, `hidden_dim`, `pretrained`
- `train`: `batch_size`, `epochs`, `lr`, `beta`, `device`, `save_interval`, `checkpoint_dir`
- `val`: `batch_size`, `checkpoint`

Some values (e.g., `--image_size`) can be overridden via CLI.

## Training
```bash
python train.py --config config/base.yaml --image_size 224
```
What you get:
- Per-epoch logs with Train/Val Loss, MSE, IB, SRCC, PLCC
- Automatic checkpoints:
  - Best: `checkpoints/best_model_epoch_*.pth`
  - Periodic: every `save_interval` epochs
  - Final: `checkpoints/final_model.pth`

## Validation & Single-Image Inference
```bash
python val.py --config config/base.yaml --image_size 224
```
- Uses `val.checkpoint` for evaluation; prints Loss, MSE, IB, SRCC, PLCC
- Contains a single-image inference example (edit `test_img_name` and `test_caption`)

## Metrics
SRCC (Spearman) and PLCC (Pearson) are computed via `d2s.utils.compute_srcc_plcc` after aggregating predictions across the dataset.

## Optional: Caption Generation (Florence-2)
- See `florence2.py` for using Microsoft Florence-2 (`microsoft/Florence-2-large`) to generate more detailed captions using the `<MORE_DETAILED_CAPTION>` task tag.
- You can extend it into a batch script that fills the third column (caption) of `train.txt` / `val.txt`.
- Tip: GPU is recommended for faster inference; CPU works but will be slow.

## Troubleshooting
- Out-of-memory (GPU):
  - Reduce `train.batch_size`
  - Use smaller vision/text backbones
- Data errors:
  - Ensure double-space separators in annotation files
  - Ensure images exist and are readable
- Dependencies:
  - Verify `requirements.txt` is installed successfully

## License
This code is provided for research and educational purposes only.
