import os
import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ===== 配置 =====
DATA_DIR = "../data/IC9600"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TXT_FILE = os.path.join(DATA_DIR, "train_more_detailed_caption.txt")  # 可改成 test.txt
OUTPUT_CSV = "entropy_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型配置
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion400m_e31"

BATCH_SIZE = 32  # 一次计算的配对数量

# ===== Step 1: 加载模型 =====
print(f"Loading OpenCLIP model {MODEL_NAME} ({PRETRAINED}) on {DEVICE}...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# ===== Step 2: 读取数据 =====
def load_dataset(txt_path):
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("  ")  # 双空格分隔
            if len(parts) >= 3:
                img_name, score, caption = parts[0], parts[1], "  ".join(parts[2:])
                samples.append((img_name, float(score), caption))
    return samples

samples = load_dataset(TXT_FILE)
print(f"Loaded {len(samples)} samples from {TXT_FILE}")

# ===== Step 3: 信息熵计算函数 =====
def compute_entropy(prob_row):
    return -np.sum(prob_row * np.log(prob_row + 1e-12))

# ===== Step 4: 批量计算 =====
entropies = []
results = []

for start in range(0, len(samples), BATCH_SIZE):
    batch_samples = samples[start:start + BATCH_SIZE]

    # 加载图片和文本
    images = []
    texts = []
    valid_indices = []
    for idx, (img_name, score, caption) in enumerate(batch_samples):
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            continue
        try:
            images.append(preprocess(Image.open(img_path).convert("RGB")))
            texts.append(caption)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[Error] Loading image {img_name}: {e}")

    if not images or not texts:
        continue

    images = torch.stack(images).to(DEVICE)
    texts = tokenizer(texts).to(DEVICE)

    # 编码特征
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 相似度矩阵 [B, B]
        sims = image_features @ text_features.T
        probs = torch.softmax(sims, dim=-1).cpu().numpy()  # 每行是 image→caption 概率分布

    # 计算每张图片的熵
    for idx_in_batch, idx_in_file in enumerate(valid_indices):
        img_name, score, caption = batch_samples[idx_in_file]
        entropy_val = compute_entropy(probs[idx_in_batch])
        entropies.append(entropy_val)
        results.append([img_name, score, caption, entropy_val])

    if (start + len(batch_samples)) % 100 == 0:
        print(f"Processed {start + len(batch_samples)}/{len(samples)} samples...")

# ===== Step 5: 保存结果 =====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "score", "caption", "entropy"])
    writer.writerows(results)

print(f"Saved entropy results to {OUTPUT_CSV}")

# ===== Step 6: 可视化 =====
plt.figure(figsize=(8, 5))
sns.histplot(entropies, kde=True, bins=30)
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.title(f"Image-Caption Entropy Distribution ({MODEL_NAME}-{PRETRAINED})")
plt.tight_layout()
plt.savefig("entropy_distribution.png", dpi=300)
