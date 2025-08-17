import os
import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 配置 =====
DATA_DIR = "/home/liushipeng/work/data/IC9600"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TXT_FILE = os.path.join(DATA_DIR, "train_caption.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion400m_e31"
BATCH_SIZE = 32

# ===== 加载模型 =====
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# ===== 读取数据 =====
def load_dataset(txt_path):
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("  ")
            if len(parts) >= 3:
                img_name, score, caption = parts[0], parts[1], "  ".join(parts[2:])
                samples.append((img_name, float(score), caption))
    return samples

samples = load_dataset(TXT_FILE)
print(f"Loaded {len(samples)} samples.")

# ===== 单模态熵函数 =====
def feature_entropy(feat_vec):
    feat_vec = feat_vec / feat_vec.norm(p=2)  # L2 归一化
    probs = torch.softmax(feat_vec, dim=-1).cpu().numpy()
    return -np.sum(probs * np.log(probs + 1e-12))

# ===== 批量计算 =====
image_entropies = []
text_entropies = []

for start in range(0, len(samples), BATCH_SIZE):
    batch_samples = samples[start:start+BATCH_SIZE]

    images = []
    texts = []
    for img_name, score, caption in batch_samples:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        try:
            images.append(preprocess(Image.open(img_path).convert("RGB")))
            texts.append(caption)
        except:
            continue

    if not images or not texts:
        continue

    images = torch.stack(images).to(DEVICE)
    texts = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

    for img_feat, txt_feat in zip(image_features, text_features):
        image_entropies.append(feature_entropy(img_feat))
        text_entropies.append(feature_entropy(txt_feat))

    if (start + BATCH_SIZE) % 100 == 0:
        print(f"Processed {start+BATCH_SIZE}/{len(samples)} samples...")

# ===== 可视化 =====
plt.figure(figsize=(8,5))
sns.kdeplot(image_entropies, shade=True, color="blue", label="Image features")
sns.kdeplot(text_entropies, shade=True, color="red", label="Text features")
plt.xlabel("Normalized Feature Entropy")
plt.ylabel("Density")
plt.title(f"{MODEL_NAME} ({PRETRAINED}) - Normalized Feature Entropy Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("feature_entropy_distribution_normalized.png", dpi=300)
