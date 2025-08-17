import os
import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.manifold import TSNE

# ===== 配置 =====
DATA_DIR = "../data/IC9600"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TXT_FILE = os.path.join(DATA_DIR, "train_more_detailed_caption.txt")
OUTPUT_CSV = "entropy_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion400m_e31"
BATCH_SIZE = 32

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
            parts = line.strip().split("  ")
            if len(parts) >= 3:
                img_name, score, caption = parts[0], parts[1], "  ".join(parts[2:])
                samples.append((img_name, float(score), caption))
    return samples

samples = load_dataset(TXT_FILE)
print(f"Loaded {len(samples)} samples from {TXT_FILE}")

# ===== Step 3: 信息熵计算函数 =====
def compute_entropy(prob_row):
    return -np.sum(prob_row * np.log(prob_row + 1e-12))

# ===== Step 4: 批量计算 + 特征收集 =====
entropies = []
results = []
image_feats_all = []
text_feats_all = []

for start in range(0, len(samples), BATCH_SIZE):
    batch_samples = samples[start:start + BATCH_SIZE]

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

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 保存特征
        image_feats_all.append(image_features.cpu().numpy())
        text_feats_all.append(text_features.cpu().numpy())

        # 相似度矩阵
        sims = image_features @ text_features.T
        probs = torch.softmax(sims, dim=-1).cpu().numpy()

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

# ===== Step 6: 可视化熵分布 =====
plt.figure(figsize=(8, 5))
sns.histplot(entropies, kde=True, bins=30)
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.title(f"Image-Caption Entropy Distribution ({MODEL_NAME}-{PRETRAINED})")
plt.tight_layout()
plt.savefig("entropy_distribution.png", dpi=300)

# ===== Step 7: 可视化特征分布 =====
# 拼接所有特征
image_feats_all = np.vstack(image_feats_all)
text_feats_all = np.vstack(text_feats_all)

# 只取前 N 条防止 t-SNE 太慢
MAX_VIS = 2000
if len(image_feats_all) > MAX_VIS:
    idx_sel = np.random.choice(len(image_feats_all), MAX_VIS, replace=False)
    image_feats_all = image_feats_all[idx_sel]
    text_feats_all = text_feats_all[idx_sel]

# 合并两个模态
all_feats = np.vstack([image_feats_all, text_feats_all])
labels = np.array([0]*len(image_feats_all) + [1]*len(text_feats_all))  # 0=image, 1=text

print("Running t-SNE for feature visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
feats_2d = tsne.fit_transform(all_feats)

# 绘图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=feats_2d[:,0], y=feats_2d[:,1],
                hue=labels, palette={0: "blue", 1: "red"}, alpha=0.6, s=20)
plt.legend(["Image", "Text"])
plt.title(f"{MODEL_NAME} ({PRETRAINED}) Feature Space t-SNE")
plt.tight_layout()
plt.savefig("feature_distribution_tsne.png", dpi=300)
