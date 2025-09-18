## Describe-to-Score: Text-Guided Image Complexity Assessment


### 0. Installation
```bash
pip install -r requirements.txt
```

#### Caption Generation
```
python tools/caption_gen.py
```

### 1. Data Preparation
Text annotations use double-space separators per line:
```
<image_name>  <score>  <caption>
```
Example:
```
0001.jpg  0.12  a cat sitting on a wooden table
```

### 2. Training
```
python train.py --img_dir="../data/IC9600/images" \
                --train_file="../data/IC9600/train_blip_caption.txt" \
                --val_file="../data/IC9600/test.txt" \
                --image_size=512 \
                --vision_encoder="resnet18"        
```

### 3. Validation
```
python val.py --img_dir="../data/IC9600/images" \
              --val_file="../data/IC9600/test.txt" \
              --image_size=512 \
              --vision_encoder="resnet18" \
              --ckpts="./D2S_R18.pth" \
              --batch_size=32
```

#### Test log on our A2000 12G
D2S ResNet18
```
Using device: cuda
Image transforms created for size: 512x512
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:13<00:00,  6.65it/s]
SRCC: 0.9521 | PLCC: 0.9548 | RMSE: 0.0496 | RMAE: 0.1963 | Time: 13.54s
```
D2S ResNet50
```
Using device: cuda
Image transforms created for size: 512x512
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:32<00:00,  2.73it/s]
SRCC: 0.9544 | PLCC: 0.9580 | RMSE: 0.0497 | RMAE: 0.1966 | Time: 32.93s
```