## [Pattern Recognition] Describe-to-Score: A text-guided framework for image complexity assessment

### [2026-08-05] :tada: Our paper has been accepted by [Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S003132032601513X)

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


#### Citation
If you are using our D2S for your research, please cite the following paper:
```
@article{liu2026describe,
  title={Describe-to-Score: A text-guided framework for image complexity assessment},
  author={Liu, Shipeng and Zhao, Liang and Chen, Dengfeng and Zhang, Zhonglin},
  journal={Pattern Recognition},
  pages={114549},
  year={2026},
  publisher={Elsevier}
}
```

We provide the trained weights `D2S_R18.pth` in this repo and `D2S_R50.pth` at [Google Drive](https://drive.google.com/file/d/1CYRuCUc-YOpg1NJXT-M0RcxgC3AAPfxX/view?usp=sharing)
