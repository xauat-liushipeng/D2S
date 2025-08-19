from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple

import torch


class FeatureBuffer:
    """
    带“影子网络（key_encoder）动量更新 + 队列重算熵”的特征缓冲区。

    设计要点：
    - 仅存储轻量的样本元信息：image_name、input_ids、attention_mask、hv、hs
    - 不直接缓存图片张量，刷新时通过 image_loader(name)->tensor 重新加载，减少内存占用
    - 提供按步增量刷新：把  target_size 样本在 refresh_steps 步内分片重算（近似 2048/N）
    """

    def __init__(
        self,
        max_size: int = 2048,
        refresh_steps: int = 50,
        target_size: int = 2048,
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        self.max_size: int = max_size
        self.buffer: Deque[Dict] = deque(maxlen=max_size)
        self.refresh_steps: int = max(1, refresh_steps)
        self.target_size: int = target_size
        self.image_loader: Optional[Callable[[str], torch.Tensor]] = image_loader
        self._refresh_ptr: int = 0

    def set_image_loader(self, image_loader: Callable[[str], torch.Tensor]) -> None:
        self.image_loader = image_loader

    def push_samples(
        self,
        image_names,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hv: torch.Tensor,
        hs: torch.Tensor,
    ) -> None:
        """
        以样本粒度压入队列；仅保存名称/文本token与当前的 hv/hs 标量。
        - image_names: List[str] 或 长度为B的序列
        - input_ids, attention_mask: [B, L]
        - hv, hs: [B]
        """
        if isinstance(image_names, torch.Tensor):
            image_names = list(image_names)

        hv = hv.detach().cpu().view(-1)
        hs = hs.detach().cpu().view(-1)
        input_ids = input_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()

        batch_size = hv.shape[0]
        for i in range(batch_size):
            name_i = image_names[i]
            # 允许 name 是 bytes/张量等，统一转 str
            name_i = name_i if isinstance(name_i, str) else str(name_i)
            self.buffer.append({
                'image_name': name_i,
                'input_ids': input_ids[i].clone(),
                'attention_mask': attention_mask[i].clone(),
                'hv': float(hv[i].item()),
                'hs': float(hs[i].item()),
            })

    # 兼容旧接口：仅压入数值，不支持刷新
    def push(self, feats: torch.Tensor) -> None:
        feats = feats.detach().cpu().view(-1)
        for f in feats:
            self.buffer.append({'image_name': '', 'input_ids': None, 'attention_mask': None, 'hv': float(f.item()), 'hs': float('nan')})

    def get(self, device: torch.device, which: str = 'hv') -> Optional[torch.Tensor]:
        """
        which: 'hv' 或 'hs'
        返回 [N] 的张量；若缺失则返回 None
        """
        if len(self.buffer) == 0:
            return None
        values = []
        for item in list(self.buffer):
            v = item.get(which, float('nan'))
            if v == v:  # 非 NaN
                values.append(v)
        if len(values) == 0:
            return None
        return torch.tensor(values, dtype=torch.float32, device=device)

    def refresh(self, key_model, device: torch.device, batch_size: int = 32, use_amp: bool = True) -> None:
        """
        使用影子网络对队列中的一段样本重新计算 hv/hs。
        - 每次调用仅刷新一个分片；分片大小 ≈ target_size / refresh_steps
        - 需先设置 image_loader
        """
        if len(self.buffer) == 0 or self.image_loader is None:
            return

        total = min(len(self.buffer), self.target_size)
        if total == 0:
            return
        chunk = max(1, total // self.refresh_steps)

        # 计算刷新起止（循环）
        start = self._refresh_ptr % total
        end = start + chunk

        # 取出要刷新的索引列表
        idxs = list(range(total))
        if end <= total:
            sel = idxs[start:end]
        else:
            sel = idxs[start:total] + idxs[0:(end - total)]

        # 组 mini-batch 前向
        key_model.eval()
        with torch.no_grad():
            i = 0
            while i < len(sel):
                j = min(i + batch_size, len(sel))
                part = sel[i:j]

                images = []
                ids_list = []
                attn_list = []
                names = []
                for k in part:
                    item = self.buffer[k]
                    img = self.image_loader(item['image_name'])  # tensor [3,H,W]
                    images.append(img)
                    ids_list.append(item['input_ids'])
                    attn_list.append(item['attention_mask'])
                    names.append(item['image_name'])

                images = torch.stack(images, dim=0).to(device, non_blocking=True)
                if ids_list[0] is not None:
                    input_ids = torch.stack(ids_list, dim=0).to(device, non_blocking=True)
                    attention_mask = torch.stack(attn_list, dim=0).to(device, non_blocking=True)
                else:
                    # 纯视觉时使用占位
                    input_ids = None
                    attention_mask = None

                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        if input_ids is not None:
                            _, hv_k, hs_k = key_model(images, input_ids, attention_mask)
                        else:
                            _, hv_k, hs_k = key_model(images)
                else:
                    if input_ids is not None:
                        _, hv_k, hs_k = key_model(images, input_ids, attention_mask)
                    else:
                        _, hv_k, hs_k = key_model(images)

                hv_k = hv_k.detach().cpu().view(-1)
                hs_k = hs_k.detach().cpu().view(-1)
                for t, k in enumerate(part):
                    self.buffer[k]['hv'] = float(hv_k[t].item())
                    self.buffer[k]['hs'] = float(hs_k[t].item())

                i = j

        self._refresh_ptr = end % total

    def __len__(self) -> int:
        return len(self.buffer)