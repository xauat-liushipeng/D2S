from collections import deque
from typing import Callable, Deque, Dict, Optional
import torch


class FeatureBuffer:
    def __init__(
        self,
        buffer_size: int = 2048,
        refresh_steps: int = 50,
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        self.buffer_size: int = buffer_size
        self.buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.refresh_steps: int = max(1, refresh_steps)
        self.image_loader: Optional[Callable[[str], torch.Tensor]] = image_loader
        self._refresh_ptr: int = 0

    def set_image_loader(self, image_loader: Callable[[str], torch.Tensor]) -> None:
        self.image_loader = image_loader

    def push_samples(
        self,
        image_names,
        tokens,
        hv: torch.Tensor,
        hs: torch.Tensor,
    ) -> None:
        if isinstance(image_names, torch.Tensor):
            image_names = list(image_names)

        hv = hv.detach().cpu().view(-1)
        hs = hs.detach().cpu().view(-1)

        batch_size = hv.shape[0]
        for i in range(batch_size):
            name_i = image_names[i]
            token_i = tokens[i]
            name_i = name_i if isinstance(name_i, str) else str(name_i)
            self.buffer.append({
                'image_name': name_i,
                'token': token_i,
                'hv': float(hv[i].item()),
                'hs': float(hs[i].item()),
            })

    def push(self, feats: torch.Tensor) -> None:
        feats = feats.detach().cpu().view(-1)
        for f in feats:
            self.buffer.append({'image_name': '', 'token': None, 'hv': float(f.item()), 'hs': float('nan')})

    def get(self, device: torch.device, which: str = 'hv') -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        values = []
        for item in list(self.buffer):
            v = item.get(which, float('nan'))
            if v == v:
                values.append(v)
        if len(values) == 0:
            return None
        return torch.tensor(values, dtype=torch.float32, device=device)

    def refresh(self, key_model, device: torch.device, batch_size: int = 32, use_amp: bool = True) -> None:
        if len(self.buffer) == 0 or self.image_loader is None:
            return

        total = min(len(self.buffer), self.buffer_size)
        if total == 0:
            return
        chunk = max(1, total // self.refresh_steps)

        start = self._refresh_ptr % total
        end = start + chunk

        idxs = list(range(total))
        if end <= total:
            sel = idxs[start:end]
        else:
            sel = idxs[start:total] + idxs[0:(end - total)]

        key_model.eval()
        with torch.no_grad():
            i = 0
            while i < len(sel):
                j = min(i + batch_size, len(sel))
                part = sel[i:j]

                images = []
                tokens = []
                names = []
                for k in part:
                    item = self.buffer[k]
                    img = self.image_loader(item['image_name'])
                    images.append(img)
                    tokens.append(item['token'])
                    names.append(item['image_name'])

                images = torch.stack(images, dim=0).to(device, non_blocking=True)
                tokens = torch.stack(tokens, dim=0).to(device, non_blocking=True)

                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _, _, _, hv_k, hs_k = key_model(images, tokens)
                else:
                    _, _, _, hv_k, hs_k = key_model(images, tokens)

                hv_k = hv_k.detach().cpu().view(-1)
                hs_k = hs_k.detach().cpu().view(-1)
                for t, k in enumerate(part):
                    self.buffer[k]['hv'] = float(hv_k[t].item())
                    self.buffer[k]['hs'] = float(hs_k[t].item())

                i = j

        self._refresh_ptr = end % total

    def __len__(self) -> int:
        return len(self.buffer)