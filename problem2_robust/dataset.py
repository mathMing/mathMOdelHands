import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MOSEIDataset(Dataset):
    def __init__(self, data_split, is_train=True, augment_missing=False, 
                 audio_mean=None, audio_std=None, vision_mean=None, vision_std=None):
        self.is_train = is_train
        self.augment_missing = augment_missing
        
        self.ids = data_split['id']
        self.text = data_split['text'].astype(np.float32)      # (N, 50, 768)
        self.audio = data_split['audio'].astype(np.float32)    # (N, 50, 74)
        self.vision = data_split['vision'].astype(np.float32)  # (N, 50, 35)
        
        # 标签
        self.cls_labels = data_split['classification_labels'].astype(np.int64).reshape(-1)
        self.reg_labels = data_split['regression_labels'].astype(np.float32).reshape(-1)
        
        # 特征标准化 (Z-score Normalization)
        if audio_mean is None:
            a_mask = np.abs(self.audio).sum(axis=-1) > 1e-5
            self.audio_mean = np.mean(self.audio[a_mask], axis=0, keepdims=True) if a_mask.any() else np.zeros((1, 74))
            self.audio_std = np.std(self.audio[a_mask], axis=0, keepdims=True) + 1e-6 if a_mask.any() else np.ones((1, 74))
        else:
            self.audio_mean = audio_mean
            self.audio_std = audio_std

        if vision_mean is None:
            v_mask = np.abs(self.vision).sum(axis=-1) > 1e-5
            self.vision_mean = np.mean(self.vision[v_mask], axis=0, keepdims=True) if v_mask.any() else np.zeros((1, 35))
            self.vision_std = np.std(self.vision[v_mask], axis=0, keepdims=True) + 1e-6 if v_mask.any() else np.ones((1, 35))
        else:
            self.vision_mean = vision_mean
            self.vision_std = vision_std

        # 应用归一化
        self.audio = np.where(np.abs(self.audio) > 1e-5, (self.audio - self.audio_mean) / self.audio_std, 0.0)
        self.vision = np.where(np.abs(self.vision) > 1e-5, (self.vision - self.vision_mean) / self.vision_std, 0.0)

    def __len__(self):
        return len(self.ids)

    def apply_temporal_masking(self, t, a, v):
        """思路四核心机制：时序连续区间动态挖空增强（模拟附件3真实缺失场景）"""
        seq_len = 50
        if np.random.rand() < 0.5: # 50% 概率触发缺失模拟
            # 真实附件3中以音视频局部缺失为主
            # 覆盖题目要求的单模态和双模态局部连续缺失。
            mods = np.random.choice(
                ['t', 'a', 'v', 't_a', 't_v', 'a_v'],
                p=[0.20, 0.20, 0.20, 0.13, 0.13, 0.14]
            )
            mlen = np.random.randint(5, 25) # 缺失 10% ~ 50% 序列长度
            st = np.random.randint(0, seq_len - mlen + 1)
            ed = st + mlen
            if 't' in mods: t[st:ed] = 0.0
            if 'a' in mods: a[st:ed] = 0.0
            if 'v' in mods: v[st:ed] = 0.0
        return t, a, v

    def __getitem__(self, idx):
        t_full = self.text[idx].copy()
        a_full = self.audio[idx].copy()
        v_full = self.vision[idx].copy()
        cls_l = self.cls_labels[idx]
        reg_l = self.reg_labels[idx]

        t_miss = t_full.copy()
        a_miss = a_full.copy()
        v_miss = v_full.copy()

        if self.is_train and self.augment_missing:
            # 应用时序连续区间动态掩码 (模拟残缺输入，供 Student 训练)
            t_miss, a_miss, v_miss = self.apply_temporal_masking(t_miss, a_miss, v_miss)

        # 动态生成有效性掩码
        mask_t_f = (np.abs(t_full).sum(axis=-1) > 1e-5).astype(np.float32)
        mask_a_f = (np.abs(a_full).sum(axis=-1) > 1e-5).astype(np.float32)
        mask_v_f = (np.abs(v_full).sum(axis=-1) > 1e-5).astype(np.float32)

        mask_t_m = (np.abs(t_miss).sum(axis=-1) > 1e-5).astype(np.float32)
        mask_a_m = (np.abs(a_miss).sum(axis=-1) > 1e-5).astype(np.float32)
        mask_v_m = (np.abs(v_miss).sum(axis=-1) > 1e-5).astype(np.float32)

        item = {
            # Student 接收的残缺特征 (若测试集则为真实输入)
            'text': torch.tensor(t_miss, dtype=torch.float32),
            'audio': torch.tensor(a_miss, dtype=torch.float32),
            'vision': torch.tensor(v_miss, dtype=torch.float32),
            'mask_t': torch.tensor(mask_t_m, dtype=torch.float32),
            'mask_a': torch.tensor(mask_a_m, dtype=torch.float32),
            'mask_v': torch.tensor(mask_v_m, dtype=torch.float32),
            # 标签
            'cls_label': torch.tensor(cls_l, dtype=torch.long),
            'reg_label': torch.tensor(reg_l, dtype=torch.float32),
            'id': self.ids[idx]
        }

        # 训练阶段同时提供 Teacher 接收的完备特征 (用于自蒸馏特征与软标签对齐)
        if self.is_train:
            item.update({
                'text_full': torch.tensor(t_full, dtype=torch.float32),
                'audio_full': torch.tensor(a_full, dtype=torch.float32),
                'vision_full': torch.tensor(v_full, dtype=torch.float32),
                'mask_t_full': torch.tensor(mask_t_f, dtype=torch.float32),
                'mask_a_full': torch.tensor(mask_a_f, dtype=torch.float32),
                'mask_v_full': torch.tensor(mask_v_f, dtype=torch.float32),
            })

        return item

def get_dataloaders(pkl_path, batch_size=64, augment_missing=True):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    train_set = MOSEIDataset(data['train'], is_train=True, augment_missing=augment_missing)
    valid_set = MOSEIDataset(data['valid'], is_train=False, augment_missing=False,
                             audio_mean=train_set.audio_mean, audio_std=train_set.audio_std,
                             vision_mean=train_set.vision_mean, vision_std=train_set.vision_std)
    test_set = MOSEIDataset(data['test'], is_train=False, augment_missing=False,
                            audio_mean=train_set.audio_mean, audio_std=train_set.audio_std,
                            vision_mean=train_set.vision_mean, vision_std=train_set.vision_std)
                            
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_set
