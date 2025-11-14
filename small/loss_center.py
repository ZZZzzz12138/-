# loss_center.py

import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=768):
        super().__init__()
        # ✅ 初始化每个类别的中心向量
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        features: [B, D]  每个样本的特征向量
        labels:   [B]     每个样本的类别 index
        """
        # 取出每个样本所属类别的中心向量
        centers_batch = self.centers[labels]  # [B, D]
        # 计算每个样本到其中心的欧式距离
        return ((features - centers_batch) ** 2).sum(dim=1).mean()
