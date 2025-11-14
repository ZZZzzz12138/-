# model_convnext.py

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ProtoNetWithAngle(nn.Module):
    def __init__(self, n_way: int):
        super().__init__()
        # ✅ ConvNeXt-Tiny 主干，加载预训练权重
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        # ✅ 去除分类头，保留 feature extractor（特征维度为 768）
        self.backbone.classifier = nn.Sequential(
            nn.Flatten()  # output shape: [B, 768]
        )

        self.n_way = n_way
        self.angle_regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, support_images, support_labels, query_images):
        # 提取特征
        z_support = self.backbone(support_images)  # [S, 768]
        z_query = self.backbone(query_images)      # [Q, 768]

        # === 分类 ===
        prototypes = []
        for i in range(self.n_way):
            class_features = z_support[support_labels == i]
            prototype = class_features.mean(dim=0, keepdim=True)  # [1, 768]
            prototypes.append(prototype)
        prototypes = torch.cat(prototypes, dim=0)  # [n_way, 768]

        # Euclidean 距离：越近越相似
        dists = torch.cdist(z_query, prototypes)  # [Q, n_way]
        class_scores = -dists  # 转为得分（负距离）

        # === 回归角度 ===
        predicted_angles = self.angle_regressor(z_query).squeeze(1)

        return class_scores, predicted_angles

