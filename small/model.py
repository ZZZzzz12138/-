# model.py
import torch
import torch.nn as nn
from torchvision import models

class ProtoNetWithAngle(nn.Module):
    """
    - backbone: ResNet18（默认 ImageNet 预训练；无外网可设 pretrained=False）
    - 分类：ProtoNet（support 计算原型，query 与原型的负欧氏距离为 logits）
    - 角度：回归 (sinθ, cosθ)，并在前向中做单位化
    """
    def __init__(self, n_way: int, pretrained: bool = True):
        super().__init__()
        self.n_way = n_way

        # === Backbone ===
        # 兼容 torchvision 新旧版本权重写法
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained)

        modules = list(resnet.children())[:-1]  # 去掉 fc，保留到 avgpool
        self.backbone = nn.Sequential(*modules)  # [B,512,1,1] -> flatten -> 512
        feat_dim = 512

        # === Head（分类 + 角度）===
        # 分类不需要额外层（直接基于原型距离得到 logits）
        # 角度回归头：2 维输出（sin, cos）未归一化
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def _embed(self, x):
        # x: [B,3,H,W] -> [B,512]
        feats = self.backbone(x)              # [B,512,1,1]
        feats = feats.flatten(1)              # [B,512]
        return feats

    @staticmethod
    def _prototypes(feats, labels, n_way):
        """根据 episode 内标签(0..n_way-1)计算每一类的原型向量"""
        D = feats.size(1)
        protos = []
        for c in range(n_way):
            mask = (labels == c)
            pc = feats[mask]
            proto = pc.mean(dim=0) if pc.numel() > 0 else torch.zeros(D, device=feats.device)
            protos.append(proto)
        protos = torch.stack(protos, dim=0)  # [n_way, D]
        return protos

    @staticmethod
    def _euclidean_logits(query, protos):
        """
        query:  [Q, D]
        protos: [C, D]
        return: [Q, C]  (负欧氏距离作为 logit)
        """
        q2 = (query**2).sum(dim=1, keepdim=True)      # [Q,1]
        p2 = (protos**2).sum(dim=1, keepdim=True).t() # [1,C]
        dist2 = q2 + p2 - 2.0 * query @ protos.t()    # [Q,C]
        logits = -dist2
        return logits

    def forward(self, support_imgs, support_labels, query_imgs):
        """
        return:
          class_scores: [Q, n_way]
          pred_unit:    [Q, 2]  (单位化后的 (sin, cos))
        """
        # 1) 支持/查询特征
        support_feats = self._embed(support_imgs)  # [S, D]
        query_feats   = self._embed(query_imgs)    # [Q, D]

        # 2) 原型
        protos = self._prototypes(support_feats, support_labels, self.n_way)  # [n_way, D]

        # 3) 分类 logits
        class_scores = self._euclidean_logits(query_feats, protos)  # [Q, n_way]

        # 4) 角度 (sin,cos) 向量（单位化）
        pred_vec  = self.head(query_feats)                            # [Q,2]
        pred_unit = pred_vec / (pred_vec.norm(dim=-1, keepdim=True) + 1e-6)

        return class_scores, pred_unit
