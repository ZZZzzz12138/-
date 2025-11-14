# dataset.py
import os
import glob
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ProtoAngleDataset(Dataset):
    """
    - 支持集：不旋转（提供“0°/正方向”的参照）
    - 查询集：随机旋转，返回旋转角（度）作为监督
    """
    def __init__(self, root_dir, image_size=224, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size

        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # 类名与索引
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 收集每类图像路径（多后缀）
        self.data = {}
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for cls in self.class_names:
            img_paths = []
            for e in exts:
                img_paths += glob.glob(os.path.join(root_dir, cls, e))
            img_paths = [p for p in img_paths if os.path.isfile(p)]
            self.data[cls] = img_paths

    def _load_rgb(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img

    def get_episode(self, n_way, k_shot, q_query):
        support_imgs, support_labels = [], []
        query_imgs, query_labels, query_angles = [], [], []

        valid_classes = [cls for cls in self.class_names
                         if len(self.data[cls]) >= (k_shot + q_query)]
        if len(valid_classes) < n_way:
            raise ValueError(f"可用类别不足：只找到 {len(valid_classes)} 个满足样本要求。")

        selected_classes = random.sample(valid_classes, n_way)
        class_label_map = {cls: i for i, cls in enumerate(selected_classes)}

        for cls in selected_classes:
            img_paths = random.sample(self.data[cls], k_shot + q_query)
            support_paths = img_paths[:k_shot]
            query_paths   = img_paths[k_shot:]

            # 支持集：不旋转
            for path in support_paths:
                try:
                    img = self._load_rgb(path)
                except Exception:
                    continue
                img = self.transform(img)
                support_imgs.append(img)
                support_labels.append(class_label_map[cls])

            # 查询集：随机旋转 + 双线性插值（白色填充可按数据风格调整）
            for path in query_paths:
                try:
                    img = self._load_rgb(path)
                except Exception:
                    continue
                angle = random.uniform(0.0, 360.0)  # 绝对角（度）
                img = TF.rotate(
                    img, angle,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=(255, 255, 255)
                )
                img = self.transform(img)
                query_imgs.append(img)
                query_labels.append(class_label_map[cls])
                query_angles.append(angle)

        return (
            torch.stack(support_imgs),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_imgs),
            torch.tensor(query_labels, dtype=torch.long),
            torch.tensor(query_angles, dtype=torch.float32),  # 单位：度
            selected_classes
        )
