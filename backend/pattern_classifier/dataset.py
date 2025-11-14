import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ProtoAngleDataset(Dataset):
    def __init__(self, root_dir, image_size=224, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        # 构建类别名与类别索引映射
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 收集每类的图像路径
        self.data = {}
        for cls in self.class_names:
            img_paths = glob.glob(os.path.join(root_dir, cls, "*.jpg"))
            self.data[cls] = img_paths

    def get_episode(self, n_way, k_shot, q_query):
        support_imgs, support_labels = [], []
        query_imgs, query_labels, query_angles = [], [], []

        # ✅ 筛选所有样本数量足够的类
        valid_classes = [cls for cls in self.class_names if len(self.data[cls]) >= (k_shot + q_query)]
        if len(valid_classes) < n_way:
            raise ValueError(f"可用类别不足：只找到 {len(valid_classes)} 个类别满足样本要求。")

        # ✅ 随机选择 n_way 个类别
        selected_classes = random.sample(valid_classes, n_way)
        class_label_map = {cls: i for i, cls in enumerate(selected_classes)}

        for cls in selected_classes:
            img_paths = random.sample(self.data[cls], k_shot + q_query)
            support_paths = img_paths[:k_shot]
            query_paths = img_paths[k_shot:]

            # ✅ 支持集图像（也进行旋转以增强鲁棒性）
            for path in support_paths:
                angle = random.uniform(0, 360)
                img = Image.open(path).convert("RGB")
                img = TF.rotate(img, angle, fill=(255, 255, 255))
                img = self.transform(img)
                support_imgs.append(img)
                support_labels.append(class_label_map[cls])

            # 查询集图像（继续保持随机旋转）
            for path in query_paths:
                angle = random.uniform(0, 360)
                img = Image.open(path).convert("RGB")
                img = TF.rotate(img, angle, fill=(255, 255, 255))
                img = self.transform(img)
                query_imgs.append(img)
                query_labels.append(class_label_map[cls])
                query_angles.append(angle)

        return (
            torch.stack(support_imgs),
            torch.tensor(support_labels),
            torch.stack(query_imgs),
            torch.tensor(query_labels),
            torch.tensor(query_angles),
            selected_classes
        )
