import torch
import torchvision.transforms as transforms

from .dataset import ProtoAngleDataset
from typing import Tuple

from .model import ProtoNetWithAngle


class PatternAndAngleClassifier:
    def __init__(self,
                 model_path: str = r"proto_angle_final.pt",
                 image_size: int = 224,
                 n_way: int = 3,
                 k_shot: int = 3,
                 q_query: int = 1,
                 device: str = None):
        """
        初始化分类器，加载预训练模型和支持集。
        """
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ProtoNetWithAngle(n_way=n_way).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 构造支持集
        dataset = ProtoAngleDataset(r"backend\raw", image_size=image_size)
        self.support_imgs, self.support_labels, _, _, _, self.class_names = dataset.get_episode(
            n_way=n_way, k_shot=k_shot, q_query=q_query
        )
        self.support_imgs = self.support_imgs.to(self.device)
        self.support_labels = self.support_labels.to(self.device)

        # 图像预处理 pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def predict(self, image) -> Tuple[str, float]:
        """
        对单张图像执行图案分类与角度预测
        输入：OpenCV 格式的图像 (BGR)
        输出：预测的类别名称，角度值（单位：度）
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            class_scores, predicted_angles = self.model(self.support_imgs, self.support_labels, img_tensor)
            pred_label = class_scores.argmax(dim=1).item()
            pred_angle = predicted_angles[0].item()

        pred_class_name = self.class_names[pred_label]
        return pred_class_name, pred_angle

