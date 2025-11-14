import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from model import ProtoNetWithAngle
import os
import glob

# ==== 配置参数 ====
N_WAY = 3
K_SHOT = 3
IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载模型 ====
print("📦 正在加载模型权重 proto_angle_final.pt ...")
model = ProtoNetWithAngle(n_way=N_WAY).to(device)
model.load_state_dict(torch.load("proto_angle_final.pt", map_location=device))
model.eval()
print("✅ 模型加载完成")

# ==== 图像预处理与支持集构建 ====
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor()
])
class_names = sorted(os.listdir("raw"))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

support_imgs, support_labels = [], []
for cls in class_names:
    img_paths = glob.glob(os.path.join("raw", cls, "*.jpg"))[:K_SHOT]
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        support_imgs.append(transform(img))
        support_labels.append(class_to_idx[cls])

support_imgs = torch.stack(support_imgs).to(device)
support_labels = torch.tensor(support_labels).to(device)
print(f"📁 支持集加载完成，共 {len(support_imgs)} 张图像")

# ==== 打开摄像头 ====
cap = cv2.VideoCapture(0)
print("📷 相机开启，按 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    # OpenCV BGR → PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        class_scores, angle_pred = model(support_imgs, support_labels, image_tensor)
        pred_label = class_scores.argmax(dim=1).item()
        pred_angle = angle_pred.item()
        distances = -class_scores.squeeze().cpu().numpy()

    pred_class = class_names[pred_label]

    # 打印信息到控制台
    print(f"✅ 类别: {pred_class} | 角度: {pred_angle:.1f}°")
    print("↪ 欧式距离到各原型中心:")
    for i, dist in enumerate(distances):
        print(f"  - {class_names[i]} : {dist:.2f}")

    # 可视化：叠加文字
    text = f"{pred_class} | {pred_angle:.1f} deg"
    vis_frame = cv2.resize(frame, (400, 400))
    cv2.putText(vis_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示窗口
    cv2.imshow("📸 Industrial Camera Inference", vis_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 用户退出")
        break

cap.release()
cv2.destroyAllWindows()
