# evaluate.py
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import ProtoAngleDataset
from model import ProtoNetWithAngle

# ==== 配置 ====
N_WAY = 3
K_SHOT = 3
Q_QUERY = 2
IMAGE_SIZE = 224
CKPT_PATH = "proto_angle_final.pt"   # 训练脚本保存的“仅权重 + meta”ckpt
EXPECTED_META = {"arch": "resnet18_sincos_protonet", "backbone": "resnet18", "head": "sincos2"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device}")

# ==== 模型 ====
model = ProtoNetWithAngle(n_way=N_WAY, pretrained=False).to(device)
model.eval()

# ==== 安全加载权重（仅当与当前架构兼容）====
def safe_load_state_dict(path, model):
    if not os.path.exists(path):
        print(f"❌ 找不到权重文件: {path}")
        return False
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # 老版本 PyTorch 不支持 weights_only
        obj = torch.load(path, map_location=device)

    state = obj.get("model_state", obj)
    meta  = obj.get("meta", {})

    def is_compatible(sd, mt):
        # 1) 优先用 meta 判断
        if mt.get("arch") == EXPECTED_META["arch"] and \
           mt.get("backbone") == EXPECTED_META["backbone"] and \
           mt.get("head") == EXPECTED_META["head"]:
            return True
        # 2) 启发式键名过滤：ConvNeXt/旧结构直接判不兼容
        keys = list(sd.keys())
        if any(k.startswith("backbone.features") for k in keys):  # ConvNeXt 风格
            return False
        if any(k.startswith("angle_regressor") for k in keys):    # 旧标量角头
            return False
        # 3) ResNet 风格粗判
        return any(k.startswith("backbone.0") or k.startswith("backbone.1") for k in keys)

    if not is_compatible(state, meta):
        print("⚠️ 检测到不兼容的权重（可能来自旧结构/ConvNeXt）。请使用由当前训练脚本导出的 ckpt。")
        return False

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ℹ️ 非严格加载：missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"📦 已加载模型权重：{path}")
    return True

print(f"📦 加载训练好的模型权重 {CKPT_PATH} ...")
if not safe_load_state_dict(CKPT_PATH, model):
    raise SystemExit(1)
print("✅ 模型加载完成，进入评估模式。")

# ==== 数据 ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
print("📁 构造一个测试任务（episode），包含支持集与查询集 ...")
dataset = ProtoAngleDataset("raw", image_size=IMAGE_SIZE, transform=transform)
support_imgs, support_labels, query_imgs, query_labels, query_angles, class_names = dataset.get_episode(
    n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY
)
print(f"✅ 支持集数量: {len(support_imgs)}，查询集数量: {len(query_imgs)}")
print(f"🎯 测试类别: {class_names}")

# 移动到设备
support_imgs   = support_imgs.to(device)
query_imgs     = query_imgs.to(device)
support_labels = support_labels.to(device)
# 注意：query_labels / query_angles 评估时在 CPU 用更方便；需要也可转 device
# query_labels_cpu/query_angles_cpu 仅用于打印与计算指标
query_labels_cpu = query_labels.clone()
query_angles_cpu = query_angles.clone()

# ==== 推理 ====
print("🔍 开始进行分类和角度预测 ...")
with torch.no_grad():
    class_scores, pred_unit = model(support_imgs, support_labels, query_imgs)  # pred_unit: [B,2] = (sin,cos)

    # 分类准确率
    pred_labels = class_scores.argmax(dim=1).cpu()
    acc = (pred_labels == query_labels_cpu).float().mean().item()

    # 角度还原（度）
    pred_deg = torch.rad2deg(torch.atan2(pred_unit[..., 0], pred_unit[..., 1])).cpu()  # (-180,180]
    # 环状 MAE（[-180,180)）
    diff = torch.remainder(pred_deg - query_angles_cpu + 180.0, 360.0) - 180.0
    mae = diff.abs().mean().item()

print(f"✅ 测试分类准确率 (Accuracy): {acc*100:.2f}%")
print(f"📐 平均角度误差 (MAE): {mae:.2f}°")

# ==== 可视化查询样本预测 ====
print("🖼️ 正在可视化部分查询图像预测结果 ...")
to_pil = transforms.ToPILImage()
n_show = min(query_imgs.shape[0], 5)

plt.figure(figsize=(15, 3))
for i in range(n_show):
    img = to_pil(query_imgs[i].cpu())
    true_cls = class_names[query_labels_cpu[i]]
    pred_cls = class_names[pred_labels[i]]

    angle_gt = float(query_angles_cpu[i].item())
    angle_pred = float(pred_deg[i].item())

    plt.subplot(1, n_show, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"T:{true_cls}\nP:{pred_cls}\nGT:{angle_gt:.1f}°\nPR:{angle_pred:.1f}°")
plt.tight_layout()
plt.show()
