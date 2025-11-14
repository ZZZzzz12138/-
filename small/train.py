# train_safe.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from tqdm import trange

from dataset import ProtoAngleDataset
from model import ProtoNetWithAngle

# ==== 配置 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device}")

# few-shot 超参
N_WAY = 3
K_SHOT = 3
Q_QUERY = 3

# 训练超参
IMAGE_SIZE = 224
EPOCHS = 1200
SAVE_INTERVAL = 50
LEARNING_RATE_BACKBONE = 1e-5
LEARNING_RATE_HEAD = 1e-3
REG_WEIGHT = 1.0  # sin/cos 回归一般设 1.0

# 路径
MODEL_FINAL = "proto_angle_final.pt"          # 仅权重
MODEL_LATEST = "checkpoints/latest.pt"        # 仅权重 ckpt
os.makedirs("checkpoints", exist_ok=True)

# ==== 可复现性（可选）====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== 数据 ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = ProtoAngleDataset("raw", image_size=IMAGE_SIZE, transform=transform)

# ==== 模型 ====
# 如果没有外网或不想下载预训练权重，可传 pretrained=False
model = ProtoNetWithAngle(n_way=N_WAY, pretrained=True).to(device)

# 分组学习率：backbone 小、head 大
optimizer = Adam([
    {"params": model.backbone.parameters(), "lr": LEARNING_RATE_BACKBONE},
    {"params": model.head.parameters(),     "lr": LEARNING_RATE_HEAD},
])

# ==== 恢复（仅加载权重；若不兼容自动跳过）====
start_epoch = 1
EXPECTED_META = {"arch": "resnet18_sincos_protonet", "backbone": "resnet18", "head": "sincos2"}
if os.path.exists(MODEL_LATEST):
    # 读取 ckpt（兼容老版本 torch）
    try:
        checkpoint = torch.load(MODEL_LATEST, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_LATEST, map_location=device)

    state = checkpoint.get("model_state", checkpoint)
    meta = checkpoint.get("meta", {})

    def is_compatible(sd, mt):
        # 1) 优先看 meta
        if mt.get("arch") == EXPECTED_META["arch"] and \
           mt.get("backbone") == EXPECTED_META["backbone"] and \
           mt.get("head") == EXPECTED_META["head"]:
            return True
        # 2) 启发式 key 检查（粗略过滤 ConvNeXt/旧 angle_regressor 命名）
        keys = list(sd.keys())
        if any(k.startswith("backbone.features") for k in keys):
            return False
        if any(k.startswith("angle_regressor") for k in keys):
            return False
        # 3) 至少要有 resnet 风格的前几层 key
        return any(k.startswith("backbone.0") or k.startswith("backbone.1") for k in keys)

    if is_compatible(state, meta):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️ 非严格加载：missing={len(missing)}, unexpected={len(unexpected)}")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"🔁 从 epoch {start_epoch} 恢复训练...（已加载兼容权重）")
    else:
        print("⚠️ 检测到不兼容的 checkpoint（架构/命名不同），跳过加载，从头训练。")

# ==== 训练 ====
print("🚀 开始训练...\n")
bar = trange(start_epoch, EPOCHS + 1)

for epoch in bar:
    model.train()

    # 采样一个 episode
    support_imgs, support_labels, query_imgs, query_labels, query_angles, _ = dataset.get_episode(
        n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY
    )
    support_imgs   = support_imgs.to(device)
    query_imgs     = query_imgs.to(device)
    support_labels = support_labels.to(device)
    query_labels   = query_labels.to(device)
    query_angles   = query_angles.to(device)  # 单位：度, shape [B]

    # 前向：分类 logits + 角度预测单位向量（[sinθ, cosθ]）
    class_scores, pred_unit = model(support_imgs, support_labels, query_imgs)  # [B,2]

    # 分类损失
    loss_cls = F.cross_entropy(class_scores, query_labels)

    # === sin/cos 回归损失 ===
    gt_sin = torch.sin(torch.deg2rad(query_angles))
    gt_cos = torch.cos(torch.deg2rad(query_angles))
    gt_vec = torch.stack([gt_sin, gt_cos], dim=-1)  # [B,2]
    loss_reg = F.mse_loss(pred_unit, gt_vec)

    total_loss = loss_cls + REG_WEIGHT * loss_reg

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 指标：还原角度并用环状 MAE（度）
    with torch.no_grad():
        pred_deg = torch.rad2deg(torch.atan2(pred_unit[..., 0], pred_unit[..., 1]))  # atan2(sin, cos)
        diff = torch.remainder(pred_deg - query_angles + 180.0, 360.0) - 180.0
        mae = diff.abs().mean().item()

        pred_labels = class_scores.argmax(dim=1)
        acc = (pred_labels == query_labels).float().mean().item()

    bar.set_description(
        f"[{epoch}/{EPOCHS}] 🎯 Acc: {acc*100:.1f}% | Cls: {loss_cls.item():.4f} | Reg: {loss_reg.item():.4f} | MAE: {mae:.2f}°"
    )

    # 偶尔打印范围，快速体检
    if epoch % 50 == 0:
        with torch.no_grad():
            pr_min, pr_max = float(pred_deg.min()), float(pred_deg.max())
            gt_min, gt_max = float(query_angles.min()), float(query_angles.max())
            print(f"\n   ↪︎ pred_deg_range: [{pr_min:.2f}, {pr_max:.2f}] | gt_deg_range: [{gt_min:.2f}, {gt_max:.2f}]")

    # ==== 只保存权重 + meta ====
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        meta = {"arch": "resnet18_sincos_protonet", "backbone": "resnet18", "head": "sincos2",
                "n_way": N_WAY, "k_shot": K_SHOT, "q_query": Q_QUERY, "img": IMAGE_SIZE}
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "meta": meta}, MODEL_LATEST)

# ==== 最终保存：仅保存权重 + meta ====
final_meta = {"arch": "resnet18_sincos_protonet", "backbone": "resnet18", "head": "sincos2",
              "n_way": N_WAY, "k_shot": K_SHOT, "q_query": Q_QUERY, "img": IMAGE_SIZE}
torch.save({"model_state": model.state_dict(), "meta": final_meta}, MODEL_FINAL)
print(f"\n✅ 训练完成，模型已保存至：{MODEL_FINAL}（仅权重）")
