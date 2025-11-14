# clip_zeroshot_predictor.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Step 1: 模型与处理器加载
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Step 2: 设置你的标签类别（英文 prompt 描述）
class_names = [
    "a nautical label with the text 'Seafarer Shipworks', an anchor on top and a ship wheel at the bottom",
    "a round vintage badge with a black ship wheel in the center and the text 'LIBERTY & AMITY'",
    "a marine label with the text 'Land & Sea', an anchor in the center, and a ribbon saying 'SINCE 1826'"
]


# Step 3: 加载图像
image_path = "test_images\test3.jpg"  # 替换为你的图像路径
image = Image.open(image_path).convert("RGB")

# Step 4: 编码图像与文本，计算相似度
inputs = processor(text=class_names, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像对文本的相似度
    probs = logits_per_image.softmax(dim=1)      # 概率分布

# Step 5: 输出预测结果
pred_idx = probs.argmax().item()
print(f"预测类别：{class_names[pred_idx]}")
print(f"所有类别概率：{probs.tolist()[0]}")
