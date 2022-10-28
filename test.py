import torch
import torchvision.transforms
from PIL import Image
from model import *

# 数据引入
image_path = "图片路径"
image = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(32, 32), torchvision.transforms.ToTensor])
image = transform(image)
# 检测部分
model = torch.load("训练模型权重")
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))