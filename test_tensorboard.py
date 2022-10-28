from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
# 输出图片
imag_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(imag_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("train", img_array, 1, dataformats="HWC")
# 输出函数情况
for i in range(100):
    writer.add_scalar("y=x", i, i)
# 在terminal中用tensorboard --logdir=文件夹名称 --port=端口号 查看
writer.close()
