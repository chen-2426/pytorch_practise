from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# transforms 该如何使用
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

# 调用tensorboard
writer = SummaryWriter("logs")

# ToTensor 的使用
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image("Tensor_image", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norm, 1)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize ->img_resize type:PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor ->img_resize type:tensor
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL ->PIL ->tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize, 1)

# RandomCrop
trans_random = transforms.RandomCrop((300, 200))
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomCrop", img_crop, i)

writer.close()
