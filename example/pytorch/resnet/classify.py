'''
 @ Libs   : python3.9 -m pip install jieba -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/7/5
 @ Desc   : 描述
'''

from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
resnet = models.resnet101(pretrained=True)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("./images/1.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()

out = resnet(batch_t)
_, indices = torch.sort(out, descending=True)
for idx in indices[0][:5]:
    print(idx)