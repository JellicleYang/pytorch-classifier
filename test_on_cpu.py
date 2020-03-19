from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import utils

from PIL import Image

image = Image.open('./data/valid/2/image_0085.jpg')
names = [1,2]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize
])

image_transformed = val_transforms(image)
print(image_transformed.size())
image_transformed = image_transformed.unsqueeze(0)
print(image_transformed.size())
Net = getattr(models, 'vgg16')
model = Net(num_classes=2)
model = torch.load('./weights/cpu_vgg16.pth.tar')
model.eval()
model.cpu()
out = model(image_transformed)
a,b = out.topk(1,1,True,True)
print(b.t())
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(percentage)