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

print('model reload')
Net = getattr(models, 'vgg16')
model = Net(num_classes=2)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
cudnn.benchmark = True
checkpoint = torch.load('./weights/vgg16_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()


image = Image.open('./data/valid/2/image_0101.jpg')
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

out = model(image_transformed)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(percentage)