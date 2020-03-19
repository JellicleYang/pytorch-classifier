import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data(train_data, valid_data, batch_size, workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
                                           transforms.RandomHorizontalFlip(),		#水平翻转
                                           transforms.ToTensor(),					#转化为张量
                                           normalize	                            #进行归一化
    ])

    val_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         normalize
    ])

    train_datasets = datasets.ImageFolder(train_data, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    val_datasets = datasets.ImageFolder(valid_data, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_datasets,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    return train_loader, val_loader