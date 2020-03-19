import torch
import torchvision.models as models


class Models():
    def __init__(self, arch, num_classes, gpus):
        self.arch = arch
        self.num_classes = num_classes
        self.gups = gpus
    
    def Model(self):
        if self.arch == 'alexnet':
            model = models.alexnet(num_classes=self.num_classes)
        if self.arch == 'vgg11':
            model = models.vgg11(num_classes=self.num_classes)
        if self.arch == 'vgg13':
            model = models.vgg13(num_classes=self.num_classes)       
        if self.arch == 'vgg16':
            model = models.vgg16(num_classes=self.num_classes)
        if self.arch == 'vgg19':
            model = models.vgg19(num_classes=self.num_classes)
        if self.arch == 'vgg11_bn':
            model = models.vgg11_bn(num_classes=self.num_classes)
        if self.arch == 'vgg13_bn':
            model = models.vgg13_bn(num_classes=self.num_classes)       
        if self.arch == 'vgg16_bn':
            model = models.vgg16_bn(num_classes=self.num_classes)
        if self.arch == 'vgg19_bn':
            model = models.vgg19_bn(num_classes=self.num_classes)
        if self.arch == 'resnet18':
            model = models.resnet18(num_classes=self.num_classes)
        if self.arch == 'resnet34':
            model = models.resnet34(num_classes=self.num_classes)
        if self.arch == 'resnet50':
            model = models.resnet50(num_classes=self.num_classes)
        if self.arch == 'resnet101':
            model = models.resnet101(num_classes=self.num_classes)
        if self.arch == 'resnet152':
            model = models.resnet152(num_classes=self.num_classes)
        if self.arch == 'squeezenet1_0':
            model = models.squeezenet1_0(num_classes=self.num_classes)
        if self.arch == 'squeezenet1_1':
            model = models.squeezenet1_1(num_classes=self.num_classes)
        if self.arch == 'densenet121':
            model = models.densenet121(num_classes=self.num_classes)
        if self.arch == 'densenet161':
            model = models.densenet161(num_classes=self.num_classes)
        if self.arch == 'densenet169':
            model = models.densenet169(num_classes=self.num_classes)
        if self.arch == 'densenet201':
            model = models.densenet201(num_classes=self.num_classes)
        if self.arch == 'inception_v1':
            # parameters 'aux_logits' maybe will make the model not work 
            model = models.googlenet(num_classes=self.num_classes)
        if self.arch == 'inception_v3':
            # parameters 'aux_logits' maybe will make the model not work
            model = models.inception_v3(num_classes=self.num_classes)
        if self.arch == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5(num_classes=self.num_classes)
        if self.arch == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0(num_classes=self.num_classes)
        if self.arch == 'shufflenet_v2_x1_5':
            model = models.shufflenet_v2_x1_5(num_classes=self.num_classes)
        if self.arch == 'shufflenet_v2_x2_0':
            model = models.shufflenet_v2_x2_0(num_classes=self.num_classes)
        if self.arch == 'mobilenet_v2':
            model = models.mobilenet_v2(num_classes=self.num_classes)
        if self.arch == 'resnext50_32x4d':
            model = models.resnext50_32x4d(num_classes=self.num_classes)
        if self.arch == 'resnext101_32x4d':
            model = models.resnext101_32x4d(num_classes=self.num_classes)
        if self.arch == 'wide_resnet50_2':
            model = models.wide_resnet50_2(num_classes=self.num_classes)
        if self.arch == 'wide_resnet101_2':
            model = models.wide_resnet101_2(num_classes=self.num_classes)
        if self.arch == 'mnasnet1_0':
            model = models.mnasnet1_0(num_classes=self.num_classes)

        model = torch.nn.DataParallel(model, device_ids=self.gups).cuda()
        return model

