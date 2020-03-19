
import torch
import torchvision.models as models
import os

# th architecture to use
model_name = 'vgg16'
model_weight = './weights/'+model_name+'_best.pth.tar'
model_type = 'cpu' 

# create the network architecture
print('read model')
model = getattr(models, model_name)
model = model(num_classes=2)

# the data parallel layer will add 'module' before each layer name
print('load weight')
checkpoint = torch.load(model_weight)

print('change data parallel layer state')
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} 

print('recover the model')
model.load_state_dict(state_dict)

print('chose the model type gpu or cpu')
if model_type == 'gpu':
    print('gpu')
    model.gpu()
else:
    print('cpu')
    model.cpu()
model.eval()
print('save model')
torch.save(model, './weights/' + model_type + '_' + model_name + '.pth.tar')