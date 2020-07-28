import ResNet
import torch
resnet101 = ResNet.resnet101(pretrained=True, b_fc = True)

print(resnet101.state_dict().keys())
temp = resnet101.state_dict()
temp.pop('fc.weight')
temp.pop('fc.bias')
torch.save(temp, 'resnet101_no_fc.bin')

resnet101 = ResNet.resnet101(pretrained=True, b_fc = False)
print(resnet101.state_dict().keys())

