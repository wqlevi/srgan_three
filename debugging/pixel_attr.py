import torch
import torchvision
class Resnet_FE(torch.nn.Module):
    def __init__(self, model):
        super(Resnet_FE, self).__init__()
        self.model = model
        self.fe = torch.nn.Sequential(*list(self.model.children())[:-1])
    def __call__(self,x):
        return self.fe(x)[:,:,None,None]
