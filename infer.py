import torch
from models.new_model import Generator, FeatureExtractor
import numpy as np
import random
from dataset import dataloader
from utils.utils import psnr_cal

random.seed(20)
img_size = 64
batch_size = 16
device = torch.device("cuda:0")

dst = dataloader(root="/big_data/qi1/Celeba/train", hr_shape=img_size)
loader = torch.utils.data.dataloader.DataLoader(dst, batch_size = batch_size, shuffle=True, num_workers=2, drop_last=True)

Gnet = Generator().to(device)
Gnet.load_state_dict(torch.load("toy-model_9_epoch.pth")['Gnet_state_dict'])
Gnet.eval()

_, im = next(enumerate(loader))
output = Gnet(im['lr'].to(device))
with torch.no_grad():
    psnr_v = psnr_cal(output.cpu().squeeze().numpy(),
            im['hr'].squeeze().numpy())
print(f"{psnr_v:.3f}dB")
plot_fn(output, im['hr'])
def plot_fn(sr,gt):
    fig, ax = plt.subplot(1,2)
    ax[0].imshow(gt[0].detach().cpu().squeeze().permute(1,2,0))
    ax[1].imshow(sr[0].detach().cpu().squeeze().permute(1,2,0))
    plt.show()
