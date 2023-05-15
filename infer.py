import torch
from models.new_model import Generator, FeatureExtractor
import numpy as np
import random
from dataset import dataloader
from utils.utils import psnr_cal
from torchvision import transforms
from PIL import Image
from ln_DDP_train import GAN
from utils.utils import load_pretrained, psnr_cal
import matplotlib.pyplot as plt

random.seed(20)
img_size = 64
batch_size = 16
device = torch.device("cuda:0")

#-----transforms-----#
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
    ])

transform_back = transforms.ToPILImage()

norm = lambda x: (x-x.min())*255/(x.max() - x.min())
norm_one = lambda x: (x-x.min())/(x.max() - x.min())

# import an example data
lr = Image.open('../SwinIR/testsets/set5/LR_bicubic/X2/img_001x2.png')
gt = Image.open('../SwinIR/testsets/set5/HR/img_001.png')
x = transform(lr)[None,:,:,:]
gt_ts = transform(gt)[None,:,:,:]


# load our weights
ckp_path = 'esrgan_2d_FE/updating_VGG19_noBN_Unet_D_SC_SN/checkpoints/last-v2.ckpt' # a Unet Discriminator version GAN, trained for 50 epoch
Gnet = Generator()
load_pretrained(Gnet, ckp_path, replace_key = 'generator.')
Gnet.eval()
pred = Gnet(x)
pred_n = norm(pred)
pred_im = transform_back(pred_n.detach().to(torch.uint8).cpu().squeeze().permute(1,2,0).numpy()).convert('RGB')
pred_im.save("SR_ours.png")
crop = pred_im.crop(pred_im.size(0)-50, 0, pred_im.size(1)+50, 100)
crop.save("crop.png")
#plt.imshow(pred.detach().squeeze().permute(1,2,0).numpy()), plt.show()


"""
PL generic loading and inferring does not work like charm :)
"""


# -----using old pytorch training----- #
with torch.no_grad():
    psnr_v = psnr_cal(pred.cpu().squeeze().numpy(),
            gt_ts.squeeze().numpy())
print(f"{psnr_v:.3f}dB")
def plot_fn(sr,gt):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(gt[0].detach().cpu().squeeze().permute(1,2,0).to(torch.int32))
    ax[0].set_title('GT')
    ax[1].imshow(sr[0].detach().cpu().squeeze().permute(1,2,0).to(torch.int32))
    ax[1].set_title('SR')
    [i.axis('off') for i in ax[:]]
    plt.show()
plot_fn(norm(pred), norm(gt_ts))

