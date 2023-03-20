#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:49:17 2022
- [x] ESRGAN training 
- [x] Recording 1st SV of conv1,2,3 
@author: qiwang
"""

import torch
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms
from models.new_model import FeatureExtractor, Generator, Discriminator, VGG19_54
from utils.utils import power_iteration, psnr_cal, print_norm_hook
import wandb
from torch.autograd import Variable
import numpy as np
import gc, random
from dataset import dataloader
from models.compute_gradient import Gradient, Lap

random.seed(10)
# params
plot_per_iter = 1000
image_size = 64
batch_size = 16
num_epochs = 10
input_channel =3
lr = 1e-4
device = torch.device("cuda:0")

# dataloaders
dataset = dataloader(root = '/big_data/qi1/Celeba/train',hr_shape = image_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=2,drop_last = True)


# models
Gnet = Generator().to(device)
Dnet = Discriminator((input_channel,image_size,image_size)).to(device)
#FE = FeatureExtractor().to(device)
FE = VGG19_54().to(device)

Gnet = torch.nn.DataParallel(Gnet, device_ids=[0,1,2])
Dnet = torch.nn.DataParallel(Dnet, device_ids=[0,1,2])
FE = torch.nn.DataParallel(FE, device_ids=[0,1,2])
model_list = [Dnet, Gnet, FE]
optimizer = [torch.optim.Adam(Dnet.parameters(),
                             lr = lr,
                             betas = (.9,.999)),
            torch.optim.Adam(list(Gnet.parameters())+list(FE.parameters()),
                             lr = lr,
                             betas = (.9,.999))]
"""
optimizer = [torch.optim.Adam(Dnet.parameters(),
                             lr = lr,
                             betas = (.9,.999)),
            torch.optim.Adam(list(Gnet.parameters())+list(FE.parameters()),
                             lr = lr,
                             betas = (.9,.999)),
            torch.optim.Adam(Gnet.parameters(),
                             lr = lr,
                             betas = (.9,.999))]
"""
            
loss = [torch.nn.L1Loss(), torch.nn.BCEWithLogitsLoss(), torch.nn.L1Loss()]
# train
def train(model, loss, optimizer):
    Dnet, Gnet, FE = model
    optimizer_D, optimizer_G= optimizer
    criterion_pixel, criterion_GAN, criterion_content = loss
    
    Dnet.train()
    Gnet.train()
    FE.train()
    Tensor = torch.cuda.FloatTensor
    valid = Variable(Tensor(np.ones((batch_size, *Dnet.module.output_shape))),requires_grad=False)
    fake = Variable(Tensor(np.zeros((batch_size, *Dnet.module.output_shape))),requires_grad=False) 
    
    #noise
    noise_mean = torch.full((batch_size,*Dnet.module.input_shape),0,dtype=torch.float32,device=device)
    sigma = 1
    
    Grad = Gradient().cuda()
    laplace = Lap().cuda()
    
    for epoch in range(num_epochs):
        sigma -= 0.1
        sigma = max(sigma,0)
        noise_std= torch.full((batch_size,*Dnet.module.input_shape),sigma,dtype=torch.float32,device=device)
        for i,imgs in enumerate(dataloader):
            imgs_hr = imgs['hr'].to(device)
            imgs_lr = imgs['lr'].to(device)
            
            #
            # G training
            #

            # register hooks
            #FE.module.vgg19_54[0].register_forward_hook(print_norm_hook)
            optimizer_G.zero_grad()
            #optimizer_FE.zero_grad()
            sr = Gnet(imgs_lr)
            loss_pixel = criterion_pixel(sr, imgs_hr)
            """
            if epoch == 100:
                loss_pixel.backward()
                optimizer_G.step()
                print(f"[{i}/{len(dataloader)}] Loss pixel:\t{loss_pixel}")
                continue
            """
            
            # MAKE SOME NOISE!!!!
            instance_noise = torch.normal(mean=noise_mean, std=noise_std).to(device)
            pred_real = Dnet(imgs_hr+instance_noise).detach()
            pred_fake = Dnet(sr+instance_noise)
            loss_GAN_G = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            sr_feature = FE(sr)
            hr_feature = FE(imgs_hr).detach()
            
            loss_content = criterion_content(sr_feature, hr_feature)
            
            loss_G = loss_content + 5e-3*loss_GAN_G + 1e-2*loss_pixel
            loss_G.backward() # -[BUG] updating both FE and G?
            #optimizer_FE.step()
            optimizer_G.step()
            
            #
            # D training
            #
            optimizer_D.zero_grad()
            pred_real = Dnet(imgs_hr+instance_noise)
            pred_fake = Dnet(sr.detach()+instance_noise)
            
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0,keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake- pred_real.mean(0,keepdim=True), fake)
            
            loss_D = (loss_real + loss_fake) /2
            loss_D.backward()
            optimizer_D.step()
           
            # 
            # metricessx
            #
            d_sv_list = []
            fe_sv_list= []
            
            with torch.no_grad():
                for j in range(len(Dnet.module.model)):
                    if Dnet.module.model[j].__str__()[0] == 'N': # meaning its a Norm layer
                        d_sv_list.append(Dnet.module.model[j].module.weight_sv.item())
                for j,m in enumerate(FE.module.conv_layers.modules()):
                    if m.__str__()[0] == 'N': # meaning its a Norm layer
                        fe_sv_list.append(m.module.weight_sv.item())
                psnr = psnr_cal(sr.cpu().squeeze().numpy(),
                                imgs_hr.cpu().squeeze().numpy())
                wandb.log({'train_D_loss':loss_D.item(),
                           'train_G_loss':loss_G.item(),
                           'train_psnr':psnr,
                           'train_SV_Gnet_conv1':Gnet.module.conv1.module.weight_sv.item(),
                           'train_SV_Gnet_conv2':Gnet.module.conv2.module.weight_sv.item(),
                           'train_SV_Gnet_conv3':Gnet.module.conv3[0].module.weight_sv.item(),
                           'train_SV_Dnet_conv1':d_sv_list[0],
                           'train_SV_Dnet_conv2':d_sv_list[1],
                           'train_SV_Dnet_conv3':d_sv_list[2],
                           'train_SV_FE_conv1':fe_sv_list[0],
                           'train_SV_FE_conv2':fe_sv_list[1],
                           'train_SV_FE_conv3':fe_sv_list[2]
                           })
            if i%plot_per_iter == 0:
                img_grid = vutils.make_grid(imgs_hr[:4], 2, padding=2, normalize=True).permute(1,2,0)
                sr_grid = vutils.make_grid(sr[:4], 2,padding=2, normalize=True).permute(1,2,0) # make color channel as last dim
                with torch.no_grad():
                    y_grad = Grad(sr[0].to(device)) # 1st derivative
                    y_lap = laplace(sr[0].cuda()) # 2nd derivative type 1
                wandb.log({'img':[wandb.Image(img_grid.detach().cpu().numpy(), caption='GT'),
                                  wandb.Image(sr_grid.detach().cpu().numpy(), caption='SR'),
                                  wandb.Image(y_grad.permute(1,2,0).detach().cpu().numpy(), caption='grad'),
                                  wandb.Image(y_lap.squeeze().permute(1,2,0).detach().cpu().numpy(), caption='laplace')]})
            
        gc.collect()
        torch.cuda.empty_cache()
    torch.save({'Gnet_state_dict':Gnet.module.state_dict(),
                'FE_state_dict':FE.state_dict()},
               'toy-model_%d_epoch.pth'%(epoch))
            
if __name__ == '__main__':
    wandb.init(project='esrgan_2d_FE',entity='wqlevi')
    train(model_list, loss, optimizer)



