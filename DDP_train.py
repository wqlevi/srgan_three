import torch
import torch.nn as nn
from tqdm import tqdm
import os, gc
from datetime import timedelta
import wandb
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.models import vgg19
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
import numpy as np
from torch.autograd import Variable
from dataset import dataloader as DataLoader
from models.new_model import Generator, Discriminator, VGG19_54, Discriminator_Unet
from utils.utils import psnr_cal

def make(opt, rank, world_size):
    n_fraction = 10
    dataset_all = DataLoader(root=opt.data_path, hr_shape=opt.image_size)
    dataset,_ = random_split(dataset_all,[len(dataset_all)//n_fraction, len(dataset_all)-len(dataset_all)//n_fraction])
    dist_sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler = dist_sampler,  batch_size = opt.batch_size,  num_workers = int(os.cpu_count()/2), drop_last = True)
    return dataloader

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    #os.environ['TORCH_CPP_LOG_LEVEL']='INFO'
    #os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f"|Rank-{rank} is initialized|")

def demo(rank, world_size, opt):
    setup(rank, world_size)
    if dist.get_rank() == 0:
        print(dist.get_backend())
        wandb.init(project=opt.model_name,entity='wqlevi')
        wandb.config.update(opt)
    dataloader = make(opt, rank, world_size)
    Gnet = Generator().to(rank)
    Dnet_orig = Discriminator((opt.input_channel,opt.image_size,opt.image_size)).to(rank)
    Dnet= nn.SyncBatchNorm.convert_sync_batchnorm(Dnet_orig)
    FE = VGG19_54(BN=False).to(rank)

    #[NOTE]detect anomaly
    #torch.autograd.set_detect_anomaly(True)

    ddp_Gnet, ddp_Dnet, ddp_FE = list(map(lambda x: DDP(x, device_ids=[rank], output_device=[rank]), [Gnet, Dnet, FE]))

    print(f'model inited for rank{rank}')
    Tensor = torch.cuda.FloatTensor
    valid = Variable(Tensor(np.ones((opt.batch_size, *Dnet.output_shape))),requires_grad=False).to(rank)
    fake = Variable(Tensor(np.zeros((opt.batch_size, *Dnet.output_shape))),requires_grad=False).to(rank) 
    
    l1 = torch.nn.L1Loss()
    l_BCE = torch.nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(ddp_Gnet.parameters(), lr = opt.lr, betas=(.9,.999))
    optimizer_D = torch.optim.Adam(ddp_Dnet.parameters(), lr = opt.lr, betas=(.9,.999))

    #noise
    noise_mean = torch.full((opt.batch_size,*Dnet.input_shape),0,dtype=torch.float32)
    sigma = 1
    anneal = sigma/opt.num_epochs # decreasing of sigma per epoch
    
    for epoch in tqdm(range(opt.num_epochs),desc='Epochs',colour='green'):
        sigma -= anneal
        sigma = max(sigma,0)
        noise_std= torch.full((opt.batch_size,*Dnet.input_shape),sigma,dtype=torch.float32)
        dataloader.sampler.set_epoch(epoch) # shuffle samples
        for i,imgs in enumerate(tqdm(dataloader, desc=f'epoch:{epoch}', colour='white')):
            imgs_hr = imgs['hr'].to(rank)
            imgs_lr = imgs['lr'].to(rank)
            #
            # G training
            #

            optimizer_G.zero_grad()
            #optimizer_FE.zero_grad()
            sr = ddp_Gnet(imgs_lr)
            loss_pixel = l1(sr, imgs_hr)
            # MAKE SOME NOISE!!!!

            instance_noise = torch.normal(mean=noise_mean, std=noise_std).to(rank)
            pred_real = ddp_Dnet(imgs_hr+instance_noise).detach() 
            pred_fake = ddp_Dnet(sr+instance_noise)
            loss_GAN_G = l_BCE(pred_fake - pred_real.mean(0, keepdim=True), valid)
            sr_feature = ddp_FE(sr)
            hr_feature = ddp_FE(imgs_hr).detach()
            
            loss_content = l1(sr_feature, hr_feature)
            
            loss_G = loss_content + 5e-3*loss_GAN_G + 1e-2*loss_pixel
            loss_G.backward() # -[BUG] updating both FE and G?
            optimizer_G.step()
            
            #
            # D training
            #
            optimizer_D.zero_grad()
            pred_real_d = ddp_Dnet(imgs_hr+instance_noise)
            pred_fake_d = ddp_Dnet(sr.detach()+instance_noise) 
            
            loss_real = l_BCE(pred_real_d - pred_fake_d.mean(0,keepdim=True), valid)
            loss_fake = l_BCE(pred_fake_d - pred_real_d.mean(0,keepdim=True), fake)
            
            loss_D = (loss_real + loss_fake) /2
            loss_D.backward() 
            optimizer_D.step()

            # 
            # metricessx
            #
            d_sv_list = []
            fe_sv_list= []
            #breakpoint()
            if rank == 0:
                with torch.no_grad():
                    #for j in range(len(Dnet.model)):
                    #    if Dnet.model[j].__str__()[0] == 'N': # meaning its a Norm layer
                    #        d_sv_list.append(Dnet.model[j].module.weight_sv.item())
                    #for j,m in enumerate(FE.conv_layers.modules()):
                    #    if m.__str__()[0] == 'N': # meaning its a Norm layer
                    #        fe_sv_list.append(m.module.weight_sv.item())
                    psnr = psnr_cal(sr.cpu().squeeze().numpy(),
                                    imgs_hr.cpu().squeeze().numpy())
                    wandb.log({'train_D_loss':loss_D.item(),
                               'train_G_loss':loss_G.item(),
                               'train_psnr':psnr,
                    #           'train_SV_Gnet_conv1':Gnet.conv1.module.weight_sv.item(),
                    #           'train_SV_Gnet_conv2':Gnet.conv2.module.weight_sv.item(),
                    #           'train_SV_Gnet_conv3':Gnet.conv3[0].module.weight_sv.item(),
                    #           'train_SV_Dnet_conv1':d_sv_list[0],
                    #           'train_SV_Dnet_conv2':d_sv_list[1],
                    #           'train_SV_Dnet_conv3':d_sv_list[2],
                    #           'train_SV_FE_conv1':fe_sv_list[0],
                    #           'train_SV_FE_conv2':fe_sv_list[1],
                    #           'train_SV_FE_conv3':fe_sv_list[2]
                               })
                if i%opt.plot_per_iter == 0:
                    img_grid = vutils.make_grid(imgs_hr[:4], 2, padding=2, normalize=True).permute(1,2,0)
                    sr_grid = vutils.make_grid(sr[:4], 2,padding=2, normalize=True).permute(1,2,0) # make color channel as last dim
                    #with torch.no_grad():
                        #y_grad = Grad(sr[0].to(device)) # 1st derivative
                        #y_lap = laplace(sr[0].cuda()) # 2nd derivative type 1
                        #y_grad = norm(y_grad)
                    wandb.log({'img':[wandb.Image(img_grid.detach().cpu().numpy(), caption='GT'),
                                      wandb.Image(sr_grid.detach().cpu().numpy(), caption='SR')
                                      #wandb.Image(y_grad.permute(1,2,0).detach().cpu().numpy(), caption='grad'),
                                      #wandb.Image(y_lap.squeeze().permute(1,2,0).detach().cpu().numpy(), caption='laplace')
                                      ]})
            
        cleanup()
        gc.collect()
        torch.cuda.empty_cache()

# spawn the task
def run_demo(opt, demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,opt),
             nprocs=world_size,
             join=True)

    #print(output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str, default='esrgan_2d_FE')
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--plot_per_iter',type=int, default=1000)
    parser.add_argument('--image_size',type=int, default=64)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--num_epochs',type=int, default=1)
    parser.add_argument('--input_channel',type=int, default=3)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--arch_type',type=str, default='VGG16')
    parser.add_argument('--multi_gpu',type=int, default=1)
    global opt
    opt = parser.parse_args()
    world_size = 4 # n_gpus
    run_demo(opt, demo, world_size)
