# TODO
# -[ ] ViT backbone
# -[ ] YAML file for config experiment
# -[x] residual connection in Unet, improving PSNR

# -[x] frequency guided Dnet(dwt Unet)
# -[ ] sobel filter for FE to extract gradient constrain  
import os, argparse
import wandb
import glob

from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
#from dataset import dataloader as DataLoader
from torch.utils.data import DataLoader, random_split
from importlib import import_module
from models.new_model import Generator, Discriminator, VGG19_54, Discriminator_Unet, Discriminator_SN_SC, FeatureExtractor
from utils import utils

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            hr_shape: int,
            data_dir:str,
            batch_size :int,
            num_workers :int=2
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.shape = hr_shape
        self.lr_shape = int(hr_shape/2)
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_shape),
            transforms.CenterCrop(self.lr_shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.CenterCrop(self.shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.files = sorted(glob.glob(self.data_dir+'/*jpg'))
    def __getitem__(self,index):
        img = Image.open(self.files[index % len(self.files)])
        im_lr = self.lr_transform(img)
        im_hr = self.hr_transform(img)
        return {"lr":im_lr, "hr":im_hr}
    def __len__(self):
        return len(self.files)
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            trainset = DataModule(self.shape, self.data_dir, self.batch_size, self.num_workers)
            self.trainset,self.validset = random_split(trainset, [len(trainset)//4, len(trainset)-len(trainset)//4])
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = True
                )
    def val_dataloader(self):
        return DataLoader(
                self.validset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                drop_last = True
                )
class GAN(L.LightningModule):
    def __init__(
            self,
            config,
            channels:int=3,
            #hr_shape:tuple,
            lr:float=.0003,
            batch_size:int=32,
            update_FE:bool=False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.standarize = lambda x: (x-x.min())/(x.max()-x.min())
        # updating attr needed
        self.fe_sv_list = []
        self.d_sv_list = []
        self.g_sv_list = []
        
        self.is_FE = config.is_FE
        self.update_FE = update_FE
        self.log_images_interval = 1000
        self.psnr_cal = peak_signal_noise_ratio
        self.ssim_cal = structural_similarity_index_measure
        self.lpips_cal = LearnedPerceptualImagePatchSimilarity(net='vgg', reduction='mean')

        #self.generator = Generator()
        mod = import_module("models.new_model")
        self.discriminator = getattr(mod, config.D_type)()
        self.generator = getattr(mod, config.G_type)()

        self.noise_anneal_epochs = 20
        self.noise_mean = torch.zeros((self.batch_size, *self.discriminator.input_shape))

        if self.is_FE:
            #self.FE = VGG19_54(arch_type=config.arch_type, BN=False)
            self.FE= getattr(mod, 'FeatureExtractor')(arch_type = config.arch_type, pretrained=False)
            if not self.update_FE:
                utils.freeze_model(self.FE)

    def forward(self,x):
        return self.generator(x)
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    def l1_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)
    #def on_save_checkpoint(self, checkpoint):
    #    checkpoint['sigma'] = self.sigma_numerics
    #def on_load_checkpoint(self, checkpoint):
    #    self.sigma_numerics = checkpoint['sigma']
    def training_step(self, batch, batch_idx):
        imgs_hr = batch['hr']
        imgs_lr = batch['lr']
        optimizer_g, optimizer_d = self.optimizers()
        self.step_sigma = 1/self.noise_anneal_epochs

        valid = torch.ones((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)
        fake = torch.zeros((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)

        sigma_numerics = 1 - self.current_epoch * self.step_sigma
        self.sigma_numerics = max(sigma_numerics, 0)
        sigma = torch.full((self.batch_size, *self.discriminator.input_shape), self.sigma_numerics)

        instancenoise = torch.normal(mean = self.noise_mean, std=sigma).type_as(imgs_lr)
        valid = valid.type_as(imgs_lr)
        fake = fake.type_as(imgs_lr)

        #---Update G---#
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(imgs_lr)

        pred_real = self.discriminator(imgs_hr + instancenoise).detach()
        pred_fake = self.discriminator(self.generated_imgs + instancenoise)
        g_adv_loss = self.adversarial_loss( pred_fake - pred_real.mean(0,keepdim=True), valid)
        g_content_loss = self.l1_loss(self.generated_imgs, imgs_hr)
        if self.is_FE:
            g_pixel_loss = self.l1_loss(self.FE(self.generated_imgs),self.FE(imgs_hr))
        else:
            g_pixel_loss = 0 # not using FE at all
        g_loss = g_content_loss + 5e-3*g_adv_loss + 1e-2*g_pixel_loss
        self.log("g_loss", g_loss, prog_bar = True)
        self.log("g_content_loss", g_content_loss.item())
        self.log("g_pixel_loss", g_pixel_loss.item())
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        #---Update D---#
        self.toggle_optimizer(optimizer_d)
        pred_real = self.discriminator(imgs_hr + instancenoise)
        pred_fake = self.discriminator(self.generated_imgs.detach() + instancenoise)
        loss_real = self.adversarial_loss(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.adversarial_loss(pred_fake - pred_real.mean(0, keepdim=True), fake)

        d_loss = (loss_real + loss_fake)/2
        self.log("d_loss", d_loss, prog_bar = True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        grid_sr = make_grid(self.generated_imgs)
        grid_hr = make_grid(imgs_hr)

        fe_sv_list, g_sv_list = [],[]
        vs_d_dict = utils.sv_2_dict(self.discriminator)
        #vs_g_dict = utils.sv_2_dict(self.generator)
        #vs_fe_dict = utils.sv_2_dict(self.FE)
        #d_sv_list = self._get_Dnet_sv()
        #fe_sv_list = self._get_FE_sv()
        #g_sv_list = self._get_Gnet_sv()

        with torch.no_grad():
            psnr = self.psnr_cal(self.generated_imgs, imgs_hr)
            ssim = self.ssim_cal(self.generated_imgs, imgs_hr)
            lpips = self.lpips_cal(self.standarize(self.generated_imgs),self.standarize(imgs_hr))
#        print(f"PSNR:{psnr}\tSSIM:{ssim}\tLPIPS:{lpips}")
        #-----Logging-----#
        self.log("PSNR", psnr)
        self.log("SSIM", ssim)
        self.log("LPIPS", lpips)
        self.log("noise",self.sigma_numerics)
        #wandb.log(vs_d_dict) # not changing much and not starting from same value for all layers
        #wandb.log(vs_fe_dict) # not changing much and not starting from same value for all layers
        [self.log(k,v) for k,v in vs_d_dict.items()] # not changing much and not starting from same value for all layers
        #[self.log(k+"_Gnet",v) for k,v in vs_g_dict.items()] # not changing much and not starting from same value for all layers
        #[self.log(k,v) for k,v in vs_fe_dict.items()] # not changing much and not starting from same value for all layers
        if batch_idx % self.log_images_interval == 0:
            self.logger.log_image("Results", [grid_sr, grid_hr], caption=["SR", "GT"])


    #def validation_step(self):
    #    """TODO, and add check_val_every_n_epoch=1(default) to Trainer"""
    #    pass
    def configure_optimizers(self):
        if self.update_FE:
            opt_g = torch.optim.Adam(list(self.FE.parameters())+list(self.generator.parameters()), lr=self.hparams.lr)
        else:
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str, default='esrgan_2d_FE')
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--plot_per_iter',type=int, default=1000)
    parser.add_argument('--image_size',type=int, default=64)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--num_epochs',type=int, default=10)
    parser.add_argument('--input_channel',type=int, default=3)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--n_gpus',type=int, default=1)
    parser.add_argument('--n_nodes',type=int, default=1)
    parser.add_argument('--use_yaml_config',action='store_true', help = 'use YAML file for parsing configurations')
    parser.add_argument('--update_FE',action='store_true', help='update only when such flag is added to execute the script') # 
    parser.add_argument('--is_FE',action='store_true', help='whether FE exist') # 
    parser.add_argument('--is_ckp',action='store_true', help='whether use ckp') # 
    parser.add_argument('--name_ckp',type=str, default="no_name")
    parser.add_argument('--arch_type',type=str, default='VGG16')
    parser.add_argument('--D_type',type=str, default="Discriminator")
    parser.add_argument('--G_type',type=str, default="Generator")

    opt = parser.parse_args()
    if opt.use_yaml_config:
        dict_yaml = utils.load_yaml(f'config/{opt.name_ckp}')
        update_opt_dict = vars(opt)
        update_opt_dict.update(dict_yaml)
        opt = argparse.Namespace(**update_opt_dict)

    wandb_logger = WandbLogger(project = opt.model_name,
            log_model = False,
            group = opt.name_ckp)

    dm = DataModule(hr_shape = opt.image_size,
            data_dir= opt.data_path,
            batch_size = opt.batch_size,
            num_workers=32)

    ckp_path = os.getcwd()+ "/" + opt.model_name + "/" + opt.name_ckp + "/checkpoints/"
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath = ckp_path,
            save_last = True,
            save_top_k = -1
            )
    trainer = L.Trainer(
            accelerator = "auto",
            devices = opt.n_gpus,
            num_nodes = opt.n_nodes,
            max_epochs = opt.num_epochs,
            strategy='ddp_find_unused_parameters_true',
            logger = wandb_logger,
            callbacks = [checkpoint_callback],
            #limit_train_batches = 0.1,
            #fast_dev_run=True
            ) # strategy flag when one model has not updating parameters


    model = GAN(
            opt,
            channels=opt.input_channel,
            hr_shape=opt.image_size,
            batch_size = opt.batch_size,
            update_FE = opt.update_FE,
            arch_type = opt.arch_type
            )
    CKP_FLAG = False
    if trainer.global_rank == 0:
        if checkpoint_callback.file_exists(ckp_path+'last.ckpt', trainer) and opt.is_ckp:
            print('\033[93m WARNING: Checkpoint loading... \033[0m')
            CKP_FLAG = True
        utils.write_config(opt)
    if CKP_FLAG:
        trainer.fit(model,
            dm, 
            ckpt_path = 'last'
            )
    else:
        trainer.fit(model,
            dm)