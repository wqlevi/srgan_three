# TODO
# -[ ] ViT backbone
# -[ ] YAML file for config experiment
# -[ ] residual connection in Unet
# -[ ] frequency guided Dnet
import os, argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
import glob
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
#from dataset import dataloader as DataLoader
from torch.utils.data import DataLoader, random_split
from models.new_model import Generator, Discriminator, VGG19_54, Discriminator_Unet
from utils.utils import psnr_cal

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
        #img = np.array(img)
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
            channels,
            hr_shape,
            lr:float=.0003,
            batch_size:int=32,
            update_FE:bool=False,
            arch_type:str='VGG16',
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.batch_size = batch_size
        # updating attr needed
        self.fe_sv_list = []
        self.d_sv_list = []
        self.g_sv_list = []

        self.update_FE = update_FE
        
        self.generator = Generator()
        self.discriminator = Discriminator_Unet()

        self.noise_mean = torch.zeros((self.batch_size, *self.discriminator.input_shape))

        self.arch_type = arch_type
        self.FE = VGG19_54(arch_type=self.arch_type, BN=False)
    def forward(self,x):
        return self.generator(x)
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    def l1_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)
    """
    @ property
    def _get_FE_sv(self):
        return self.fe_sv_list
    @property
    def _get_Dnet_sv(self):
        return self.d_sv_list
    @property
    def _get_Gnet_sv(self):
        return self.g_sv_list
    """
    #@_get_FE_sv.setter
    def _get_FE_sv(self):
        fe_sv_list = []
        for j,m in enumerate(self.FE.conv_layers.modules()):
            if m.__str__()[0] == 'N': # meaning its a Norm layer
                fe_sv_list.append(m.module.weight_sv.item())
        return fe_sv_list

    #@_get_Dnet_sv.setter
    """
    def _get_Dnet_sv(self):
        d_sv_list = []
        for i,m in enumerate(self.discriminator.model):
            if m.__str__()[0] == 'N': # meaning its a Norm layer
                d_sv_list.append(m.module.weight_sv.item())
        return d_sv_list
    """

    """
    def _get_Gnet_sv(self):
        g_sv_list = []
        g_sv_list.append([self.generator.conv1.module.weight_sv.item(),
            self.generator.conv2.module.weight_sv.item(),
            self.generator.conv3.module.weight_sv.item()])
        return g_sv_list
    """

    def training_step(self, batch, batch_idx):
        imgs_hr = batch['hr']
        imgs_lr = batch['lr']
        optimizer_g, optimizer_d = self.optimizers()
        self.step_sigma = 1/self.trainer.max_epochs

        valid = torch.ones((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)
        fake = torch.zeros((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)

        sigma_numerics = 1 - self.current_epoch * self.step_sigma
        sigma_numerics = max(sigma_numerics, 0)
        sigma = torch.full((self.batch_size, *self.discriminator.input_shape), sigma_numerics)

        instancenoise = torch.normal(mean = self.noise_mean, std=sigma).type_as(imgs_lr)
        valid = valid.type_as(imgs_lr)
        fake = fake.type_as(imgs_lr)

        #---Update G---#
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(imgs_lr)

        pred_real = self.discriminator(imgs_hr + instancenoise).detach()
        pred_fake = self.discriminator(self.generated_imgs + instancenoise)
        g_adv_loss = self.adversarial_loss( pred_fake - pred_real.mean(0,keepdim=True), valid)
        g_pixel_loss = self.l1_loss(self.FE(self.generated_imgs),self.FE(imgs_hr))
        g_content_loss = self.l1_loss(self.generated_imgs, imgs_hr)
        g_loss = g_content_loss + 5e-3*g_adv_loss + 1e-2*g_pixel_loss
        self.log("g_loss", g_loss, prog_bar = True)
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
        #d_sv_list = self._get_Dnet_sv()
        fe_sv_list = self._get_FE_sv()
        #g_sv_list = self._get_Gnet_sv()

        psnr = psnr_cal(self.generated_imgs.detach().cpu().squeeze().numpy(), imgs_hr.detach().cpu().squeeze().numpy())
        #-----Logging-----#
        self.log("PNSR", psnr)
        self.log("train_SV_FE_conv1", fe_sv_list[0])
        self.log("train_SV_FE_conv2", fe_sv_list[1])
        self.log("train_SV_FE_conv3", fe_sv_list[2])
        #self.log("train_SV_Gnet_conv1", g_sv_list[0])
        #self.log("train_SV_Gnet_conv2", g_sv_list[1])
        #self.log("train_SV_Gnet_conv3", g_sv_list[2])
        #self.log("train_SV_Dnet_conv1", d_sv_list[0])
        #self.log("train_SV_Dnet_conv2", d_sv_list[1])
        #self.log("train_SV_Dnet_conv3", d_sv_list[2])
        if batch_idx % 100 == 0:
            self.logger.log_image("Results", [grid_sr, grid_hr], caption=["SR", "GT"])


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
    parser.add_argument('--arch_type',type=str, default='VGG16')
    parser.add_argument('--n_gpus',type=int, default=1)
    parser.add_argument('--update_FE',action='store_true', help='update only when such flag is added to execute the script') # 
    parser.add_argument('--name_ckp',type=str, default="no_name")

    opt = parser.parse_args()

    wandb_logger = WandbLogger(project = opt.model_name,
            log_model = True)
    dm = DataModule(hr_shape = opt.image_size,
            data_dir= opt.data_path,
            batch_size = opt.batch_size,
            num_workers=32)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath = os.getcwd()+ "/" + opt.model_name + "/" + opt.name_ckp + "/checkpoints/",
            save_last = True,
            save_top_k = -1
            )
    trainer = L.Trainer(
            accelerator = "auto",
            devices = opt.n_gpus,
            max_epochs = opt.num_epochs,
            strategy='ddp_find_unused_parameters_true',
            logger = wandb_logger,
            callbacks = [checkpoint_callback],
            ) # strategy flag when one model has not updating parameters
    model = GAN(channels=opt.input_channel,
            hr_shape=opt.image_size,
            batch_size = opt.batch_size,
            update_FE = opt.update_FE,
            arch_type = opt.arch_type
            )
    trainer.fit(model,
            dm, 
            #ckpt_path = 'last'
            )
