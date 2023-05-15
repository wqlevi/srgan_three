import torch
from lightning.pytorch.utilities import model_summary as MS
import lightning as L
from importlib import import_module
import sys
sys.path.append('../../NTIRE-2021-Dehazing-DWGAN')
sys.path.append('../')
from model import *
from models.new_model import Generator, Discriminator, VGG19_54, Discriminator_Unet, Discriminator_SN_SC

class P_G(dwt_UNet, L.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(16,3,64,64)

G = P_G()
print(MS.ModelSummary(G, max_depth = -1))
