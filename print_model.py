import torch
from lightning.pytorch.utilities import model_summary as MS
import lightning as L
from models.new_model import Generator, Discriminator, VGG19_54, Discriminator_Unet

class P_G(Discriminator_Unet, L.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(16,3,64,64)

G = P_G()
print(MS.ModelSummary(G, max_depth = -1))
