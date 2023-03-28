import torch
from models.new_model import Generator

if __name__ == '__main__':
    def print_hook(m,i,o):
        print("inside"+m.__class__.__name__)

    G = Generator().to("cuda:0")
    G = torch.nn.DataParallel(G, device_ids=[range(torch.cuda.device_count())])
    G.module.conv1.register_forward_hook(print_hook)
    x_sample = torch.ones([16,3,32,32], device=torch.device("cuda:0"))
    y_result = G(x_sample)
    print(f"resulting tensor in shape of: {y_result.shape}")
