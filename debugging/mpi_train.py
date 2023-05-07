import torch
import os
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.models import vgg19
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()
def demo(rank, world_size):
    #print(f'Running basic DDP on rank {rank}')
    setup(rank, world_size)
    print(dist.get_backend())
    if rank == 0:
        wandb.init(project='test',entity='wqlevi')
    m = vgg19(weights='DEFAULT').to(rank)
    ddp_model = DDP(m, device_ids=[rank])
    output = ddp_model(torch.randn(20,3,64,64))

    print(output)
    cleanup()
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args = (world_size,),
             nprocs=world_size,
             join = True)
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus-1
    run_demo(demo, world_size)
