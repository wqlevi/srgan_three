import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from skimage.metrics import structural_similarity, normalized_root_mse
import lpips


norm = lambda x: ((x-x.min())/(x.max() - x.min()))*2-1
def proj(x,y):
    # project vector x to vector y
    return torch.mm(y,x.t()) * y / torch.mm(y,y.t())
def gram_schmidt(x, ys):
    # get the othorgonal of vector x to its y-projection
    for y in ys:
        x = x-proj(x,y)
    return x

def power_iteration(W,u_,update=True,eps=1e-12):
# SVS output the largest SV of sqrt(weight*weight.T)
    us, vs, svs = [], [] ,[]
    for i,u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u,W) # Weight * randn in N(0,1)
            v = F.normalize(gram_schmidt(v, vs), eps=eps) # cal l_2 norm of weight projected onto previous weights
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u,us),eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us ,vs
            
def psnr_cal(img,gt):
    mse = np.mean((img-gt)**2)
    if mse == 0:
        return float('inf')
    data_range = gt.max()-gt.min()
    return 20*np.log10(data_range) - 10*np.log10(mse)

def print_norm_hook(module, inp, outp):
    print("Inside "+module.__class__.__name__+" forward")
    print("Norm of 0-th kernele 0-th"+str(module.weight.data.norm()))

"""
def ssim_cal(img,gt):
    if not (isinstance(image,np.ndarray) and isinstance(gt,np.ndarray)):
        raise ValueError("both inputs should be in numpy,ndarray type")
    if not image.ndim == gt.ndim:
        raise ValueError("dimensiom of the inputs should be the same")

    data_range = np.max(gt) - np.min(gt)
    if image.ndim==4: # C,H,W,L
        return structural_similarity(image.transpose(1,2,3,0), gt.transpose(1,2,3,0), data_range = data_range,channel_axis=-1)
    elif image.ndim==3: # H,W,L Batch_size = 1
        return structural_similarity(image, gt, data_range = data_range, channel_axis=-1)
"""

def lpips_cal(img, gt):
    """data are both normalized to range [-1,1]"""
    lpips_fn = lpips.LPIPS(net='vgg')
    return lpips_fn(norm(img), norm(gt))

def load_pretrained(model,pretrain_path,replace_key:str='module.',key_dict:str='state_dict'):
    '''
    Parameters
    ----------
        model: model of network
        pretrain_path : str | path to store checkpoint
        key_dict : str | key string of the model(e.g. FE_state_dict, etc.)
    '''
    pretrains = torch.load(pretrain_path)
    net_dict = model.state_dict()
    if not len(pretrains.keys()) > 10:
        print("not c11")
        pretrain_dict = {k.replace(replace_key,""): v for k, v in pretrains['%s'%key_dict].items() if k.replace(replace_key,"") in net_dict.keys()}
    else:
        pretrain_dict = pretrains
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    return model


def sv_2_dict(model)->dict:
    sv_dict = {}
    counter = 0
    for k,v in model.named_modules():
        if isinstance(v, nn.Conv2d):
            sv_dict['Conv_sv_%i'%counter] = v.weight_sv.data.item()
            counter += 1
    return sv_dict
def freeze_model(model):
    for params in model.parameters():
        params.requires_grad =False
