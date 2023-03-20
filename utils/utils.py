import torch
import torch.nn.functional as F
import numpy as np
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
