import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height//4, hr_width//4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))            
])
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
])
    self.file = sorted(glob.glob(root+'/*.*'))
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.file)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return {'lr':img_lr, 'hr':img_hr}
    def __len__(self):
        return len(self.file)
