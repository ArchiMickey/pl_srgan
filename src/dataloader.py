import pytorch_lightning as pl
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

from loguru import logger
from icecream import ic


# Helper function for image preprocessing
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class SRGANDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.crop_size = cfg['dm']['crop_size']
        self.upscale_factor = cfg['dm']['upscale_factor']
        self.datapath = cfg['datapath']
        self.train_datapath = self.datapath['train']
        self.val_datapath = self.datapath['val']
        ic(self.crop_size, self.upscale_factor, self.datapath, self.train_datapath, self.val_datapath)
        
        self.train_hr_transform = train_hr_transform(self.crop_size)
        self.train_lr_transform = train_lr_transform(self.crop_size, self.upscale_factor)        
    
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.srgan_train = []
            self.srgan_val = []
            
            for path in self.train_datapath:
                self.srgan_train += [join(path, x) for x in listdir(path) if is_image_file(x)]
            
            for path in self.val_datapath:
                self.srgan_val += [join(path, x) for x in listdir(path) if is_image_file(x)]

if __name__ == '__main__':
    from loguru import logger
    from icecream import ic
    dm = SRGANDataModule()