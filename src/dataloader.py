from collections import defaultdict
import pytorch_lightning as pl
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

from loguru import logger
from icecream import ic
from tqdm import tqdm


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

def verify_dataset(dataset: list, crop_size):
    for datapath in dataset:
        img = Image.open(datapath)
        w, h = img.size
        if w < crop_size or h < crop_size:
            dataset.remove(datapath)
            logger.info(f'Skipping {datapath} due to small size')

class SRGANDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.batch_size = cfg['batch_size']
        self.crop_size = cfg['dm']['crop_size']
        self.upscale_factor = cfg['upscale_factor']
        self.datapath = cfg['datapath']
        self.train_datapath = self.datapath['train']
        self.val_datapath = self.datapath['val']
        self.num_workers = cfg['dm']['num_workers']
    
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.srgan_train = []
            self.srgan_val = []
            
            for path in self.train_datapath:
                logger.info(f'Loading train data from {path}:')
                for x in tqdm(listdir(path)):
                    if is_image_file(x):
                        self.srgan_train += [join(path, x)]
            
            for path in self.val_datapath:
                logger.info(f'Loading val data from {path}:')
                for x in tqdm(listdir(path)):
                    if is_image_file(x):
                        self.srgan_val += [join(path, x)]
            
            verify_dataset(self.srgan_train, self.crop_size)
            verify_dataset(self.srgan_val, self.crop_size)
            
    
    def train_dataloader(self):
        ds = TrainDataset(self.srgan_train, self.crop_size, self.upscale_factor)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        ds = ValDataset(self.srgan_val, self.upscale_factor)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=val_collate)

class TrainDataset(Dataset):
    def __init__(self, image_filenames, crop_size, upscale_factor) -> None:
        super(TrainDataset, self).__init__()
        self.image_filenames = image_filenames
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        
        self.train_hr_transform = train_hr_transform(self.crop_size)
        self.train_lr_transform = train_lr_transform(self.crop_size, self.upscale_factor)
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        try:
            hr_image = self.train_hr_transform(Image.open(filename))
        except ValueError:
            ic(filename)
            raise ValueError
        lr_image = self.train_lr_transform(hr_image)
        return {'filename': filename,'hr_img': hr_image, 'lr_img': lr_image}
    
    def __len__(self):
        return len(self.image_filenames)

class ValDataset(Dataset):
    def __init__(self, image_filenames, upscale_factor) -> None:
        super(ValDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = image_filenames
        
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        hr_image = Image.open(filename)
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return {
            'filename': filename,
            'lr_img': ToTensor()(lr_image),
            'hr_img': ToTensor()(hr_restore_img),
            'original_img': ToTensor()(hr_image)
        }
    
    def __len__(self):
        return len(self.image_filenames)

def val_collate(batch):
    ret = defaultdict(list)
    for data in batch:
        for k, v in data.items():
            ret[k].append(v)
    return ret

if __name__ == '__main__':
    from loguru import logger
    from icecream import ic
    dm = SRGANDataModule()