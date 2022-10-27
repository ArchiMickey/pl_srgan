from math import log10
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from torch.autograd import Variable

from icecream import ic
import wandb

from .dataloader import display_transform
from .loss import GeneratorLoss
from .model import Generator, Discriminator
from .ssim import ssim


class SRGAN(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.batch_size = cfg['batch_size']
        self.lr = cfg['lr']
        self.generator = Generator(cfg['upscale_factor'])
        self.discriminator = Discriminator()
        
        self.generator_criterion = GeneratorLoss()
            
    def training_step(self, batch, batch_idx, optimizer_idx):    
        real_img = Variable(batch['hr_img'])
        z = Variable(batch['lr_img'])
        
        gen_output = self.generator(z)
        if self.global_step % self.trainer.log_every_n_steps == 0:
            lr_img = batch['lr_img'][0].cpu().detach()
            hr_img = real_img[0].cpu().detach()
            sr_img = gen_output[0].cpu().detach()
            train_img = torch.stack([
                display_transform()(lr_img),
                display_transform()(hr_img),
                display_transform()(sr_img),
            ])
            wandb.log({'train/img': wandb.Image(train_img)})
            
        if optimizer_idx == 0:
            # Update G network
            fake_out = self.discriminator(gen_output).mean()
            g_loss = self.generator_criterion(fake_out, gen_output, real_img)
            
            self.log('train/g_loss', g_loss)
            
            return g_loss
        
        if optimizer_idx == 1:
            # Update Discriminator network
            real_out = self.discriminator(real_img).mean()
            fake_out = self.discriminator(gen_output.detach()).mean()
            d_loss = (1 - real_out) ** 2 + fake_out ** 2
            
            self.log('train/d_loss', d_loss)
            
            return d_loss
        
        

    def validation_step(self, batch, batch_idx):
        lr = batch['lr_img']
        hr = batch['hr_img']
        mses = []
        ssims = []
        psnrs = []
        for i in range(len(lr)):
            sr = self.generator(lr[i].unsqueeze(0))
            hr_img = hr[i].unsqueeze(0)
            mse_loss = ((sr - hr_img) ** 2).data.mean()
            ssim_loss = ssim(sr, hr_img).item()
            psnr_loss = 10 * log10((hr_img.max()**2) / mse_loss)
            
            mses.append(mse_loss.cpu())
            ssims.append(ssim_loss)
            psnrs.append(psnr_loss)
        
        if batch_idx == 0:
            lr_img = lr[len(lr) - 1].cpu().detach()
            hr_img = hr[len(hr) - 1].cpu().detach()
            sr_img = sr[0].cpu().detach()
            val_img = torch.stack([
                display_transform()(lr_img),
                display_transform()(hr_img),
                display_transform()(sr_img),
            ])
            wandb.log({'val/img': wandb.Image(val_img)})
        
        loss_dict = {
            'val/mse_loss': np.mean(mses),
            'val/ssim_loss': np.mean(ssims),
            'val/psnr_loss': np.mean(psnrs),
        }
        
        self.log_dict(loss_dict)
        
        return loss_dict

    def configure_optimizers(self):
        gen_opt = optim.Adam(self.generator.parameters(), lr=self.lr)
        dis_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        return gen_opt, dis_opt
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
