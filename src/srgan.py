from math import log10
import pytorch_lightning as pl
import torch
from torch import optim
from torch.autograd import Variable

from icecream import ic
import wandb

from .loss import GeneratorLoss
from .model import Generator, Discriminator
from .ssim import ssim


class SRGAN(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.batch_size = cfg['batch_size']
        self.generator = Generator(cfg['upscale_factor'])
        self.discriminator = Discriminator()
        
        self.generator_criterion = GeneratorLoss()
        
        self.automatic_optimization = False
    
    def training_step(self, batch):
        gen_opt, dis_opt = self.optimizers()
        real_img = Variable(batch['hr_img'])
        z = Variable(batch['lr_img'])
        loss_dict = {}
        
        gen_output = self.generator(z)
        wandb.log({'gen_output': wandb.Image(gen_output[0])})
        
        # Update Discriminator network
        real_out = self.discriminator(real_img).mean()
        fake_out = self.discriminator(gen_output.detach()).mean()
        d_loss = (1 - real_out) ** 2 + fake_out ** 2
        
        dis_opt.zero_grad()
        self.manual_backward(d_loss)
        dis_opt.step()
        
        # Update G network
        fake_out = self.discriminator(gen_output).mean()
        g_loss = self.generator_criterion(fake_out, gen_output, real_img)
        

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()
        
        loss_dict |= {'train/d_loss': d_loss, 'train/g_loss': g_loss}
        
        self.log_dict(loss_dict, prog_bar=True)
        
        return loss_dict

    def validation_step(self, batch, *args, **kwargs):
        lr = batch['lr_img']
        hr = batch['hr_img']
        
        sr = self.generator(lr)
        mse_loss = ((sr - hr) ** 2).data.mean()
        ssim_loss = ssim(sr, hr).item()
        psnr_loss = 10 * log10((hr.max()**2) / mse_loss)
        
        loss_dict = {
            'val/mse_loss': mse_loss,
            'val/ssim_loss': ssim_loss,
            'val/psnr_loss': psnr_loss
        }
        
        self.log_dict(loss_dict)
        
        return loss_dict

    def configure_optimizers(self):
        gen_opt = optim.Adam(self.generator.parameters())
        dis_opt = optim.Adam(self.discriminator.parameters())
        
        return gen_opt, dis_opt
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
