from math import log10
import pytorch_lightning as pl
from torch import optim
from torch.autograd import Variable

from icecream import ic

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
        
        loss_dict |= {'d_loss': d_loss, 'g_loss': g_loss}
        
        self.log_dict(loss_dict, prog_bar=True)
        
        return loss_dict

    def validation_step(self, batch, *args, **kwargs):
        lr = batch['lr_img']
        hr = batch['hr_img']
        
        sr = self.generator(lr)
        mse_loss = ((sr - hr) ** 2).data.mean()
        ssim_loss = ssim(sr, hr).item()
        psnr_loss = 10 * log10((hr.max()**2) / mse_loss)
        
        return {
            'mse_loss': mse_loss,
            'ssim_loss': ssim_loss,
            'psnr_loss': psnr_loss
        }

    def configure_optimizers(self):
        gen_opt = optim.Adam(self.generator.parameters())
        dis_opt = optim.Adam(self.discriminator.parameters())
        
        return gen_opt, dis_opt
