import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from loguru import logger
from icecream import ic

from src.dataloader import SRGANDataModule
from src.srgan import SRGAN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(version_base=None, config_path="src/config", config_name="debug")
def main(cfg: DictConfig) -> None:
    ic(dict(cfg))
    dm = SRGANDataModule(cfg)
    model = SRGAN(cfg)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()