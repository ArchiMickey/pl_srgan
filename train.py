import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from loguru import logger
from icecream import ic

from src.dataloader import SRGANDataModule
from src.srgan import SRGAN

@hydra.main(version_base=None, config_path="src/config", config_name="debug")
def main(cfg: DictConfig) -> None:
    ic(cfg)
    dm = SRGANDataModule(cfg)
    model = SRGAN(cfg)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()