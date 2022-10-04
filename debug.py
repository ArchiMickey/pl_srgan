import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from loguru import logger
from icecream import ic

from src.dataloader import SRGANDataModule

@hydra.main(version_base=None, config_path="src/config", config_name="debug")
def main(cfg: DictConfig) -> None:
    ic(cfg)
    SRGANDataModule(cfg)

if __name__ == '__main__':
    main()