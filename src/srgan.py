import pytorch_lightning as pl


class SRGAN(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        