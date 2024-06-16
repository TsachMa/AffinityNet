from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

import sys
sys.path.append('../../')

from gnn import GCN

class AffinityNet(pl.LightningModule):
    def __init__(self, in_channels: int) -> None:
        self.model = GCN(in_channels=in_channels)
        self.criteria = nn.MSELoss()

    def forward(self, batch: Any) -> Any:
        output = self.model(batch)
        loss = self.criteria(output, batch['affinity'])
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    


