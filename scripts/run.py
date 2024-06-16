import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
import numpy as np
import h5py
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import random
from scipy import stats
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer

import sys
sys.path.append("../")
from src.model.gnn import GCN
from model.model import AffinityNet
from src.data.datasets import GCN_Dataset

def main():
    
    training_data, validation_data, checkpoint_path = 'data/training_data.h5', 'data/validation_data.h5', 'checkpoints/model.pth'
    
    # define parameters
    epochs = 300                   # number of training epochs
    batch_size = 7                 # batch size to use for training
    learning_rate = 0.001          # learning rate to use for training
    gather_width = 128             # gather width
    prop_iter = 4                  # number of propagation interations
    dist_cutoff = 3.5              # common cutoff for donor-to-acceptor distance for energetically significant H bonds in proteins is 3.5 Å
    feature_size = 20              # number of features: 19 + Van der Waals radius

    # construct model
    model = AffinityNet(3)

    training_dataset = GCN_Dataset(data_file=training_data)
    validation_dataset = GCN_Dataset(data_file=validation_data)
        
    training_dataloader = DataLoader(training_dataset, batch_size=256, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=256, shuffle=False, drop_last=True)
    
    #fit model
    trainer = Trainer(max_epochs=300, gpus=1)
    trainer.fit(model, training_dataloader, validation_dataloader)

    # save model
    torch.save(model.state_dict(), checkpoint_path)
    
if __name__ == '__main__':
    main()






