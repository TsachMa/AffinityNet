import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import torch.nn as nn

#TODO: Parse preprocessed data; Training

class EdgeEmbeddingMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super(EdgeEmbeddingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
""" Define a helper module for reshaping tensors """
class View(nn.Module):
	
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
	
    def forward(self, x):
        return x.view(*self.shape)
    

class CNN(nn.Module):
    def __conv_filter__(self, in_channels, out_channels, kernel_size, stride, padding):
        conv_filter = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True), 
                                    nn.ReLU(inplace=True), 
                                    nn.BatchNorm3d(out_channels))
        return conv_filter

    def __se_block__(self, channels):
        se_block = nn.Sequential(nn.AdaptiveAvgPool3d(1), View((-1,channels)), 
                                 nn.Linear(channels, channels//16, bias=False), 
                                 nn.ReLU(), 
                                 nn.Linear(channels//16, channels, bias=False), 
                                 nn.Sigmoid(), View((-1,channels,1,1,1)))
        return se_block
    
    def __init__(self, node_feat_dim, edge_feat_dim, edge_embedding_dim, mlp_hidden_dim):
        super(CNN, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.enhanced_dim = node_feat_dim+edge_embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        # Define the MLPs
        self.mlp1 = EdgeEmbeddingMLP(edge_feat_dim, edge_embedding_dim, mlp_hidden_dim)
        self.mlp2 = EdgeEmbeddingMLP(edge_feat_dim * 2, edge_embedding_dim, mlp_hidden_dim)
        self.mlp3 = EdgeEmbeddingMLP(edge_feat_dim * 3, edge_embedding_dim, mlp_hidden_dim)
        self.mlp4 = EdgeEmbeddingMLP(edge_feat_dim * 4, edge_embedding_dim, mlp_hidden_dim)

        self.conv_block1 = self.__conv_filter__(self.enhanced_dim,64,9,2,3)
        self.se_block1=self.__se_block__(64)
        self.res_block1 = self.__conv_filter__(64, 64, 7, 1, 3)
        self.res_block2 = self.__conv_filter__(64, 64, 7, 1, 3)
        self.conv_block2=self.__conv_filter__(64, 128, 7, 3, 3)
        self.se_block2=self.__se_block__(128)
        self.max_pool = nn.MaxPool3d(2)
        self.conv_block3=self.__conv_filter__(128, 256, 5, 2, 2)
        self.se_block3=self.__se_block__(256)
        self.linear1 = nn.Linear(2048, 100)
        torch.nn.init.normal_(self.linear1.weight, 0, 1)
        self.relu=nn.ReLU()
        self.linear1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
        self.linear2 = nn.Linear(100, 1)
        torch.nn.init.normal_(self.linear2.weight, 0, 1)

    def forward(self, x, node_features, adj_list):

        # Update node features with edge embeddings
        node_features = self.update_node_features(node_features, adj_list)
        
        # SE block 1
        conv1 = self.conv_block1(x)
        squeeze1 = self.se_block1(conv1)
        se1 = conv1 * squeeze1.expand_as(conv1) 

        # Residual blocks
        conv1_res1 = self.res_block1(se1)
        conv1_res12 = conv1_res1 + se1
        conv1_res2 = self.res_block2(conv1_res12)
        conv1_res2_2 = conv1_res2 + se1

        # SE block 2
        conv2 = self.conv_block2(conv1_res2_2)
        squeeze2 = self.se_block2(conv2)
        se2 = conv2 * squeeze2.expand_as(conv2) 

        # Pooling layer
        pool2 = self.max_pool(se2)

        # SE block 3
        conv3 = self.conv_block3(pool2)
        squeeze3 = self.se_block3(conv3)
        se3 = conv3 * squeeze3.expand_as(conv3) 

        # Flatten
        flatten = se3.view(se3.size(0), -1)

        # Linear layer 1
        linear1_z = self.linear1(flatten)
        linear1_y = self.relu(linear1_z)
        linear1 = self.linear1_bn(linear1_y) if linear1_y.shape[0] > 1 else linear1_y

        # Linear layer 2
        linear2_z = self.linear2(linear1)

        return linear2_z, flatten
    
    def update_node_features(self, node_features, adj_list):
        updated_node_features = []
        for i in range(node_features.shape[0]):
            node_feature = node_features[i]

            # Gather all edges connected to this node
            connected_edges = []
            for edge in adj_list:
                if edge[0] == i or edge[1] == i:
                    connected_edges.append(edge[2])

            # Concatenate edge features and use the appropriate MLP
            if len(connected_edges) == 1:
                concatenated_edges = torch.tensor(connected_edges[0]).float()
                edge_embedding = self.mlp1(concatenated_edges.unsqueeze(0)).squeeze(0)
            elif len(connected_edges) == 2:
                concatenated_edges = torch.cat([torch.tensor(edge).float() for edge in connected_edges])
                edge_embedding = self.mlp2(concatenated_edges.unsqueeze(0)).squeeze(0)
            elif len(connected_edges) == 3:
                concatenated_edges = torch.cat([torch.tensor(edge).float() for edge in connected_edges])
                edge_embedding = self.mlp3(concatenated_edges.unsqueeze(0)).squeeze(0)
            elif len(connected_edges) >= 4:
                concatenated_edges = torch.cat([torch.tensor(edge).float() for edge in connected_edges[:4]])
                edge_embedding = self.mlp4(concatenated_edges.unsqueeze(0)).squeeze(0)
            else:
                edge_embedding = torch.zeros(self.edge_embedding_dim)

            # Concatenate the node feature with the edge embedding
            updated_feature = torch.cat((node_feature, edge_embedding), dim=0)
            updated_node_features.append(updated_feature)

        updated_node_features = torch.stack(updated_node_features)
        return updated_node_features





