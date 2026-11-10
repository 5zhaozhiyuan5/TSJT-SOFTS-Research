import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from typing import Union, Tuple, Optional

import time
from torch.nn.utils import weight_norm
import copy

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution, IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
        )

        self.regressor = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.regressor(x)  
        return x, y


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)  
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)  
        return self.elu(x)
        
class BCEclassifier(nn.Module):
    def __init__(self, input_size=36, hidden_size=36, num_layers=3): 
        super(BCEclassifier, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return x   
        
class GPClassificationModel(ApproximateGP):
    def __init__(self, input_dim, inducing_points):
        feature_dim = input_dim  

        variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),       
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),       
            nn.ELU()
        )

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

