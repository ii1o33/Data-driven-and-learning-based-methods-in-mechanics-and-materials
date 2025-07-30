# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:26:11 2025

@author: socce
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import time
import random
import optuna  # Added Optuna for hyperparameter tuning
import optuna.visualization as vis

############################################################################################
# Set random seed for reproducibility
############################################################################################
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

############################################################################################
# Define your loss function here
############################################################################################
class Lossfunc(object):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, pred, target):
        return self.criterion(pred, target)

############################################################################################
# This reads the matlab data from the .mat file provided
############################################################################################
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2, 0, 1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2, 0, 1)
        return torch.tensor(stress, dtype=torch.float32)

############################################################################################
# Define data normalizer
############################################################################################
class DataNormalizer(object):
    def __init__(self, data):
        data_2d = data.reshape(-1, data.shape[-1])
        self.mean = data_2d.mean(dim=0)
        self.std = data_2d.std(dim=0) + 1e-8

    def encode(self, data):
        return (data - self.mean) / self.std

    def decode(self, data):
        return data * self.std + self.mean

############################################################################################
# Define network your neural network for the constitutive model below (Single Hidden Layer)
############################################################################################
class Const_Net(nn.Module):
    def __init__(self, n_comp_in=1, n_comp_out=1, width=128, n_timestep=50):
        super(Const_Net, self).__init__()
        self.n_comp_in = n_comp_in
        self.n_comp_out = n_comp_out

        # Single hidden layer architecture (width defines the number of neurons)
        self.net = nn.Sequential(
            nn.Linear(n_comp_in * n_timestep, width),    # Input layer → Hidden layer
            nn.ReLU(),                                   # Activation function
            nn.Linear(width, n_comp_out * n_timestep)    # Hidden layer → Output layer
        )

    #Forward propagation from input to output layer
    def forward(self, x):
        batch_size, nsteps, _ = x.shape
        flat_x = x.reshape(batch_size, -1)
        out = self.net(flat_x)
        out = out.view(batch_size, nsteps, self.n_comp_out)
        return out
    
############################################################################################
# Custom He Normal weight initialization
############################################################################################
def init_weights_he_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

############################################################################################
# Data processing for Material C (1D uniaxial deformation data)
############################################################################################
path = 'Material_C.mat'  # Updated for Material C
data_reader = MatRead(path)
strain = data_reader.get_strain()   # Use 1 component for Material C
stress = data_reader.get_stress()   # Use 1 component for Material C

n_samples = strain.shape[0]
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
ntrain = int(train_ratio * n_samples)
nval = int(val_ratio * n_samples)
ntest = n_samples - ntrain - nval

train_strain, train_stress = strain[:ntrain], stress[:ntrain]
val_strain, val_stress = strain[ntrain:ntrain + nval], stress[ntrain:ntrain + nval]
test_strain, test_stress = strain[ntrain + nval:], stress[ntrain + nval:]

strain_normalizer = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.encode(train_strain)
val_strain_encode = strain_normalizer.encode(val_strain)
test_strain_encode = strain_normalizer.encode(test_strain)

stress_normalizer = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.encode(train_stress)
val_stress_encode = stress_normalizer.encode(val_stress)
test_stress_encode = stress_normalizer.encode(test_stress)

############################################################################################
# Optuna hyperparameter tuning (tuning only width now)
############################################################################################
def objective(trial):
    width = trial.suggest_int('width', 30, 500)               # Only width now defines hidden layer size
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
    val_set = Data.TensorDataset(val_strain_encode, val_stress_encode)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ⚡ Updated network call
    net = Const_Net(n_comp_in=1, n_comp_out=1, width=width, n_timestep=50)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = Lossfunc()

    epochs = 100
    for epoch in range(epochs):
        net.train()
        for input_batch, target_batch in train_loader:
            output = net(input_batch)
            loss = loss_func(output, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                output = net(input_batch)
                val_loss += loss_func(output, target_batch).item()
        val_loss /= len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


# Running Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print(f"Best hyperparameters: {study.best_params}")

fig = 1
if fig == 1:
    # Visualizations
    fig1 = vis.plot_parallel_coordinate(study)
    fig1.show(renderer="browser")
    
    fig2 = vis.plot_slice(study)
    fig2.show(renderer="browser")
    
    fig3 = vis.plot_optimization_history(study)
    fig3.show(renderer="browser")
    
    fig4 = vis.plot_contour(study, params=['width', 'lr', 'batch_size'])
    fig4.show(renderer="browser")
