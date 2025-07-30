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
# Define loss function (BCEWithLogitsLoss for binary classification)
############################################################################################
class Lossfunc(object):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target):
        return self.criterion(pred, target)

############################################################################################
# Data reader for the Eiffel tower dataset
############################################################################################
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    def get_load(self):
        load_data = np.array(self.data['load_apply'])
        if load_data.shape[0] == 20 and load_data.shape[1] == 1000:
            load_data = load_data.T  # shape -> [1000, 20]
        return torch.tensor(load_data, dtype=torch.float32)

    def get_result(self):
        result_data = np.array(self.data['result'])
        if result_data.shape[0] == 1 and result_data.shape[1] == 1000:
            result_data = result_data.T  # shape -> [1000, 1]
        return torch.tensor(result_data, dtype=torch.float32)

############################################################################################
# Data normaliser
############################################################################################
class DataNormalizer(object):
    def __init__(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-8

    def encode(self, data):
        return (data - self.mean) / self.std

    def decode(self, data):
        return data * self.std + self.mean

############################################################################################
# Define a U-Net style fully connected network for the constitutive model
############################################################################################
class UNetFCNN(nn.Module):

    def __init__(self, input_dim=20, hidden_layers=2, width=64):
        super(UNetFCNN, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU()
        )
        # Extra encoder layer(s)
        self.enc2 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Linear(width, width)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU()
        )
        # Output layer outputs a single logit for BCEWithLogitsLoss
        self.dec2 = nn.Linear(width, 1)

    def forward(self, x):
        # Encoder forward pass
        skip1 = self.enc1(x)      # First skip connection
        skip2 = self.enc2(skip1)  # Second skip connection

        # Bottleneck
        b = self.bottleneck(skip2)

        # Decoder forward pass (incorporate skip connections)
        up1 = self.dec1(b + skip2)
        out = self.dec2(up1 + skip1)

        return out

############################################################################################
# Load and prepare data
############################################################################################
path = 'Eiffel_data.mat'
data_reader = MatRead(path)
X_all = data_reader.get_load()
y_all = data_reader.get_result()

n_samples = X_all.shape[0]
train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
ntrain = int(train_ratio * n_samples)
nval = int(val_ratio * n_samples)

indices = torch.randperm(n_samples)
train_idx = indices[:ntrain]
val_idx = indices[ntrain:ntrain + nval]
test_idx = indices[ntrain + nval:]

train_X, train_y = X_all[train_idx], y_all[train_idx]
val_X, val_y = X_all[val_idx], y_all[val_idx]
test_X, test_y = X_all[test_idx], y_all[test_idx]

normalizer = DataNormalizer(train_X)
train_X_encode = normalizer.encode(train_X)
val_X_encode = normalizer.encode(val_X)
test_X_encode = normalizer.encode(test_X)

############################################################################################
# Optuna hyperparameter tuning
############################################################################################
def accuracy_fn(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum()
    acc = correct / labels.shape[0]
    return acc.item()

def objective(trial):
    hidden_layers = 4#trial.suggest_int('hidden_layers', 1, 10)
    width = trial.suggest_int('width', 64, 512)
    lr = trial.suggest_loguniform('lr', 1e-5, 5e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    train_set = Data.TensorDataset(train_X_encode, train_y)
    val_set = Data.TensorDataset(val_X_encode, val_y)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    net = UNetFCNN(input_dim=20, hidden_layers=hidden_layers, width=width)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = Lossfunc()

    epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        net.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            output = net(input_batch)
            loss = loss_func(output, target_batch)
            loss.backward()
            optimizer.step()

        # Validation phase
        net.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                output = net(input_batch)
                val_loss += loss_func(output, target_batch).item()
                val_acc += accuracy_fn(output, target_batch)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        trial.report(val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss

############################################################################################
# Running the Optuna study
############################################################################################
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best hyperparameters: {study.best_params}")

############################################################################################
# Visualising the hyperparameter optimisation results
############################################################################################
fig = 0
if fig == 1:
    fig1 = vis.plot_optimization_history(study)
    fig1.show(renderer="browser")
    
    fig2 = vis.plot_param_importances(study)
    fig2.show(renderer="browser")
    
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.show(renderer="browser")
    
    fig4 = vis.plot_slice(study)
    fig4.show(renderer="browser")