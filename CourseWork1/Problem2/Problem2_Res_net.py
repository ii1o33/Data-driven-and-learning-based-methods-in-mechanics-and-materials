import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import time
import random
from sklearn.model_selection import KFold
from itertools import zip_longest

############################################################################################
# User input
############################################################################################
n_epoch = 2000  # number of epochs
n_splits = 8    # number of folds for k-fold cross-validation

############################################################################################
# User input for hyperparameters
############################################################################################
user_hidden = 10                # number of hidden layers
user_width = 229                # neurons per hidden layer
user_lr = 0.00014346
user_BatchSize = 16

############################################################################################
# Set random seed for reproducibility and hyperparameter tuning
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
# Define loss function
############################################################################################
class Lossfunc(object):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target):
        return self.criterion(pred, target)

############################################################################################
# Data reader for .mat files
############################################################################################
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')
    
        # load_apply is of size (1000 x 20) in MATLAB,
        # but h5py might store it with reversed shape
        # so,transpose if necessary to ensure shape [1000, 20].
    def get_load(self):
        load_data = np.array(self.data['load_apply'])
        if load_data.shape[0] == 20 and load_data.shape[1] == 1000:
            load_data = load_data.T
        return torch.tensor(load_data, dtype=torch.float32)

    def get_result(self):
        result_data = np.array(self.data['result'])
        if result_data.shape[0] == 1 and result_data.shape[1] == 1000:
            result_data = result_data.T
        return torch.tensor(result_data, dtype=torch.float32)

############################################################################################
#Define data normalizer
#normalise only the input features, not the binary target.
############################################################################################
class DataNormalizer(object):
    def __init__(self, data):
        # data is shape (N, 20)
        self.mean = data.mean(dim=0)        # dimension [20]
        self.std = data.std(dim=0) + 1e-8   # dimension [20]

    def encode(self, data):
        return (data - self.mean) / self.std

    def decode(self, data):
        return data * self.std + self.mean

############################################################################################
# Define a residual block for the ResNet-based architecture
############################################################################################
class ResBlock(nn.Module):
    def __init__(self, width):
        super(ResBlock, self).__init__()
        self.fc = nn.Linear(width, width)           # Single hidden layer
        self.relu = nn.ReLU()                       # Single activation fcn

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = out + identity                        # Skip connection
        out = self.relu(out)
        return out

############################################################################################
# Define network your neural network for the constitutive model below ( ResNet)
############################################################################################
class FCNN(nn.Module):
    def __init__(self, input_dim=20, hidden_layers=2, width=64):
        """
        input_dim = 20 because load_apply has shape [N, 20].
        hidden_layers = number of hidden layers to stack.
        width = number of neurons in each hidden layer.
        """
        super(FCNN, self).__init__()

        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        # Additional hidden layers with skip connections
        # We replace the simple linear+ReLU with ResBlocks
        for _ in range(hidden_layers - 1):
            layers.append(ResBlock(width))

        # Final layer outputs a single logit for BCEWithLogitsLoss
        layers.append(nn.Linear(width, 1))

        # Combine into a Sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, 20)
        output shape: (batch_size, 1)  (logits for BCEWithLogitsLoss)
        """
        return self.net(x)

############################################################################################
# Accuracy computation function
############################################################################################
def accuracy_fn(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum()
    acc = correct / labels.shape[0]
    return acc.item()

############################################################################################
# Utility function to pad lists of different lengths
############################################################################################
def pad_lists(list_of_lists):
    """Pad shorter lists with np.nan for proper averaging later."""
    return np.array(list(zip_longest(*list_of_lists, fillvalue=np.nan))).T

############################################################################################
# Main k-fold cross-validation process with learning curve plotting
############################################################################################
# Data file path
path = 'Eiffel_data.mat'
data_reader = MatRead(path)
X_all = data_reader.get_load()   #distribution of loading
y_all = data_reader.get_result() #fail or survive

# Define k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store results across folds
fold_results = []
all_train_loss, all_val_loss = [], []
all_train_acc, all_val_acc = [], []

# Perform training for each fold
fold_idx = 1
for train_index, val_index in kf.split(X_all):
    print(f"\n--- Fold {fold_idx}/{n_splits} ---")

    # Split data into training and validation sets
    train_X, val_X = X_all[train_index], X_all[val_index]
    train_y, val_y = y_all[train_index], y_all[val_index]

    # Normalize based on training data
    normalizer = DataNormalizer(train_X)
    train_X_encode = normalizer.encode(train_X)
    val_X_encode = normalizer.encode(val_X)

    # Create data loaders
    train_set = Data.TensorDataset(train_X_encode, train_y)
    val_set = Data.TensorDataset(val_X_encode, val_y)
    train_loader = Data.DataLoader(train_set, batch_size=user_BatchSize, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=user_BatchSize, shuffle=False)

    # Initialize model, loss function, optimizer, and scheduler
    net = FCNN(input_dim=20, hidden_layers=user_hidden, width=user_width)
    loss_func = Lossfunc()
    optimizer = torch.optim.Adam(net.parameters(), lr=user_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Early stopping variables
    best_val_loss = float('inf')
    patience, patience_counter = 15, 0

    # Store per-epoch loss and accuracy
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    # Training loop per fold
    for epoch in range(n_epoch):
        net.train()
        trainloss, trainacc = 0.0, 0.0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            output_logits = net(input_batch)
            loss = loss_func(output_logits, target_batch)
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
            trainacc += accuracy_fn(output_logits, target_batch)

        trainloss /= len(train_loader)
        trainacc /= len(train_loader)

        # Validation loop
        net.eval()
        valloss, valacc = 0.0, 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                output_logits = net(input_batch)
                loss_ = loss_func(output_logits, target_batch)
                valloss += loss_.item()
                valacc += accuracy_fn(output_logits, target_batch)

        valloss /= len(val_loader)
        valacc /= len(val_loader)

        scheduler.step()

        # Append metrics per epoch
        train_loss_list.append(trainloss)
        val_loss_list.append(valloss)
        train_acc_list.append(trainacc)
        val_acc_list.append(valacc)

        # Early stopping check
        if valloss < best_val_loss:
            best_val_loss = valloss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {trainloss:.6f}, Val Loss: {valloss:.6f}, Train Acc: {trainacc:.3f}, Val Acc: {valacc:.3f}")

    # Store fold metrics
    all_train_loss.append(train_loss_list)
    all_val_loss.append(val_loss_list)
    all_train_acc.append(train_acc_list)
    all_val_acc.append(val_acc_list)

    print(f"Fold {fold_idx} results: Best Val Loss: {best_val_loss:.6f}, Val Acc: {valacc:.3f}")
    fold_results.append((best_val_loss, valacc))
    fold_idx += 1

# Handle inhomogeneous lengths by padding
all_train_loss_padded = pad_lists(all_train_loss)
all_val_loss_padded = pad_lists(all_val_loss)
all_train_acc_padded = pad_lists(all_train_acc)
all_val_acc_padded = pad_lists(all_val_acc)

# Compute average learning curves across all folds, ignoring NaNs
avg_train_loss = np.nanmean(all_train_loss_padded, axis=0)
avg_val_loss = np.nanmean(all_val_loss_padded, axis=0)
avg_train_acc = np.nanmean(all_train_acc_padded, axis=0)
avg_val_acc = np.nanmean(all_val_acc_padded, axis=0)

# Plot learning curves for loss
plt.figure(figsize=(12, 5))
plt.plot(avg_train_loss, label='Average Train Loss')
plt.plot(avg_val_loss, label='Average Validation Loss')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
# Adding the text box
textstr = '\n'.join((
    f'{"Average Validation Loss:":<25}{0.032825:>10.6f}',
))
props = dict(boxstyle='round', alpha=0.5, facecolor='white')
plt.gca().text(0.53, 0.65, textstr, transform=plt.gca().transAxes,
               fontsize=16, fontfamily='monospace', bbox=props)
plt.tight_layout()
plt.show()


# Plot learning curves for accuracy
plt.figure(figsize=(12, 5))
plt.plot(avg_train_acc, label='Average Train Accuracy')
plt.plot(avg_val_acc, label='Average Validation Accuracy')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
# Adding the text box
textstr = '\n'.join((
    f'{"Average Validation Accuracy:":<25}{0.981:>10.6f}',
))
props = dict(boxstyle='round', alpha=0.5, facecolor='white')
plt.gca().text(0.5, 0.65, textstr, transform=plt.gca().transAxes,
               fontsize=16, fontfamily='monospace', bbox=props)
plt.tight_layout()
plt.show()

# Final averaged cross-validation results
avg_val_loss_final = np.nanmean([res[0] for res in fold_results])
avg_val_acc_final = np.nanmean([res[1] for res in fold_results])

print("\n--- Final Cross-Validation Results ---")
print(f"Average Validation Loss: {avg_val_loss_final:.6f}")
print(f"Average Validation Accuracy: {avg_val_acc_final:.3f}")


