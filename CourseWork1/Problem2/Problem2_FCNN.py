import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py
import time
import random

############################################################################################
# User input
############################################################################################
n_epoch = 2000  # number of epochs

############################################################################################
# User input for hyperparameters
############################################################################################
user_hidden = 1                # number of hidden layers
user_width = 475                # neurons per hidden layer
user_lr = 0.00030
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
# Define your loss function here
# For a binary classification task (fail=0, survive=1), use nn.BCEWithLogitsLoss.
############################################################################################
class Lossfunc(object):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

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

    def get_load(self):
        # load_apply is of size (1000 x 20) in MATLAB,
        # but h5py might store it with reversed shape
        # so,transpose if necessary to ensure shape [1000, 20].
        load_data = np.array(self.data['load_apply'])
        if load_data.shape[0] == 20 and load_data.shape[1] == 1000:
            load_data = load_data.T  # shape -> [1000, 20]
        return torch.tensor(load_data, dtype=torch.float32)

    def get_result(self):
        # result is 1000×1
        result_data = np.array(self.data['result'])
        if result_data.shape[0] == 1 and result_data.shape[1] == 1000:
            result_data = result_data.T  # shape -> [1000, 1]
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
# Define network your neural network for the constitutive model below
############################################################################################
class FCNN(nn.Module):
    def __init__(self, input_dim=20, hidden_layers=1, width=64):
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

        # Additional hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Final layer outputs a single logit for BCEWithLogitsLoss
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, 20)
        output shape: (batch_size, 1)  (logits for BCEWithLogitsLoss)
        """
        return self.net(x)

############################################################################################
# Custom He Normal weight initialisation
############################################################################################
def init_weights_he_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

############################################################################################
# Data processing for Eiffel tower problem (binary classification)
############################################################################################
# Data file path
path = 'Eiffel_data.mat'
data_reader = MatRead(path)
X_all = data_reader.get_load()   #distribution of loading
y_all = data_reader.get_result() #fail or survive


# Split data into train and test
n_samples = X_all.shape[0]
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

ntrain = int(train_ratio * n_samples)
nval   = int(val_ratio   * n_samples)
ntest  = n_samples - ntrain - nval

# Shuffle indices for random splitting
indices = torch.randperm(n_samples)
train_idx = indices[:ntrain]
val_idx   = indices[ntrain:ntrain + nval]
test_idx  = indices[ntrain + nval:]

train_X = X_all[train_idx]
train_y = y_all[train_idx]
val_X   = X_all[val_idx]
val_y   = y_all[val_idx]
test_X  = X_all[test_idx]
test_y  = y_all[test_idx]

# Normalise the input features using only the training set
normalizer = DataNormalizer(train_X)
train_X_encode = normalizer.encode(train_X)
val_X_encode   = normalizer.encode(val_X)
test_X_encode  = normalizer.encode(test_X)
# Note y is not normalise because it is 0/1

# Create DataLoaders
batch_size = user_BatchSize

train_set = Data.TensorDataset(train_X_encode, train_y)
val_set   = Data.TensorDataset(val_X_encode,   val_y)
test_set  = Data.TensorDataset(test_X_encode,  test_y)

train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = Data.DataLoader(val_set,   batch_size=batch_size, shuffle=False)
test_loader  = Data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)

############################################################################################
# Define, train, and test network
############################################################################################
# Create neural network
net = FCNN(input_dim=20, hidden_layers=user_hidden, width=user_width)

#Apply He Normal weight initialisation
# net.apply(init_weights_he_normal)

# Count trainable parameters
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)

# Define loss, optimiser, and scheduler      
loss_func = Lossfunc()
optimizer = torch.optim.Adam(net.parameters(), lr=user_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#####################################################################################################################
# Train network
#####################################################################################################################
# Number of training epochs
epochs = n_epoch

#variable definition for early stopping
patience = 10
patience_counter = 0
best_val_loss = float('inf')

print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_val_list   = []
acc_train_list  = []
acc_val_list    = []

#Start timing
start_cpu = time.process_time()
start_wall = time.time()
start_io = time.perf_counter()

#Definition of accuracy
def accuracy_fn(logits, labels):
    """
    Compute classification accuracy.
    logits: (batch_size, 1)
    labels: (batch_size, 1)
    apply a sigmoid threshold at 0.5
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum()
    acc = correct / labels.shape[0]
    return acc.item()

for epoch in range(epochs):
    net.train()
    trainloss = 0.0
    trainacc  = 0.0

    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        output_logits = net(input_batch)                    # Forward pass in encoded (normalised) space
        loss = loss_func(output_logits, target_batch)       # Compute loss on normalised data
        loss.backward()                                     # Backprop and update
        optimizer.step()

        trainloss += loss.item()                            # Average training loss over all batches
        # compute training accuracy for this batch
        batch_acc = accuracy_fn(output_logits, target_batch)
        trainacc  += batch_acc                              # Average training accuracy over all batches

    trainloss /= len(train_loader)                          # Average training loss over all batches
    trainacc  /= len(train_loader)                          # Average training accuracy over all batches

    # Evaluate on validation set
    net.eval()
    valloss = 0.0
    valacc  = 0.0
    with torch.no_grad():                                    #To ensure no gradient calculation is performed and stored so that no parameters are updated.
        for input_batch, target_batch in val_loader:
            output_logits = net(input_batch)
            loss_ = loss_func(output_logits, target_batch)
            valloss += loss_.item()                             # Average validation loss over all batches
            valacc  += accuracy_fn(output_logits, target_batch) # Average validation accuracy over all batches

    valloss /= len(val_loader)                                  # Average validation loss over all batches
    valacc  /= len(val_loader)                                  # Average validation accuracy over all batches

    # Step the scheduler
    scheduler.step()

    # Early stopping
    if valloss < best_val_loss:
        best_val_loss = valloss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Print train loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, "
              f"Train Loss: {trainloss:.6f}, Train Acc: {trainacc:.3f}, "
              f"Val Loss: {valloss:.6f}, Val Acc: {valacc:.3f}")

    loss_train_list.append(trainloss)
    loss_val_list.append(valloss)
    acc_train_list.append(trainacc)
    acc_val_list.append(valacc)

print("Final Train Loss: {:.6f}".format(trainloss))
print("Final Validation Loss:  {:.6f}".format(valloss))
print("Final Validation Accuracy: {:.3f}".format(valacc))

# Compute test loss and accuracy
net.eval()
testloss = 0.0
testacc  = 0.0
with torch.no_grad():
    for input_batch, target_batch in test_loader:
        output_logits = net(input_batch)
        loss_ = loss_func(output_logits, target_batch)
        testloss += loss_.item()
        testacc  += accuracy_fn(output_logits, target_batch)

testloss /= len(test_loader)
testacc  /= len(test_loader)

print("Final Test Loss:  {:.6f}".format(testloss))
print("Final Test Accuracy: {:.3f}".format(testacc))


#Finish timing and print them
end_cpu = time.process_time()
cpu_time = end_cpu - start_cpu
print(f"CPU Time: {cpu_time:.6f} seconds")
end_wall = time.time()
wall_time = end_wall - start_wall
print(f"Wall Time: {wall_time:.6f} seconds")
end_io = time.perf_counter()
io_time = end_io - start_io
print(f"I/O Read Time: {io_time:.6f} seconds")

#####################################################################################################################
# Plot results
#####################################################################################################################
# First figure: Loss
plt.figure(figsize=(12, 5))
plt.plot(loss_train_list, label='Train Loss')
plt.plot(loss_val_list,   label='Validation Loss')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss (BCE)', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
# Adding the text box
textstr = '\n'.join((
    f'{"Final Train Loss:":<25}{0.081854:>10.6f}',
    f'{"Final Validation Loss:":<25}{0.0104049:>10.6f}',
    f'{"Final Test Loss:":<25}{0.073377:>10.6f}'
))
props = dict(boxstyle='round', alpha=0.5, facecolor='white')
plt.gca().text(0.53, 0.55, textstr, transform=plt.gca().transAxes,
               fontsize=16, fontfamily='monospace', bbox=props)
plt.tight_layout()
plt.show()

# Second figure: Accuracy
plt.figure(figsize=(12, 5))
plt.plot(acc_train_list, label='Train Accuracy')
plt.plot(acc_val_list,   label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
# Adding the text box
textstr = '\n'.join((
    f'{"Final Validation Accuracy:":<25}{0.969:>10.6f}',
    f'{"Final Test Accuracy:":<25}{0.981:>10.6f}'
))
props = dict(boxstyle='round', alpha=0.5, facecolor='white')
plt.gca().text(0.53, 0.65, textstr, transform=plt.gca().transAxes,
               fontsize=16, fontfamily='monospace', bbox=props)
plt.tight_layout()
plt.show()


