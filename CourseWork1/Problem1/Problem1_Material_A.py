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
n_epoch = 2000

############################################################################################
# User input for hyperparameters
############################################################################################
user_hidden = 10                #used for testing double hidden layer
user_width = 405              
user_lr = 0.000761176
user_BatchSize = 16
#128,128,0.001,20 for example comparison

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
############################################################################################
class Lossfunc(object):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()         #Mean Squared Error (MSE) loss for regression.
        
    def __call__(self, pred, target):
        return self.criterion(pred, target)   #Calculate MSE between stress predicted by NN and target (true) stress

############################################################################################
# This reads the matlab data from the .mat file provided
############################################################################################
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    # Strain and Stress are stored with shape (nsamples, ncomponents, ntimesteps) in MATLAB file,
    # but here it is read as (ncomponents, ntimesteps, nsamples).
    # We transpose to get (nsamples, ntimesteps, ncomponents).
    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2, 0, 1)
        return torch.tensor(strain, dtype=torch.float32)

    # Same logic for 'stress'
    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2, 0, 1)
        return torch.tensor(stress, dtype=torch.float32)

############################################################################################
# Define data normalizer
############################################################################################
class DataNormalizer(object):
    def __init__(self, data):
        data_2d = data.reshape(-1, data.shape[-1])  #Flatten the data into 2D
        self.mean = data_2d.mean(dim=0)             #Calculate mean and std of the data
        self.std = data_2d.std(dim=0) + 1e-8        # Avoid division by zero

    def encode(self, data):
        return (data - self.mean) / self.std        #Normalizes input data using stored mean and std.

    def decode(self, data):
        return data * self.std + self.mean          #Converts normalized data back to original scale.

############################################################################################
# Define network your neural network for the constitutive model below
############################################################################################
class Const_Net(nn.Module):
    def __init__(self, n_comp_in=3, n_comp_out=4, hidden=64, width=128, n_timestep=50):
        super(Const_Net, self).__init__()
        self.n_comp_in = n_comp_in
        self.n_comp_out = n_comp_out
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(n_comp_in * n_timestep, width),       # Single hidden layer
            nn.ReLU(),                                      # Activation function (non-linearity)
            nn.Linear(width, n_comp_out * n_timestep)      # Output layer
        )

    # Forward propagation from input to output layer
    def forward(self, x):
        batch_size, nsteps, _ = x.shape
        flat_x = x.reshape(batch_size, -1)
        out = self.net(flat_x)                             # "out" here is the approximated stress by NN
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
# Data processing accounting for plane strain condition
############################################################################################
# Read data from .mat file
path = 'Material_A.mat' #Define your data path here
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# For plane strain:
# Strain: ε11, ε22, ε12 -> indices [0, 1, 3]
# Stress: σ11, σ22, σ33, σ12 -> indices [0, 1, 2, 3]
strain_reduced = strain[:, :, [0, 1, 3]]
stress_reduced = stress[:, :, [0, 1, 2, 3]]

# Split data into train and test
n_samples = strain_reduced.shape[0]
train_ratio = 0.7   #70% of the total data forms the train dataset
val_ratio = 0.15    #15% of the total data forms the validation dataset
test_ratio = 0.15   #15% of the total data forms the test dataset

ntrain = int(train_ratio * n_samples)
nval = int(val_ratio * n_samples)
ntest = n_samples - ntrain - nval

train_strain = strain_reduced[:ntrain]
train_stress = stress_reduced[:ntrain]
val_strain = strain_reduced[ntrain:ntrain + nval]
val_stress = stress_reduced[ntrain:ntrain + nval]
test_strain = strain_reduced[ntrain + nval:]
test_stress = stress_reduced[ntrain + nval:]

#Normalise the data for strain (Fit the normalizer using only the training data and apply it to the test and validatino data.)
strain_normalizer = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.encode(train_strain)
val_strain_encode = strain_normalizer.encode(val_strain)
test_strain_encode = strain_normalizer.encode(test_strain)

#Normalise the data for stress data (Fit the normalizer using only the training data and apply it to the test data.)
stress_normalizer = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.encode(train_stress)
val_stress_encode = stress_normalizer.encode(val_stress)
test_stress_encode = stress_normalizer.encode(test_stress)

# Create data loader
batch_size = user_BatchSize
train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
val_set = Data.TensorDataset(val_strain_encode, val_stress_encode)
test_set = Data.TensorDataset(test_strain_encode, test_stress_encode)

train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = Data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

############################################################################################
# Define and train network
############################################################################################
# Create Nueral network
#Plain strain so out of 6, only non-zero 3 strain and 4 stress data per timestep and per sample are used
#number of nodes for hidden layer is determined by tunning with Optuna library (the result of this tunning is specied at the beginning of the script)
net = Const_Net(n_comp_in=3, n_comp_out=4, hidden=user_hidden, width=user_width, n_timestep=50)

# Apply He Normal weight initialization
#net.apply(init_weights_he_normal)

# Count trainable parameters
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)

# Define loss, optimizer, and optional scheduler      
loss_func = Lossfunc()
optimizer = torch.optim.Adam(net.parameters(), lr=user_lr, weight_decay=1e-5)  #weight_decay=1e-5: Controls the strength of L2 regularization.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


#####################################################################################################################
# Train network
#####################################################################################################################
# Number of training epochs
epochs = n_epoch

#initialisation of variables for early stopping
patience, patience_counter = 10, 0
best_val_loss = float('inf')

print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_val_list = []


#Start timing for main iteration
start_cpu = time.process_time()
start_wall = time.time()
start_io = time.perf_counter()

for epoch in range(epochs):
    net.train(True)
    trainloss = 0.0

    for input_batch, target_batch in train_loader:
        output_encode = net(input_batch)                    # Forward pass in encoded (normalised) space
        loss = loss_func(output_encode, target_batch)        # Compute loss on normalised data
        optimizer.zero_grad()                                   
        loss.backward()                                     # Backprop and update
        optimizer.step()
        trainloss += loss.item()

    trainloss /= len(train_loader)                          # Average training loss over all batches

    # Evaluate on test set
    net.eval()
    valloss = 0.0
    with torch.no_grad():                                    #To ensure no gradient calculation is performed and stored so that no parameters are updated.
        for input_batch, target_batch in val_loader:
            output_encode = net(input_batch)
            loss_ = loss_func(output_encode, target_batch)
            valloss += loss_.item()
    valloss /= len(val_loader)

    # Step the scheduler
    scheduler.step()

    #Early stopping when no further improvement (over 5 epoches) in validation loss is made in order to avoid overfitting.
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
        print("Epoch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, trainloss, valloss))

    loss_train_list.append(trainloss)
    loss_val_list.append(valloss)
    
        
print("Final Train Loss: {:.6f}".format(trainloss))
print("Final Validation Loss:  {:.6f}".format(valloss))

net.eval()
testloss = 0.0
with torch.no_grad():
    for input_batch, target_batch in test_loader:
        output_encode = net(input_batch)
        loss_ = loss_func(output_encode, target_batch)
        testloss += loss_.item()
testloss /= len(test_loader)

print("Final Test Loss:  {:.6f}".format(testloss))


#End timing and print CPU, wall and input/ouput read times    
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
#Plot train and test losses
plt.figure(figsize=(12, 8))
plt.plot(loss_train_list, label='Train Loss')
plt.plot(loss_val_list, label='Validation Loss')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss (MSE)', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
# Adding the text box
textstr = '\n'.join((
    f'{"Final Train Loss:":<25}{0.001293:>10.6f}',
    f'{"Final Validation Loss:":<25}{0.002928:>10.6f}',
    f'{"Final Test Loss:":<25}{0.002836:>10.6f}'
))
props = dict(boxstyle='round', alpha=0.5, facecolor='white')
plt.gca().text(0.53, 0.65, textstr, transform=plt.gca().transAxes,
               fontsize=16, fontfamily='monospace', bbox=props)
plt.tight_layout()
plt.show()

sample_idx = 0
with torch.no_grad():
    input_sample = test_strain_encode[sample_idx:sample_idx+1]
    pred_stress_encoded = net(input_sample)
    pred_stress = stress_normalizer.decode(pred_stress_encoded)
    true_stress = test_stress[sample_idx:sample_idx+1]

#Plot the comparison between the true and approximated stress.
plt.figure(figsize=(12,8))
plt.plot(pred_stress[0, :, 0].numpy(), label='NN Prediction')
plt.plot(true_stress[0, :, 0].numpy(), label='True Stress', linestyle='--')
plt.xlabel('Time Step',fontsize=16)
plt.ylabel('Stress Component [σ11]',fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
plt.tight_layout()
plt.show()

