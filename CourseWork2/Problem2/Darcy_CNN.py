import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py

# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Define network  
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(CNN, self).__init__()
        features = init_features

        # Encoder 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 3
        self.encoder3 = nn.Sequential(
            nn.Conv2d(features*2, features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*4, features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*4, features*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*8, features*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Global context branch in the bottleneck
        # Here, we set context_channels = features*4 (this is a design choice)
        self.fc = nn.Linear(features*8, features*4)
        # Fuse the concatenated bottleneck and context: (features*8 + features*4) -> features*8
        self.global_context_conv = nn.Conv2d(features*8 + features*4, features*8, kernel_size=1)

        # Decoder part

        # Up 3
        self.up3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(features*8, features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*4, features*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 2
        self.up2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(features*4, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, features*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up 1
        self.up1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, Nx, Ny]
        x = x.unsqueeze(1)  # -> [B, 1, Nx, Ny]

        # --- Encoder ---
        enc1 = self.encoder1(x)                    # [B, features, Nx, Ny]
        enc2_in = self.pool1(enc1)                   # downsample
        enc2_out = self.encoder2(enc2_in)           # [B, features*2, Nx/2, Ny/2]
        enc3_in = self.pool2(enc2_out)
        enc3_out = self.encoder3(enc3_in)           # [B, features*4, Nx/4, Ny/4]
        enc4_in = self.pool3(enc3_out)

        # --- Bottleneck ---
        bottleneck = self.bottleneck(enc4_in)       # [B, features*8, H, W] where H = Nx/8, W = Ny/8

        # --- Global Context Branch ---
        # Global average pooling to obtain a context vector [B, features*8, 1, 1]
        global_context = torch.nn.functional.adaptive_avg_pool2d(bottleneck, (1, 1))
        # Flatten to [B, features*8] and pass through FC layer to get [B, features*4]
        global_context = global_context.view(bottleneck.size(0), -1)
        global_context = self.fc(global_context)
        # Reshape to [B, features*4, 1, 1] and expand to match bottleneck's spatial dims
        global_context = global_context.view(bottleneck.size(0), -1, 1, 1)
        global_context = global_context.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))
        
        # Concatenate global context with bottleneck features
        bottleneck = torch.cat((bottleneck, global_context), dim=1)  # [B, features*8 + features*4, H, W]
        # Fuse via 1x1 conv back to [B, features*8, H, W]
        bottleneck = self.global_context_conv(bottleneck)

        # --- Decoder ---
        dec3 = self.up3(bottleneck)                # [B, features*4, H*2, W*2]
        dec3 = torch.cat((dec3, enc3_out), dim=1)    # concatenate with corresponding encoder output
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)                      # [B, features*2, ...]
        dec2 = torch.cat((dec2, enc2_out), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)                      # [B, features, Nx, Ny]
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.conv(dec1)                      # [B, out_channels, Nx, Ny]
        out = out.squeeze(1)                       # [B, Nx, Ny]
        return out


if __name__ == '__main__':
    ############################# Data processing #############################
    # Read data from mat
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    # Create data loader
    batch_size = 20
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #############################
    # Create RNN instance, define loss function and optimizer
    channel_width = 64
    net = CNN().float()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7, verbose=True, threshold=1e-4, threshold_mode='rel')

    # Train network
    epochs = 200 # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    start_time = time()
    
    loss_train_list = []
    loss_test_list = []
    x = []
    for epoch in range(epochs):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            output = net(input) # Forward
            output = u_normalizer.decode(output)
            l = loss_func(output, target) # Calculate loss

            optimizer.zero_grad() # Clear gradients
            l.backward() # Backward
            optimizer.step() # Update parameters
            #scheduler.step() # Update learning rate

            trainloss += l.item()    

        # Test
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        # I am using "torch.optim.lr_scheduler.ReduceLROnPlateau". Hence, testloss value is fed in for scheduler step.
        scheduler.step(testloss)
        
        # Print train loss every 10 epochs
        if epoch % 10 == 0:
            print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss))

        loss_train_list.append(trainloss/len(train_loader))
        loss_test_list.append(testloss)
        x.append(epoch)

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Traing time: {}'.format(total_time_str))
    print("Train loss:{}".format(trainloss/len(train_loader)))
    print("Test loss:{}".format(testloss))
    
    ############################# Plot #############################
    plt.figure(1)
    plt.plot(x, loss_train_list, label='Train loss')
    plt.plot(x, loss_test_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()