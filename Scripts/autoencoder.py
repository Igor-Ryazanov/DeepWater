import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, final_kernel = 7):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = final_kernel)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = final_kernel),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train_autoencoder(model, train_data, num_epochs=5, batch_size=5, learning_rate=1e-3):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.manual_seed(0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_data = torch.from_numpy(train_data)
    train_data = train_data.to(device)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)
    outputs = []

    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            rec = model(data)
            loss = criterion(rec, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, total_loss))
        outputs.append((epoch, data, rec),)
    return outputs

def test_reconstruction(model, test_frame):
    test_reconst = model(torch.from_numpy(test_frame).to(device).view([1, test_frame.shape[0], test_frame.shape[1], test_frame.shape[2]]))
    return test_reconst.cpu().detach().numpy().reshape(test_frame.shape)

def compress_wakes(wakes, model):
    compressed_wakes = []
    for wake in wakes:
        wake_image = torch.from_numpy(wake[:,:-1,:]).to(device).view([1] + list(wake[:,:-1,:].shape))
        compressed_wake = model.encoder(wake_image).cpu().detach().numpy()
        compressed_wakes.append(compressed_wake)
    return np.array(compressed_wakes)

def decompress_wakes(seeds, model, comperssed_shape = [1, 64, 1, 9], depth = 28, points = 60):
    decompressed_wakes = []
    for seed in seeds:
        seed_tensor = torch.from_numpy(seed).to(device).view(comperssed_shape)
        decompressed_wake = model.decoder(seed_tensor).cpu().detach().numpy().reshape(4, depth, points)
        decompressed_wakes.append(decompressed_wake)
    return np.array(decompressed_wakes)
