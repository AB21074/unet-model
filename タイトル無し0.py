import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from pathlib import Path as plib
from csv_io import CsvIO

class UNetEncoder(nn.Module):
    def __init__(self, input_dim):
        super(UNetEncoder, self).__init__()
        
        # 全結合層で高次元の特徴ベクトルに変換
        self.fc = nn.Linear(input_dim, 256*4*4)
        
        # Reshape層に相当するために後で形状を変換
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # プーリング層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # 全結合層で高次元の特徴ベクトルに変換
        x = F.relu(self.fc(x))
        
        # Reshapeして(8, 8, 2)の形状に変換
        x = x.view(-1, 2, 8, 8)
        
        # 畳み込み層とプーリング層
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        return x

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        
        # アップサンプリング層
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(64, 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.upconv1(x))
        x = self.conv1(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, input_dim):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(input_dim)
        self.decoder = UNetDecoder()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = 4
model = UNet(input_dim)

# Test encoder output shape
sample_coords = torch.randn(2, input_dim)
encoded_output = model.encoder(sample_coords)
print("Encoded Output Shape:", encoded_output.shape)

# Test decoder output shape
decoded_output = model.decoder(encoded_output)
print("Decoded Output Shape:", decoded_output.shape)

'''
#画像の読み込み
def load_image(file_path):
    image = Image.open(file_path).convert('L')  # グレースケールで読み込み
    image = np.array(image, dtype=np.float32) / 255.0  # 0-1にスケーリング
    return image

#画像と座標のデータの読み込み
def load_data(image_dir, coordinates):
    images = []
    for i, coord in enumerate(coordinates):
        image_path = os.path.join(image_dir, f'p{i:05d}.jpg')
        image = load_image(image_path)
        images.append(image)
    return np.array(images)

#座標情報をｃｓｖから読み込み
data_name = 'test_line_data'
directry = plib(r'D:\pic_dataset') / data_name
csvpath = directry / (data_name + '.csv')
#座標情報
coordinates = np.genfromtxt(csvpath, delimiter=',', usecols=(1, 2, 3, 4))

# 画像データの読み込み
images = load_data(directry, coordinates)

# 座標データをテンソルに変換
coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
# 画像データをテンソルに変換（チャネルを追加）
images_tensor = torch.tensor(images[:, np.newaxis, :, :], dtype=torch.float32)

# データセットの作成
dataset = TensorDataset(coordinates_tensor, images_tensor)

# データローダーの作成
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


print("Coordinates Tensor Shape:", coordinates_tensor.shape)
print("Images Tensor Shape:", images_tensor.shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for coords, img in dataloader:
        optimizer.zero_grad()
        output = model(coords)
        loss = criterion(output, img.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
'''























