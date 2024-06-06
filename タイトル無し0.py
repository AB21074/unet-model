import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from pathlib import Path as plib
from csv_io import CsvIO


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