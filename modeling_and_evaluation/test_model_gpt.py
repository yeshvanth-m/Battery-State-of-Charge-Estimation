import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
PROCESSED_DATA_DIR = '../dataset/LG_HG2_processed'
TEMPERATURE = '25degC'  # Change to any available temperature folder
FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Power [W]', 'CC_Capacity [Ah]']
LABEL_COL = 'SOC [-]'
MODEL_PATH = "soc_cnn_model.pth"
BATCH_SIZE = 64

# --- Model Definition (copy from your test_model.py) ---
import torch.nn as nn
class SoCCNN1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size=3, dropout=0.1):
        super(SoCCNN1D, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_channels))
        for i in range(1, num_layers):
            layer_channels = hidden_channels // (2 ** i)
            self.conv_layers.append(nn.Conv1d(hidden_channels // (2 ** (i - 1)), layer_channels, kernel_size, padding=kernel_size//2))
            self.batch_norm_layers.append(nn.BatchNorm1d(layer_channels))
        self.output_layer = nn.Conv1d(hidden_channels // (2 ** (num_layers - 1)), 1, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for conv_layer, batch_norm_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(conv_layer(x)))
            x = self.dropout(x)
        x = self.output_layer(x)
        x = x.squeeze()
        return x

# --- Load one test file ---
test_folder = os.path.join(PROCESSED_DATA_DIR, TEMPERATURE)
test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv') and 'Charge' not in f and 'Dis' not in f]
test_file = test_files[0]  # Pick the first test file
df = pd.read_csv(os.path.join(test_folder, test_file))

# --- Preprocess ---
df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']
df['CC_Capacity [Ah]'] = (df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600).cumsum()
scaler = StandardScaler()
df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

# --- Prepare tensors ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32).to(device)
inputs = inputs.unsqueeze(2)  # Shape: (N, features, 1)

# --- Load model ---
hidden_size = 128
num_layers = 1
model = SoCCNN1D(input_channels=len(FEATURE_COLS), hidden_channels=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
model.eval()

# --- Get predictions ---
with torch.no_grad():
    outputs = model(inputs)
    outputs = outputs.cpu().numpy()

# --- Print results ---
print(f"Test file: {test_file}")
print("Predicted SOC values:")
print(outputs)