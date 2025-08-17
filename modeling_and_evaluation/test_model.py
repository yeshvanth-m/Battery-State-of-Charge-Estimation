import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_DIR = '../dataset/LG_HG2_processed'
FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Power [W]', 'CC_Capacity [Ah]']
LABEL_COL = 'SOC [-]'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Power [W]', 'CC_Capacity [Ah]']
model_path = "soc_cnn_model.pth"
temperatures_to_process = ['25degC', '0degC', 'n10degC', 'n20degC', '10degC', '40degC']   

best_hyperparams = {'learning_rate': 0.00022558427829869434, 'hidden_size': 128, 'num_layers': 1, 'dropout_rate': 0.4512710193068582}
hidden_size = best_hyperparams['hidden_size']
num_layers = best_hyperparams['num_layers']

class BatteryDatasetCNN1D(Dataset):
    def __init__(self, data_tensor, labels_tensor, filenames=None, times=None):
        self.features = data_tensor
        self.labels = labels_tensor
        self.filenames = filenames 
        self.times = times 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].unsqueeze(1)
        label = self.labels[idx]
        filename = self.filenames[idx]
        time = self.times[idx]  
        return feature, label, filename, time
    
    def get_unique_filenames(self):
        return set(self.filenames)
    
    def get_times(self):
        return self.times

def filter_data_by_filenames(df, filenames):
    return df[df['SourceFile'].isin(filenames)]

# Function to load data
def load_data(directory, temperatures):
    frames = []    
    for temp_folder in os.listdir(directory):
        if temp_folder in temperatures:
            temp_path = os.path.join(directory, temp_folder)
            for file in os.listdir(temp_path):
                if 'Charge' in file or 'Dis' in file:
                    continue  # Skip constant charge and discharge files
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))
                    df['SourceFile'] = file

                    # Calculate power
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']

                    # Initialize CC_Capacity [Ah] column
                    df['CC_Capacity [Ah]'] = 0.0

                    # Integrating current over time to calculate cumulative capacity
                    df['CC_Capacity [Ah]'] = (df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600).cumsum()

                    frames.append(df)
    return pd.concat(frames, ignore_index=True)

data = load_data(PROCESSED_DATA_DIR, temperatures_to_process)
scaler = StandardScaler()
data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])

unique_files = np.array(list(set(data['SourceFile'])))

# Convert to tensors and load into GPU memory
data_tensor = torch.tensor(data[FEATURE_COLS].values, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(data[LABEL_COL].values, dtype=torch.float32).to(device)
filenames_tensor = data['SourceFile'].values

unique_files = np.array(list(set(data['SourceFile'])))
train_files, temp_files = train_test_split(unique_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

test_data = filter_data_by_filenames(data, test_files)

test_dataset = BatteryDatasetCNN1D(
    torch.tensor(test_data[FEATURE_COLS].values, dtype=torch.float32).to(device),
    torch.tensor(test_data[LABEL_COL].values, dtype=torch.float32).to(device),
    test_data['SourceFile'].values,
    test_data['Time [s]'].values  
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model
class SoCCNN1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size=3, dropout=0.1):
        super(SoCCNN1D, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # First convolutional layer
        self.conv_layers.append(nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_channels))

        # Dynamically add convolutional layers
        for i in range(1, num_layers):
            layer_channels = hidden_channels // (2 ** i)
            self.conv_layers.append(nn.Conv1d(hidden_channels // (2 ** (i - 1)), layer_channels, kernel_size, padding=kernel_size//2))
            self.batch_norm_layers.append(nn.BatchNorm1d(layer_channels))

        # Output layer - adjust as needed
        self.output_layer = nn.Conv1d(hidden_channels // (2 ** (num_layers - 1)), 1, kernel_size, padding=kernel_size//2)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv_layer, batch_norm_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(conv_layer(x)))
            x = self.dropout(x)

        x = self.output_layer(x)
        x = x.squeeze()
        return x
    

def load_model(model_path, input_size):
    model = SoCCNN1D(input_channels=input_size, hidden_channels=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

loaded_model = load_model(model_path, len(FEATURE_COLS))

#for name, param in loaded_model.named_parameters():
#    print(f"{name}: {param.shape}")
#    print(param.data)   # or param.detach().cpu().numpy() for NumPy
def test_model(model, test_loader, device):

    # 5 features in the same order as FEATURE_COLS
    in_soc = [-0.6601, 0.4437, -1.4243, 0.4374, -0.6009]  # looks like z-scores already

    x = torch.tensor(in_soc, dtype=torch.float32, device=device).view(1, len(FEATURE_COLS), 1)
    model.eval()
    with torch.no_grad():
        out_soc = model(x).item()   # scalar
    print(f"Predicted SOC: {out_soc:.6f}")

    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels, _, _ in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs)
            outputs = model(inputs)
            #print(outputs)
            test_predictions.extend(outputs.cpu().view(-1).tolist())
            test_labels.extend(labels.cpu().view(-1).tolist())

    return test_predictions, test_labels

# Evaluate the model
test_predictions, test_labels = test_model(loaded_model, test_loader, device)

# Convert predictions and labels to numpy arrays for error calculation
test_predictions_np = np.array(test_predictions)
test_labels_np = np.array(test_labels)

# Calculate MSE and MAE
mse = mean_squared_error(test_labels_np, test_predictions_np)
mae = mean_absolute_error(test_labels_np, test_predictions_np)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")

plt.figure(figsize=(8, 8))
plt.scatter(test_labels, test_predictions, alpha=0.5)
plt.xlabel('True Values [SOC]')
plt.ylabel('Predictions [SOC]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color='red') 
plt.title('Predicted SOC vs True SOC')
plt.show()