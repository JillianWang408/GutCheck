import torch
import neurokit2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ConfidenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layer 1
        self.cnv1 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5), # play with kernel size during train
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(2),
            #nn.Dropout(0.05) # play with during train
        )

        # conv layer 2
        self.cnv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3), # play with kernel size during train
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3) # play with during train
        )

        # third conv layer - likely too little data to use
        #self.cn3 = nn.Sequential(
        #    nn.Conv1d(32, 64, kernel_size=5), # play with kernel size during train
        #    nn.LazyBatchNorm1d(),
        #    nn.ReLU(),
        #    nn.MaxPool1d(2),
        #    nn.Dropout(0.4)
        #)

        self.global_pool = nn.AdaptiveAvgPool1d(4)

        # output layer
        self.out = nn.Sequential(
            nn.Linear(32*4, 64), # switch to (64, 64) if 3 conv layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x): #Shape of x: [batch_size, channels, time], e.g., [32, 4, 256]
        x = self.cnv1(x)
        x = self.cnv2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x    

# Training function
def train(model, train_loader, val_loader=None, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

        if val_loader:
            validate(model, val_loader, criterion, device)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# to be implemented
def preprocess(df):
    coll = [
        1 #placeholder
    ]
    df['conf'] = pd.DataFrame(np.array(coll).T)

# Example dataset setup
def get_data_loaders(df):
    X = df[:256]
    y = df["conf"]  # confidence score in [0, 1]

    dataset = TensorDataset(X, y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

# Usage
if __name__ == "__main__":
    model = ConfidenceCNN()
    train_loader, val_loader = get_data_loaders()
    train(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")
