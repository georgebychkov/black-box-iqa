#!/usr/bin/env python

import torch
from torch import nn
import time
from copy import deepcopy


NUM_EPOCHS = 100

# Small networks don't benefit from gpu acceleration,
# so I'll leave it like this even though you may have gpu. Feel free to change this line
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(16, 150, 1, batch_first=True)
        self.lin1 = nn.Linear(150, 50)
        self.lin2 = nn.Linear(50, 1)
        self.lin3 = nn.Linear(10, 1)
        self.bn = nn.BatchNorm1d(100)
        self.to(device)
        self.float()
        self.best_params = None

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)

        return x.squeeze(-1)

    def predict_on_frames(self, scaled_data):
        tt = torch.tensor(scaled_data).to(device).float()
        result = self(tt)
        return result.cpu().detach().numpy()

    def fit(self, train_video_dataset, test_video_dataset):
        from torch.utils.data import Dataset, DataLoader

        class RNNDataset(Dataset):
            def __init__(self, video_dataset):
                super(RNNDataset).__init__()
                self.X = []
                self.all_features = []
                self.y = []
                self.labels = []
                for X, y, is_clean in video_dataset.data:
                    for i in range(0, len(X), 10):
                        self.X.append(X[i: i + 10])
                        self.y.append(y[i: i + 10])
                        self.labels.append(is_clean)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.labels[idx]

        batch_size = 64
        features = 16
        train_dataloader = DataLoader(
            RNNDataset(train_video_dataset), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(
            RNNDataset(test_video_dataset), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), 0.001)
        best_test_loss = float("inf")
        for epoch in range(NUM_EPOCHS):
            self.train()
            total_loss = 0
            total_count = 0
            for X, y, labels in train_dataloader:
                initial_shape = X.shape
                X = X.to(device).float()
                y = y.to(device).float()
                labels = labels.to(device).unsqueeze(1)
                output = self(X)
                loss = torch.abs(output - y).sum()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                total_count += y.size(0) * y.size(1)
                optimizer.step()
            avg_train_loss = total_loss / total_count
            print("Epoch [%d], train loss: [%5f]" %(epoch, avg_train_loss))
            self.eval()
            total_loss = 0
            total_count = 0
            with torch.no_grad():
                for X, y, labels in test_dataloader:
                    initial_shape = X.shape
                    X = X.to(device).float()
                    y = y.to(device).float()
                    labels = labels.to(device).unsqueeze(1)
                    output = self(X)
                    loss = torch.abs(output - y).sum()
                    total_loss += loss.item()
                    total_count += y.size(0) * y.size(1)
                avg_test_loss = total_loss / total_count
                print("Epoch [%d], test loss: [%5f]" %
                      (epoch, avg_test_loss))
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    self.best_params = deepcopy(self.state_dict())
        self.load_state_dict(self.best_params)
        print("best test loss: %5f"%best_test_loss)

