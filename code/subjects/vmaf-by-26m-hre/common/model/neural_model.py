#!/usr/bin/env python

import torch
from torch import nn
import time
from copy import deepcopy


NUM_EPOCHS = 30

# Small networks don't benefit from gpu acceleration,
# so I'll leave it like this even though you may have gpu. Feel free to change this line
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(output, clean, attacked):
    diff = torch.relu(attacked - clean)
    target = clean - diff * 0.5
    return torch.abs(output - target).sum()

class NeuralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(16, 200)
        self.lin2 = nn.Linear(200, 80)
        self.lin3 = nn.Linear(80, 1)
        self.to(device)
        self.float()
        self.best_params = None
        self.dump_path = "torch_models/torch_%d.pt" % (time.time())

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        return self.lin3(x)

    def predict_on_frames(self, scaled_data):
        tt = torch.tensor(scaled_data).to(device).float()
        result = self(tt)
        return result

    def fit(self, train_features, test_features, verbose=True):
        from torch.utils.data import Dataset, DataLoader

        class NeuralDataset(Dataset):
            def __init__(self, videos_list):
                super(NeuralDataset).__init__()
                self.X = []
                self.clean = []
                self.attacked = []
                for X, clean, attacked in videos_list:
                    for i in range(0, len(X)):
                        self.X.append(X[i])
                        self.attacked.append(attacked[i])
                        self.clean.append(clean[i])

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.clean[idx],  self.attacked[idx]

        train_dataloader = DataLoader(
            NeuralDataset(train_features), batch_size=32, shuffle=True)
        test_dataloader = DataLoader(
            NeuralDataset(test_features), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), 0.001)
        best_test_loss = float("inf")
        for epoch in range(NUM_EPOCHS):
            self.train()
            total_loss = 0
            total_count = 0
            for X, clean, attacked in train_dataloader:
                X = X.to(device).float()
                clean = torch.unsqueeze(clean.to(device).float(), 1)
                attacked = torch.unsqueeze(attacked.to(device).float(), 1)
                output = self(X)
                loss = loss_fn(output, clean, attacked)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                total_count += clean.size(0)
                optimizer.step()
            avg_train_loss = total_loss / total_count
            if verbose:
                print("Epoch [%d], train loss: [%5f]" %(epoch, avg_train_loss))
            self.eval()
            total_loss = 0
            total_count = 0
            with torch.no_grad():
                for X, clean, attacked in test_dataloader:
                    X = X.to(device).float()
                    clean = torch.unsqueeze(clean.to(device).float(), 1)
                    attacked = torch.unsqueeze(attacked.to(device).float(), 1)
                    output = self(X)
                    loss = loss_fn(output, clean, attacked)
                    total_loss += loss.item()
                    total_count += clean.size(0)
            avg_test_loss = total_loss / total_count
            if verbose:
                print("Epoch [%d], test loss: [%5f]" % (epoch, avg_test_loss))
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                self.best_params = deepcopy(self.state_dict())
        self.load_state_dict(self.best_params)
        if verbose:
            print("best test loss: %5f"%best_test_loss)


                