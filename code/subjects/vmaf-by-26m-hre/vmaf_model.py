import os
import torch
import pickle

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path="/src/vmaf"):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=device)
        self.lower_better = False

    def forward(self, ref, dist):
        return self.model.predict(dist, ref)

