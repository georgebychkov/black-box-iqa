import os
import torch
import pickle
from torchvision.utils import save_image
import hashlib
import pandas as pd
import shlex
import subprocess

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path="/src/vmaf"):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lower_better = False
        self.full_reference = True

    def forward(self, ref, dist):
        orig_path = os.path.join(os.getcwd(), "orig.png")
        save_image(ref, orig_path)
        dist_path = os.path.join(os.getcwd(), "dist.png")
        save_image(dist, dist_path)

        vqmt_opt = "vqmt -quiet -metr vmaf over Y -set disable_clip=true -set model_preset=vmaf_v061_neg" + \
            " -csv-file %s -orig %s -in %s  -threads 0" % ('res.csv', orig_path, dist_path)
        
        subprocess.run(shlex.split(vqmt_opt))
        
        res = float(pd.read_csv('res.csv')['Netflix VMAF_VMAF061_float_neg'].iloc[-1])
        return torch.tensor(res)

