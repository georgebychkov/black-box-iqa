from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
import numpy as np
from option.config import Config
from model.model_main import IQARegression
from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
sys.path.remove(str(Path(__file__).parent))


config = Config({

    # model for PIPAL (NTIRE2021 Challenge)
    "n_enc_seq": 21*21,                 # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 21*21,                 # feature map dimension (H x W) from backbone
    "n_layer": 1,                       # number of encoder/decoder layers
    "d_hidn": 128,                      # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,                       # feed forward hidden layer dimension
    "d_MLP_head": 128,                  # hidden layer of final MLP 
    "n_head": 4,                        # number of head (in multi-head attention)
    "d_head": 128,                      # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,                     # dropout ratio of transformer
    "emb_dropout": 0.1,                 # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,                      # dimension of final prediction
    "crop_size": 192,                   # input image crop size


    # ensemble in test
    "test_ensemble": True,
    "n_ensemble": 20
})

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path, backbone_path):
        super().__init__()
        self.device = device
        
        self.model_transformer = IQARegression(config)
        self.model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background', weights_path=backbone_path)
        
        self.save_output = SaveOutput()
        self.hook_handles = []
        for layer in self.model_backbone.modules():
            if isinstance(layer, Mixed_5b):
                handle = layer.register_forward_hook(self.save_output)
                self.hook_handles.append(handle)
            elif isinstance(layer, Block35):
                handle = layer.register_forward_hook(self.save_output)
                self.hook_handles.append(handle)

        self.model_backbone.eval().to(device)
        
        self.model_transformer.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model_transformer.eval().to(device)
        
        self.lower_better = False

    
    def forward(self, ref, dist):
        
        ref = (ref - 0.5) / 0.5
        dist = (dist - 0.5) / 0.5
        
        enc_inputs = torch.ones(1, config.n_enc_seq + 1).to(self.device)
        dec_inputs = torch.ones(1, config.n_dec_seq + 1).to(self.device)
        
        score = 0
        for i in range(config.n_ensemble):
            n, c, h, w = ref.shape
            new_h = config.crop_size
            new_w = config.crop_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            
            r_img_crop = ref[..., top: top+new_h, left: left+new_w]
            d_img_crop = dist[..., top: top+new_h, left: left+new_w]
            
            # backbone feature map (ref)
            x_ref = self.model_backbone(r_img_crop)
            feat_ref = torch.cat(
                (self.save_output.outputs[0],
                self.save_output.outputs[2],
                self.save_output.outputs[4],
                self.save_output.outputs[6],
                self.save_output.outputs[8],
                self.save_output.outputs[10]),
                dim=1
            ) # feat_ref: n_batch x (320*6) x 21 x 21
            # clear list (for saving feature map of d_img)
            self.save_output.outputs.clear()
            
            # backbone feature map (dis)
            x_dis = self.model_backbone(d_img_crop)
            feat_dis = torch.cat(
                (self.save_output.outputs[0],
                self.save_output.outputs[2],
                self.save_output.outputs[4],
                self.save_output.outputs[6],
                self.save_output.outputs[8],
                self.save_output.outputs[10]),
                dim=1
            ) # feat_ref: n_batch x (320*6) x 21 x 21
            # clear list (for saving feature map of r_img in next iteration)
            self.save_output.outputs.clear()
            
            feat_diff = feat_ref - feat_dis
            enc_inputs_embed = feat_diff
            dec_inputs_embed = feat_ref
            score += self.model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
                
        score /= config.n_ensemble
        
        return score
        
        