import collections

import numpy as np

import torch
from torch import nn
import torchvision
from torchvision import transforms

def plcc(x, y):
    """
    Pearson’s linear correlation coefficient.
    :param x: score vector 1, with n raw scores.
    :param y: score vector 2, with n raw scores.
    :return: PLCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    x = x - np.average(x)
    y = y - np.average(y)
    numerator = np.dot(x, y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    return numerator / denominator


def srocc(x, y):
    """
    Spearman's rank order correlation coefficient.
    :param x: score vector 1, with n raw scores.
    :param y: score vector 2, with n raw scores.
    :return: SROCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    rank_x = x.argsort().argsort()
    rank_y = y.argsort().argsort()
    return plcc(rank_x, rank_y)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        model = torchvision.models.vgg16(pretrained=False, num_classes=1)
        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    [
                        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
                    ],
                    model.features
                )
            )
        )

        self.classifier = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    ['fc6_m', 'relu6_m', 'drop6_m', 'fc7_m', 'relu7_m', 'drop7_m', 'fc8_m'],
                    model.classifier
                )
            )
        )

    def load_model(self, file, debug: bool = False):
        """
        Load model file.
        :param file: the model file to load.
        :param debug: indicate if output the debug info.
        """
        state_dict = torch.load(file)

        dict_to_load = dict()
        for k, v in state_dict.items():  # "v" is parameter and "k" is its name
            for l, p in self.named_parameters():  # "p" is parameter and "l" is its name
                # use parameter's name to match state_dict's params and model's params
                split_k, split_l = k.split('.'), l.split('.')
                if (split_k[0] in split_l[1]) and (split_k[1] == split_l[2]):
                    dict_to_load[l] = torch.from_numpy(np.array(v)).view_as(p)
                    if debug:  # output debug info
                        print(f"match: {split_k} and {split_l}.")

        self.load_state_dict(dict_to_load)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)  # dont use adaptive avg pooling
        out = self.classifier(out)
        return out



class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = Vgg16()
        model.load_model(model_path)
        model.eval().to(device)
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        # transforms.Compose doesn't accept torch tensors
        out = self.model(
            transforms.Resize([224, 224])(image)
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        
