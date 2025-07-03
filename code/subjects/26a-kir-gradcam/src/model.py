import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torchvision
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision.models._api import WeightsEnum
from torchvision.transforms import functional as TF
from torchvision import transforms

EPS = 1e-8


class EvalMode:
    def __init__(self, model: nn.Module) -> None:
        self.old_mode = model.training
        self.model = model

    def __enter__(self):
        self.model.eval()

    def __exit__(self, *args):
        self.model.train(self.old_mode)


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


@dataclass
class VanillaOutput:
    pred_score: Tensor
    gradcam: Optional[Tensor] = None
    pred_saliency: Optional[Tensor] = None

    def detach(self):
        """Inplace detach all attributes"""
        self.pred_score = self.pred_score.detach()
        if self.gradcam is not None:
            self.gradcam = self.gradcam.detach()
        if self.pred_saliency is not None:
            self.pred_saliency = self.pred_saliency.detach()

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))


class VanillaArch(nn.Module):
    def __init__(
        self,
        backbone_type: str = "efficientnet_b0",
        backbone_weights: str = "imagenet",
        saliency_mode: str = "none",
        fusion: str = "none",
        gradcam_mode: str = "draw",
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
        freeze_backbone: bool = False,
        calculate_gradcam_in_eval_mode: bool = False,
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.backbone_weights = backbone_weights
        self.saliency_mode = saliency_mode
        self.fusion = fusion
        self.gradcam_mode = gradcam_mode
        self.calculate_gradcam_in_eval_mode = calculate_gradcam_in_eval_mode

        self._setup_backbone()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels_size, 1024),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        if self.saliency_mode == "output":
            self.saliency_subnet = nn.Sequential(
                nn.Conv2d(
                    self.channels_size,
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(),
            )
        if self.saliency_mode in ["input", "output"]:
            if self.fusion == "multiple_concat":
                len_backbone = len(self.depths)

                concat_convs = [
                    nn.Conv2d(self.depths[i] + 1, self.depths[i], 1, 1, 0)
                    if self.depths[i] is not None
                    else nn.Identity()
                    for i in range(len_backbone)
                ]

                for i in range(len(self.depths)):
                    size = self.depths[i]
                    if size:
                        concat_convs[i].weight.data = (
                            torch.cat([torch.eye(size), torch.zeros(size, 1)], dim=1)
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                        )
                        concat_convs[i].bias.data = torch.zeros(size)

                self.concat_convs = nn.Sequential(*concat_convs)
            elif self.fusion in ["multiply", "multiply_weighted"]:
                self.saliency_linop = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            elif self.fusion == "small_adapter":
                self.adapter = nn.Sequential(nn.Conv2d(3 + 1, 32, 7, 2, 3))
                self.adapter[0].weight.data *= 0
            elif self.fusion == "big_adapter":
                self.adapter = nn.Sequential(
                    nn.Conv2d(3 + 1, 32, 7, 2, 3, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                )
            elif self.fusion == "ritm_like":
                self.adapter = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    ScaleLayer(init_value=0.05, lr_mult=1),
                )

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def _setup_backbone(self):
        def get_state_dict(self, *args, **kwargs):
            #kwargs.pop("check_hash")
            return load_state_dict_from_url(self.url, *args, **kwargs)

        WeightsEnum.get_state_dict = get_state_dict

        # checks
        assert self.backbone_type in ["resnet50", "efficientnet_b0"]

        allowed_weights = ["none", "imagenet"]
        if self.backbone_type == "resnet50":
            allowed_weights.append("transalnet")
        assert self.backbone_weights in allowed_weights

        # init backbone and load weights
        if self.backbone_type == "resnet50":
            backbone = nn.Sequential(
                *list(
                    torchvision.models.resnet50(
                        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                        if self.backbone_weights == "imagenet"
                        else None
                    ).children()
                )[:-2]
            )
            if self.backbone_weights == "transalnet":
                print("Using TranSalNet checkpoint")
                backbone.load_state_dict(
                    torch.load("pretrained_models/resnet50_transalnet.pth")
                )

            self.channels_size = 2048
            self.depths = [3, 64, None, None, None, 256, 512, 1024, 2048]
        elif self.backbone_type == "efficientnet_b0":
            backbone = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if self.backbone_weights == "imagenet"
                else None
            ).features
            self.channels_size = 1280
            self.depths = [3, 32, 16, 24, 40, 80, 112, 192, 320, 1280]

        self.backbone = nn.Sequential(nn.Identity(), *backbone)

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    # ========== TRAINING AND INFERENCE LOGIC ==========

    def _fuse(self, x: Tensor, sal: Tensor) -> Tensor:
        sal = self.saliency_linop(sal)
        if self.fusion == "multiply":
            sal = TF.resize(sal, x.shape[2:])
            x = x * sal
        else:
            alpha = 1e-4
            x = (x * (alpha + sal)) / (alpha + sal.sum())
        return x

    def _concat_saliency(self, x: Tensor, sal: Tensor, i: int) -> Tensor:
        if not isinstance(self.concat_convs[i], nn.Identity):
            shape = x.shape[2:]
            sal = TF.resize(sal, shape)
            x = torch.cat([x, sal], dim=1)
            x = self.concat_convs[i](x)
        return x

    def build_gradcam(self, image, saliency, features, y) -> Tensor:
        mode_before = self.training
        with EvalMode(self) if self.calculate_gradcam_in_eval_mode else nullcontext():
            with torch.set_grad_enabled(True):
                mode_after = self.training
                if mode_before != mode_after:
                    features, pred_sal = self.extact_features(image, saliency)
                    y = self.head(features)

                grads = []
                for i in range(len(y)):
                    grads.append(
                        torch.autograd.grad(y[i], features, retain_graph=True)[0][i]
                    )
                grads = torch.stack(grads)
                w = grads.mean(axis=(-2, -1))
                cam = features * w.reshape(w.size(0), w.size(1), 1, 1)
                cam = cam.mean(axis=1, keepdim=True)
                cam = torch.where(cam > 0, cam, 0)
                min_for_sample = cam.amin(axis=(1, 2, 3), keepdim=True)
                max_for_sample = cam.amax(axis=(1, 2, 3), keepdim=True)
                cam = (cam - min_for_sample) / (max_for_sample - min_for_sample + EPS)
                if cam.isnan().any():
                    print("Found NAN in gradcam")
        return cam

    def extact_features(self, image, saliency=None) -> Tuple[Tensor, Tensor]:
        image = self.preprocess(image)
        pred_sal = None

        if self.saliency_mode == "none":
            x = self.backbone(image)
        elif self.saliency_mode == "input":
            if self.fusion in ["multiply", "multiply_weighted"]:
                x = self.backbone(image)
                x = self._fuse(x, saliency)
            elif self.fusion == "multiple_concat":
                x = image
                for i, layer in enumerate(self.backbone):
                    x = layer(x)
                    x = self._concat_saliency(x, saliency, i)
            elif self.fusion in ["big_adapter", "small_adapter"]:
                stacked = torch.cat([image, saliency], dim=1)
                x = self.backbone[:2](image) + self.adapter(stacked)
                x = self.backbone[2:](x)
            elif self.fusion == "ritm_like":
                x = self.backbone[:2](image) + self.adapter(saliency)
                x = self.backbone[2:](x)
        elif self.saliency_mode == "output":
            x = self.backbone(image)
            pred_sal = self.saliency_subnet(x)
            if self.fusion in ["multiply", "multiply_weighted"]:
                x = self._fuse(x, pred_sal)

        return x, pred_sal

    def forward(self, image, saliency=None) -> VanillaOutput:
        features, pred_sal = self.extact_features(image, saliency)
        y = self.head(features)
        output = VanillaOutput(pred_score=y)

        if self.saliency_mode == "output":
            output.pred_saliency = pred_sal

        if self.gradcam_mode != "none":
            gradcam = self.build_gradcam(image, saliency, features, y)
            output.gradcam = gradcam

        return output

    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def get_parameters_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device

        model = VanillaArch(
            saliency_mode="none",
            fusion="none",
            gradcam_mode="none",
            calculate_gradcam_in_eval_mode=False
        )
        model.load_state_dict(torch.load(model_path, map_location=device)['params'])
        model.eval().to(device)
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        out = self.model(transforms.Resize([384, 512])(image)).pred_score
        
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out