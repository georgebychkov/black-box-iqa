from src.fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from src.fastvqa.models import DiViDeAddEvaluator
import torch

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}


def sigmoid_rescale(score, device=None, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    #print(f"Inferring with model [{model}]:")
    score = 1 / (1 + torch.exp(-x))
    return score


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        opt = {'backbone': {'fragments': {'checkpoint': False, 'pretrained': None}},
                'backbone_size': 'swin_tiny_grpb',
                'backbone_preserve_keys': 'fragments',
                'divide_head': False,
                'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}
        model = DiViDeAddEvaluator(**opt)
        model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
        model.eval()
        self.sample_args = {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 4}
        self.sample_type = 'fragments'
        self.num_clips = 4

        self.model = model.to(device)
        self.lower_better = False

    def forward(self, image, inference=False):
        torch.manual_seed(torch.initial_seed())
        image = image.to(self.device)
        image = image.permute(1, 0, 2, 3) * 255 
        if image.shape[1] == 1:
          image = torch.concatenate([image for _ in range(32)], axis=1)
        if image.shape[1] == 4:
          image = torch.concatenate([image for _ in range(8)], axis=1)
        if image.shape[1] == 8:
          image = torch.concatenate([image, image, image, image], axis=1)
        sampled_video = get_spatial_fragments(image, **self.sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]).to(self.device), torch.FloatTensor([58.395, 57.12, 57.375]).to(self.device)
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        sampled_video = sampled_video.reshape(sampled_video.shape[0], self.num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples = {}
        vsamples[self.sample_type] = sampled_video.to(self.device)
        result = self.model(vsamples) 
        score = sigmoid_rescale(result.mean(), device=self.device, model='FAST-VQA')
        return score
