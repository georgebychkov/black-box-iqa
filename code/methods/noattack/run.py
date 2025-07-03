import torch

from evaluate import predict, compress
from read_dataset import iter_images, get_batch
import json
import csv
import importlib
import math
import numpy as np
from torchvision import transforms
from itertools import islice


def get_noise(shape):
    x = torch.rand(shape)
    return x

def is_valid_number(x):
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))



def run(model, dataset_path, test_dataset, metric_name, batch_size=1, save_path='res.csv', is_fr=False, device='cpu'):
    
    if model.lower_better:
        boundary_bad_metric = float('-inf')
        boundary_good_metric = float('+inf')
        good_reduce = min
        bad_reduce = max
    else:
        boundary_bad_metric = float('+inf')
        boundary_good_metric = float('-inf')
        good_reduce = max
        bad_reduce = min
    jpeg_quality = 98
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['metric_name', 'test_dataset', 'boundary_bad_metric', 'boundary_good_metric']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        video_iter = iter_images(dataset_path)
        while True:
            images, video_name, fn, path, video_iter, is_video = get_batch(video_iter, batch_size)
            if images is None:
                break
            images = np.stack(images)
            orig_images = images
            images = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
            images = images.to(device)
        

            if is_fr:    
                try:
                    good_metric = predict(images, images, model=model, device=device)
                except torch._C._LinAlgError: 
                    good_metric = None
                if not is_valid_number(good_metric):
                    changed_images = compress(orig_images, jpeg_quality, return_torch=True)
                    good_metric = predict(images, changed_images, model=model, device=device)
                boundary_good_metric = good_reduce(boundary_good_metric, good_metric)

                noise = get_noise(images.shape)
                boundary_bad_metric = bad_reduce(boundary_bad_metric, predict(images.contiguous(), noise, model=model, device=device))
                boundary_bad_metric = bad_reduce(boundary_bad_metric, predict(images, compress(orig_images, 10, return_torch=True).contiguous(), model=model, device=device))
                


            else:
                w = images.shape[3]
                images = transforms.Resize(w // 2)(images)
                boundary_good_metric = good_reduce(boundary_good_metric, predict(images.contiguous(), model=model, device=device))
                
                noise = get_noise(images.shape)
                boundary_bad_metric = bad_reduce(boundary_bad_metric, predict(noise, model=model, device=device))
                boundary_bad_metric = bad_reduce(boundary_bad_metric, predict(compress(orig_images, 10, return_torch=True).contiguous(), model=model, device=device))

        writer.writerow({
            'metric_name': metric_name,
            'test_dataset': test_dataset,
            'boundary_bad_metric' : boundary_bad_metric,
            'boundary_good_metric' : boundary_good_metric
            })


def test_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--dump-path", type=str, default=None)
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--save-path", type=str, default='res.csv')
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument('--video-metric', action='store_true')
    parser.add_argument("--codecs", type=str, nargs='*', default=[])
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    batch_size = 4 if args.video_metric else 1
    for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
        run(
            model,
            dataset_path,
            test_dataset,
            args.metric,
            save_path=args.save_path,
            is_fr=is_fr,
            device=args.device,
            batch_size=batch_size
        )
        
        
        
if __name__ == "__main__":
    test_main()

