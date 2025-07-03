import torch
import cv2
import os
import csv
import json
import importlib
import time
import numpy as np
from read_dataset import to_numpy, iter_images, get_batch
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE
from frozendict import frozendict
from functools import partial
import subprocess
import inspect
import torchvision


def apply_attack(model, attack_callback, dist_images, ref_images=None, metric_range=100, device='cpu', q=80, variable_params={}, metric_name=""):
    
    sig = inspect.signature(attack_callback)
    if 'resize' in sig.parameters:
        transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(299, 299))
                    ])
        shape = dist_images.shape
        dist_images = transforms(dist_images)
        if ref_images is not None:
            ref_images = transforms(ref_images)
    
    predict_metric = predict if ref_images is None else partial(predict, ref_images)
    clear_metric = predict_metric(compress(dist_images, q), model=model, device=device)
    if clear_metric is None:
        return None
    
    t0 = time.time()

    if ref_images is None:
        if "metric_name" in sig.parameters:
            attacked_images = attack_callback(dist_images.clone(), model=model, metric_range=metric_range, device=device, metric_name=metric_name, **variable_params)
        else:
            attacked_images = attack_callback(dist_images.clone(), model=model, metric_range=metric_range, device=device, **variable_params)
    else:
        if "metric_name" in sig.parameters:
            attacked_images = attack_callback(dist_images.clone(), ref_images.clone().detach().contiguous().to(device), model=model, metric_range=metric_range, device=device, metric_name=metric_name, q=q,
                                          **variable_params)
        else:
            attacked_images = attack_callback(dist_images.clone(), ref_images.clone().detach().contiguous().to(device), model=model, metric_range=metric_range, device=device, q=q,
                                          **variable_params)
    attack_time = time.time() - t0
    if attacked_images is None:
        return None
    attacked_metric = predict_metric(attacked_images, model=model, device=device)
    if attacked_metric is None:
        return None
    if 'resize' in sig.parameters:
        transforms = torchvision.transforms.Resize(size=(shape[-2:]))
        attacked_images = transforms(attacked_images).numpy().transpose(0, 2, 3, 1)
    return attacked_images, clear_metric, attacked_metric, attack_time
    

def run(model, dataset_path, test_dataset, attack_callback, save_path='res.csv', is_fr=False, jpeg_quality=None, codecs=[], batch_size=1, metric_range=100, device='cpu', dump_path=None, dump_freq=500, metric_name=""):
    codecs.append('rawvideo')
    time_sum = 0
    attack_num = 0
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'start_frame', 'end_frame', 'jpeg_quality', 'codec', 'test_dataset', 'clear', 'attacked', 'lower_better', 'rel_gain', 'psnr', 'ssim', 'mse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        video_iter = iter_images(dataset_path)
        prev_path = None
        prev_video_name = None
        encoders = dict()
        is_video = None
        global_i = 0
        while True:
            images, video_name, fn, video_path, video_iter, received_video = get_batch(video_iter, batch_size)
            
            if is_video is None:
                is_video = received_video
            
            if video_name != prev_video_name:
                if is_video:
                    for config, encoder in encoders.items():
                        encoder.close()
                        if config['codec'] != 'rawvideo':
                            for encoded_metric, start, end, psnr, ssim, mse in eval_encoded_video(model, encoder.fn, orig_video_path=prev_path, is_fr=is_fr, batch_size=batch_size, device=device):
                                writer.writerow({
                                    'image_name': f'{prev_video_name}',
                                    'start_frame': start,
                                    'end_frame': end,
                                    'clear': None,
                                    'attacked': encoded_metric,
                                    'jpeg_quality': config['jpeg_quality'] if 'jpeg_quality' in config else None,
                                    'codec': config['codec'],
                                    'lower_better' : model.lower_better,
                                    'rel_gain': None,
                                    'test_dataset': test_dataset,
                                    'psnr' : psnr,
                                    'ssim' : ssim,
                                    'mse' : mse
                                    })
                        os.system(f"vqmt -metr vmaf -metr bfm -orig {prev_path} -in {encoder.fn} -csv-file /artifacts/vqmt_{prev_video_name}_{config['codec']}.csv")
                        os.remove(encoder.fn)
                    encoders = dict()
                    if received_video is not None:
                        for codec in codecs:
                            if is_fr:
                                for q in jpeg_quality:
                                    encoders[frozendict(codec=codec, jpeg_quality=q)] = Encoder(fn=f'{video_name}_{codec}_{q}.mkv', codec=codec)
                            else:
                                encoders[frozendict(codec=codec)] = Encoder(fn=f'{video_name}_{codec}.mkv', codec=codec)
                        prev_path = video_path
                        prev_video_name = video_name
                        is_video = received_video
                local_i = 0
                
            if images is None:
                break
            images = np.stack(images)
            orig_images = images
            images = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
            images = images.to(device)
            success_attack = True
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_images = compress(orig_images, 100, return_torch=True).to(device)
                    attack_result = apply_attack(
                        model,
                        attack_callback,
                        jpeg_images,
                        ref_images=images,
                        metric_range=metric_range,
                        device=device,
                        q=q,
                        metric_name=metric_name
                        )
                    if attack_result is not None:
                        attacked_images, clear_metric, attacked_metric, attack_time = attack_result
                    else:
                        success_attack = False
                        break
                    time_sum += attack_time
                    attack_num += 1
                    for config, encoder in encoders.items():
                        if config['jpeg_quality'] == q:
                            encoder.add_frames(to_numpy(attacked_images))
                    writer.writerow({
                        'image_name': f'{video_name}',
                        'start_frame': local_i if is_video else None,
                        'end_frame': (local_i + len(images)) if is_video else None,
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'jpeg_quality': q,
                        'codec': None,
                        'lower_better' : model.lower_better,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'test_dataset': test_dataset,
                        'psnr' : PSNR(jpeg_images, attacked_images),
                        'ssim' : SSIM(jpeg_images, attacked_images),
                        'mse' : MSE(jpeg_images, attacked_images)
                        })

                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}-jpeg{q}_orig.png'), to_numpy(jpeg_images).squeeze(0) * 255)
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}-jpeg{q}.png'), to_numpy(attacked_images).squeeze(0) * 255)
            else:
                attack_result = apply_attack(
                    model,
                    attack_callback,
                    images.contiguous(),
                    metric_range=metric_range,
                    device=device,
                    metric_name=metric_name
                    )
                if attack_result is not None:
                    attacked_images, clear_metric, attacked_metric, attack_time = attack_result
                else:
                    success_attack = False
                if success_attack:
                    time_sum += attack_time
                    attack_num += 1
                    for config, encoder in encoders.items():
                        encoder.add_frames(to_numpy(attacked_images))
                            
                            
                    writer.writerow({
                        'image_name': f'{video_name}',
                        'start_frame': local_i if is_video else None,
                        'end_frame': (local_i + len(images)) if is_video else None,
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'jpeg_quality': None,
                        'codec': None,
                        'lower_better' : model.lower_better,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'test_dataset': test_dataset,
                        'psnr' : PSNR(orig_images, attacked_images),
                        'ssim' : SSIM(orig_images, attacked_images),
                        'mse' : MSE(orig_images, attacked_images)
                        })
    
                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}.png'), to_numpy(attacked_images).squeeze(0) * 255)
            local_i += batch_size
            global_i += batch_size
            if not success_attack:
                for config, encoder in encoders.items():
                    encoder.add_frames(to_numpy(orig_images))

    if attack_num == 0:
        return None
    return time_sum / attack_num * 1000


def test_main(attack_callback):
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
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--job-name', type=str, required=True)
    args = parser.parse_args()
    subprocess.run('pip list', shell=True, check=True)
    
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else abs(bounds_metric['high'] - bounds_metric['low'])
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    batch_size = 4 if args.video_metric else 1
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
        mean_time = run(
                        model,
                        dataset_path,
                        test_dataset,
                        attack_callback=attack_callback,
                        save_path=args.save_path,
                        is_fr=is_fr,
                        jpeg_quality=args.jpeg_quality,
                        codecs=args.codecs,
                        batch_size=batch_size,
                        metric_range=metric_range,
                        device=args.device,
                        dump_path=args.dump_path,
                        dump_freq=args.dump_freq,
                        metric_name=args.metric
                    )
        write_log(args.log_file, test_dataset, mean_time)
    