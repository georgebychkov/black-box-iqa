import cv2
import os
import csv
import json
import importlib
import time
import numpy as np
from read_dataset import iter_images, get_batch, to_numpy
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE
from frozendict import frozendict


def train_main(train_callback):
    import importlib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-train", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--save-dir", type=str, default="./")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
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
    for train_dataset, path_train in zip(args.train_dataset, args.path_train):
        uap = train_callback(model, path_train, batch_size=args.batch_size, is_fr=is_fr, jpeg_quality=args.jpeg_quality, metric_range=metric_range, device=args.device)
        cv2.imwrite(os.path.join(args.save_dir, f'{train_dataset}.png'), (uap + 0.1) * 255)
        np.save(os.path.join(args.save_dir, f'{train_dataset}.npy'), uap)
        

  
def run(model, uap, dataset_path, train_dataset, test_dataset, amplitude=[0.2], is_fr=False, jpeg_quality=None, codecs=[], batch_size=1, save_path='res.csv', device='cpu', dump_path=None, dump_freq=500):
    codecs.append('rawvideo')
    if isinstance(uap, str):
        uap = np.load(uap)
        
    time_sum = 0
    attack_num = 0
    
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'start_frame', 'end_frame', 'train_dataset', 'test_dataset', 'clear', 'attacked', 'jpeg_quality', 'codec', 'lower_better', 'amplitude', 'rel_gain', 'psnr', 'ssim', 'mse']
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
                                    'train_dataset': train_dataset,
                                    'test_dataset': test_dataset,
                                    'amplitude': config['amplitude'],
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
                                    for k in amplitude:
                                        encoders[frozendict(codec=codec, jpeg_quality=q, amplitude=k)] = Encoder(fn=f'{video_name}_{codec}_{q}_{k}.mkv', codec=codec)
                            else:
                                for k in amplitude:
                                    encoders[frozendict(codec=codec, amplitude=k)] = Encoder(fn=f'{video_name}_{codec}_{k}.mkv', codec=codec)
                        prev_path = video_path
                        prev_video_name = video_name
                        is_video = received_video
                local_i = 0
                
            if images is None:
                break
            images = np.stack(images)
            orig_images = images

            h, w = orig_images[0].shape[:2]

            uap_h, uap_w = uap.shape[0], uap.shape[1]
            uap_resized = np.tile(uap,(h // uap_h + 1, w // uap_w + 1, 1))[:h, :w, :]
            success_attack = True
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_images = compress(orig_images, q)
                    clear_metric = predict(orig_images, jpeg_images, model=model, device=device)
                    if clear_metric is None:
                        success_attack = False
                        break
                    for k in amplitude:
                        
                        t0 = time.time()
                        attacked_images = orig_images + uap_resized * k
                        attacked_images = np.clip(attacked_images, 0, 1)
                        attacked_images = compress(attacked_images, q)
                        time_sum += time.time() - t0
                        attack_num += 1
                        
                        attacked_metric = predict(orig_images, attacked_images, model=model, device=device)
                        if attacked_metric is None:
                            success_attack = False
                            break
                        for config, encoder in encoders.items():
                            if config['jpeg_quality'] == q and config['amplitude'] == k:
                                encoder.add_frames(to_numpy(attacked_images))
                        writer.writerow({
                            'image_name': f'{video_name}',
                            'start_frame': local_i if is_video else None,
                            'end_frame': (local_i + len(images)) if is_video else None,
                            'clear': clear_metric,
                            'attacked': attacked_metric,
                            'jpeg_quality': q,
                            'codec': None,
                            'lower_better': model.lower_better,
                            'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'amplitude': k,
                            'psnr' : PSNR(jpeg_images, attacked_images),
                            'ssim' : SSIM(jpeg_images, attacked_images),
                            'mse' : MSE(jpeg_images, attacked_images)
                            })
                        if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                            cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{k}_{fn}-jpeg{q}.png'), attacked_images * 255)
                    if not success_attack:
                        break
                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{fn}-jpeg{q}_orig.png'), jpeg_images * 255)            

            else:
                clear_metric = predict(orig_images, model=model, device=device)
                if clear_metric is None:
                    success_attack = False
                else:
                    for k in amplitude:
                        
                        t0 = time.time()
                        attacked_images = orig_images + uap_resized * k
                        attacked_images = np.clip(attacked_images, 0, 1)
                        time_sum += time.time() - t0
                        attack_num += 1
                        
                        attacked_metric = predict(attacked_images, model=model, device=device)
                        if attacked_metric is None:
                            success_attack = False
                            break
                        for config, encoder in encoders.items():
                            if config['amplitude'] == k:
                                encoder.add_frames(to_numpy(attacked_images))
                        writer.writerow({
                            'image_name': f'{video_name}',
                            'start_frame': local_i if is_video else None,
                            'end_frame': (local_i + len(images)) if is_video else None,
                            'clear': clear_metric,
                            'attacked': attacked_metric,
                            'jpeg_quality': None,
                            'codec': None,
                            'lower_better': model.lower_better,
                            'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'amplitude': k,
                            'psnr' : PSNR(orig_images, attacked_images),
                            'ssim' : SSIM(orig_images, attacked_images),
                            'mse' : MSE(orig_images, attacked_images)
                            })
                        if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                            cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{k}_{fn}.png'), attacked_images * 255)
            local_i += batch_size
            global_i += batch_size
            if not success_attack:
                for config, encoder in encoders.items():
                    encoder.add_frames(to_numpy(orig_images))
    if attack_num == 0:
        return None
    return time_sum / attack_num * 1000
            
    
                    
def test_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--uap-path", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--dump-path", type=str, default=None)
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument("--save-path", type=str, default='res.csv')
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--amplitude", type=float, default=[0.2], nargs='+')
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument('--video-metric', action='store_true')
    parser.add_argument("--codecs", type=str, nargs='*', default=[])
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--job-name', type=str, required=True)
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
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    for train_dataset, uap_path in zip(args.train_dataset, args.uap_path):
        for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
            mean_time = run(
                            model,
                            uap_path,
                            dataset_path,
                            train_dataset,
                            test_dataset,
                            amplitude=args.amplitude,
                            is_fr=is_fr,
                            jpeg_quality=args.jpeg_quality,
                            codecs=args.codecs,
                            batch_size=batch_size,
                            save_path=args.save_path,
                            device=args.device,
                            dump_path=args.dump_path,
                            dump_freq=args.dump_freq
                        )
            write_log(args.log_file, test_dataset, mean_time)
            
