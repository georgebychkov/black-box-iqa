import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
import torch
from torchvision.utils import save_image


def load_from_csv(key, header_len):
    df = pd.read_csv(key)[header_len:].to_numpy(float)[:, 1:]
    return df

def load_video(orig, dist, storage_path: str = "/storage/", use_cached=True):
    import hashlib
    global counter
    use_cached = False
    orig_path = os.path.join(os.getcwd(), str(hashlib.sha256(f'orig{load_video.counter}'.encode()).hexdigest()) + ".png")
    save_image(orig, orig_path)
    dist_path = os.path.join(os.getcwd(), str(hashlib.sha256(f'dist{load_video.counter}'.encode()).hexdigest()) + ".png")
    save_image(dist, dist_path)    
    orig = os.path.abspath(orig_path)
    dist = os.path.abspath(dist_path)
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
    if not os.path.exists(orig):
        raise FileNotFoundError("orig : " + orig + " doesn't exist")
    if not os.path.exists(dist):
        raise FileNotFoundError("dist : " + dist + " doesn't exist")
    key1 = storage_path + (orig + "_" + dist).replace(os.sep, "_") + ".psnr.csv"
    key2 = storage_path + (orig + "_" + dist).replace(os.sep, "_") + ".csv"
    key1 = os.path.join(storage_path, str(hashlib.sha256(key1.encode()).hexdigest()) + ".csv")
    key2 = os.path.join(storage_path + str(hashlib.sha256(key2.encode()).hexdigest()) + ".csv")
    if not os.path.exists(key1) or not use_cached:
        os.system("vqmt -metr psnr over Y 1920x1080 YUV420P" +
                  " -csv-file %s -resize to 1920x1080 -orig %s -in %s  -threads 0" % (key1, orig, dist))

    if not os.path.exists(key2) or not use_cached:
        os.system("vqmt -metr vmaf over Y -set model_preset=standard_features" +
                  " -set disable_clip=true -metr ssim over Y 1920x1080 YUV420P" +
                  " -csv-file %s -resize to 1920x1080 -orig %s -in %s -threads 0" % (key2, orig, dist))

    val = np.concatenate(
        [load_from_csv(key1, 13), load_from_csv(key2, 12)], axis=1)
    os.remove(key1)
    os.remove(key2)
    os.remove(orig_path)
    os.remove(dist_path)
    load_video.counter += 1
    return val

load_video.counter = 0