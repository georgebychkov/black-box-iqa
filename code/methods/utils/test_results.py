import pandas as pd
import numpy as np

def basic_sanity_check(csv_path):
    # print(f'sanity check {metric_name}, {attack}')
    df = pd.read_csv(csv_path)
    THRESH = 1e-9
    if pd.to_numeric(df['mse'], errors='coerce').max() < THRESH:
        print(f'[Sanity Check Failed] Max MSE < 1e-9')
        return False
    if 1.0 - pd.to_numeric(df['ssim'], errors='coerce').min()< THRESH:
        print(f'[Sanity Check Failed] 1.0 - min. SSIM < 1e-9')
        return False
    if pd.to_numeric(df['psnr'], errors='coerce').min() > 90:
        print(f'[Sanity Check Failed] Min PSNR > 90')
        return False
    if df['psnr'].isna().sum() > 0 or df['mse'].isna().sum() > 0 or df['ssim'].isna().sum() > 0:
        print(f'[Sanity Check Failed] PSNR/MSE/SSIM columns contains NaNs')
        return False
    if np.isinf(df['psnr']).sum() > 0 or np.isinf(df['mse']).sum() > 0 or np.isinf(df['ssim']).sum() > 0:
        print(f'[Sanity Check Failed] PSNR/MSE/SSIM columns contains Infs')
        return False
    if np.mean(np.abs(df['attacked'] - df['clear'])) < THRESH:
        print('[Sanity Check Failed] Clear and attacked values are indistinguishable ({THRESH})')
        return False
    return True