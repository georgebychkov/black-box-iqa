import os
import traceback
import sys
sys.path.append('..')

import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable

DEVICE = os.environ.get("DEVICE", "cuda:0")
print("DEVICE=" + DEVICE)

def tensorinfo(arr):
    return f"type={type(arr)} dtype={arr.dtype}, shape={arr.shape}, min={arr.min()}, max={arr.max()}"

def adv(ref_im, dist_im, metric_fn, lower_better, full_reference, eps=10/255, alpha=1/255, iters=10):
    ref = transforms.ToTensor()(ref_im).float().unsqueeze_(0).to(DEVICE)
    dist = transforms.ToTensor()(dist_im).float().unsqueeze_(0).to(DEVICE)
    p = torch.zeros_like(ref).to(DEVICE)
    p = Variable(p, requires_grad=True)

    for i in range(iters):
        res = ref + p
        res.data.clamp_(0., 1.)
        if full_reference:
            score = metric_fn(res, dist).mean()
        else:
            score = metric_fn(res).mean()
        if i == 0 or i == iters-1:
            print("score:", score.item())
        if not lower_better:
            loss = -score
        else:
            loss = score
        loss.backward()
        g = p.grad
        g = torch.sign(g)
        p.data -= alpha * g
        p.data.clamp_(-eps, +eps)
        p.grad.zero_()
    
    return score.item()

def test(model):
    image_dir = "../../methods/framework-25e_shu/test_ims/"
    images = sorted(os.listdir(image_dir))
    ref_im = cv2.cvtColor(cv2.imread(image_dir+images[0]), cv2.COLOR_BGR2RGB)

    cv2.imwrite("compr_0.jpg", ref_im, [cv2.IMWRITE_JPEG_QUALITY, 20])
    dist_im = cv2.cvtColor(cv2.imread('compr_0.jpg'), cv2.COLOR_BGR2RGB)
    os.remove("compr_0.jpg")
    
    print("I-FGSM run 1")
    adv(ref_im, dist_im, model, model.lower_better, model.full_reference)
    print("I-FGSM run 2")
    adv(ref_im, dist_im, model, model.lower_better, model.full_reference)
    
    cv2.imwrite("compr_0.jpg", ref_im, [cv2.IMWRITE_JPEG_QUALITY, 10])
    dist2_im = cv2.cvtColor(cv2.imread('compr_0.jpg'), cv2.COLOR_BGR2RGB)
    os.remove("compr_0.jpg")
    
    t = [transforms.ToTensor()(im).float().unsqueeze(0).to(DEVICE) for im in [ref_im, dist_im, dist2_im]]
    try:
        if model.full_reference:
            out1 = model(t[0], t[1])
            out2 = model(t[1], t[2])
        else:
            out1 = model(t[0])
            out2 = model(t[1])
        out = torch.stack([out1, out2, (out1 + out2) / 2]).detach().cpu().numpy()
        print("No batching:", out[0], out[1], "average:", out[2])
        if model.full_reference:
            out = model( torch.cat([t[0], t[1]]), torch.cat([t[1], t[2]]) )
        else:
            out = model( torch.cat([t[0], t[1]]) )
        print("Batching:", out.detach().cpu().numpy())
    except Exception as error:
        print("Batch error")
        traceback.print_exc()
    
def main():
    from src.fsim_model import MetricModel
    print("*** FSIM")
    test(MetricModel(DEVICE))
    
    from src.cw_ssim_model import MetricModel
    print("*** CW-SSIM")
    test(MetricModel(DEVICE))
    
    from src.vif_model import MetricModel
    print("*** VIF")
    test(MetricModel(DEVICE))
    
    from src.gmsd_model import MetricModel
    print("*** GMSD")
    test(MetricModel(DEVICE))
    
    from src.nlpd_model import MetricModel
    print("*** NLPD")
    test(MetricModel(DEVICE))
    
    from src.vsi_model import MetricModel
    print("*** VSI")
    test(MetricModel(DEVICE))
    
    from src.mad_model import MetricModel
    print("*** MAD")
    test(MetricModel(DEVICE))
    
    from src.musiq_model import MetricModel
    print("*** MUSIQ")
    test(MetricModel(DEVICE))
    
    from src.dbcnn_model import MetricModel
    print("*** DBCNN")
    test(MetricModel(DEVICE))
    
    from src.brisque_model import MetricModel
    print("*** BRISQUE")
    test(MetricModel(DEVICE))
    
    from src.pieapp_model import MetricModel
    print("*** PIEAPP")
    test(MetricModel(DEVICE))
    
    from src.niqe_model import MetricModel
    print("*** NIQE")
    test(MetricModel(DEVICE))
    
    from src.ilniqe_model import MetricModel
    print("*** ILNIQE")
    test(MetricModel(DEVICE))
    
    
    
    from paq2piq.src.paq2piq_standalone import MetricModel
    print("*** PAQ2PIQ")
    test(MetricModel(DEVICE))
    
    from spaq.src.model import MetricModel
    print("*** SPAQ")
    test(MetricModel(DEVICE))
    
    from unique.src.model import MetricModel
    print("*** UNIQUE")
    test(MetricModel(DEVICE))
    

if __name__ == "__main__":
   main()
