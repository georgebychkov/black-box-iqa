
# ShiftTolerant-LPIPS

**Shift-tolerant Perceptual Similarity Metric**

[Abhijay Ghildyal](https://abhijay9.github.io/), [Feng Liu](http://web.cecs.pdx.edu/~fliu/). In ECCV, 2022. [[Arxiv]](https://arxiv.org/abs/2207.13686)

<img src="https://abhijay9.github.io/images/stlpips_teaser.gif" width=300>

```python
from stlpips_pytorch import stlpips
from stlpips_pytorch import utils

path0 = "<dir>/ShiftTolerant-LPIPS/imgs/ex_p0.png"
path1 = "<dir>/ShiftTolerant-LPIPS/imgs/ex_ref.png"

img0 = utils.im2tensor(utils.load_image(path0))
img1 = utils.im2tensor(utils.load_image(path1))

stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")

stlpips_metric(img0,img1)
# 0.7777554988861084
```

## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{ghildyal2022stlpips,
  title={Shift-tolerant Perceptual Similarity Metric},
  author={Ghildyal, Abhijay and Liu, Feng},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## Acknowledgements
This repository borrows from [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [Anti-aliasedCNNs](https://github.com/adobe/antialiased-cnns), and [CNNsWithoutBorders](https://github.com/oskyhn/CNNs-Without-Borders). We thank the authors of these repositories for their incredible work and inspiration.
