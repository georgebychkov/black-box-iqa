# cd ../IQA-PyTorch
unix2dos ../ckdn/patches/ckdn.patch && patch -Np1 < ../ckdn/patches/ckdn.patch
python setup.py develop
