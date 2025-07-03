# cd ../IQA-PyTorch
unix2dos ../ahiq/patches/ahiq.patch && patch -Np1 < ../ahiq/patches/ahiq.patch
python setup.py develop
