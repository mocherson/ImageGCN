# ImageGCN
This repository is the official PyTorch implementation of the experiments in the following paper: 

Mao, Chengsheng, Liang Yao, and Yuan Luo. "Imagegcn: Multi-relational image graph convolutional networks for disease identification with chest x-rays." IEEE Transactions on Medical Imaging 41.8 (2022): 1990-2003.

## Test run


```
python run_sglayer.py --path <data path> --neibor relation -e res50  --gpu 0 --batch-size 16   --train-percent 0.7
```

