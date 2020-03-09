# ImageGCN
This repository is the official PyTorch implementation of the experiments in the following paper: 

Mao, Chengsheng, Liang Yao, and Yuan Luo. "Imagegcn: Multi-relational image graph convolutional networks for disease identification with chest x-rays." arXiv preprint arXiv:1904.00325 (2019).

## Test run


```
python run_sglayer.py --neibor relation -e res50  --gpu 0 --batch-size 16   --train-percent 0.7
```

