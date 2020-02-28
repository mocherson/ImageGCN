from collections import Counter
import numpy as np
import pandas as pd
import torch
import shutil


class FlexCounter(Counter):
    def __truediv__(self,a):
        return FlexCounter({key:value/a if a else 0 for key, value in self.items()})
    
    def __mul__(self,a):
        return FlexCounter({key:value*a for key, value in self.items()})
    
    def __pow__(self,a):
        return FlexCounter({key:value**a for key, value in self.items()})
    
    
class FlexDict(dict):
    def __truediv__(self,a):
        return FlexDict({key:value/a if a else 0 for key, value in self.items()})
    
    def __mul__(self,a):
        return FlexDict({key:value*a for key, value in self.items()})
    
    def __pow__(self,a):
        return FlexDict({key:value**a for key, value in self.items()})
    
    def __add__(self, a):
        if isinstance(a, dict):
            for key, value in a.items():
                self[key] = (torch.cat((self[key], value)) if isinstance(value, torch.Tensor) \
                            else np.concatenate([self[key], value]) if isinstance(value, np.ndarray)  \
                            else self[key] | value if isinstance(value, pd.Index)  \
                            else self[key]+value )  \
                            if key in self else value
            return self
        else:
            raise Error('undifined function')
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint','best'))
        





