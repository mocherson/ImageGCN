r""""Contains definitions of the methods used by the _DataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
from torch._six import container_abcs, string_classes, int_classes
from collections import Counter
from .selfdefine import FlexCounter
import numpy as np
from heapq import nlargest
from .preprocessing import adj_norm

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):   
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))
    
def mycollate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        if 'dataset' not in  batch[0] or batch[0]['dataset'].neib_samp not in ('sampling', 'best', 'relation'):
            return {key: default_collate([d[key] for d in batch]) for key in batch[0] if key not in ['weight','impt','dataset']}
        relations = batch[0]['dataset'].tr_grp
        if batch[0]['dataset'].neib_samp == 'relation':
            nodes2 = sum([d['impt'] for d in batch],[])
        else:
            w= sum([d['weight'] for d in batch], Counter())
            [w.pop(d['index'], None)  for d in batch]         
            if batch[0]['dataset'].neib_samp == 'sampling':
                p = FlexCounter(w)/sum(w.values())
                nodes2 = np.random.choice(list(p.keys()), batch[0]['dataset'].k, replace=False, p=list(p.values()))
            elif batch[0]['dataset'].neib_samp == 'best':
                nodes2 = nlargest(batch[0]['dataset'].k, w, key = w.get) 
            
        neib_batch = [batch[0]['dataset']._getimage(x,True,1) for x in nodes2]
        [(d.pop('weight', None), d.pop('dataset', None)) for d in batch]
        batch = neib_batch + batch
        coll = default_collate(batch)
        adj_mats = {r: np.zeros((len(batch), len(batch))) for r in relations}
        for r in relations:
            for i, b1 in enumerate(coll[r]):
                for j, b2 in enumerate(coll[r]):
                    if i!=j:
                        adj_mats[r][i,j] = 1 if b1==b2 else 0
            adj_mats[r] = adj_norm(adj_mats[r])            
        coll['adj'] = adj_mats
        coll['k'] = len(nodes2)
        return coll
            
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))
