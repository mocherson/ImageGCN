import numpy as np
import scipy.sparse as sp
import torch
from collections import Counter
# from selfdefine import *
# from collate import default_collate


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_to_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(coords, values, shape)

def to_sparse(x):
    return x if x.is_sparse else x.to_sparse()

def to_dense(x):
    return x if not x.is_sparse else x.to_dense()

def str2value(code):
    exec('global i; i = %s' % code)
    global i
    return i

def issymmetric(mat):
    if torch.is_tensor(mat):
        mat = mat.to_dense().cpu().numpy() if mat.is_sparse else mat.cpu().numpy()
    if mat.shape!=mat.T.shape:
        return False
    if sp.issparse(mat):
        return not (mat!=mat.T).todense().any()
    else:
        return not (mat!=mat.T).any()
    
def adj_norm(adj, issym=True):
  
    adj = sp.csc_matrix(adj)
    if issym:
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        with np.errstate(divide='ignore'):
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv)
    else:
        rowsum = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.nan_to_num(np.power(rowsum, -1)).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj)
        
    return adj_normalized

def maxminnorm(df):
    min=df.min(axis=0)
    max=df.max(axis=0)
    return (df-min)/(max-min)

def adj_from_series(s, groups=True, norm=True):
    if groups:
        return s.groupby(s).groups
    else:
        grps = s.groupby(s).indices
        adj = np.zeros((len(s), len(s)))
        for grp in grps.values():
            adj[np.ix_(grp,grp)]=1/len(grp)

        return adj
    
# def sample_adj(k, relations, dataset, data):
#     # print('importance sampling')
#     w = Counter()
#     for r, grp in relations.items():
#         w = sum([(FlexCounter(grp[i])/(len(grp[i])-1))**2 for i in (data[r] if isinstance(data[r], list) else data[r].tolist() )], w)
#     [w.pop(x,None) for x in data['index'].tolist()]    
#     p = FlexCounter(w)/sum(w.values())
#     np.random.seed(0)
#     nodes2 = np.random.choice(list(p.keys()), k, replace=False, p=list(p.values()))
#     dataset.getbyindex=True
#     batch2 = [dataset[i] for i in nodes2]    
#     dataset.getbyindex=False
#     adj_mats = {r: np.zeros((len(data['index']), len(nodes2))) for r in relations}
#     for r, grp in relations.items():
#         for i, dr in enumerate(data[r] if isinstance(data[r], list) else data[r].tolist() ):
#             for j, b in enumerate(batch2):
#                 adj_mats[r][i,j] = 1/((len(grp[dr])-1)*p[b['index']]*k) if b[r]==dr else 0
#     return adj_mats, default_collate(batch2)

    
def sample_adj(k, relations,  dataset, data):
  
    w = dataset.impt.loc[data['index'].tolist()].sum(0) 
    w = w[w.index.difference(data['index'].tolist())]
    p = w/w.sum()
    np.random.seed(0)
    nodes2 = np.random.choice(list(p.index), k, replace=False, p=list(p))
    nodes = nodes2.tolist()+data['index'].tolist()
    dataset.getbyindex=True
    batch2 = [dataset[i] for i in nodes]    
    dataset.getbyindex=False
    adj_mats = {r: np.zeros((len(nodes), len(nodes))) for r in relations}
    for r in relations:
        for i, b1 in enumerate(batch2):
            for j, b2 in enumerate(batch2):
                if i!=j:
                    adj_mats[r][i,j] = 1 if b1[r]==b2[r] else 0
        adj_mats[r] = adj_norm(adj_mats[r])
    return adj_mats, default_collate(batch2)

def adj_imp(k, relations, dataset, data):

    w = dataset.impt.loc[data['index'].tolist()].sum(0)  
    w = w[w.index.difference(data['index'].tolist())]
    nodes2 = w.nlargest(k).index
    nodes = nodes2.tolist()+data['index'].tolist()
    dataset.getbyindex=True
    batch2 = [dataset[i] for i in nodes]    
    dataset.getbyindex=False
    adj_mats = {r: np.zeros((len(nodes), len(nodes))) for r in relations}
    for r in relations:
        for i, b1 in enumerate(batch2):
            for j, b2 in enumerate(batch2):
                if i!=j:
                    adj_mats[r][i,j] = 1 if b1[r]==b2[r] else 0
        adj_mats[r] = adj_norm(adj_mats[r])
    return adj_mats, default_collate(batch2)

def adj_rdsel(relations, dataset, data):
    nodes2=[]
    for r, grp in relations.items():
        for i, idx in zip(data[r] if isinstance(data[r], list) else data[r].tolist(), data['index'].tolist() ):
            if i in grp: 
                if idx in grp[i]:
                    if not grp[i].drop(idx).empty :
                        nodes2.append(np.random.choice(grp[i].drop(idx) ) ) 
                else:
                    nodes2.append(np.random.choice(grp[i]) )
    
    nodes = nodes2+data['index'].tolist()
    dataset.getbyindex=True
    batch2 = [dataset[i] for i in nodes]    
    dataset.getbyindex=False
    adj_mats = {r: np.zeros((len(nodes), len(nodes))) for r in relations}
    for r in relations:
        for i, b1 in enumerate(batch2):
            for j, b2 in enumerate(batch2):
                if i!=j:
                    adj_mats[r][i,j] = 1 if b1[r]==b2[r] else 0
        adj_mats[r] = adj_norm(adj_mats[r])
    return adj_mats, default_collate(batch2)
    
    
    
    
    
    
