import scipy.sparse as sp
import copy
import torch

from torch import nn
import torch.nn.functional as F

from layers import *

from utility.preprocessing import str2value, issymmetric, to_dense, to_sparse



class MRGCN(nn.Module):
    """
    multi-relational GCN
    """
    def __init__(self, relations, in_dim=None, out_dim=None, enc='alex', inchannel=3, share_encoder='partly', selfweight=1):
        super(MRGCN, self).__init__()
        self.selfweight = selfweight
        if enc == 'alex':
            self.encoder = MyAlexNet(inchannel=inchannel).features
        if enc == 'singlealex':
            self.encoder = MyAlexNet(inchannel=inchannel).features
        elif enc=='singleres50':
            self.encoder = MyResNet50(inchannel=inchannel).features
        elif enc=='singlevgg16bn':
            self.encoder = MyVggNet16_bn(inchannel=inchannel).features
        elif enc=='singlevgg16':
            self.encoder = MyVggNet16(inchannel=inchannel).features
        elif enc=='singledens161':
            self.encoder=MyDensNet161(inchannel=inchannel).features
        elif enc=='singledens201':
            self.encoder=MyDensNet201(inchannel=inchannel).features
        elif enc=='singledens121':
            self.encoder=MyDensNet121(inchannel=inchannel).features
            
        self.share_encoder= share_encoder
        if share_encoder == 'totally':
            self.gcn =  ImageGraphConvolution(self.encoder, out_dim=out_dim, inchannel=inchannel) if enc else GraphConvolution(in_dim, out_dim)
        elif share_encoder == 'not':    
            self.gcn = nn.ModuleDict({str(i): ImageGraphConvolution(enc=copy.deepcopy(self.encoder), out_dim=out_dim, inchannel=inchannel)  \
                                      for i in relations+['self']}) if enc else  \
                                    nn.ModuleDict({str(i): GraphConvolution(in_dim, out_dim) for i in relations+['self']})
        elif share_encoder == 'partly':  
            self.gcn = nn.ModuleDict({str(i): GraphConvolution(in_dim, out_dim) for i in relations+['self']})
        else:
            raise Error('share_encoder %s is not defined, it must be "totally", "not" or  "partly"'%(share_encoder))
    
    def forward(self, fea_in, k, adj_mats ):          
        if self.share_encoder == 'totally':
            adj = sum(adj_mats.values())
            adj[:,k:] +=self.selfweight*torch.eye(len(fea_in)-k).cuda()
            fea = self.gcn(fea_in , adj)
        elif self.share_encoder == 'not':     
            fea_out = fea_in[k:]
            fea = self.gcn['self'](fea_out, self.selfweight)
            for i, adj in adj_mats.items():
                fea = fea + self.gcn[i](fea_in, adj)
        elif self.share_encoder == 'partly': 
            fea_in = self.encoder(fea_in).squeeze()
            fea_out = fea_in[k:]
            fea = self.gcn['self'](fea_out, self.selfweight)
            for i, adj in adj_mats.items():
                fea = fea + self.gcn[i](fea_in, adj)
            
        return fea
    

class ImageGCN(nn.Module):
    def __init__(self, hid_dims, out_dims, relations, encoder='alex', inchannel=3, share_encoder=False, dropout=0.1):
        super(ImageGCN, self).__init__()
        self.imagelayer = MRGCN(relations, enc=encoder, inchannel=inchannel, share_encoder=share_encoder )
        self.denselayer = MRGCN(relations, hid_dims, out_dims, share_encoder=share_encoder)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, fea, adj_mats2, adj_mats1, k=8 ):
        fea2 = self.imagelayer(fea, fea, adj_mats1)
        fea2 = fea2.view(-1, 1024)
        fea2 = self.relu(self.dropout(fea2))
        fea = self.denselayer(fea2, fea2[k:], adj_mats2)
        return fea
    
class SingleLayerImageGCN(nn.Module):
    def __init__(self, relations, encoder='singlealex', in_dim=1024,out_dim=14, inchannel=3, share_encoder='partly'):
        super(SingleLayerImageGCN, self).__init__()
        self.out_dim = out_dim
        self.layer = MRGCN(relations, enc=encoder,in_dim=in_dim,out_dim=out_dim, inchannel=inchannel, share_encoder=share_encoder )        
    
    def forward(self, fea,  adj_mats, k ):
        fea2 = self.layer(fea, k, adj_mats)
        fea2 = fea2.view(-1, self.out_dim)

        return fea2
    
class W_BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(W_BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        s=int(np.prod(target.size()))
        pos_weight = (target==0).sum()/ (target!=0).sum()
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=pos_weight)
     
class W_BCELossWithNA(nn.Module):
    def __init__(self, reduction='mean'):
        super(W_BCELossWithNA, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):       
        pos_weight = (target==0).sum()/ (target==1).sum()
        return F.binary_cross_entropy_with_logits(input[target!=-1], target[target!=-1], reduction=self.reduction, pos_weight=pos_weight)


class MedGAE(nn.Module):
    def __init__(self, in_dim,  out_dims, edge_decoder, dropout=0.5):
        super(MedGAE, self).__init__()
        self.encoder = nn.ModuleList( [HGCN(in_dim, out_dims[0])] )
        for i in range(len(out_dims)):
            if i+1<len(out_dims):
                self.encoder.append( HGCN(out_dims[i], out_dims[i+1]) )

        self.decoder = nn.ModuleDict({})
        for nodes, (dec, num) in edge_decoder.items():
            if dec == 'innerproduct':
                decoder = InnerProductDecoder(out_dims[-1], num)
            elif dec == 'distmult':
                decoder = DistMultDecoder(out_dims[-1], num)
            elif dec == 'bilinear':
                decoder = BilinearDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'dedicom':
                decoder = DEDICOMDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'symbilinear':
                decoder = BilinearDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'symdedicom':
                decoder = DEDICOMDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'symMLP':
                decoder = MLPDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'MLP':
                decoder = MLPDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'symlinear':
                decoder = LinearDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'linear':
                decoder = LinearDecoder(out_dims[-1], num, issymmetric=False)    
            else:
                raise ValueError('Unknown decoder type')
            self.decoder[str(nodes)] = decoder
            self.dropout = dropout
    
    def encode(self, fea_mats, adj_mats ):
        for i, m in enumerate(self.encoder):
            fea_mats = m(fea_mats, adj_mats)
            if i+1<len(self.encoder):
                fea_mats = DictReLU()(fea_mats)
                fea_mats = DictDropout(p=self.dropout)(fea_mats)

        return fea_mats
    
    def decode(self, z):
        adj_recon={}
        for nodes, decoder in self.decoder.items():
            nodes = str2value(nodes)            
            adj_recon[nodes] = [decoder(z[nodes[0]], z[nodes[1]], i) for i in range(decoder.num_types)]
            
        return adj_recon
    
    def forward(self, fea_mats, adj_mats, adj_masks):
        adj_mats = copy.deepcopy(adj_mats)
        for key, masks in  adj_masks.items():
            adj_mats[key] = [to_sparse(masks[i]).float()*to_sparse(adj_mats[key][i]) for i in range(len(masks))]
            
        z = self.encode(fea_mats, adj_mats)
        adj_recon = self.decode(z)
        return adj_recon, z
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MatWithNALoss(nn.Module):
    def __init__(self, reduction='mean', pos_weight=None):
        super(MatWithNALoss, self).__init__()
        self.reduction  = reduction
        self.pos_weight = pos_weight
        
    def forward(self, input, target, mask, losstype= None ):
        # losstype in {'BCE', 'MSE'}
        input = input.view(-1)
        target = target.view(-1)
        if not isinstance(mask, (int,float)):
            mask = to_dense(mask.byte()).view(-1)
            target = target[mask]
            input = input[mask]
        pos_weight= self.pos_weight if self.pos_weight else (target==0).sum()/ (target!=0).sum()
        if not losstype: 
            losstype = 'BCE' if (target>=0).all() and (target<=1).all() else 'MSE' 
        
        if losstype=='BCE':
            loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=pos_weight)
        elif losstype=='MSE':
            loss = F.mse_loss(input, target, reduction=self.reduction)
        elif losstype=='L1':
            loss = F.l1_loss(input, target, reduction=self.reduction)
        else:
            raise ValueError('Undefined loss type.')
        # print("using loss type "+losstype+':'+str(loss.item()))
        return loss
    
class MultiMatLoss(nn.Module):
    def __init__(self, reduction='mean', pos_weight=None):
        super(MultiMatLoss, self).__init__()
        self.losscls = MatWithNALoss(pos_weight = pos_weight)
        self.reduction  = reduction
        
        
    def forward(self, adj_recon, adj_mats, adj_masks, adj_losstype=None ):
        loss=0
        for key, adj in adj_recon.items():
            # print('computing loss for type:'+str(key))
            for i in range(len(adj)):
                input = adj_recon[key][i]
                target = to_dense(adj_mats[key][i])                
                mask = adj_masks[key][i].byte() if key in adj_masks else 1
                losstype = adj_losstype[key][i] if isinstance(adj_losstype, dict) else adj_losstype
                loss += losstype[1]*self.losscls(input, target, mask, losstype[0])
        return loss
 