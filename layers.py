import math
import numpy as np
import scipy.sparse as sp

import torch

from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models
from utility.preprocessing import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, norm='', bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)
        self.norm=norm
        
    def forward(self, input, adj=1.0):
        input = to_dense(input) 
        support = self.linear(input)
        if isinstance(adj, (float, int)):
            output = support*adj  
        else:
            adj = adj_norm(adj, True) if self.norm=='symmetric' else adj_norm(adj, False) if self.norm=='asymmetric' else adj
            output = torch.matmul(adj, support) 
        return output

    def __repr__(self):
        return self.__class__.__name__ +'(in_features={}, out_features={}, bias={}, norm={})'.format(
            self.in_features, self.out_features, self.bias, self.norm )
    
class ImageGraphConvolution(nn.Module):
    """
    GCN layer for image data
    """

    def __init__(self, enc , inchannel=3):
        super(ImageGraphConvolution, self).__init__()
        self.encoder = enc
        self.classifier = nn.Linear(1024, 14)

    def forward(self, input, adj=1.0):
        x = self.encoder(input).squeeze()
        x = x.view(-1,1024)
        support = self.classifier(x) 
        if isinstance(adj, (float, int)):
            output = support*adj  
        else:
            output = torch.spmm(adj, support) 
        return output

class MyAlexNet(nn.Module):
    def __init__(self,outnum=14, gpsize=4, inchannel=3):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        if inchannel!=3:
            original_model.features[0]=nn.Conv2d(1,64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.features = original_model.features        
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
    
class MyResNet50(nn.Module):
    def __init__(self,outnum=14,gpsize=4, inchannel=3):
        super(MyResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True) 
        if inchannel!=3:
            original_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyVggNet16_bn(nn.Module):
    def __init__(self, outnum=14, gpsize=4,inchannel=3):
        super(MyVggNet16_bn, self).__init__()
        original_model = models.vgg16_bn(pretrained=True)
        if inchannel!=3:
            original_model.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyVggNet16(nn.Module):
    def __init__(self, outnum=14,gpsize=4,inchannel=3):
        super(MyVggNet16, self).__init__()
        original_model = models.vgg16(pretrained=True)
        if inchannel!=3:
            original_model.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), 
                                                         nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet161(nn.Module):
    def __init__(self, outnum=14,gpsize=4, inchannel=3):
        super(MyDensNet161, self).__init__()
        original_model = models.densenet161(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(2208, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet201(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet201, self).__init__()
        original_model = models.densenet201(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1920, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x
    
class MyDensNet121(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet121, self).__init__()
        original_model = models.densenet121(pretrained=True)
        if inchannel!=3:
            original_model.features.conv0=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = original_model.features
        self.features.add_module('transit',nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                         nn.ReLU(inplace=True), nn.MaxPool2d(2,padding=1)))
        self.features.add_module('gpool',nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,1024)
        x = self.classifier(x)
        return x    
    
class DictReLU(nn.ReLU):        
    def forward(self , input):
        return {key: F.relu(fea) for key, fea in input.items()} if isinstance(input, dict) else F.relu(input)
    
class DictDropout(nn.Dropout):        
    def forward(self , input):
        if isinstance(input, dict):
            return {key: F.dropout(fea, self.p, self.training, self.inplace) for key, fea in input.items()}  
        else: 
            return F.dropout(input, self.p, self.training, self.inplace)
    
            
class DEDICOMDecoder(nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x:x):
        super(DEDICOMDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight_global = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.weight_local = Parameter(torch.FloatTensor(num_types, input_dim))
        self.reset_parameters()
        if issymmetric:
            self.weight_global = self.weight_global + self.weight_global.t()
        
    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight_global.size(1))
        self.weight_global.data.uniform_(-stdv, stdv)
        self.weight_local.data.uniform_(-stdv, stdv)


    def forward(self, input1, input2, type_index):
        relation = torch.diag(self.weight_local[type_index])
        product1 = torch.mm(input1, relation)
        product2 = torch.mm(product1, self.weight_global)
        product3 = torch.mm(product2, relation)
        outputs = torch.mm(product3, input2.transpose(0,1))
        outputs = outputs + self.bias

        return self.act(outputs)


class DistMultDecoder(nn.Module):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_types, bias=True, act=lambda x:x ):
        super(DistMultDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight = Parameter(torch.FloatTensor(num_types, input_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, type_index):
        relation = torch.diag(self.weight[type_index])
        intermediate_product = torch.mm(input1, relation)
        outputs = torch.mm(intermediate_product, input2.transpose(0,1))
        outputs = outputs + self.bias

        return self.act(outputs)


class BilinearDecoder(nn.Module):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x:x):
        super(BilinearDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight = Parameter(torch.FloatTensor(num_types, input_dim, input_dim))
        self.reset_parameters()                
        
    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)        

    def forward(self, input1, input2, type_index):
        self.wt = self.weight + self.weight.transpose(1,2) if issymmetric else self.weight
        intermediate_product = torch.mm(input1, self.wt[type_index])
        outputs = torch.mm(intermediate_product, input2.transpose(0,1))
        outputs = outputs + self.bias
        return self.act(outputs)
    
class LinearDecoder(nn.Module):
    """Linear Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x:x):
        super(LinearDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.layer = nn.Linear(input_dim, 1, bias) if issymmetric else nn.Linear(input_dim*2, 1, bias)             
              

    def forward(self, input1, input2, type_index):
        outputs = []
        for input in input2:
            if self.issymmetric:
                output = self.layer(input1)+self.layer(input.expand_as(input1)) 
            else:
                output = self.layer(torch.cat([input1,input.expand_as(input1)], dim=1))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return self.act(outputs)

    
class MLPDecoder(nn.Module):
    """multi-layer perceptron Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_types, hid_dim=20, issymmetric=True, bias=True, act=lambda x:x):
        super(MLPDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.layer1 = nn.Linear(input_dim, hid_dim, bias) if issymmetric else nn.Linear(input_dim*2, hid_dim, bias)
        self.layer2 = nn.Linear(hid_dim, 1, bias)                
              

    def forward(self, input1, input2, type_index):
        outputs = []
        for input in input2:
            if self.issymmetric:
                output = self.layer1(input1)+self.layer1(input.expand_as(input1)) 
            else:
                output = self.layer1(torch.cat([input1,input.expand_as(input1)], dim=1))
            output = F.relu(output)
            outputs.append( self.layer2(output))
        outputs = torch.cat(outputs, dim=1)
        return self.act(outputs)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim=None, num_types=None, bias=True, act=lambda x:x):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0

    def forward(self, input1, input2, type_index=None):
        outputs = torch.mm(input1, input2.transpose(0,1))
        outputs = outputs + self.bias
        return self.act(outputs)
