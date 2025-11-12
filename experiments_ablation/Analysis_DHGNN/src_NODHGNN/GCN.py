import math
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init






class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

    
class GraphConvolution_wmask_learnadj(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution_wmask_learnadj, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features//8)
        self.fc2 = nn.Linear(in_features, in_features//8)
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.fc1.bias.data.uniform_(-stdv, stdv)
        
        stdv = 1. / math.sqrt(self.fc2.weight.size(1))
        self.fc2.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.fc2.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        sim = torch.bmm(self.fc1(input), self.fc2(input).transpose(1, 2)) / torch.sqrt(torch.tensor(self.in_features//8, dtype=input.dtype, device=input.device))
        sim = sim + ((1.0-mask.unsqueeze(-1))*-1e12).to(dtype=input.dtype)
        adj = F.softmax(sim, dim=-1)*mask.unsqueeze(-1)

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
    
    
class GCN(nn.Module):
    def __init__(self, list_key_dims, kvgraph_dim, kvgraph_layernum, kvgraph_drop):
        super(GCN, self).__init__()
        self.list_key_dims = list_key_dims
        self.kvgraph_dim = kvgraph_dim
        self.kvgraph_layernum = kvgraph_layernum
        self.kvgraph_drop = kvgraph_drop
        
        self.linear_list = nn.ModuleList([])
        for i in range(len(self.list_key_dims)):
            self.linear_list.append(nn.Linear(self.list_key_dims[i], self.kvgraph_dim))
            
        self.module_list = nn.ModuleList([])
        for i in range(self.kvgraph_layernum):
            self.module_list.append(GraphConvolution_wmask_learnadj(kvgraph_dim, kvgraph_dim))

    def forward(self, list_x, list_mask):
        #list_x: [(N, D'), (N, D'),..., (N, D')]  list_mask: [(N,), (N,),..., (N,)]
        #out: (N, D)
        x = torch.stack([self.linear_list[i](list_x[i]) for i in range(len(self.list_key_dims))], dim=1)
        mask = torch.stack(list_mask, dim=1)
        x = x * mask.unsqueeze(dim=-1)
        
        original_x = x.sum(dim=1) / mask.unsqueeze(-1).sum(dim=1) #N D
        for i in range(self.kvgraph_layernum):
            x = F.dropout(F.relu(self.module_list[i](x, mask)), self.kvgraph_drop)*mask.unsqueeze(dim=-1)
        x = original_x + x.sum(dim=1) / mask.unsqueeze(-1).sum(dim=1) #N D
        return x
# ####check
# list_x = [torch.randn(2, 7), torch.randn(2, 11), torch.randn(2, 5)]
# list_mask = [torch.tensor([1,1]), torch.tensor([0,1]), torch.tensor([1,0])]
# model = GCN([7, 11, 5], 128, 2, 0.1)
# x = model(list_x, list_mask)
# print(x)
# print(x.shape)