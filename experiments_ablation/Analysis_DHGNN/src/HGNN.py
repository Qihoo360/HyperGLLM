import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init



# define cluster function
def cluster_and_get_labels(C, data):
    model = KMeans(n_clusters=C, n_init='auto', random_state=0)
    return model.fit_predict(data)
    
def generate_H_from_x(x, C_list):
    #x: n D
    data = x.detach().to(torch.float32).cpu().numpy()
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    data = imputer.fit_transform(data)
    if not isinstance(C_list, list):
        if data.shape[0]<C_list:
            C_list = data.shape[0]
        n_clusters = C_list
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
        cluster_labels = kmeans.fit_predict(data)  # shape: (n,)
        # One-hot 
        one_hot = np.eye(n_clusters)[cluster_labels]  # shape: (n, C)
        if data.shape[0]<C_list:
            res = np.random.randint(0, 2, size=(one_hot.shape[0], C_list-data.shape[0]))
            one_hot = np.concatenate([one_hot, res], axis=1)
        return one_hot
    
#     # sequential execution
#     H = []
#     for i in range(len(C_list)):
#         if data.shape[0]<C_list[i]:
#             C_list[i] = data.shape[0]
#         n_clusters = C_list[i]
#         kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
#         cluster_labels = kmeans.fit_predict(data)  # shape: (n,)
#         # One-hot
#         one_hot = np.eye(n_clusters)[cluster_labels]  # shape: (n, C)
#         if data.shape[0]<C_list[i]:
#             res = np.random.randint(0, 2, size=(one_hot.shape[0], C_list[i]-data.shape[0]))
#             one_hot = np.concatenate([one_hot, res], axis=1)
#         H.append(one_hot)
#     H = np.concatenate(H, axis=-1)
    
    # parallel execution
    C_list_run = [data.shape[0] if data.shape[0]<C else C for C in C_list]
    cluster_labels = Parallel(n_jobs=-1)(
        delayed(cluster_and_get_labels)(C, data) for C in C_list_run
    )
    H = [np.eye(C)[cluster_labels[i]] for i, C in enumerate(C_list_run)]
    H = [np.concatenate([H[i], np.random.randint(0, 2, size=(H[i].shape[0], C-data.shape[0]))], axis=1) if data.shape[0]<C else H[i] for i, C in enumerate(C_list)]
    H = np.concatenate(H, axis=-1)
    return H
# ####check
# x = torch.randn(11, 512); C_list=[2,4]
# H = generate_H_from_x(x, C_list)
# print(H)



def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.where(DV > 0, np.power(DV, -0.5), 0)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

    

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    

class HGNN(nn.Module):
    def __init__(self, in_ch, hypergraph_dim, hypergraph_layernum, hypergraph_drop, hidden_size, C_list, topK):
        super(HGNN, self).__init__()
        self.hypergraph_dim = hypergraph_dim
        self.hypergraph_layernum = hypergraph_layernum
        self.hypergraph_drop = hypergraph_drop
        self.C_list = C_list
        self.topK = topK
        self.hidden_size = hidden_size
        self.module_list = nn.ModuleList([])
        for i in range(self.hypergraph_layernum):
            if i == 0:
                self.module_list.append(HGNN_conv(in_ch, hypergraph_dim))
            else:
                self.module_list.append(HGNN_conv(hypergraph_dim, hypergraph_dim))
        self.adapter = nn.Linear(self.hypergraph_dim, self.hidden_size)

    def forward(self, x):
        #x: (N, D)
        #out: (K, D)
        H_sub = np.ones((x.shape[0],1))
        G_sub = torch.from_numpy(generate_G_from_H(H_sub)).to(dtype=x.dtype, device=x.device)
        
        H = generate_H_from_x(x, self.C_list)
        G = torch.from_numpy(generate_G_from_H(H)).to(dtype=x.dtype, device=x.device)
        H = torch.from_numpy(H).to(dtype=x.dtype, device=x.device)
        
        original_x = x.clone()
        for i in range(self.hypergraph_layernum):
            x_sub = F.dropout(F.relu(self.module_list[i](x, G_sub)), self.hypergraph_drop)
            x_i = F.dropout(F.relu(self.module_list[i](x, G)), self.hypergraph_drop)
            x = x_i - x_sub
        x = (H.T @ (original_x + x)) / H.sum(dim=0, keepdim=True).T
        x = self.adapter(x)
        return x # shape: (K, D)
# # ####check
# x = torch.randn(11, 128).cuda()
# C_list = [2, 4]; topK=5
# model = HGNN(128,128,2,0.1,C_list,topK).cuda()
# out = model(x)
# print(out)
# print(out.shape)
