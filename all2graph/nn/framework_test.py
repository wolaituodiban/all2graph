import dgl
from dgl.nn.pytorch import GATConv
import numpy as np
import torch as th

if __name__ == '__main__':
    u = [0, 1, 0, 0, 1]
    v = [0, 1, 2, 3, 2]
    g = dgl.heterograph({('A', 'r', 'B'): (u, v), ('A', 'r', 'C'): (u, v)})
    print(g)
    u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    gatv2conv = GATConv((5, 10), 2, 3)
    res = gatv2conv(g, (u_feat, v_feat))
    print(res.shape)

