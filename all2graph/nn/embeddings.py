import torch
import dgl


class Embedding(torch.nn.Module):
    def __init__(self, embeding):
        super().__init__()
