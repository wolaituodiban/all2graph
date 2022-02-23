import torch


def default_collate(batches):
    fst_item = batches[0]
    if isinstance(fst_item, torch.Tensor):
        if len(fst_item.shape) == 0:
            return torch.stack(batches)
        else:
            return torch.cat(batches, dim=0)
    elif isinstance(fst_item, list):
        return [default_collate(tensors) for tensors in zip(*batches)]
    elif isinstance(fst_item, dict):
        return {key: default_collate([batch[key] for batch in batches]) for key in batches[0]}
    raise TypeError('only accept "tensor", "list", "dict"')