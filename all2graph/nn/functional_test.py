import all2graph as ag
import torch


def test_tensor_op_speed():
    feat = torch.rand(10000, 8)
    weight = torch.rand(10000, 8, 8)
    bias = torch.rand(10000, 8)
    with ag.Timer('*'):
        for _ in range(1000):
            out1 = ag.nn.nodewise_linear(feat, weight, bias)

    with ag.Timer('matmul'):
        for _ in range(1000):
            out2 = ag.nn.nodewise_linear(feat, weight, bias, use_matmul=True)

    assert (out1 == out2).all()


if __name__ == "__main__":
    test_tensor_op_speed()
