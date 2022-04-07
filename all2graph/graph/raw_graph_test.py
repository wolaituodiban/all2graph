import all2graph as ag
import matplotlib.pyplot as plt


def test_add_kv_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'b')
    graph.add_kv_(0, 'a', [1, 2, 3, 4])
    graph.add_kv_(0, 'b', 'b')
    graph.add_kv_(0, 'b', 'haha')
    graph.add_kv_(0, 'c', 'hehe')
    graph.add_kv_(0, 'd', 'hihi')
    graph._assert()
    graph.draw()
    plt.title('test_add_kv_')
    plt.show()


def test_add_targets_():
    graph = ag.graph.RawGraph()
    graph.add_targets_({'c', 'd', 'e_f'})
    graph._assert()
    print(graph)


def test_add_local_foreign_key_():
    graph = ag.graph.RawGraph()
    graph.add_local_foreign_key_(0, 'a', 'b')
    graph.add_local_foreign_key_(0, 'a', 'b')
    graph.add_local_foreign_key_(1, 'a', 'b')
    graph._assert()
    assert graph.num_nodes == 2, graph._local_foreign_keys
    graph.draw()
    plt.title('test_add_local_foreign_key_')
    plt.show()


def test_add_glocal_foreign_key_():
    graph = ag.graph.RawGraph()
    graph.add_global_foreign_key_(0, 'a', 'b')
    graph.add_global_foreign_key_(1, 'a', 'b')
    graph._assert()
    assert graph.num_nodes == 1
    graph.draw()
    plt.title('test_add_gid_')
    plt.show()


def test_add_edge_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'c')
    graph.add_kv_(0, 'a', 'b')
    graph.add_edge_(0, 1)
    graph._assert()
    graph.draw()
    plt.title('test_add_edge_')
    plt.show()


def test_add_edges_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'c')
    graph.add_kv_(0, 'a', 'b')
    graph.add_kv_(0, 'a', 'd')
    graph.add_edges_([0, 1], [1, 2])
    graph._assert()
    graph.draw()
    plt.title('test_add_edges_')
    plt.show()


def test_add_dense_edges_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'c')
    graph.add_kv_(0, 'a', 'b')
    graph.add_kv_(0, 'a', 'd')
    graph.add_dense_edges_([0, 1, 2])
    graph._assert()
    graph.draw()
    plt.title('test_add_dense_edges_')
    plt.show()


if __name__ == '__main__':
    test_add_kv_()
    test_add_targets_()
    test_add_local_foreign_key_()
    test_add_glocal_foreign_key_()
    test_add_edge_()
    test_add_edges_()
    test_add_dense_edges_()
