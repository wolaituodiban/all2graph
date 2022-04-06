import all2graph as ag
import matplotlib.pyplot as plt


def test_add_kv_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_kv_('a', 'b')
    graph.add_kv_('a', [1, 2, 3, 4])
    graph.add_kv_('b', 'b')
    graph.add_kv_('b', 'haha')
    graph.add_kv_('c', 'hehe')
    graph.add_kv_('d', 'hihi')
    graph._assert()
    graph.draw()
    plt.title('test_add_kv_')
    plt.show()


def test_add_targets_():
    graph = ag.graph.RawGraph()
    graph.add_targets_({'c', 'd', 'e_f'})
    graph._assert()
    print(graph)


def test_add_lid_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_lid_('a', 'b')
    graph.add_lid_('a', 'b')
    graph.add_sample_()
    graph.add_lid_('a', 'b')
    graph._assert()
    assert graph.num_nodes == 2, graph._lids
    graph.draw()
    plt.title('test_add_lid_')
    plt.show()


def test_add_gid_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_gid_('a', 'b')
    graph.add_sample_()
    graph.add_gid_('a', 'b')
    graph._assert()
    assert graph.num_nodes == 1
    graph.draw()
    plt.title('test_add_gid_')
    plt.show()


def test_add_edge_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_kv_('a', 'c')
    graph.add_kv_('a', 'b')
    graph.add_edge_(0, 1)
    graph._assert()
    graph.draw()
    plt.title('test_add_edge_')
    plt.show()


def test_add_edges_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_kv_('a', 'c')
    graph.add_kv_('a', 'b')
    graph.add_kv_('a', 'd')
    graph.add_edges_([0, 1], [1, 2])
    graph._assert()
    graph.draw()
    plt.title('test_add_edges_')
    plt.show()


def test_add_dense_edges_():
    graph = ag.graph.RawGraph()
    graph.add_sample_()
    graph.add_kv_('a', 'c')
    graph.add_kv_('a', 'b')
    graph.add_kv_('a', 'd')
    graph.add_dense_edges_([0, 1, 2])
    graph._assert()
    graph.draw()
    plt.title('test_add_dense_edges_')
    plt.show()


def test_mete_info():
    pass


if __name__ == '__main__':
    test_add_kv_()
    test_add_targets_()
    test_add_lid_()
    test_add_gid_()
    test_add_edge_()
    test_add_edges_()
    test_mete_info()
