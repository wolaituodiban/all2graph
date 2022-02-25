import all2graph as ag
import matplotlib.pyplot as plt


def test_add_kv_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'b', True)
    graph.add_kv_(0, 'a', [1, 2, 3, 4], True)
    graph.add_kv_(0, ('a', 'b'), 'b', False)
    graph.add_kv_(0, ('a', 'b'), 'c', True)
    graph._assert()
    graph.draw()
    plt.title('test_add_kv_')
    plt.show()


def test_add_targets_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'b', True)
    graph.add_kv_(1, 'a', [1, 2, 3, 4], True)
    graph.add_kv_(0, ('a', 'b'), 'b', False)
    graph.add_kv_(1, ('a', 'b'), 'c', True)
    graph.add_targets_(['c'])
    graph._assert()
    graph.draw()
    plt.title('test_add_targets_')
    plt.show()


def test_add_lid_():
    graph = ag.graph.RawGraph()
    graph.add_lid_(0, 'a', 'b', True)
    graph.add_lid_(0, 'a', 'b', False)
    graph.add_lid_(1, 'a', 'b', True)
    graph._assert()
    graph.draw()
    plt.title('test_add_lid_')
    plt.show()


def test_add_gid_():
    graph = ag.graph.RawGraph()
    graph.add_gid_('a', 'b', True)
    graph.add_gid_('a', 'b', False)
    graph._assert()
    graph.draw()
    plt.title('test_add_gid_')
    plt.show()


def test_add_edge_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'c', False)
    graph.add_kv_(0, 'a', 'b', True)
    graph.add_edge_(0, 1, bidirectional=True)
    graph._assert()
    graph.draw()
    plt.title('test_add_edge_')
    plt.show()


def test_add_edges_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'c', False)
    graph.add_kv_(0, 'a', 'b', True)
    graph.add_kv_(0, 'a', 'd', True)
    graph.add_edges_([0, 1], [1, 2])
    graph._assert()
    graph.draw()
    plt.title('test_add_edges_')
    plt.show()


def test_add_edges_for_seq_():
    graph = ag.graph.RawGraph()
    for i in range(7):
        graph.add_kv_(0, ('a', 'b', 'c'), i, False)
    graph.add_edges_for_seq_([0, 1, 2])
    graph.add_edges_for_seq_([3, 4, 5, 6], degree=2, r_degree=1)
    graph._assert()
    graph.draw()
    plt.title('test_add_edges_for_seq_')
    plt.show()


def test_add_edges_for_seq_by_key_():
    graph = ag.graph.RawGraph()
    for i in range(3):
        graph.add_kv_(0, 'a', i, False)
    for i in range(4):
        graph.add_kv_(0, 'b', i, False)
    graph.add_edges_for_seq_by_key_('a')
    graph.add_edges_for_seq_by_key_('b', degree=2, r_degree=1)
    graph._assert()
    graph.draw()
    plt.title('test_add_edges_for_seq_by_key_')
    plt.show()


def test_to_simple_():
    graph = ag.graph.RawGraph()
    graph.add_kv_(0, 'a', 'b', False)
    graph.add_kv_(0, 'a', 'c', False)
    graph.add_edges_([0, 0], [1, 1])
    assert graph.num_edges(ag.VALUE2VALUE) == 2
    graph.to_simple_()
    assert graph.num_edges(ag.VALUE2VALUE) == 1
    graph._assert()
    graph.draw()
    plt.title('test_to_sample_')
    plt.show()


if __name__ == '__main__':
    test_add_kv_()
    test_add_targets_()
    test_add_lid_()
    test_add_gid_()
    test_add_edge_()
    test_add_edges_()
    test_add_edges_for_seq_()
    test_add_edges_for_seq_by_key_()
    test_to_simple_()
