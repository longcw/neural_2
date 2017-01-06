import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np


def pruning(N0, I, B, P, increasing):
    nodes = np.arange(N0)
    edges = list(itertools.permutations(nodes, 2))
    E0 = len(edges)
    p = int(P / I)

    rc = 1 - pow(B / float(E0), 1. / I)
    if increasing == 0:
        rs = [rc] * 10
    else:
        rs = [0.1, 0.15, 0.20, 0.25, rc]
        for i in range(5):
            r = 1. - (1-rc)**2 / (1 - rs[4-i])
            rs.append(r)
        if increasing < 0:
            rs = list(reversed(rs))
    # print rc
    # print(rs)

    # init fully connected network
    dg = nx.DiGraph()
    dg.add_nodes_from(nodes)
    dg.add_edges_from(edges)

    # training and test set
    train_srcs = np.random.choice(nodes[:int(N0 / 2)], P)
    train_dests = np.random.choice(nodes[int(N0 / 2):], P)

    # train
    unroutable = 0
    for i in range(I):
        r = rs[i]
        tracks = dict()

        for j in range(i*p, i*p+p):
            src = train_srcs[j]
            dest = train_dests[j]
            try:
                path = nx.shortest_path(dg, src, dest)
            except nx.exception.NetworkXNoPath:
                path = []
                unroutable += 1
            for k in range(len(path)-1):
                key = (path[k], path[k+1])
                tracks.setdefault(key, 0)
                tracks[key] += 1

        # pruning
        pruning_num = int(np.round(len(dg.edges()) * r))

        tracked_edges = set(tracks.keys())
        all_edges = set(dg.edges())
        untracked_edges = all_edges - tracked_edges
        untracked_edges = list(untracked_edges)
        un_num = min(len(untracked_edges), pruning_num)
        un_idx = np.random.choice(len(untracked_edges), un_num, replace=False)
        for k in un_idx:
            key = untracked_edges[k]
            dg.remove_edge(key[0], key[1])

        pruning_num -= un_num
        if pruning_num == 0:
            continue
        sorted_tracks = sorted(tracks.items(), key=operator.itemgetter(1))
        for k in range(pruning_num):
            key, _ = sorted_tracks[k]
            dg.remove_edge(key[0], key[1])

    # print('cost: {}'.format(len(dg.edges())))
    return dg


def test(dg, N0, P):
    nodes = np.arange(N0)
    test_srcs = np.random.choice(nodes[:int(N0 / 2)], P)
    test_dests = np.random.choice(nodes[int(N0 / 2):], P)

    # test
    unroutable = 0
    efficiency = 0
    robustness = 0
    for i in range(P):
        src = test_srcs[i]
        dest = test_dests[i]

        # efficiency
        try:
            path = nx.shortest_path(dg, src, dest)
            eff = len(path)
        except nx.exception.NetworkXNoPath:
            unroutable += 1
            eff = 25
        efficiency += eff

        # robustness
        k = approx.node_connectivity(dg, src, dest)
        robustness += k

    efficiency /= float(P)
    robustness /= float(P)
    return unroutable, efficiency, robustness


if __name__ == '__main__':
    # settings
    N0 = 1000
    I = 10
    P = 5000
    # B = 2000
    increasing = 0
    print('type:{}'.format(increasing))
    for B in range(2000, 5001, 500):
        dg = pruning(N0, I, B, P, increasing=increasing)
        print('cost: {}'.format(len(dg.edges())))
        unroutable, efficiency, robustness = test(dg, N0, P)
        print('unroutable: {}'.format(unroutable))
        print('efficiency: {}'.format(efficiency))
        print('robustness: {}'.format(robustness))

