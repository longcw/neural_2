import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np


# settings
N0 = 1000
I = 10
B = 5000
P = 5000

nodes = np.arange(N0)
edges = list(itertools.permutations(nodes, 2))
E0 = len(edges)
rc = 1 - pow(B / float(E0), 1. / I)
p = int(P / I)

# init fully connected network
dg = nx.DiGraph()
dg.add_nodes_from(nodes)
dg.add_edges_from(edges)

# training and test set
train_srcs = np.random.choice(nodes[:int(N0 / 2)], P)
train_dests = np.random.choice(nodes[int(N0 / 2):], P)

test_srcs = np.random.choice(nodes[:int(N0 / 2)], P)
test_dests = np.random.choice(nodes[int(N0 / 2):], P)

# train
unroutable = 0
for i in range(I):
    r = rc
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
    un_num = min(len(untracked_edges), pruning_num)
    for k, key in enumerate(untracked_edges):
        if k >= un_num:
            break
        dg.remove_edge(key[0], key[1])

    pruning_num -= un_num
    if pruning_num == 0:
        continue
    sorted_tracks = sorted(tracks.items(), key=operator.itemgetter(1))
    for k in range(pruning_num):
        key, _ = sorted_tracks[k]
        dg.remove_edge(key[0], key[1])

print('cost: {}'.format(len(dg.edges())))

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

print('unroutable: {}'.format(unroutable))
print('efficiency: {}'.format(efficiency))
print('robustness: {}'.format(robustness))

# # draw
# nx.draw(dg)
# plt.show()
