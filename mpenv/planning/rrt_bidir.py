import numpy as np
from tqdm import tqdm

import pinocchio as pin


class Node:
    def __init__(self, point, parent):
        if not (parent is None or isinstance(parent, Node)):
            raise ValueError("Parent should be None or Node type")
        self.parent = parent
        self.point = point

    def path_from_root(self):
        node = self
        path = []
        while node is not None:
            path.append(node.point)
            node = node.parent
        return path[::-1]


def nearest_neighbor(x, nodes, distance_fn):
    dist = [distance_fn(x, n.point) for n in nodes]
    idx = np.argmin(dist)
    return nodes[idx]


def rrt_bidir(start, goal, sample_fn, expand_fn, distance_fn, close_fn, iterations):
    nodes_ab = [[], []]
    for i, x in enumerate((start, goal)):
        node = Node(x, parent=None)
        nodes_ab[i].append(node)
    solution = {"points": [], "collisions": [], "n_samples": 0, "n_collisions": 0}
    growing_index = 0
    # for i in tqdm(range(iterations), ncols=80):
    for i in range(iterations):
        nodes_a, nodes_b = nodes_ab[growing_index], nodes_ab[1 - growing_index]
        x_rand = sample_fn()
        # grows tree_a toward x_rand
        node_a = nearest_neighbor(x_rand, nodes_a, distance_fn)
        solution["n_samples"] += 1
        x_a = node_a.point
        # path_a = interpolate_fn(x_a, x_a_new)
        x_a_new, col_free_a = expand_fn(x_a, x_rand)
        if col_free_a and not close_fn(x_a, x_a_new):
            node_a_new = Node(x_a_new, parent=node_a)
            nodes_ab[growing_index].append(node_a_new)
            # grows tree_b toward x_a_new
            node_b = nearest_neighbor(x_a_new, nodes_b, distance_fn)
            solution["n_samples"] += 1
            x_b = node_b.point
            # path_b = interpolate_fn(x_b, x_b_new)
            x_b_new, col_free_b = expand_fn(x_b, x_a_new)
            if col_free_b and not close_fn(x_b, x_b_new):
                node_b_new = Node(x_b_new, parent=node_b)
                nodes_ab[1 - growing_index].append(node_b_new)
            # if the two trees are connected, stop the algorithm
            if close_fn(x_a_new, x_b_new):
                if growing_index == 1:
                    node_a_new, node_b_new = node_b_new, node_a_new
                seq_start_a = node_a_new.path_from_root()
                seq_b_goal = node_b_new.path_from_root()[::-1]
                seq = seq_start_a + seq_b_goal[1:]
                solution["points"] = seq
                return True, solution, nodes_ab, 2 * i

        if len(nodes_ab[0]) == len(nodes_ab[1]):
            growing_index = np.random.binomial(1, 0.5)
        elif len(nodes_ab[0]) > len(nodes_ab[1]):
            growing_index = 1 - growing_index

    return (
        False,
        {"collisions": solution["collisions"]},
        nodes_ab,
        2 * iterations,
    )
