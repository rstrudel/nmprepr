import numpy as np


def shorten(path, expand_fn, interpolate_fn, distance_fn):
    path = list(path)
    current_idx = 0
    target_idx = len(path) - 1
    it = 0

    # path shortcut
    while current_idx < target_idx:
        point = path[current_idx]
        target = path[target_idx]
        q_stop, free = expand_fn(point, target, limit_growth=False)
        if free:
            path = path[: current_idx + 1] + path[target_idx:]
            target_idx = current_idx
            current_idx = 0
        else:
            current_idx += 1
        it += 1

    # random configurations shortcut
    # for i in range(200):
    #     i0, i1 = np.random.choice(len(path) - 1, size=2, replace=False)
    #     if i1 < i0:
    #         i0_old = i0
    #         i0 = i1
    #         i1 = i0_old
    #     t0, t1 = np.random.uniform(0, 1, size=2)
    #     q0 = interpolate_fn(path[i0], path[i0 + 1], t0)
    #     q1 = interpolate_fn(path[i1], path[i1 + 1], t1)
    #     q_stop, free = expand_fn(q0, q1, limit_growth=False)
    #     if free:
    #         path = path[: i0 + 1] + [q0, q1] + path[i1 + 1 :]
    #     it += 1
    # trimmed_path = []
    # for i in range(len(path) - 1):
    #     if not np.allclose(path[i].q, path[i + 1].q):
    #         trimmed_path.append(path[i])

    return path, it


def limit_step_size(path, arange_fn, step_size):
    n = len(path)
    new_path = []
    for i in range(n - 1):
        new_path += arange_fn(path[i], path[i + 1], step_size)
    return new_path
