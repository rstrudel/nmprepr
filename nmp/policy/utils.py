import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvas
from tqdm import tqdm

import torch


def load(log_dir, exp_name, cpu, stochastic):
    data = torch.load(
        os.path.join(log_dir, exp_name),
        map_location=torch.device("cpu" if cpu else "cuda"),
    )
    policy_keys = ["evaluation/policy", "trainer/policy"]
    policy = None
    for key in policy_keys:
        if key in data:
            policy = data[key]
            break
    if policy is None:
        raise ValueError(f"Policy not found in the keys: {data.keys()}")
    policy.stochastic_policy.eval()
    if stochastic:
        policy = policy.stochastic_policy
    return policy


def runs(rollout_fn, process_path, episodes):
    gen = range(episodes)
    if episodes > 1:
        gen = tqdm(gen)
    paths = []
    for i in gen:
        path = rollout_fn()
        paths.append(path)
        process_path(path)
    return paths


def evaluate(rollout_fn, episodes):
    returns = []
    rewards = []
    n_steps = []
    lengths = []
    successes = []
    paths_states = []

    def process_path(path):
        obs = path["observations"]
        n = obs.shape[0]
        length = 0
        path_states = []
        for i in range(n):
            q0 = obs[i]["achieved_q"]
            if i < n - 1:
                q1 = obs[i + 1]["achieved_q"]
                length += np.linalg.norm(q1 - q0)
            path_states.append(q0[:2])
        paths_states.append(path_states)
        lengths.append(length)
        successes.append(path["env_infos"]["success"][-1])
        rewards.append(path["rewards"])
        returns.append(np.sum(path["rewards"]))
        n_steps.append(len(path["rewards"]))

    paths = runs(rollout_fn, process_path, episodes)
    returns = np.array(returns)
    successes = np.array(successes)
    print(f"Mean {returns.mean()} Max {returns.max()}, Min {returns.min()}")
    success_rate = sum(successes) / episodes
    mean_steps = np.array(n_steps).mean()
    mean_length = np.array(lengths).mean()
    max_length = np.array(lengths).max()
    print("mean steps", mean_steps)
    print("mean length", mean_length)
    collisions = 0
    for r in rewards:
        collisions += np.sum(r < -0.1)
    collisions /= episodes

    # if hasattr(policy.stochastic_policy, "hist_features"):
    #     features = np.array(policy.stochastic_policy.hist_features)[:, 0]
    #     show_features(features, lengths, successes)

    return success_rate, collisions, paths_states


def render(env, rollout_fn):
    count = []

    def process_path(path):
        print(f"Episode cumulative reward: {np.sum(path['rewards'])}")
        n_steps = len(path["terminals"])
        if path["env_infos"]["success"][-1]:
            print(f"Success after {n_steps} steps.")
        else:
            print(f"Failure after {n_steps} steps.")
        env.render()
        time.sleep(2)
        print(len(count))
        count.append(1)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])

    while True:
        path = runs(rollout_fn, process_path, 1)
    return path


def pca(X, d):
    mean = X.mean(axis=0)
    X -= mean
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    components = vh
    pca_proj = X.dot(components[:d].T)
    X += mean

    return pca_proj


def show_features(features, lengths, successes):
    features_success = []
    colors = []
    start_col = np.array([0, 0, 1, 0.7])
    end_col = np.array([1, 0, 0, 0.7])
    counter = 0
    for length, success in zip(lengths, successes):
        if success:
            features_success.append(features[counter : counter + length])
            # features_success.append(features[counter + length - 1 : counter + length])
            color_range = (
                start_col
                + np.linspace(0, 1, length)[:, None] * (end_col - start_col)[None, :]
            )
            colors.append(color_range)
            # color_range = end_col
            # colors.append(color_range[None, :])
        counter += length
    features = np.concatenate(features_success, axis=0)
    colors = np.concatenate(colors, axis=0)
    proj = pca(features, 2)
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, s=5)
    plt.show()
