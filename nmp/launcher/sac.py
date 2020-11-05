import gym
import torch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
import mpenv.envs
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
    MdpPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


from nmp.launcher import utils


def get_replay_buffer(variant, expl_env):
    """
    Define replay buffer specific to the mode
    """
    mode = variant["mode"]
    if mode == "vanilla":
        replay_buffer = EnvReplayBuffer(
            env=expl_env, **variant["replay_buffer_kwargs"],
        )

    elif mode == "her":
        replay_buffer = ObsDictRelabelingBuffer(
            env=expl_env, **variant["her"], **variant["replay_buffer_kwargs"]
        )

    return replay_buffer


def get_networks(variant, expl_env):
    """
    Define Q networks and policy network
    """
    qf_kwargs = variant["qf_kwargs"]
    policy_kwargs = variant["policy_kwargs"]
    shared_base = None

    qf_class, qf_kwargs = utils.get_q_network(variant["archi"], qf_kwargs, expl_env)
    policy_class, policy_kwargs = utils.get_policy_network(
        variant["archi"], policy_kwargs, expl_env, "tanhgaussian"
    )

    qf1 = qf_class(**qf_kwargs)
    qf2 = qf_class(**qf_kwargs)
    target_qf1 = qf_class(**qf_kwargs)
    target_qf2 = qf_class(**qf_kwargs)
    policy = policy_class(**policy_kwargs)
    print("Policy:")
    print(policy)

    nets = [qf1, qf2, target_qf1, target_qf2, policy, shared_base]
    print(f"Q function num parameters: {qf1.num_params()}")
    print(f"Policy num parameters: {policy.num_params()}")

    return nets


def get_path_collector(variant, expl_env, eval_env, policy, eval_policy):
    """
    Define path collector
    """
    mode = variant["mode"]
    if mode == "vanilla":
        expl_path_collector = MdpPathCollector(expl_env, policy)
        eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    elif mode == "her":
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
        )
        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            eval_policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
        )
    return expl_path_collector, eval_path_collector


def sac(variant):
    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    expl_env.seed(variant["seed"])
    eval_env.set_eval()

    mode = variant["mode"]
    archi = variant["archi"]
    if mode == "her":
        variant["her"] = dict(
            observation_key="observation",
            desired_goal_key="desired_goal",
            achieved_goal_key="achieved_goal",
            representation_goal_key="representation_goal",
        )

    replay_buffer = get_replay_buffer(variant, expl_env)
    qf1, qf2, target_qf1, target_qf2, policy, shared_base = get_networks(
        variant, expl_env
    )
    expl_policy = policy
    eval_policy = MakeDeterministic(policy)

    expl_path_collector, eval_path_collector = get_path_collector(
        variant, expl_env, eval_env, expl_policy, eval_policy
    )

    mode = variant["mode"]
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"],
    )
    if mode == "her":
        trainer = HERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
    )

    algorithm.to(ptu.device)
    algorithm.train()
