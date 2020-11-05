import torch
import torch.nn.functional as F

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy

from nmp.model.cnn import CNN
from nmp.model.pointnet import PointNet
from nmp.policy.tanh_gaussian import (
    TanhGaussianCNNPolicy,
    TanhGaussianPointNetPolicy,
)


ARCHI = {
    "mlp": {"vanilla": FlattenMlp, "tanhgaussian": TanhGaussianPolicy},
    "pointnet": {"vanilla": PointNet, "tanhgaussian": TanhGaussianPointNetPolicy},
    "cnn": {"vanilla": CNN, "tanhgaussian": TanhGaussianCNNPolicy},
}


def archi_to_network(archi_name, function_type):
    allowed_function_type = ["vanilla", "tanhgaussian"]
    if function_type not in allowed_function_type:
        raise ValueError(f"Function name should be in {allowed_function_type}")
    return ARCHI[archi_name][function_type]


def get_policy_network(archi, kwargs, env, policy_type):
    action_dim = env.action_space.low.size
    obs_dim = env.observation_space.spaces["observation"].low.size
    goal_dim = env.observation_space.spaces["representation_goal"].low.size
    if policy_type == "tanhgaussian":
        kwargs["obs_dim"] = obs_dim + goal_dim
        kwargs["action_dim"] = action_dim
    else:
        kwargs["output_size"] = action_dim

    if archi != "kinnet":
        kwargs["hidden_sizes"] = [kwargs.pop("hidden_dim")] * kwargs.pop("n_layers")

    if archi != "mlp":
        robot_props = env.robot_props
        obs_indices = env.obs_indices
        obstacles_dim = env.obstacles_dim
        coordinate_frame = env.coordinate_frame

    policy_class = archi_to_network(archi, policy_type)
    if archi == "mlp":
        if policy_type == "vanilla":
            kwargs["input_size"] = obs_dim + goal_dim
    elif "pointnet" in archi:
        obstacle_point_dim = env.obstacle_point_dim
        kwargs["q_action_dim"] = 0
        kwargs["robot_props"] = robot_props
        kwargs["obstacle_point_dim"] = obstacle_point_dim
        kwargs["input_indices"] = obs_indices
        kwargs["hidden_activation"] = F.elu
        kwargs["coordinate_frame"] = coordinate_frame
        # kwargs["hidden_activation"] = torch.sin
    elif archi == "cnn":
        kwargs.pop("obs_dim", None)
        kwargs["q_action_dim"] = 0
        kwargs["conv_sizes"] = (1, 16, 32, 64)
        kwargs["fc_sizes"] = (256, 256)
        kwargs["input_indices"] = obs_indices
        kwargs["robot_props"] = robot_props
        kwargs["coordinate_frame"] = coordinate_frame
        kwargs.pop("hidden_sizes")
    elif archi == "voxnet":
        kwargs["q_action_dim"] = 0
        kwargs["input_indices"] = obs_indices
        kwargs["robot_props"] = robot_props
        kwargs["coordinate_frame"] = coordinate_frame
    else:
        raise ValueError(f"Unknown network archi: {archi}")

    return policy_class, kwargs


def get_q_network(archi, kwargs, env, classification=False):
    action_dim = env.action_space.low.size
    obs_dim = env.observation_space.spaces["observation"].low.size
    goal_dim = env.observation_space.spaces["representation_goal"].low.size
    kwargs["output_size"] = 1
    q_action_dim = action_dim

    if archi != "kinnet":
        kwargs["hidden_sizes"] = [kwargs.pop("hidden_dim")] * kwargs.pop("n_layers")

    if archi != "mlp":
        robot_props = env.robot_props
        obs_indices = env.obs_indices
        obstacles_dim = env.obstacles_dim
        coordinate_frame = env.coordinate_frame

    qf_class = archi_to_network(archi, "vanilla")
    if archi == "mlp":
        kwargs["input_size"] = obs_dim + goal_dim + q_action_dim
    elif "pointnet" in archi:
        obstacle_point_dim = env.obstacle_point_dim
        kwargs["q_action_dim"] = q_action_dim
        kwargs["robot_props"] = robot_props
        kwargs["obstacle_point_dim"] = obstacle_point_dim
        kwargs["input_indices"] = obs_indices
        kwargs["hidden_activation"] = F.elu
        kwargs["coordinate_frame"] = coordinate_frame
        # kwargs["hidden_activation"] = torch.sin
    elif archi == "cnn":
        kwargs["q_action_dim"] = q_action_dim
        kwargs["conv_sizes"] = (1, 16, 32, 64)
        kwargs["fc_sizes"] = (256, 256)
        kwargs["input_indices"] = obs_indices
        kwargs["robot_props"] = robot_props
        kwargs["coordinate_frame"] = coordinate_frame
        kwargs.pop("hidden_sizes", None)
    elif archi == "voxnet":
        kwargs.pop("obs_dim", None)
        kwargs["q_action_dim"] = q_action_dim
        kwargs["input_indices"] = obs_indices
        kwargs["robot_props"] = robot_props
        kwargs["coordinate_frame"] = coordinate_frame
    else:
        raise ValueError(f"Unknown network archi: {archi}")

    return qf_class, kwargs
