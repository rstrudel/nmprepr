import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmp.model.mlp_block import MLPBlock

# from nmp.model.utils import SublayerConnection, clones
from rlkit.torch import pytorch_util as ptu


def identity(x):
    return x


class PointNet(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_sizes,
        robot_props,
        obstacle_point_dim,
        q_action_dim,
        input_indices,
        coordinate_frame,
        output_activation=identity,
        init_w=3e-3,
        hidden_activation=F.elu,
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_indices = input_indices
        self.link_dim = robot_props[coordinate_frame]["link_dim"]
        self.config_dim = robot_props[coordinate_frame]["config_dim"]
        self.goal_dim = robot_props[coordinate_frame]["goal_rep_dim"]
        self.obstacle_point_dim = obstacle_point_dim
        self.coordinate_frame = coordinate_frame
        self.output_activation = output_activation

        self.q_action_dim = q_action_dim
        self.blocks_sizes = get_blocks_sizes(
            self.obstacle_point_dim,
            self.config_dim,
            self.goal_dim,
            self.q_action_dim,
            self.hidden_sizes,
            self.coordinate_frame,
        )

        self.block0 = MLPBlock(
            self.blocks_sizes[0],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        self.block1 = MLPBlock(
            self.blocks_sizes[1],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        self.init_last_fc(output_size, init_w)

    def init_last_fc(self, output_size, init_w=3e-3):
        self.last_fc = nn.Linear(self.hidden_sizes[-1], output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *input, return_features=False):
        obstacles, links, goal, action = process_input(
            self.input_indices, self.obstacle_point_dim, self.coordinate_frame, *input
        )
        batch_size = obstacles.shape[0]

        if self.coordinate_frame == "local":
            # early action integration
            h = torch.cat((obstacles, action), dim=2)
            # late action integration
            # h = obstacles
        elif self.coordinate_frame == "global":
            h = torch.cat((obstacles, links, goal, action), dim=2)

        h = self.block0(h)
        h = torch.max(h, 1)[0]

        if self.coordinate_frame == "local":
            if self.goal_dim > 0:
                h = torch.cat((h, goal), dim=1)

        h = self.block1(h)

        if return_features:
            return h

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()


def get_blocks_sizes(
    obstacle_point_dim,
    config_dim,
    goal_dim,
    q_action_dim,
    hidden_sizes,
    coordinate_frame,
):
    if coordinate_frame == "local":
        # early action integration
        obstacles_sizes = [obstacle_point_dim + q_action_dim] + hidden_sizes
        global_sizes = [hidden_sizes[0] + goal_dim] + hidden_sizes
    elif coordinate_frame == "global":
        obstacles_sizes = [
            obstacle_point_dim + config_dim + q_action_dim + goal_dim
        ] + hidden_sizes
        global_sizes = [hidden_sizes[0]] + hidden_sizes

    return obstacles_sizes, global_sizes


def process_input(input_indices, obstacle_point_dim, coordinate_frame, *input):
    """
    input: s or (s, a)
    BS x N
    """
    if len(input) > 1:
        out, action = input
    else:
        out, action = input[0], None

    batch_size = out.shape[0]
    obstacles = out[:, input_indices["obstacles"]]
    obstacles = obstacles.view(batch_size, -1, obstacle_point_dim)
    n_obsts = obstacles.shape[1]

    if action is None:
        action = torch.zeros((batch_size, 0), device=obstacles.device)
    # ealry action integration
    action = action.unsqueeze(1).expand(-1, n_obsts, -1)

    goal = out[:, input_indices["goal"]]
    if coordinate_frame == "global":
        goal = goal.unsqueeze(1).expand(batch_size, n_obsts, goal.shape[-1])

    links = None
    if coordinate_frame == "global":
        links = out[:, input_indices["robot"]]
        links = links.unsqueeze(1).expand(-1, n_obsts, -1)

    return obstacles, links, goal, action
