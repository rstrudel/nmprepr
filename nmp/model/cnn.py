import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)


class CNN(nn.Module):
    def __init__(
        self,
        conv_sizes,
        fc_sizes,
        output_size,
        input_indices,
        robot_props,
        q_action_dim,
        coordinate_frame,
        init_w=3e-3,
        hidden_init=nn.init.xavier_uniform_,
        fc_hidden_init=ptu.fanin_init,
    ):
        super().__init__()
        self.conv_sizes = conv_sizes
        self.output_size = output_size
        self.input_indices = input_indices
        self.config_dim = robot_props["config_dim"]
        self.goal_dim = robot_props["goal_rep_dim"]
        self.q_action_dim = q_action_dim
        self.coordinate_frame = coordinate_frame

        kernel_sizes = [3, 3, 3]
        strides = [1, 2, 2]
        padding = 0
        cnn = []
        for i in range(len(conv_sizes) - 1):
            conv_filter = nn.Conv2d(
                conv_sizes[i],
                conv_sizes[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=padding,
            )
            # normalization_layer = nn.BatchNorm2d(conv_sizes[i + 1])
            cnn.append(conv_filter)
            # cnn.append(normalization_layer)
            if i < len(conv_sizes) - 2:
                cnn.append(nn.ReLU())
        cnn += [nn.MaxPool2d(kernel_size=6), Flatten()]
        self.cnn = nn.ModuleList(cnn)
        fc_head = []
        if coordinate_frame == "local":
            fc_sizes = [256 + self.goal_dim + self.q_action_dim] + list(fc_sizes)
        elif coordinate_frame == "global":
            # fc_sizes = [256 + self.config_dim + self.q_action_dim] + list(fc_sizes)
            fc_sizes = [
                256 + self.config_dim + self.goal_dim + self.q_action_dim
            ] + list(fc_sizes)
        self.fc_sizes = fc_sizes
        for i in range(len(fc_sizes) - 1):
            fc_layer = nn.Linear(fc_sizes[i], fc_sizes[i + 1])
            # normalization_layer = nn.BatchNorm1d(fc_sizes[i + 1])
            # fc_head += [fc_layer, normalization_layer, nn.ReLU()]
            fc_head += [fc_layer, nn.ReLU()]
        self.fc_head = nn.ModuleList(fc_head)

        self.last_fc = nn.Linear(self.fc_sizes[-1], output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def process_input(self, *input):
        """
        input: s or (s, a)
        BS x N
        """
        if len(input) > 1:
            x, action = input
        else:
            x, action = input[0], None

        batch_size = x.shape[0]

        obstacles = x[:, self.input_indices["obstacles"]]
        # assume the image is always a square
        im_size = np.sqrt(obstacles.size(1)).astype(int)
        obstacles = obstacles.reshape(-1, 1, im_size, im_size)

        if action is None:
            action = torch.zeros((batch_size, 0), device=obstacles.device)

        links = x[:, self.input_indices["robot"]]
        goal = x[:, self.input_indices["goal"]]

        return obstacles, links, goal, action

    def forward(self, *input, return_preactivations=False, return_features=False):
        obstacles, links, goal, action = self.process_input(*input)

        h_obst = obstacles
        for elem in self.cnn:
            h_obst = elem(h_obst)

        if self.coordinate_frame == "local":
            h = torch.cat((h_obst, goal, action), dim=1)
        elif self.coordinate_frame == "global":
            # h = torch.cat((h_obst, links, action), dim=1)
            h = torch.cat((h_obst, links, goal, action), dim=1)
        for elem in self.fc_head:
            h = elem(h)

        if return_features:
            return h

        preactivation = self.last_fc(h)
        output = preactivation

        if return_preactivations:
            return output, preactivation
        else:
            return output

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()
