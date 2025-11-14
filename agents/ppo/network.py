import torch.nn as nn

policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128])
)