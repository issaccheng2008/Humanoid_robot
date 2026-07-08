from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def forward_velocity_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_command: float = 0.10,
) -> torch.Tensor:
    """Reward actually moving forward when a forward command is given.

    This is intentionally simple. It helps break the standing-still local optimum.
    The reward saturates at 1.0, so the robot is not rewarded for running much faster
    than the command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    commanded_vx = torch.clamp(command[:, 0], min=0.0)
    actual_vx = asset.data.root_lin_vel_b[:, 0]

    active = (commanded_vx > min_command).float()
    normalized_vx = actual_vx / (commanded_vx + 1.0e-6)

    return active * torch.clamp(normalized_vx, min=0.0, max=1.0)


def standing_still_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_command: float = 0.10,
    min_speed: float = 0.12,
) -> torch.Tensor:
    """Penalize standing still when the command asks the robot to move forward.

    This directly attacks the behavior you described: the policy stands or shuffles
    instead of producing forward locomotion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    commanded_vx = torch.clamp(command[:, 0], min=0.0)
    actual_vx = asset.data.root_lin_vel_b[:, 0]

    active = (commanded_vx > min_command).float()

    # Positive when actual forward speed is too small.
    speed_deficit = torch.clamp(min_speed - actual_vx, min=0.0)

    return active * speed_deficit