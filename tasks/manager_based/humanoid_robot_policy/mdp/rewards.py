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

def _ground_contact_force_norms(env: ManagerBasedRLEnv, sensor_names: list[str]) -> torch.Tensor:
    """Return max filtered ground-contact force per sensor and environment."""
    force_norms = []
    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        forces = getattr(sensor.data, "force_matrix_w_history", None)
        if forces is None:
            forces = sensor.data.force_matrix_w
        # Shapes are typically (num_envs, history, bodies, filters, 3) or
        # (num_envs, bodies, filters, 3).  Reducing every non-env/non-vector
        # dimension produces one ground-contact force magnitude per env.
        force_norm = torch.linalg.norm(forces, dim=-1)
        reduce_dims = tuple(range(1, force_norm.ndim))
        force_norms.append(torch.amax(force_norm, dim=reduce_dims))
    return torch.stack(force_norms, dim=1)


def ground_contact_count(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float,
) -> torch.Tensor:
    """Penalize only non-foot body contacts with the ground plane.

    The configured sensors are filtered to ``/World/ground``, so self-collision
    contacts between robot links do not contribute to this reward term.
    """
    force_norms = _ground_contact_force_norms(env, sensor_names)
    return torch.sum(force_norms > threshold, dim=1).float()


def illegal_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float,
) -> torch.Tensor:
    """Terminate only when a non-foot body touches the ground plane."""
    force_norms = _ground_contact_force_norms(env, sensor_names)
    return torch.any(force_norms > threshold, dim=1)
