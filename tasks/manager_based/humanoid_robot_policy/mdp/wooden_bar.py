# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command, observation, event, reward, and termination terms for the wooden bar task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


WOODEN_BAR_LENGTH = 0.35
WOODEN_BAR_HEIGHT = 0.02
WOODEN_BAR_DEFAULT_DISTANCE = 0.40


class ForwardYawVelocityCommand(UniformVelocityCommand):
    """Generate only forward-velocity and yaw-rate commands.

    A three-component buffer is retained internally because the inherited
    metrics and visualizer use SE(2), but the public command is two-dimensional
    and never exposes a lateral-velocity component to the policy.
    """

    cfg: ForwardYawVelocityCommandCfg

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b[:, (0, 2)]

    def _resample_command(self, env_ids: Sequence[int]):
        samples = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = samples.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = samples.uniform_(*self.cfg.ranges.ang_vel_z)
        self.is_standing_env[env_ids] = samples.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        super()._update_command()

        active = getattr(self._env, "_wooden_bar_active", None)
        crossed = getattr(self._env, "_wooden_bar_crossed", None)
        episode_enabled = getattr(self._env, "_wooden_bar_episode_enabled", None)
        if active is None or crossed is None or episode_enabled is None:
            return

        obstacle_envs = episode_enabled & ~crossed
        self.is_standing_env[obstacle_envs] = False
        self.vel_command_b[obstacle_envs, 0] = self.cfg.bar_forward_speed
        self.vel_command_b[obstacle_envs, 1] = 0.0

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        desired_scale, desired_quat = self._resolve_xy_velocity_to_arrow(self.vel_command_b[:, :2])
        actual_scale, actual_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, desired_quat, desired_scale)
        self.current_vel_visualizer.visualize(base_pos_w, actual_quat, actual_scale)


@configclass
class ForwardYawVelocityCommandCfg(CommandTermCfg):
    """Configuration for a forward-velocity and yaw-rate command.

    The public command contains only ``(lin_vel_x, ang_vel_z)``. A lateral
    velocity command is deliberately not part of this command space.
    """

    class_type: type = ForwardYawVelocityCommand
    asset_name: str = MISSING
    rel_standing_envs: float = 0.0
    bar_forward_speed: float = 0.7

    # These attributes keep the class compatible with the visualization and
    # initialization implemented by UniformVelocityCommand.
    heading_command: bool = False
    heading_control_stiffness: float = 0.0
    rel_heading_envs: float = 0.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )


def _env_ids_tensor(env: ManagerBasedEnv, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_wooden_bar_state(env: ManagerBasedEnv):
    if hasattr(env, "_wooden_bar_active"):
        return

    num_envs = env.num_envs
    device = env.device
    env._wooden_bar_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._wooden_bar_crossed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._wooden_bar_episode_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._wooden_bar_reference_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._wooden_bar_success_rewarded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._wooden_bar_spawn_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    env._wooden_bar_reference_pos_w = torch.zeros(num_envs, 3, device=device)
    env._wooden_bar_reference_quat_w = torch.zeros(num_envs, 4, device=device)
    env._wooden_bar_reference_quat_w[:, 0] = 1.0
    env._wooden_bar_forward_w = torch.zeros(num_envs, 2, device=device)
    env._wooden_bar_forward_w[:, 0] = 1.0
    env._wooden_bar_episode_start_pos_w = torch.zeros(num_envs, 3, device=device)
    env._wooden_bar_previous_distance = torch.full(
        (num_envs,), WOODEN_BAR_DEFAULT_DISTANCE, device=device
    )
    env._wooden_bar_spawn_probability = 0.0


def wooden_bar_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_step: int,
    full_probability_step: int,
) -> float:
    """Ramp obstacle-episode probability from zero to one after locomotion pretraining."""
    del env_ids
    _ensure_wooden_bar_state(env)
    denominator = max(full_probability_step - start_step, 1)
    probability = (env.common_step_counter - start_step) / denominator
    env._wooden_bar_spawn_probability = float(max(0.0, min(1.0, probability)))
    return env._wooden_bar_spawn_probability


def reset_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Hide the bar and sample whether the new episode belongs to the obstacle curriculum."""
    _ensure_wooden_bar_state(env)
    env_ids = _env_ids_tensor(env, env_ids)
    bar: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    hidden_pose = torch.zeros(len(env_ids), 7, device=env.device)
    hidden_pose[:, :3] = env.scene.env_origins[env_ids]
    hidden_pose[:, 2] -= 1.0
    hidden_pose[:, 3] = 1.0
    bar.write_root_pose_to_sim(hidden_pose, env_ids=env_ids)
    bar.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)

    probability = env._wooden_bar_spawn_probability
    env._wooden_bar_episode_enabled[env_ids] = torch.rand(len(env_ids), device=env.device) < probability
    env._wooden_bar_active[env_ids] = False
    env._wooden_bar_crossed[env_ids] = False
    env._wooden_bar_reference_ready[env_ids] = False
    env._wooden_bar_success_rewarded[env_ids] = False
    env._wooden_bar_spawn_step[env_ids] = -1
    env._wooden_bar_previous_distance[env_ids] = WOODEN_BAR_DEFAULT_DISTANCE
    env._wooden_bar_episode_start_pos_w[env_ids] = robot.data.root_pos_w[env_ids]


def spawn_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_range: tuple[float, float] = (0.30, 0.40),
    minimum_episode_time_s: float = 2.0,
):
    """Place one bar 30-40 cm ahead after the robot has begun walking."""
    _ensure_wooden_bar_state(env)
    env_ids = _env_ids_tensor(env, env_ids)
    robot: Articulation = env.scene[robot_cfg.name]
    old_enough = env.episode_length_buf[env_ids] * env.step_dt >= minimum_episode_time_s
    moved_from_start = torch.linalg.norm(
        robot.data.root_pos_w[env_ids, :2] - env._wooden_bar_episode_start_pos_w[env_ids, :2],
        dim=-1,
    ) > 0.10
    eligible = (
        env._wooden_bar_episode_enabled[env_ids]
        & ~env._wooden_bar_active[env_ids]
        & old_enough
        & moved_from_start
    )
    env_ids = env_ids[eligible]
    if len(env_ids) == 0:
        return

    bar: RigidObject = env.scene[asset_cfg.name]
    heading = robot.data.heading_w[env_ids]
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    distance = torch.empty(len(env_ids), device=env.device).uniform_(*distance_range)

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = robot.data.root_pos_w[env_ids, :2] + distance.unsqueeze(-1) * forward
    # Leave 6 mm under the bar so it settles onto the existing +/-5 mm terrain.
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + 0.5 * WOODEN_BAR_HEIGHT + 0.006
    pose[:, 3] = torch.cos(0.5 * heading)
    pose[:, 6] = torch.sin(0.5 * heading)

    bar.write_root_pose_to_sim(pose, env_ids=env_ids)
    bar.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)

    env._wooden_bar_active[env_ids] = True
    env._wooden_bar_crossed[env_ids] = False
    env._wooden_bar_reference_ready[env_ids] = False
    env._wooden_bar_success_rewarded[env_ids] = False
    env._wooden_bar_spawn_step[env_ids] = env.episode_length_buf[env_ids]
    env._wooden_bar_reference_pos_w[env_ids] = pose[:, :3]
    env._wooden_bar_reference_quat_w[env_ids] = pose[:, 3:7]
    env._wooden_bar_forward_w[env_ids] = forward
    env._wooden_bar_previous_distance[env_ids] = distance


def _update_reference_after_settling(
    env: ManagerBasedRLEnv,
    settle_time_s: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
):
    _ensure_wooden_bar_state(env)
    age_s = (env.episode_length_buf - env._wooden_bar_spawn_step) * env.step_dt
    newly_ready = env._wooden_bar_active & ~env._wooden_bar_reference_ready & (age_s >= settle_time_s)
    if not torch.any(newly_ready):
        return

    bar: RigidObject = env.scene[asset_cfg.name]
    env._wooden_bar_reference_pos_w[newly_ready] = bar.data.root_pos_w[newly_ready]
    env._wooden_bar_reference_quat_w[newly_ready] = bar.data.root_quat_w[newly_ready]
    env._wooden_bar_reference_ready[newly_ready] = True


def _wooden_bar_moved_mask(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
    position_tolerance: float = 0.002,
    rotation_tolerance: float = math.radians(2.0),
) -> torch.Tensor:
    _update_reference_after_settling(env, asset_cfg=asset_cfg)
    bar: RigidObject = env.scene[asset_cfg.name]

    position_error = torch.linalg.norm(bar.data.root_pos_w - env._wooden_bar_reference_pos_w, dim=-1)
    quat_dot = torch.abs(torch.sum(bar.data.root_quat_w * env._wooden_bar_reference_quat_w, dim=-1))
    rotation_error = 2.0 * torch.acos(torch.clamp(quat_dot, 0.0, 1.0))
    return (
        env._wooden_bar_active
        & env._wooden_bar_reference_ready
        & ((position_error > position_tolerance) | (rotation_error > rotation_tolerance))
    )


def _update_crossed_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    forward_clearance: float = 0.06,
    lateral_margin: float = 0.03,
) -> torch.Tensor:
    _ensure_wooden_bar_state(env)
    robot: Articulation = env.scene[robot_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    relative_xy = foot_pos_w[:, :, :2] - env._wooden_bar_reference_pos_w[:, None, :2]
    forward = env._wooden_bar_forward_w[:, None, :]
    lateral = torch.stack((-forward[:, :, 1], forward[:, :, 0]), dim=-1)
    foot_forward = torch.sum(relative_xy * forward, dim=-1)
    foot_lateral = torch.sum(relative_xy * lateral, dim=-1)

    both_feet_past = torch.all(foot_forward > forward_clearance, dim=1)
    both_feet_inside_bar = torch.all(
        torch.abs(foot_lateral) < 0.5 * WOODEN_BAR_LENGTH + lateral_margin,
        dim=1,
    )
    newly_crossed = (
        env._wooden_bar_active
        & env._wooden_bar_reference_ready
        & ~env._wooden_bar_crossed
        & both_feet_past
        & both_feet_inside_bar
    )
    env._wooden_bar_crossed |= newly_crossed
    return newly_crossed


def wooden_bar_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return forward distance to the bar, or 0.40 m while absent/after crossing."""
    _ensure_wooden_bar_state(env)
    _update_crossed_state(env, foot_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    relative_xy = env._wooden_bar_reference_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.sum(relative_xy * env._wooden_bar_forward_w, dim=-1)

    observation = torch.full((env.num_envs,), WOODEN_BAR_DEFAULT_DISTANCE, device=env.device)
    visible = env._wooden_bar_active & ~env._wooden_bar_crossed
    observation[visible] = torch.clamp(distance[visible], min=0.0, max=WOODEN_BAR_DEFAULT_DISTANCE)
    return observation.unsqueeze(-1)


def forward_yaw_velocity_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return ``(lin_vel_x, ang_vel_z)`` and keep obstacle episodes moving until crossing."""
    command = env.command_manager.get_command(command_name).clone()
    _ensure_wooden_bar_state(env)
    obstacle_envs = env._wooden_bar_episode_enabled & ~env._wooden_bar_crossed
    command[obstacle_envs, 0] = 0.7
    return command


def track_forward_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking only the commanded forward velocity."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = forward_yaw_velocity_command(env, command_name)
    velocity_yaw_frame = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    error = torch.square(command[:, 0] - velocity_yaw_frame[:, 0])
    return torch.exp(-error / std**2)


def track_yaw_rate_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the yaw-rate component of the two-dimensional command."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = forward_yaw_velocity_command(env, command_name)
    error = torch.square(command[:, 1] - robot.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift without treating lateral velocity as a command."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 1])


def wooden_bar_crossing_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Give dense reward for forward progress toward and across an active bar."""
    _ensure_wooden_bar_state(env)
    robot: Articulation = env.scene[robot_cfg.name]
    relative_xy = env._wooden_bar_reference_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.sum(relative_xy * env._wooden_bar_forward_w, dim=-1)
    crossing = env._wooden_bar_active & ~env._wooden_bar_crossed
    progress = torch.clamp(env._wooden_bar_previous_distance - distance, min=0.0, max=0.03) / 0.03
    progress = progress * crossing.float()
    env._wooden_bar_previous_distance[crossing] = distance[crossing]
    return progress


def wooden_bar_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    crossing_window: float = 0.08,
    target_clearance: float = 0.04,
) -> torch.Tensor:
    """Reward lifting each foot while it is directly over the bar plane."""
    _ensure_wooden_bar_state(env)
    robot: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids]
    relative_xy = foot_pos_w[:, :, :2] - env._wooden_bar_reference_pos_w[:, None, :2]
    longitudinal = torch.sum(relative_xy * env._wooden_bar_forward_w[:, None, :], dim=-1)
    over_bar = torch.abs(longitudinal) < crossing_window
    clearance = foot_pos_w[:, :, 2] - (
        env._wooden_bar_reference_pos_w[:, None, 2] + 0.5 * WOODEN_BAR_HEIGHT
    )
    clearance_reward = torch.clamp(clearance / target_clearance, min=0.0, max=1.0)
    active = (env._wooden_bar_active & ~env._wooden_bar_crossed).unsqueeze(-1)
    return torch.sum(clearance_reward * over_bar.float() * active.float(), dim=1)


def wooden_bar_crossing_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
) -> torch.Tensor:
    """Return a one-time success reward after both feet cross without moving the bar."""
    _update_crossed_state(env, robot_cfg)
    moved = _wooden_bar_moved_mask(env, asset_cfg=asset_cfg)
    success = env._wooden_bar_crossed & ~env._wooden_bar_success_rewarded & ~moved
    env._wooden_bar_success_rewarded |= success
    return success.float()


def wooden_bar_moved(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("wooden_bar"),
    position_tolerance: float = 0.002,
    rotation_tolerance: float = math.radians(2.0),
) -> torch.Tensor:
    """Terminate if contact translates or rotates the settled wooden bar."""
    return _wooden_bar_moved_mask(
        env,
        asset_cfg=asset_cfg,
        position_tolerance=position_tolerance,
        rotation_tolerance=rotation_tolerance,
    )


def wooden_bar_timeout(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    timeout_s: float = 20.0,
) -> torch.Tensor:
    """Terminate if the robot has not fully crossed within 20 seconds of appearance."""
    _update_crossed_state(env, robot_cfg)
    elapsed_s = (env.episode_length_buf - env._wooden_bar_spawn_step) * env.step_dt
    return env._wooden_bar_active & ~env._wooden_bar_crossed & (elapsed_s > timeout_s)
