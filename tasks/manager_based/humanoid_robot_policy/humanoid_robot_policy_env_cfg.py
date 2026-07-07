# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based locomotion environment config for the custom humanoid robot.

Suggested file name:
    humanoid_robot_policy_env_cfg.py

This file is designed for the project:
    Humanoid_Robot_Policy

Competition rule:
    Only l_ankle_roll_link and r_ankle_roll_link are allowed to touch the ground.
    If any other robot body touches the ground, the episode terminates.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .humanoid_robot import HUMANOID_ROBOT_CFG


##
# Robot-specific names
##

BASE_BODY_NAME = "base_link"

# These are the only two links allowed to touch the ground.
FOOT_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

# Competition rule:
# if any of these bodies touches the ground, the robot is out.
NON_FOOT_CONTACT_BODY_NAMES = [
    "base_link",

    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",

    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
]

LEG_JOINT_NAMES = [
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",

    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]

ANKLE_JOINT_NAMES = [
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

YAW_ROLL_JOINT_NAMES = [
    ".*_leg_yaw_joint",
    ".*_leg_roll_joint",
]


##
# Scene definition
##

@configclass
class HumanoidRobotPolicySceneCfg(InteractiveSceneCfg):
    """Scene configuration for the custom humanoid walking task."""

    # Ground plane.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot.
    robot: ArticulationCfg = HUMANOID_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Contact sensor over the entire robot.
    #
    # This must cover all bodies because we need to detect illegal contact from
    # any non-foot link.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Light.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


##
# MDP: Commands
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    The command tells the policy what walking velocity to track.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Start easy: slow forward walking.
            lin_vel_x=(0.0, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


##
# MDP: Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    The policy outputs joint-position targets for the robot's actuated leg joints.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )


##
# MDP: Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the policy network."""

        # Base motion.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # Orientation relative to gravity.
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Commanded walking velocity.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Joint state.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # Previous action.
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """Post initialization."""
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP: Events
##

@configclass
class EventCfg:
    """Configuration for events.

    Events handle startup/reset randomization.
    This first version is conservative for debugging.
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Keep default standing pose at first.
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# MDP: Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -------------------------------------------------------------------------
    # Task rewards
    # -------------------------------------------------------------------------

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # Encourage biped stepping.
    # This uses only the two legal foot contact links.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.20,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            "threshold": 0.4,
        },
    )

    # -------------------------------------------------------------------------
    # Competition contact rule
    # -------------------------------------------------------------------------

    # Strong penalty when the episode terminates early.
    # Since illegal non-foot contact is a termination condition, this heavily
    # punishes touching the ground with anything except the two feet.
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    # Direct penalty for non-foot body contact.
    # This gives the policy a learning signal before/alongside termination.
    illegal_non_foot_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=NON_FOOT_CONTACT_BODY_NAMES,
            ),
            # Start with 1.0. If false terminations occur, try 5.0 while debugging.
            "threshold": 1.0,
        },
    )

    # -------------------------------------------------------------------------
    # Stability rewards / penalties
    # -------------------------------------------------------------------------

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=0.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # Penalize legal feet sliding.
    # This applies only to the two allowed contact links.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.10,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Joint / action penalties
    # -------------------------------------------------------------------------

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ANKLE_JOINT_NAMES,
            )
        },
    )

    # Softly discourage excessive leg yaw/roll motion at the beginning.
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=YAW_ROLL_JOINT_NAMES,
            )
        },
    )


##
# MDP: Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # Competition rule:
    # terminate immediately when anything except the two feet touches the ground.
    illegal_non_foot_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=NON_FOOT_CONTACT_BODY_NAMES,
            ),
            # Start with 1.0. If false terminations occur, try 5.0 while debugging.
            "threshold": 1.0,
        },
    )


##
# MDP: Curriculum
##

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Empty for now because this first version trains on a flat plane.
    Add terrain curriculum later after the robot can stand/walk.
    """

    pass


##
# Environment configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the custom humanoid velocity-tracking environment."""

    # Scene settings.
    scene: HumanoidRobotPolicySceneCfg = HumanoidRobotPolicySceneCfg(
        num_envs=1024,
        env_spacing=2.5,
    )

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings.
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # General settings.
        self.decimation = 4
        self.episode_length_s = 20.0

        # Simulation settings.
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Sensor update periods.
        self.scene.contact_forces.update_period = self.sim.dt

        # Viewer.
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)


##
# Play / visualization configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg_PLAY(HumanoidRobotPolicyEnvCfg):
    """Smaller, cleaner configuration for policy playback."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Longer episode for watching the policy.
        self.episode_length_s = 40.0

        # Fixed walking command for playback.
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable observation noise during playback.
        self.observations.policy.enable_corruption = False