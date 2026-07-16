# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based rough-terrain locomotion environment config for the custom humanoid robot.

Suggested file name:
    humanoid_robot_policy_env_cfg.py

This file is designed for the project:
    Humanoid_Robot_Policy

Self-collision note:
    Fall detection is based on root orientation and root height. Contact sensors
    are used only for foot stepping rewards, not whole-body fall detection.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from .humanoid_robot import HUMANOID_ROBOT_CFG

SMALL_RANDOM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    # Size of each generated terrain patch.
    size=(8.0, 8.0),

    # Flat border around the complete terrain grid.
    border_width=10.0,

    # Creates 10 × 20 = 200 terrain patches.
    num_rows=10,
    num_cols=20,

    # Resolution of the generated terrain.
    horizontal_scale=0.05,  # one mesh point every 5 cm
    vertical_scale=0.001,   # height resolution of 1 mm

    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,

    sub_terrains={
        "small_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,

            # Ground elevation varies from -5 mm to +5 mm.
            noise_range=(-0.005, 0.005),

            # Heights are generated in 1 mm increments.
            noise_step=0.001,

            # Random samples are generated every 10 cm and interpolated.
            # This produces smoother deviations instead of sharp noise.
            downsampled_scale=0.10,

            # Flat padding around each individual patch.
            border_width=0.25,
        ),
    },
)



##
# Robot-specific names
##

BASE_BODY_NAME = "base_link"

FOOT_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

TARGET_BASE_HEIGHT = 0.32
MIN_BASE_HEIGHT = 0.20
MAX_BASE_TILT = math.radians(65.0)

WOODEN_BAR_CURRICULUM_START_STEP = 20_000
WOODEN_BAR_CURRICULUM_FULL_STEP = 50_000

LEG_JOINT_NAMES = [
    ".*_leg_pitch_joint",
    ".*_leg_roll_joint",
    ".*_leg_yaw_joint",
    ".*_knee_pitch_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
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
    """Scene configuration for rough-terrain walking, turning, and wooden-bar crossing."""

    # Randomly rough ground used to improve locomotion robustness.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",

        # Generate small, smooth height variations instead of using an infinite plane.
        terrain_type="generator",
        terrain_generator=SMALL_RANDOM_ROUGH_TERRAIN_CFG,

        collision_group=-1,

        # Ground friction remains fixed.
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

    # Competition obstacle: 350 mm long with a 10 x 20 mm cross-section.
    # Use the rules' more difficult, upright 20 mm-high orientation.
    wooden_bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WoodenBar",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.35, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.02, 0.02),
            ),
        ),
        # The reset event keeps the obstacle hidden until its scheduled appearance.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    # Contact sensor used for foot stepping rewards only.
    # Do not use this for fall detection when self-collision is enabled,
    # because self-collision also produces contact forces.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
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
    """Command specifications for the MDP."""

    base_velocity = mdp.ForwardYawVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),

        # Increase stop-command sampling from 1% to 20%. Obstacle episodes keep
        # moving until the bar is crossed, so a stop cannot block its appearance.
        rel_standing_envs=0.20,
        bar_forward_speed=0.7,

        debug_vis=True,
        ranges=mdp.ForwardYawVelocityCommandCfg.Ranges(
            # Moving commands always request 0.7 m/s; only yaw rate is randomized.
            lin_vel_x=(0.7, 0.7),
            ang_vel_z=(-0.5, 0.5),
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
            func=mdp.forward_yaw_velocity_command,
            params={"command_name": "base_velocity"},
        )

        # Forward distance to the obstacle. It is 0.40 m before appearance and
        # after a successful crossing, with +/-5 mm simulated sensor noise.
        wooden_bar_distance = ObsTerm(
            func=mdp.wooden_bar_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            },
            noise=Unoise(n_min=-0.005, n_max=0.005),
            clip=(0.0, 0.405),
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

    # Randomize the physics material of the two feet.
    randomize_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),

            # One random material is assigned to each environment.
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.1),

            # Keep feet non-bouncy.
            "restitution_range": (0.0, 0.1),

            # Discretize the random range into material buckets.
            "num_buckets": 64,

            # Prevent physically inconsistent combinations such as
            # dynamic friction being greater than static friction.
            "make_consistent": True,
        },
    )

    reset_wooden_bar = EventTerm(
        func=mdp.reset_wooden_bar,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("wooden_bar"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # The interval is per-environment. The event function guarantees at most one
    # appearance per episode and refuses to spawn before two seconds have elapsed.
    spawn_wooden_bar = EventTerm(
        func=mdp.spawn_wooden_bar,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("wooden_bar"),
            "robot_cfg": SceneEntityCfg("robot"),
            "distance_range": (0.30, 0.40),
            "minimum_episode_time_s": 2.0,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the bipedal humanoid walking MDP.

    This is G1-like, but modified to fight two early local optima:
    1. standing still
    2. shuffling/sliding feet without real stepping
    """

    # -------------------------------------------------------------------------
    # Main task rewards
    # -------------------------------------------------------------------------

    # Stronger and sharper than your current version.
    # Your old std=0.5 was too forgiving, so standing still could still get reward.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_forward_velocity_exp,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "std": 0.35,
        },
    )

    # Track the sampled yaw-rate commands so the policy learns turning together
    # with forward locomotion.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_yaw_rate_exp,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "std": 0.3,
        },
    )

    # Lateral motion is no longer commanded, but sideways drift is still undesirable.
    lateral_velocity_l2 = RewTerm(
        func=mdp.lateral_velocity_l2,
        weight=-0.5,
    )

    wooden_bar_progress = RewTerm(
        func=mdp.wooden_bar_crossing_progress,
        weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    wooden_bar_foot_clearance = RewTerm(
        func=mdp.wooden_bar_foot_clearance,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "crossing_window": 0.08,
            "target_clearance": 0.04,
        },
    )

    wooden_bar_success = RewTerm(
        func=mdp.wooden_bar_crossing_success,
        weight=20.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("wooden_bar"),
        },
    )

    # -------------------------------------------------------------------------
    # Anti-shuffling / stepping terms
    # -------------------------------------------------------------------------

    # G1-like foot timing reward using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            # 0.4 is G1-like. For your smaller/custom robot, start slightly lower.
            "threshold": 0.25,
        },
    )

    # G1-like foot slide penalty using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.35,
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
    # Contact / termination terms
    # -------------------------------------------------------------------------

    # Strong penalty for early termination, G1-style.
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    # Disabled: fall is detected by root orientation and root height, not contact forces.
    illegal_non_foot_contact = None

    # -------------------------------------------------------------------------
    # Stability terms
    # -------------------------------------------------------------------------

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # Penalize unnecessary vertical base motion on the mildly rough terrain.
    # The terrain variation is small enough that proper foot lift is still possible.
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2,
    )

    # -------------------------------------------------------------------------
    # Joint / action penalties
    # -------------------------------------------------------------------------

    # Make these weaker at first.
    # If they are too strong, the easiest solution is "do not move".
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    # Slightly weaker than G1 at first. Increase later when walking works.
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.002,
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

    # Do not penalize leg pitch/knee pitch too much, because those are needed
    # for stepping. Only softly discourage sideways/yaw flailing.
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=YAW_ROLL_JOINT_NAMES,
            )
        },
    )

    #both feet airborn penalty
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
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

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": MAX_BASE_TILT,
        },
    )

    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
        },
    )

    wooden_bar_moved = DoneTerm(
        func=mdp.wooden_bar_moved,
        params={
            "asset_cfg": SceneEntityCfg("wooden_bar"),
            "position_tolerance": 0.002,
            "rotation_tolerance": math.radians(2.0),
        },
    )

    wooden_bar_timeout = DoneTerm(
        func=mdp.wooden_bar_timeout,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "timeout_s": 20.0,
        },
    )


##
# MDP: Curriculum
##

@configclass
class CurriculumCfg:
    """Begin obstacle training only after the walking/turning pretraining phase."""

    wooden_bar = CurrTerm(
        func=mdp.wooden_bar_curriculum,
        params={
            "start_step": WOODEN_BAR_CURRICULUM_START_STEP,
            "full_probability_step": WOODEN_BAR_CURRICULUM_FULL_STEP,
        },
    )


##
# Environment configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for rough-terrain forward-velocity and yaw-rate tracking."""

    # Scene settings.
    scene: HumanoidRobotPolicySceneCfg = HumanoidRobotPolicySceneCfg(
        num_envs=2048,
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
        # Allows the delayed obstacle appearance plus the full 20-second
        # crossing window, while remaining above the requested 30 seconds.
        self.episode_length_s = 35.0

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
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5

        # Longer episode for watching the policy.
        self.episode_length_s = 50.0

        # Fixed walking command for playback.
        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Playback starts directly at the obstacle stage so the trained crossing
        # behavior can be inspected without waiting for the training curriculum.
        self.curriculum.wooden_bar.params["start_step"] = -1
        self.curriculum.wooden_bar.params["full_probability_step"] = 0

        # Disable observation noise during playback.
        self.observations.policy.enable_corruption = False
