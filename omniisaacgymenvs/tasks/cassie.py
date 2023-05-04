from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cassie import Cassie
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math


class CassieTask(RLTask):
    def __init__(
            self,
            name: str,  # name of the Task
            sim_config,  # SimConfig instance for parsing cfg
            env,  # env instance of VecEnvBase or inherited class
            offset=None  # transform offset in World
    ) -> None:

        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._init_cfg()
        # call parent class’s __init__
        RLTask.__init__(self, name, env)
        self._prepare_reward_function()
        self._init_buffers()
        return

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_robots()  # add a robot actor to the stage
        super().set_up_scene(
            scene)  # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/cassie", name="cassie_view")
        scene.add(self._robots)

    def get_robots(self):
        # applies articulation settings from the task configuration yaml file
        robot = Cassie(prim_path=self.default_zero_env_path + "/cassie", name="Cassie",
                       translation=self._robot_translation)
        self._sim_config.apply_articulation_settings(
            "Cassie",
            get_prim_at_path(robot.prim_path),
            self._sim_config.parse_actor_config("Cassie"),
        )
        # Configure joint properties
        joint_paths = ["pelvis/hip_abduction_left",
                       "left_pelvis_rotation/hip_rotation_left",
                       "left_hip/hip_flexion_left",
                       "left_thigh/thigh_joint_left",
                       "left_shin/ankle_joint_left",
                       "left_tarsus/toe_joint_left",

                       "pelvis/hip_abduction_right",
                       "right_pelvis_rotation/hip_rotation_right",
                       "right_hip/hip_flexion_right",
                       "right_thigh/thigh_joint_right",
                       "right_shin/ankle_joint_right",
                       "right_tarsus/toe_joint_right",
                       ]

        index = 0
        for joint_path in joint_paths:
            set_drive(f"{robot.prim_path}/{joint_path}", drive_type="angular", target_type="position",
                      target_value=0, stiffness=self.Kp[index], damping=self.Kd[index], max_force=self.max_force)
            index += 1

        self.default_dof_pos = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self._device,
                                           requires_grad=False)
        dof_names = robot.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )

        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device,
                                        requires_grad=False)

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # 重置环境
        self.perform_reset()
        # 执行动作
        self.apply_action(actions)

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            # update buffer data
            self.update_buffers()
            # 获取观测
            self.get_states()
            self.get_observations()
            # 计算奖励
            self.compute_rewards()
            # 判断done
            self.is_done()
            # 附加信息
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def perform_reset(self) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def apply_action(self, actions) -> None:
        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self.actions[:] = actions.clone().to(self._device)
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt
        self.current_targets[:] = tensor_clamp(current_targets, self.robot_dof_lower_limits,
                                               self.robot_dof_upper_limits)
        self._robots.set_joint_position_targets(self.current_targets, indices)

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel * self.ang_vel_scale,
                self.projected_gravity,
                self.commands_scaled,
                (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                self.dof_vel * self.dof_vel_scale,
                self.actions
            ),
            dim=-1,
        )
        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def update_buffers(self):
        self.dof_pos = self._robots.get_joint_positions(clone=False)
        self.dof_vel = self._robots.get_joint_velocities(clone=False)
        actions_scaled = self.action_scale * self.actions
        self.torques = self.Kp_tensor * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.Kd_tensor * self.dof_vel
        # self.actions = torch.zeros_like(self.dof_pos)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)

        base_vel = self._robots.get_velocities(clone=False)
        base_lin_vel = base_vel[:, 0:3]
        base_ang_vel = base_vel[:, 3:6]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, base_lin_vel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, base_ang_vel)

        self.projected_gravity = quat_rotate(self.base_quat, self.gravity_vec)

        self.commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

    def compute_rewards(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.rew_functions)):
            name = self.rew_names[i]
            rew = self.rew_functions[i]() * self.rew_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.rew_scales:
            rew = self._reward_termination() * self.rew_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        self.fallen_over = self.is_base_below_threshold(threshold=self._reset_threshold, ground_heights=0.0)
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
        return

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities

        self.current_targets[env_ids] = dof_pos[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._robots.set_joint_positions(dof_pos, indices)
        self._robots.set_joint_velocities(dof_vel, indices)

        self._robots.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(),
                                     indices)
        self._robots.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.

    def is_base_below_threshold(self, threshold, ground_heights):
        base_pos, _ = self._robots.get_world_poses()
        base_heights = base_pos[:, 2]
        base_heights -= ground_heights
        return (base_heights[:] < threshold)

    def _init_cfg(self):
        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.max_force = self._task_cfg["env"]["control"]["maxForce"]

        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._robot_translation = torch.tensor(self._task_cfg["env"]["baseInitState"]["pos"])
        self._reset_threshold = self._task_cfg["env"]["resetThreshold"]

    def _init_buffers(self):
        # init later used data
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self._device, requires_grad=False)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.torques = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros_like(self.dof_pos)
        self.last_actions = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)

        self.base_pos = torch.zeros(self.num_envs, 3,
                                    dtype=torch.float, device=self._device, requires_grad=False)
        self.base_quat = torch.zeros(self.num_envs, 4,
                                     dtype=torch.float, device=self._device, requires_grad=False)
        self.base_lin_vel = torch.zeros(self.num_envs, 3,
                                        dtype=torch.float, device=self._device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3,
                                        dtype=torch.float, device=self._device, requires_grad=False)

        self.projected_gravity = torch.zeros(self.num_envs, 3,
                                             dtype=torch.float, device=self._device, requires_grad=False)

        self.Kp_tensor = torch.tensor(self.Kp, dtype=torch.float, device=self._device, requires_grad=False)
        self.Kd_tensor = torch.tensor(self.Kd, dtype=torch.float, device=self._device, requires_grad=False)

    def _prepare_reward_function(self):
        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        # remove zero scales + multiply non-zero ones by dt
        for key in self.rew_scales.keys():
            scale = self.rew_scales[key]
            if scale == 0:
                self.rew_scales.pop(key)
            else:
                self.rew_scales[key] *= self.dt
        # prepare list of functions
        self.rew_functions = []
        self.rew_names = []
        for name, scale in self.rew_scales.items():
            if name == "termination":
                continue
            self.rew_names.append(name)
            name = '_reward_' + name
            self.rew_functions.append(getattr(self, name))
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self._device, requires_grad=False)
                             for name in self.rew_scales.keys()}

    # ---------------------reward functions--------------------------------------------------
    def _reward_lin_vel_xy(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_z(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_joint_acc(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torque(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
