from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.go1 import Go1

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class Go2Task(RLTask):
    def __init__(
        self,
        name: str,                # name of the Task
        sim_config,    # SimConfig instance for parsing cfg
        env,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_observations = 4
        self._num_actions = 11
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_envs = 5

        self._robot_positions = torch.tensor([0.0, 0.0, 0.5])


        # call parent class’s __init__
        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_robot() # add a robot actor to the stage
        super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/go1", name="go1_view", reset_xform_properties=False)
        scene.add(self._robots)

    def get_robot(self):
        # applies articulation settings from the task configuration yaml file
        robot = Go1(prim_path=self.default_zero_env_path + "/go1", name="Go1", translation=self._robot_positions)
        self._sim_config.apply_articulation_settings(
            "go1",
            get_prim_at_path(robot.prim_path),
            self._sim_config.parse_actor_config("go1"),
        )

    def post_reset(self):
        # implement any logic required for simulation on-start here
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # self.perform_reset()
        # self.apply_action(actions)
        # self._robots.set_joint_position_targets(torch.tensor([1.0, 0.0, 1.0, 2.0, 0.0, 0.0]), torch.tensor([0]))
        pass


    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)
        self.obs_buf = torch.cat(
            (
                dof_pos,
                dof_vel,
            ),
            dim=-1,
        )
        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # self.rew_buf = self.compute_rewards()
        return torch.tensor([1.0])


    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # self.reset_buf = self.compute_resets()
        return torch.tensor([0.0])
