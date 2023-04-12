from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur10 import UR10

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class UR10Task(RLTask):
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
        self._num_envs = 1
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])


        # call parent class’s __init__
        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_ur10() # add a robot actor to the stage
        super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        self._ur10s = ArticulationView(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view", reset_xform_properties=False)
        scene.add(self._ur10s)

    def get_ur10(self):
        # applies articulation settings from the task configuration yaml file
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="UR10", translation=self._cartpole_positions)
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    def post_reset(self):
        # implement any logic required for simulation on-start here
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # self.perform_reset()
        # self.apply_action(actions)
        self._ur10s.set_joint_position_targets(torch.tensor([1.0, 0.0, 1.0, 2.0, 0.0, 0.0]), torch.tensor([0]))


    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        # self.obs_buf = self.compute_observations()
        return torch.tensor([1.0, 0.0, 0.0])


    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # self.rew_buf = self.compute_rewards()
        return torch.tensor([1.0])


    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # self.reset_buf = self.compute_resets()
        return torch.tensor([0.0])
