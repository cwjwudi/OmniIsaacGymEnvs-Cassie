from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.m03 import M03
from omniisaacgymenvs.robots.articulations.views.m03_view import UR10View
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class M03Task(RLTask):
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
        self._num_envs = 4
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])


        # call parent class’s __init__
        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        # implement environment setup here
        self.get_m03() # add a robot actor to the stage
        super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        # self._m03s = ArticulationView(prim_paths_expr="/World/envs/.*/M03", name="m03_view", reset_xform_properties=False)
        # scene.add(self._m03s) # add view to scene for initialization
        self._m03s = ArticulationView(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view", reset_xform_properties=False)
        scene.add(self._m03s)

    def get_m03(self):
        # applies articulation settings from the task configuration yaml file
        ur10 = M03(prim_path=self.default_zero_env_path + "/ur10", name="UR10", translation=self._cartpole_positions)
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    # def set_up_scene(self, scene) -> None:
    #     self.get_cartpole()
    #     super().set_up_scene(scene)
    #     self._cartpoles = ArticulationView(prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False)
    #     scene.add(self._cartpoles)
    #     return

    # def get_cartpole(self):
    #     cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
    #     # applies articulation settings from the task configuration yaml file
    #     self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))



    def post_reset(self):
        # implement any logic required for simulation on-start here
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # self.perform_reset()
        # self.apply_action(actions)
        self._m03s.set_joint_position_targets(torch.tensor([1.0, 0.0, 1.0, 2.0, 0.0, 0.0]), torch.tensor([0]))


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
