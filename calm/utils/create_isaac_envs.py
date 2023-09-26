import os
import yaml

import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import gym
from gym.spaces import Tuple as GymTuple

from tairlearning.sim_envs.isaac_envs.tasks.ase_torch import ASETorch

from tairlearning.sim_envs.isaac_envs.tasks.humanoid_strike import HumanoidStrike

from tairlearning.sim_envs.isaac_envs.tasks.vec_task_wrappers import VecTaskPythonWrapper
from tairlearning.env_wrappers.drill_wrapper.z_action_wrapper import ZActionWrapper

SIM_TIMESTEP = 1.0 / 60.0


def parse_sim_params(env_config, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = env_config["slices"]

    if env_config["physics_engine"] == gymapi.SIM_FLEX:
        if env_config["device"] != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif env_config["physics_engine"] == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = env_config["use_gpu"]
        sim_params.physx.num_subscenes = env_config["subscenes"]
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = env_config["use_gpu_pipeline"]
    sim_params.physx.use_gpu = env_config["use_gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if env_config["physics_engine"] == gymapi.SIM_PHYSX and env_config["num_threads"] > 0:
        sim_params.physx.num_threads = env_config["num_threads"]

    return sim_params


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        # valid for actor
        self.observation_space = GymTuple([env.observation_space])
        self.action_space = GymTuple([env.action_space])

    def reset(self, env_ids=None, **kwargs):
        obs = self.env.reset(env_ids=env_ids)  # do not pass args to reset()
        return (obs,)

    def step(self, action):
        obs, rwd, done, info = super(SingleAgentWrapper, self).step(action[0])
        if "post_process_data" in info:
            info["post_process_data"] = (info["post_process_data"],)
        return (obs,), (rwd,), done, info


def create_isaac_env(**env_config):
    arena_id = env_config["arena_id"]

    headless = env_config.setdefault("headless", False)
    episode_length = env_config.setdefault("episode_length", 0)
    num_envs = env_config.setdefault("num_envs", 0)
    cfg_env = env_config.setdefault("cfg_env", "")
    seed = env_config.setdefault("seed", -1)
    motion_file = env_config.setdefault("motion_file", "")
    env_config.setdefault("rl_device", "cuda:0")
    env_config.setdefault("sim_device", "cuda:0")
    env_config.setdefault("pipeline", "gpu")
    env_config.setdefault("graphics_device_id", 0)
    env_config.setdefault("flex", False)
    env_config.setdefault("physx", False)
    env_config.setdefault("num_threads", 0)
    env_config.setdefault("subscenes", 0)
    env_config.setdefault("slices", None)
    env_config["sim_device_type"], env_config["compute_device_id"] = gymutil.parse_device_str(env_config["sim_device"])
    pipeline = env_config["pipeline"].lower()

    assert (pipeline == 'cpu' or pipeline in (
        'gpu', 'cuda')), f"Invalid pipeline '{env_config['pipeline']}'. Should be either cpu or gpu."
    env_config["use_gpu_pipeline"] = (pipeline in ('gpu', 'cuda'))

    if env_config["sim_device_type"] != 'cuda' and env_config["flex"]:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        env_config["sim_device"] = 'cuda:0'
        env_config["sim_device_type"], env_config["compute_device_id"] = gymutil.parse_device_str(
            env_config["sim_device"])

    if (env_config["sim_device_type"] != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        env_config["pipeline"] = 'CPU'
        env_config["use_gpu_pipeline"] = False

    # Default to PhysX
    env_config["physics_engine"] = gymapi.SIM_PHYSX
    env_config["use_gpu"] = (env_config["sim_device_type"] == 'cuda')

    if env_config["flex"]:
        env_config["physics_engine"] = gymapi.SIM_FLEX

    if env_config["slices"] is None:
        env_config["slices"] = env_config["subscenes"]
    env_config["device_id"] = env_config["compute_device_id"]
    env_config["device"] = env_config["sim_device_type"] if env_config["use_gpu_pipeline"] else 'cpu'

    with open(os.path.join(os.getcwd(), cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if num_envs > 0:
        cfg["env"]["numEnvs"] = num_envs

    if episode_length > 0:
        cfg["env"]["episodeLength"] = episode_length

    cfg["name"] = arena_id
    cfg["headless"] = headless

    cfg["task"] = {"randomize": False}
    cfg["seed"] = seed
    if motion_file:
        cfg['env']['motion_file'] = motion_file

    sim_params = parse_sim_params(env_config, cfg)

    # create native task and pass custom config
    device_id = env_config["device_id"]
    rl_device = env_config["rl_device"]
    cfg["seed"] = -1
    cfg["env"].update(env_config)
    if env_config["arena_id"] in ["HumanoidTrackingGetup-v0", 'HumanoidTrackingBoxing-v0']:
        cfg["env"]["dynamic_sample_strategy"] = env_config.get("dynamic_sample_strategy", None)
        cfg["env"]["dynamic_sample_save_path"] = env_config.get("dynamic_sample_save_path", None)
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidTrackingGetup(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = SingleAgentWrapper(env)
    elif env_config["arena_id"] == 'HumanoidTrackingParkour-v0':
        cfg["env"]["dynamic_sample_strategy"] = env_config.get("dynamic_sample_strategy", None)
        cfg["env"]["dynamic_sample_save_path"] = env_config.get("dynamic_sample_save_path", None)
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidTrackingFull(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = SingleAgentWrapper(env)
    elif env_config["arena_id"] == "HumanoidTrackingPrior-v0" or env_config["arena_id"] == 'HumanoidTrackingBoxingPrior-v0':
        cfg["env"]["dynamic_sample_strategy"] = env_config.get("dynamic_sample_strategy", None)
        cfg["env"]["dynamic_sample_save_path"] = env_config.get("dynamic_sample_save_path", None)
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidTrackingPrior(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = ZActionWrapper(env, z_len=env_config['num_embeddings'], z_type='Discrete')
        env = SingleAgentWrapper(env)
    elif env_config["arena_id"] == "HumanoidPlay-v0":
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidViewMotion(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
    elif env_config["arena_id"] == "HumanoidPlayPair-v0":
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidViewMotionPair(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
    elif env_config["arena_id"] == "HumanoidPlayGraph-v0":
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidViewMotionGraph(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
    elif env_config["arena_id"] == "HumanoidHeading-v0":
        cfg["env"]["dynamic_sample_strategy"] = env_config.get("dynamic_sample_strategy", None)
        cfg["env"]["dynamic_sample_save_path"] = env_config.get("dynamic_sample_save_path", None)
        cfg["env"]['motion_index'] = env_config.get("motion_index", None)
        task = HumanoidHeading(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = ZActionWrapper(env, z_len=env_config['num_embeddings'], z_type='Discrete')
        env = SingleAgentWrapper(env)
    elif env_config["arena_id"] == "HumanoidStrike-v0":
        task = HumanoidStrike(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)

        env = ZActionWrapper(env, z_len=env_config['num_embeddings'], z_type='Discrete')
        env = SingleAgentWrapper(env)
    elif env_config["arena_id"] == "HumanoidFighting-v0" or env_config["arena_id"] == "HumanoidBoxingFighting-v0":
        if env_config["arena_id"] == "HumanoidFighting-v0":
            from tairlearning.sim_envs.isaac_envs.tasks.humanoid_fighting import HumanoidFighting
        else:
            from tairlearning.sim_envs.isaac_envs.tasks.humanoid_fighting_boxing import HumanoidFighting
        task = HumanoidFighting(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = ZActionWrapper(env, z_len=env_config['num_embeddings'], z_type='Discrete')
    elif env_config["arena_id"] == "HumanoidStrike2-v0":
        task = HumanoidStrike(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=env_config["physics_engine"],
            device_type=env_config["device"],
            device_id=device_id,
            headless=env_config["headless"])
        env = VecTaskPythonWrapper(task, rl_device, np.inf, 1.0)
        env = ASETorch(env)
        env = ZActionWrapper(env, z_len=64, z_type='Box')
        env = SingleAgentWrapper(env)
    else:
        raise ValueError

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    return env
