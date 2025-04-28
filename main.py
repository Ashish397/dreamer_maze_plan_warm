# ============================================================================
#   TO-DO
# ----------------------------------------------------------------------------
# - Seed
# - Make videos

#Secondaries
# - Maybe gen model predictive videos?

# ============================================================================
#   IMPORTS
# ----------------------------------------------------------------------------

import shutil
import zipfile
import datetime

import subprocess

import gc
import os
import pathlib
import random 
import torch

import ruamel.yaml as yaml

from pathlib import Path
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import ToTensorImage
from types import SimpleNamespace

import dreamer
import tools

from torchrl.envs import (
    Compose,
    InitTracker,
    ObservationNorm,
    PermuteTransform,
    RewardScaling,
    StepCounter,
    TransformedEnv,
)

# ============================================================================
#   FUNCTIONS
# ----------------------------------------------------------------------------

def load_data(replay_buffer, data_path, batch_length=-1, randomize=True, indices_to_exclude=None):
    """
    Load data from a memmapped TensorDict and populate the given replay buffer.

    Args:
        replay_buffer (ReplayBuffer): The buffer to populate with loaded samples.
        data_path (str or Path): Path to the memmapped TensorDict.
        batch_length (int): Number of trajectories to load. If -1, load all.
        randomize (bool): Whether to shuffle the loaded indices.
        indices_to_exclude (int or list[int], optional): Indices to skip during loading.

    Raises:
        ValueError: If indices_to_exclude is not None, int, or list of ints.
    """
    
    # Load the TensorDict from the memmap
    loaded = TensorDict.load_memmap(data_path)

    # Ensure indices_to_exclude is a list
    if indices_to_exclude is None:
        indices_to_exclude = []
    elif isinstance(indices_to_exclude, int):
        indices_to_exclude = [indices_to_exclude]
    elif not isinstance(indices_to_exclude, list):
        raise ValueError("indices_to_exclude must be an int or a list of ints")
    
    if batch_length > 0:
        # Create a list of indices to iterate over, excluding specified indices
        indices = [i for i in range(batch_length) if i not in indices_to_exclude]
    else:
        # Create a list of indices to iterate over, excluding specified indices
        indices = [i for i in range(len(loaded)) if i not in indices_to_exclude]

    # Randomize the indices if requested
    if randomize:
        random.shuffle(indices)

    # Extend the replay buffer with data at each index
    for i in indices:
        replay_buffer.extend(loaded[i:i+1])

    del loaded

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return commit + ("-dirty" if dirty else "")
    except Exception as e:
        return f"Git info unavailable: {e}"
    
def make_config(path="configs.yaml"):
    """
    Load and parse the configuration from a YAML file into a SimpleNamespace.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        SimpleNamespace: Config object with attribute access to nested values.
    """
    def recursive_update(base, update):
        """
        Recursively merges `update` into `base`, modifying `base` in-place.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    def dict_to_namespace(d):
        """
        Converts a nested dictionary into a nested SimpleNamespace.
        """
        return SimpleNamespace(**{k: v for k, v in d.items()})

    yaml_parser = yaml.YAML(typ='safe', pure=True)
    config_yaml = yaml_parser.load((pathlib.Path(path)).read_text())
    config = {}
    recursive_update(config, config_yaml['defaults'])
    return dict_to_namespace(config)

def main():
    # ============================================================================
    #   SETUP
    # ----------------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device(0)
        print('CUDA initialised')
    else:
        print("WARNING - CUDA not available - no hardware acceleration being used")

    config = make_config()
    tools.set_seed_everywhere(config.seed)

    experiment_name = config.exp_name
    exp_date = config.exp_date
    path_root = config.path_root
    data_path_preloaded = config.data_path_preloaded

    results_path = f"{path_root}\\{experiment_name}\\results"
    save_path = f"{path_root}\\{experiment_name}\\saved_models"
    logdir = Path(f"{path_root}\\{experiment_name}\\logs")
    data_path = f"{path_root}\\{experiment_name}\\dataset"

    for p in [results_path, f"{results_path}\\{exp_date}", save_path, logdir, data_path]:
        os.makedirs(p, exist_ok=True)

    # Create experiment folder
    root_dir = Path(os.getcwd())
    experiments_dir = root_dir / "experiments"
    experiment_folder = experiments_dir / config.exp_date

    # Delete if exists, then recreate
    if experiment_folder.exists():
        shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)


    # ============================================================================
    #   GLOBALS
    # ----------------------------------------------------------------------------

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=int(config.dataset_size / config.time_limit),
        ),
        prefetch=1,
        batch_size=torch.Size([256, config.time_limit]),
    )

    # Initialize temp buffer (if needed for short-term storage)
    temp_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.time_limit),
    )

    logger = tools.Logger(logdir/exp_date, 0)

    #Make env
    env = TransformedEnv(
        GymEnv(config.env_name, **config.maze_config),
        Compose(
            ToTensorImage(in_keys=["pixels"], from_int=True), 
            ObservationNorm(in_keys=["pixels"], loc=0, scale=1),
            PermuteTransform((-2, -1, -3), in_keys=["pixels"]),
            InitTracker(),
            RewardScaling(loc=0.0, scale=1),
            StepCounter(),
        ),
    )

    env_tens = env.reset()

    obs_space = {
        'image': env_tens['pixels'][:,:64],
        'map': env_tens['pixels'][:,64:],
    }

    agent = dreamer.Dreamer(
        obs_space,
        config,
        logger,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)

    torch.cuda.empty_cache()
    gc.collect()


    # ============================================================================
    #   MAKE RANDOM DATA - takes up to 60 minutes
    # ----------------------------------------------------------------------------
    #add this bit on

    # ============================================================================
    #   LOAD DATA - takes up to 6 minutes
    # ----------------------------------------------------------------------------

    load_data(
        replay_buffer=replay_buffer,
        data_path=data_path_preloaded,
    )

    # ============================================================================
    #   REPRODUCABILITY
    # ----------------------------------------------------------------------------

    # Archive current configs.yaml + all .py files into a zip
    archive_path = experiment_folder / f"{config.exp_date}_code_snapshot.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add configs.yaml
        config_path = root_dir / "configs.yaml"
        zipf.write(config_path, arcname="configs.yaml")
        
        # Add all .py files from the project root
        for py_file in root_dir.glob("*.py"):
            zipf.write(py_file, arcname=py_file.name)

    # Get current datetime
    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    # Write README
    with open(experiment_folder / "README.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Name: {exp_date}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"Start Time: {time_str}\n")
        f.write(f"Git: {get_git_commit()}\n")
        f.write(f"Notes:")

    print("Git commit:", get_git_commit())

    # ============================================================================
    #   RUN
    # ----------------------------------------------------------------------------

    state = None
    while agent._step < config.steps:
        logger.write()
        print("Start training.")
        agent.train()
        state = tools.simulate(
            agent,
            env,
            replay_buffer,
            temp_buffer,
            logger,
            steps = agent._config.time_limit,
            state = state,
            config=config)
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(agent._step)


if __name__ == "__main__":
    main()