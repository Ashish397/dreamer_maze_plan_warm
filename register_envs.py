import gymnasium as gym
from gymnasium.envs.registration import register

from miniworld_patches import MazeCA

# Register your custom MazeCA
register(
    id="MiniWorld-MazeCA-v0",
    entry_point="miniworld_patches:MazeCA",  # module:function_or_class
    max_episode_steps=4096,  # or whatever you like
)

print("Custom MiniWorld-MazeCA-v0 registered!")