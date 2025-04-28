from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box
from miniworld.params import DEFAULT_PARAMS
import numpy as np
import random
from gymnasium import spaces, utils
from typing import Optional, Tuple
from miniworld.entity import Agent, Entity
from gymnasium.core import ObsType
import math
import pyglet

from pyglet.gl import (
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    glEnable,
)

from miniworld.opengl import FrameBuffer

# Modified map class (because your map class was adjusted)
from skimage.draw import line_aa

KERNEL = np.ones([3,3])

# Default wall height for room
DEFAULT_WALL_HEIGHT = 2.74

# Texture size/density in texels/meter
TEX_DENSITY = 512


class Map:
    def __init__(self, obs_height, obs_width, fluff, reward_mul=1, decay=1.0):
        self.fluff = fluff
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.decay = decay
        self.reward_mul = reward_mul
        self.map = np.zeros([self.obs_height, self.obs_width])

    def __call__(self):
        return self.map

    def update(self, pos, scale=27, offset=1):
        self.map = self.map * self.decay
        count_pre = np.sum(self.map)
        y, _, x = np.round((pos / scale) * self.obs_height).astype(int)
        y += offset
        x += offset
        self.last_x = x
        self.last_y = y
        self.map[x, y] = 1.0
        if self.fluff:
            self.map[x-self.fluff:x+self.fluff, y-self.fluff:y+self.fluff] = 1.0
        count_post = np.sum(self.map)
        count_delta = count_post - count_pre
        if self.fluff > 1:
            count = count_delta / (self.reward_mul * self.fluff ** 2)
        else:
            count = count_delta
        return count
    
    def show_map(self):
        return self.map

    def show_all(self, dir, line_len=12):
        self.curr_pos = np.zeros([self.obs_height, self.obs_width])
        self.curr_dir = np.zeros([self.obs_height, self.obs_width])
        self.curr_pos[self.last_x-1:self.last_x+1, self.last_y-1:self.last_y+1] = 1.0
        self.dir_y, _, self.dir_x = np.round(dir * line_len).astype(int)
        rr, cc, val = line_aa(self.last_x, self.last_y, self.last_x + self.dir_x, self.last_y + self.dir_y)
        upper_lim = self.obs_height
        lower_lim = -1
        mask = (lower_lim < rr) & (rr < upper_lim) & (lower_lim < cc) & (cc < upper_lim)
        rr, cc, val = rr[mask], cc[mask], val[mask]
        self.curr_dir[rr, cc] = val
        all_ims = np.stack([self.map, self.curr_pos, self.curr_dir], axis=-1) * 255
        return all_ims.astype(np.uint8)
    
    def coverage(self):
        return np.mean(self.map)

class MiniWorldEnv_c(MiniWorldEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        max_episode_steps=10000,
        obs_width=64,
        obs_height=64,
        obs_channels=3,
        window_width=800,
        window_height=600,
        include_maps=True,
        patience=1000,
        decay_param=1.0,
        fluff=2,
        porosity=0.0,
        params=DEFAULT_PARAMS,
        domain_rand=False,
        render_mode=None,
        view="agent",
    ):
        # Override attributes
        self.porosity = porosity
        self.include_maps = include_maps
        self.decay_param = decay_param
        self.fluff = fluff
        self.max_patience = patience
        self.obs_channels = obs_channels
        self.obs_width = obs_width
        self.obs_height = obs_height

        # Init the parent MiniWorldEnv
        super().__init__(
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            window_width=window_width,
            window_height=window_height,
            params=params,
            domain_rand=domain_rand,
            render_mode=render_mode,
            view=view,
        )

        # Override action_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        # Fix observation space if maps included
        if self.include_maps:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.obs_height, self.obs_width * 2, obs_channels),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.obs_height, self.obs_width, obs_channels),
                dtype=np.uint8
            )

        # Initialize histoire
        self.histoire = Map(
            obs_height=self.obs_height,
            obs_width=self.obs_width,
            fluff=self.fluff,
            decay=self.decay_param,
        )


    def nearby(self, ent0, ent1=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 is None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        new_dist = dist - (ent0.radius + ent1.radius + self.max_forward_step)
        if new_dist < 0:
            new_dist = 0
        elif new_dist > 10:
            new_dist = 0
        else:
            new_dist = int(((10 - new_dist) ** 2) * 2.55) 
        return new_dist

    def move_agent(self, fwd_dist):
        fwd_dist *= 0.5
        next_pos = self.agent.pos + self.agent.dir_vec * fwd_dist
        if np.any(next_pos[[0, 2]] > 25.2) or np.any(next_pos[[0, 2]] < 0.6):
            return False
        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False
        self.agent.pos = next_pos
        return True

    def strafe_agent(self, fwd_dist):
        fwd_dist *= 0.2
        right_theta = np.arcsin(self.agent.dir_vec[0]) + np.pi / 2
        new_vec = np.array([np.sin(right_theta), 0, -np.cos(right_theta)])
        next_pos = self.agent.pos + new_vec * fwd_dist
        if np.any(next_pos[[0, 2]] > 25.2) or np.any(next_pos[[0, 2]] < 0.6):
            return False
        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False
        self.agent.pos = next_pos
        return True

    def turn_agent(self, turn_angle):
        self.agent.dir += turn_angle * 0.25
        return True

    def step(self, action):
        """
        Perform one action and update the simulation

        Multiple rewards here - the lazy reward punishes turning as it returns a 0 reward -0.1 / 0
            - sticky reward gives the agent a reward for repeating the previous action 0.5 / 0
            - IC reward rewards getting a box, increases by 50 each time, resets to 0 upon new episode
              (Sum(+50))/0
            - progress reward rewards moving 9/-4
            - expl reward rewards exploring
        """

        self.step_count += 1
        IC_reward = 0

        self.move_agent(action[0])
        self.strafe_agent(action[1])
        self.turn_agent(action[2])

        # Generate the current camera image
        obs = self.render_obs()

        expl_reward = 10 * self.histoire.update(self.agent.pos)

        if self.include_maps:
            all_maps = self.histoire.show_all(self.agent.dir_vec)
            #axis -1 for older than dreamer models - NOTE
            obs = np.concatenate([obs, all_maps], axis=-2)

        termination = False
        truncation = False

        to_remove = []
        color = np.zeros(3)
        for i, this_box in enumerate(self.boxes):
            if self.near(this_box):
                IC_reward += 50
                self.entities.remove(this_box)
                to_remove.append(i)
                self.patience_count = 0
            else:
                if this_box.color == 'red':
                    color[0] = self.nearby(this_box)
                elif this_box.color == 'green':
                    color[1] = self.nearby(this_box)
                elif this_box.color == 'blue':
                    color[2] = self.nearby(this_box)

        nearby_reward = sum(color / 76.5)

        obs[27:29, 28:36] = color    # Top edge (two pixels thick)
        obs[35:37, 28:36] = color    # Bottom edge (two pixels thick)

        for i in to_remove[::-1]:
            self.boxes.pop(i)

        if len(self.boxes) == 0:
            IC_reward += 150
            termination = True

        # If the maximum time step count is reached
        if self.step_count >= 4096: #self.max_episode_steps:
            truncation = True

        reward = -10 + expl_reward + IC_reward + nearby_reward
        
        return obs, reward, termination, truncation, {}
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0
        self.patience_count = 0
        self.IC_reward = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        self.histoire = Map(obs_height=self.obs_height, obs_width=self.obs_width, fluff = self.fluff, decay = self.decay_param)

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []
        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        if self.include_maps:
            _ = self.histoire.update(self.agent.pos) 
            all_maps = self.histoire.show_all(self.agent.dir_vec)
            #axis -1 for older than dreamer models - NOTE
            obs = np.concatenate([obs, all_maps], axis=-2)

        # Return first observation
        return obs, {}

class MazeCA(MiniWorldEnv_c, utils.EzPickle):
    def __init__(self, num_rows=8, num_cols=8, room_size=3, max_episode_steps=None, **kwargs):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            porosity=kwargs.pop("porosity", 0.0),
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        for j in range(self.num_rows):
            for i in range(self.num_cols):
                room = rows[j][i]

                # Look only at right and bottom neighbors to avoid duplicating connections
                if i < self.num_cols - 1:
                    neighbor = rows[j][i + 1]
                    if self.np_random.random() < self.porosity:
                        self.connect_rooms(
                            room, neighbor, min_z=room.min_z, max_z=room.max_z
                        )

                if j < self.num_rows - 1:
                    neighbor = rows[j + 1][i]
                    if self.np_random.random() < self.porosity:
                        self.connect_rooms(
                            room, neighbor, min_x=room.min_x, max_x=room.max_x
                        )
                        
        self.boxes = []
        self.boxes.append(self.place_entity(Box(color="red")))
        self.boxes.append(self.place_entity(Box(color="green")))
        self.boxes.append(self.place_entity(Box(color="blue")))

        self.place_agent()
