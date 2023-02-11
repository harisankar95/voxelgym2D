"""Base class for all environments"""

import logging
import math
import os
from collections import OrderedDict
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import cv2
import gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from skimage import draw


# based on https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    """Colors for different log levels."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_ + reset,
        logging.INFO: grey + format_ + reset,
        logging.WARNING: yellow + format_ + reset,
        logging.ERROR: red + format_ + reset,
        logging.CRITICAL: bold_red + format_ + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter_ = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter_.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = CustomFormatter()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class BaseEnv(gym.Env):
    """Base class for all environments"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        mapfile: str = "200x200x200_dense.npy",
        view_size: int = 21,
        max_collisions: int = 0,
        max_steps: int = 60,
        show_path: bool = True,
        multi_output: bool = False,
        partial_reward: bool = True,
        image_size: int = 42,
        test_mode: bool = False,
        inference_mode: bool = False,
    ):
        """
        Parameters
        ----------
        mapfile : str
            name of the map file in the maps folder
        view_size : int
            size of the view window for observation
        max_collisions : int
            maximum number of collisions allowed before episode ends
        max_steps : int
            maximum number of steps allowed before episode ends
        show_path : bool
            whether to show the last travesed action path in the observation
        multi_output : bool
            whether to add additional outputs in the observation
        partial_reward : bool
            whether to give rewards for each step
        image_size : int
            size of the image to be returned as observation
        test_mode : bool
            whether to run in test mode, for evaluation during training
        inference_mode : bool
            whether to run in inference mode
        """

        super().__init__()
        # current file path
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # load map
        self.grid_map = np.load(os.path.join(__location__, "maps", mapfile))
        self.world_size = self.grid_map.shape

        if view_size < 10:
            logger.warning("view_size should be at least 10, setting to 10")
            self.view_size = 10
        elif view_size > np.amin(self.world_size):
            logger.warning(
                "view_size should be less than half of the world size, setting to %s",
                int((np.amin(self.world_size) - 2) / 2),
            )
            self.view_size = int((np.amin(self.world_size) - 2) / 2)
        else:
            self.view_size = view_size

        view_ratio = image_size / view_size  # should be even
        if view_ratio % 2 != 0:
            logger.warning("view_ratio should be even, setting to %s", (int(view_ratio) + 1))
            self.image_size = int(view_ratio + 1) * view_size
        else:
            self.image_size = image_size

        # set the agent to middle of the world map
        self._new_world = np.ones((int(2 * self.view_size), int(2 * self.view_size)))
        self._new_world_center: np.ndarray = np.array([self.world_size[1] / 2, self.world_size[0] / 2], dtype=np.int32)
        self._next_new_world_center: np.ndarray = np.array(
            [self.world_size[1] / 2, self.world_size[0] / 2], dtype=np.int32
        )
        self._agent_location: np.ndarray = np.array([self.view_size, self.view_size], dtype=np.int32)
        self._target_location: np.ndarray = np.array([self.view_size, self.view_size], dtype=np.int32)
        self.min_distance_to_target = 0.0
        self.cost_astar = 0.0
        self.min_cost_astar = 0.0
        self.astar_runs = 0
        self._path: List = []
        self.astarPath: List = []
        self.action = [-1, -1]
        self.action_cost = 0.0

        self.obs_world = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
        self.base_obs_world = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        self.num_targets_reached = 0
        self.current_step = 0
        self.num_collisions = 0
        self.test_mode = test_mode
        self.inference_mode = inference_mode
        # set back the agent to middle of the world map after these many targets
        self.soft_reset_freq = 1500
        if self.test_mode:
            self.soft_reset_freq = 50

        self.target_num = -1
        self.max_collisions = max_collisions
        self.max_steps = max_steps
        self.show_path = show_path
        self.partial_reward = partial_reward

        # action space
        self.delta = list(product([-1, 0, 1], repeat=2))
        self.delta.remove((0, 0))

        # math constants
        self.SQRT2 = math.sqrt(2)
        self.EXP1 = math.exp(1)
        self.EXP1_1 = self.EXP1 - 1

        self.seed()
        self.action_space = None

        self.multi_output = multi_output
        if self.multi_output:
            self.observation_space = gym.spaces.Dict(
                {
                    "delta": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "world": gym.spaces.Box(
                        low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8
                    ),
                }
            )
        else:
            # 3 channel 2D numpy arrays: CxWXH
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8
            )

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger

        Returns
        -------
        logger : logging.Logger
            logger object
        """
        return logger

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """
        Sets the seed for this env's random number generator(s).

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator(s), by default None

        Returns
        -------
            Returns the list of seed used in this env's random number generators
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def make_astar_matrix(self) -> None:
        """
        Creates the astar matrix for the current world map and sets the astar grid
        """
        _astar_matrix = np.abs(1.0 - self._new_world).astype(np.int32)
        self._astar_grid = Grid(matrix=_astar_matrix.tolist())

    def run_astar(self, target: np.ndarray) -> Tuple[List[Tuple[int, int]], float, int]:
        """
        Runs the A* algorithm on the current world map and returns the path,
        path cost and number of nodes visited

        Parameters
        ----------
        target : np.ndarray
            target location

        Returns
        -------
        path : List[Tuple[int, int]]
            path from agent to target
        path_cost : float
            cost of the path
        runs : int
            number of nodes visited
        """

        self._astar_grid.cleanup()
        start = self._astar_grid.node(self._agent_location[0], self._agent_location[1])
        end = self._astar_grid.node(target[0], target[1])

        if self.inference_mode:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        else:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        path, runs = finder.find_path(start, end, self._astar_grid)
        if len(path) > 0:
            path_cost = end.g
        else:
            path_cost = np.inf
        return path, path_cost, runs

    def slice_grid_map(self) -> None:
        """
        Slices the grid map into a 2D numpy array
        of size (2*view_size, 2*view_size)
        """

        # slice the grid map into a 2D numpy array
        self._new_world = np.ones((int(2 * self.view_size), int(2 * self.view_size)))
        y_min = int(max(self._new_world_center[0] - self.view_size, 0))
        y_max = int(min(self._new_world_center[0] + self.view_size, self.world_size[0]))
        x_min = int(max(self._new_world_center[1] - self.view_size, 0))
        x_max = int(min(self._new_world_center[1] + self.view_size, self.world_size[1]))

        y_min_new = int(self.view_size - self._new_world_center[0] + y_min)
        y_max_new = int(self.view_size + y_max - self._new_world_center[0])
        x_min_new = int(self.view_size - self._new_world_center[1] + x_min)
        x_max_new = int(self.view_size + x_max - self._new_world_center[1])

        self._new_world[y_min_new:y_max_new, x_min_new:x_max_new] = self.grid_map[y_min:y_max, x_min:x_max]

        # set the edges to 1
        self._new_world[0, :-1] = self._new_world[:-1, -1] = self._new_world[-1, 1:] = self._new_world[1:, 0] = 1
        # create the new astar grid
        self.make_astar_matrix()

    def find_target(self) -> np.ndarray:
        """
        Finds a target location for the agent to move to

        Returns
        -------
        target_location : np.ndarray
            target location
        """

        # 10% of the time find easy target (close to agent)
        easy_target = True if (self.np_random.rand() < 0.10) else False
        colliding = True
        count = 0
        while colliding:
            nearby_clear = True
            if not easy_target:
                straight_path = True
            target_location = self._agent_location
            while np.array_equal(target_location, self._agent_location):
                if not easy_target:
                    random_r = self.np_random.randint(6, int(self.SQRT2 * self.view_size - 1))
                else:
                    random_r = self.np_random.randint(2, int((self.SQRT2 * self.view_size - 1) / 2))
                random_theta = self.np_random.uniform(-np.pi, np.pi)
                random_location = np.array([random_r * np.cos(random_theta), random_r * np.sin(random_theta)])
                random_location = np.round(random_location).astype(np.int32)
                target_location = random_location + self._agent_location
                if np.amin(target_location) < 0 or np.amax(target_location) >= int(2 * self.view_size):
                    target_location = self._agent_location
            if self._new_world[target_location[1], target_location[0]] == 0:
                spacing = 1
                # check immediate neighbors for obstacles
                y_min = int(max(target_location[1] - spacing, 0))
                y_max = int(min(target_location[1] + spacing + 1, 2 * self.view_size))
                x_min = int(max(target_location[0] - spacing, 0))
                x_max = int(min(target_location[0] + spacing + 1, 2 * self.view_size))
                if np.count_nonzero(self._new_world[y_min:y_max, x_min:x_max] == 1.0) > 0:
                    nearby_clear = False

                if nearby_clear:
                    if not easy_target:
                        # check if its a straight path
                        rr, cc = draw.line(
                            self._agent_location[0],
                            self._agent_location[1],
                            target_location[0],
                            target_location[1],
                        )
                        straight_line = list(zip(rr, cc))
                        for pt in straight_line:
                            if self._new_world[pt[1], pt[0]] == 1:
                                straight_path = False
                                break

                        if not straight_path:
                            # Astar search to find the shortest path to the target
                            self.astarPath, self.cost_astar, self.astar_runs = self.run_astar(target_location)
                            if len(self.astarPath) > 0 and self.astar_runs > 60:
                                colliding = False
                    else:
                        # Astar search to find the shortest path to the target
                        self.astarPath, self.cost_astar, self.astar_runs = self.run_astar(target_location)
                        if len(self.astarPath) > 0:
                            colliding = False
            count += 1
            if count > 9999:
                logger.info(
                    "   ---Target not set in %s tries! Setting agent back to reset conditions!",
                    count,
                )
                count = 0
                self.soft_reset()
        return target_location

    def _get_info(self) -> Dict:
        """
        Returns the info dictionary for the current step of the episode

        Returns
        -------
        info : Dict
            info dictionary
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "target reached": np.array_equal(self._target_location, self._agent_location),
            "action": self.action,
            "action path": self._path,
            "action cost": self.action_cost,
            "astar path": self.astarPath,
            "astar cost": self.cost_astar,
            "astar runs": self.astar_runs,
            "current step": self.current_step,
            "no. of  collisions": self.num_collisions,
            "grid map": self._new_world,
        }

    # base observation is the world map which remains constant throughout the episode
    def _create_base_obs(self) -> None:
        """
        Creates the base observation for the episode which can be reused throughout the episode
        """
        self.base_obs_world = np.zeros((3, int(2 * self.view_size), int(2 * self.view_size)), dtype=np.float32)
        self.base_obs_world[0, :, :] = self.base_obs_world[1, :, :] = 0.3 * self._new_world * 255.0
        self.base_obs_world[2, :, :] = self._new_world * 255.0

        self.base_obs_world[:, self._target_location[1], self._target_location[0]] = 0.3 * 255.0
        self.base_obs_world[1, self._target_location[1], self._target_location[0]] = 1.0 * 255.0

    # observation is the world map with the agent location marked and the path taken
    # returns either a numpy array of shape (3, 2*view_size, 2*view_size)
    # or a OrderedDict of numpy arrays
    def _get_obs(self) -> Union[np.ndarray, OrderedDict]:
        """
        Returns the observation for the current step of the episode

        Returns
        -------
        obs : Union[np.ndarray, OrderedDict]
            observation for the current step of the episode
        """
        self.obs_world = np.copy(self.base_obs_world)
        self.obs_world[:, self._agent_location[1], self._agent_location[0]] = 0.3 * 255.0
        self.obs_world[0, self._agent_location[1], self._agent_location[0]] = 1.0 * 255.0

        if self.show_path and (len(self._path) != 0):
            for pt in self._path:
                if self.obs_world[2, pt[1], pt[0]] == 0:
                    self.obs_world[:, pt[1], pt[0]] = 0.6 * 255.0

        # make channel last
        self.obs_world = np.moveaxis(self.obs_world, 0, 2)
        self.obs_world = cv2.resize(
            self.obs_world,
            dsize=(self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        self.obs_world = np.moveaxis(self.obs_world, -1, 0)
        if not self.multi_output:
            return self.obs_world.astype(dtype=np.uint8)

        return OrderedDict(
            {
                # normalize delta
                "delta": ((self._target_location - self._agent_location) / (2 * self.view_size - 1)).astype(np.float32),
                "world": self.obs_world.astype(dtype=np.uint8),
            }
        )

    # set the agent's location at the center of the map
    def soft_reset(self) -> None:
        """Moves the agent to the center of the map and resets the target"""
        self._new_world_center = np.array([self.world_size[1] / 2, self.world_size[0] / 2], dtype=np.int32)
        self.slice_grid_map()
        self._agent_location = np.array([self.view_size, self.view_size], dtype=np.int32)
        self.target_num = 0

    def reset(
        self, return_info: bool = False
    ) -> Union[Tuple[Union[np.ndarray, OrderedDict], Dict], np.ndarray, OrderedDict]:
        """
        Resets the environment to the initial state and returns the initial observation

        Parameters
        ----------
        return_info : bool, optional
            whether to return the info dictionary, by default False

        Returns
        -------
        obs : np.ndarray or OrderedDict
            observation from manystep environment
        info : Dict (optional)
            info dictionary (optional) of the last step in the stack
        """
        self.target_num += 1
        if (self.target_num % self.soft_reset_freq) == 0:
            self.soft_reset()
            if self.test_mode:
                self.seed(1327455)
        else:
            self._new_world_center = self._next_new_world_center

        self.current_step = 0
        self.num_collisions = 0
        self.slice_grid_map()
        self._agent_location = np.array([self.view_size, self.view_size], dtype=np.int32)
        # We will sample the target's location randomly until it does
        # not coincide with the agent's location
        self._target_location = self.find_target()
        self._next_new_world_center = self._new_world_center + (self._target_location - self._agent_location)[::-1]
        self.min_distance_to_target = float(np.linalg.norm(self._target_location - self._agent_location))
        self.min_cost_astar = self.cost_astar
        self._path = []
        self.action = [-1, -1]
        self.action_cost = 0.0

        self._create_base_obs()
        observation = self._get_obs()
        return (observation, self._get_info()) if return_info else observation

    def _compute_reward(self, completion_reward: bool = False):
        """
        Computes the reward for the current step of the episode
        """
        raise NotImplementedError

    def _take_action(self, action: np.ndarray) -> Tuple[List, bool]:
        """
        Takes the action and returns the new agent lo
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Closes all open matplotlib figures
        """
        plt.close("all")

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, OrderedDict], float, bool, Dict]:
        """
        Takes a step in the environment and returns the observation, reward, done and info

        Parameters
        ----------
        action : np.ndarray
            the action to take

        Returns
        -------
        Observation, Reward, Done, Info : Tuple[Union[np.ndarray, OrderedDict], float, bool, Dict]
            the observation, reward, done and info
        """
        done = False
        valid_action_path, collision = self._take_action(action)
        self.current_step += 1

        self._agent_location = np.array(valid_action_path[-1], dtype=np.int32)

        reward = 0.0
        self._path = []

        if collision:
            self.num_collisions += 1
            reward = -1.0
        else:  # only do it if the agent moves
            self._path = valid_action_path[:-1]  # remove the agent location
            if np.array_equal(self._target_location, self._agent_location):
                done = True
                reward += self._compute_reward(completion_reward=True)
                self.num_targets_reached += 1
                logger.info(
                    "%s Target reached in %s steps, Collisions : %s",
                    self.ordinal(self.num_targets_reached),
                    self.current_step,
                    self.num_collisions,
                )
            else:
                reward += self._compute_reward(completion_reward=False)
                self.min_distance_to_target = float(np.linalg.norm(self._target_location - self._agent_location))
                self.min_cost_astar = self.cost_astar

        if self.num_collisions > self.max_collisions:
            done = True
            if not self.partial_reward:
                reward += -self.cost_astar / 100
            logger.info(" --Max Collisions! Collisions: %s/%s", self.num_collisions, self.max_collisions + 1)
        elif (not done) and (self.current_step > self.max_steps):
            done = True
            if not self.partial_reward:
                reward += -self.cost_astar / 100
            logger.info(
                " --Max Steps: %s/%s!",
                self.current_step,
                self.max_steps + 1,
            )

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    def render(self, mode="human") -> None:
        """
        Renders the environment

        Parameters
        ----------
        mode : str, optional
            the mode to render in, by default "human"

        Returns
        -------
        None
        """
        plt.title("Voxelgym")
        plt.imshow(np.moveaxis(self.obs_world.astype(np.uint8), 0, 2))

        palette = [[77, 77, 255], [77, 255, 77], [255, 77, 77], [144, 144, 144]]
        classes = ["obstacles", "target", "agent", "last visited"]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=np.array(palette[i]) / 255.0, label=classes[i]) for i in range(4)]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    # based on https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    @staticmethod
    def ordinal(num: int) -> str:
        """
        Returns the ordinal of a number

        Parameters
        ----------
        num : int
            the number to get the ordinal of

        Returns
        -------
        str
            the ordinal of the number
        """
        return f"{num}{'tsnrhtdd'[(num//10%10!=1)*(num%10<4)*num%10::4]}"
