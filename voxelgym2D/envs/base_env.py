"""Base class for all environments"""

import math
import os
import sys
from collections import OrderedDict
from itertools import product
from logging import Formatter, Logger, LogRecord, StreamHandler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from gymnasium.utils import seeding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.core.node import GridNode
from pathfinding.finder.a_star import AStarFinder
from skimage import draw


class CustomFormatter(Formatter):
    """Custom formatter with colors for different log levels."""

    _format = "%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d"
    # different color for different log level
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    white = "\x1b[37;1m"
    green = "\x1b[32;1m"
    yellow = "\x1b[33;1m"
    purple = "\x1b[35;1m"
    red = "\x1b[31;1m"
    reset = "\x1b[0m"
    COLORED_FORMATS = {
        "DEBUG": white + _format + reset,
        "INFO": green + _format + reset,
        "WARNING": yellow + _format + reset,
        "ERROR": purple + _format + reset,
        "CRITICAL": red + _format + reset,
    }

    def format(self, record: LogRecord) -> str:
        """
        Format the log record.

        Parameters
        ----------
        record : LogRecord
            Log record

        Returns
        -------
        str
            Formatted log record
        """
        log_fmt = self.COLORED_FORMATS.get(record.levelname)
        formatter = Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomLogger:
    """Logger class."""

    def __init__(self, name: str, log_level: str = "ERROR"):
        """
        Initialize the logger.

        Parameters
        ----------
        name : str
            Name of the logger.
        log_level : str, optional
            Log level, by default "INFO"
        """
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {log_level}")
        self.log_level = log_level
        self.name = name
        self._logger: Logger
        self._setup_logger()

    def _setup_logger(self):
        """Setup the logger."""
        self._logger = Logger(self.name)
        self._logger.setLevel(self.log_level)
        formatter = CustomFormatter()

        # log to stdout
        handler = StreamHandler(sys.stdout)
        handler.setLevel(self.log_level)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def get_logger(self) -> Logger:
        """
        Get the logger

        Returns
        -------
        logging.Logger
            Logger
        """
        return self._logger


class BaseEnv(gym.Env):
    """Base class for all environments"""

    metadata: Dict[str, Any] = {"render_modes": ["None"], "render_fps": 1}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        mapfile: str = "600x600.npy",
        view_size: int = 21,
        image_size: int = 42,
        max_collisions: int = 0,
        max_steps: int = 60,
        show_path: bool = True,
        multi_output: bool = False,
        partial_reward: bool = True,
        inference_mode: bool = False,
        log_level: str = "ERROR",
    ):
        """
        Parameters
        ----------
        render_mode : Optional[str], optional
            render mode, by default None
        mapfile : str
            name of the map file in the maps folder
        view_size : int
            size of the view window for observation
        image_size : int
            size of the image to be returned as observation
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
        inference_mode : bool
            whether to run in inference mode
        log_level : str, optional
            log level, by default "ERROR". One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        """
        self.render_mode = "None" if render_mode is None else render_mode
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}")

        # current file path
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # load map
        self.grid_map = np.load(os.path.join(__location__, "maps", mapfile))
        self.MAX_VAL_UINT = int(0.9 * np.iinfo(np.uint32).max)  # 0.9 to avoid overflow

        # an arry to keep track of the start and target location
        self.start_locations, self.target_locations = self._start_end_counts()
        self.world_size = self.grid_map.shape

        # initialize the logger
        self.logger = CustomLogger(self.__class__.__name__, log_level=log_level).get_logger()

        if view_size < 16:
            self.logger.warning("view_size should be at least 16, setting to 16")
            self.view_size = 16
        elif view_size > np.amin(self.world_size):
            self.logger.warning(
                "view_size should be less than half of the world size, setting to %s",
                int((np.amin(self.world_size) - 2) / 2),
            )
            self.view_size = int((np.amin(self.world_size) - 2) / 2)
        else:
            self.view_size = view_size

        view_ratio = image_size / view_size  # should be even
        if view_ratio % 2 != 0:
            self.logger.warning("view_ratio should be even, setting to %s", (int(view_ratio) + 1))
            self.image_size = int(view_ratio + 1) * view_size
        else:
            self.image_size = image_size

        # set the agent to middle of the world map
        self._new_world = np.ones((int(2 * self.view_size), int(2 * self.view_size)))
        # place holder for target locations for the sliced view (_new_world)
        self._new_target_locations = np.ones((int(2 * self.view_size), int(2 * self.view_size)), dtype=np.uint32)
        self._new_world_center: np.ndarray = np.array([self.world_size[1] / 2, self.world_size[0] / 2], dtype=np.int32)
        self._next_new_world_center: np.ndarray = np.array(
            [self.world_size[1] / 2, self.world_size[0] / 2], dtype=np.int32
        )
        # placeholder for mapping function
        self._mapping = lambda x, y: (x, y)
        # create a list of locations where the agent can be placed
        # 25 possible locations for agent
        factor = 5
        self.possible_start_locations = np.array(
            list(
                product(
                    np.linspace(
                        start=2 * self.view_size / factor,
                        stop=2 * self.view_size,
                        num=factor,
                        endpoint=False,
                        dtype=int,
                    ),
                    np.linspace(
                        start=2 * self.view_size / factor,
                        stop=2 * self.view_size,
                        num=factor,
                        endpoint=False,
                        dtype=int,
                    ),
                )
            )
        )

        self._agent_location: np.ndarray = np.array([self.view_size, self.view_size], dtype=np.int32)
        self._target_location: np.ndarray = np.array([self.view_size, self.view_size], dtype=np.int32)
        self.ini_distance_to_target = 0.0
        self.min_distance_to_target = 0.0
        self.cost_astar = 0.0
        self.ini_cost_astar = 0.0
        self.min_cost_astar = 0.0
        self.astar_runs = 0
        self._path: List = []
        self.ini_astarPath: List = []
        self.astarPath: List = []
        self._astar_grid = Grid()
        self.action = [-1, -1]
        self.action_cost = 0.0

        self.obs_world = np.zeros((3, self.image_size, self.image_size))
        self.obs_world_astar = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
        self.base_obs_world = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        self.num_targets_reached = 0
        self.current_step = 0
        self.num_collisions = 0
        self.inference_mode = inference_mode
        # set back the agent to a new location than the previous
        # target location after these many episodes
        self.soft_reset_freq = 10

        self.target_num = 0
        self.max_collisions = max_collisions
        self.max_steps = max_steps
        self.show_path = show_path
        self.partial_reward = partial_reward

        # action space
        self.delta = list(product([-1, 0, 1], repeat=2))
        self.delta.remove((0, 0))

        # random number generator for shuffling the possible target locations
        seed_seq = np.random.SeedSequence()
        self.random_gen: np.random.Generator = np.random.Generator(np.random.PCG64(seed_seq))

        # math constants
        self.SQRT2 = math.sqrt(2)
        self.EXP1 = math.exp(1)
        self.EXP1_1 = self.EXP1 - 1

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

    def get_logger(self) -> Logger:
        """
        Returns the logger

        Returns
        -------
        logger : Logger
            logger object
        """
        return self.logger

    @staticmethod
    def find_obstacle_neighbor_count(grid_map: np.ndarray) -> np.ndarray:
        """
        Finds the number of neighboring obstacles for each cell in the grid map

        Parameters
        ----------
        grid_map : np.ndarray
            grid map with obstacles marked as 1s and free cells marked as 0s

        Returns
        -------
        neighbors : np.ndarray
            number of neighboring obstacles for each cell in the grid map
        """
        # add a border of 1s around the grid map
        padded_grid_map = np.pad(grid_map, pad_width=1, mode="constant", constant_values=0)
        # get the neighbors of all cells
        neighbors = (
            padded_grid_map[:-2, :-2]
            + padded_grid_map[:-2, 1:-1]
            + padded_grid_map[:-2, 2:]
            + padded_grid_map[1:-1, :-2]
            + padded_grid_map[1:-1, 2:]
            + padded_grid_map[2:, :-2]
            + padded_grid_map[2:, 1:-1]
            + padded_grid_map[2:, 2:]
        )
        # return the count of neighboring obstacles
        return neighbors

    def _start_end_counts(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create arrays to keep track of the start and end cell counts

        Returns
        -------
        start_counts : np.ndarray
            shape like self.grid_map with the count of start cells
        end_counts : np.ndarray
            shape like self.grid_map with the count of end cells
        """
        neighbors = self.find_obstacle_neighbor_count(self.grid_map)

        # find all cells with value 0 and without any neighbors with value > 0
        possible_start_cells = (np.logical_and(self.grid_map == 0, neighbors == 0)).astype(np.uint32)
        # change all cells with value 0 to max value of int as they are not possible start cells
        possible_start_cells[possible_start_cells == 0] = self.MAX_VAL_UINT
        # end cells are same as possible start cells
        possible_end_cells = np.copy(possible_start_cells)
        return possible_start_cells, possible_end_cells

    def _make_astar_matrix(self) -> None:
        """
        Creates the astar matrix for the current world map and sets the astar grid
        """
        # set the astar matrix to 1 for all cells with value 0 in the world map and 0 otherwise
        _astar_matrix = np.abs(1.0 - self._new_world)
        self._astar_grid = Grid(matrix=_astar_matrix.tolist())

    def _run_astar(self, target: np.ndarray) -> Tuple[List[Tuple[int, int]], float, int]:
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
        path_w_node, runs = finder.find_path(start, end, self._astar_grid)
        path_cost = np.inf
        path = []
        if len(path_w_node) > 0:
            for node in path_w_node:
                is_gn = isinstance(node, GridNode)
                x, y = (node.x, node.y) if is_gn else node[:2]
                path.append((x, y))

            path_cost = 0.0
            for dx in np.array(path[1:]) - np.array(path[:-1]):
                path_cost += np.sqrt(np.sum(dx**2))

        return path, path_cost, runs

    def _slice_grid_map(self) -> Tuple[Callable, Union[np.ndarray, None]]:
        """
        Slices the grid map into a 2D numpy array of size (2*view_size, 2*view_size)
        Generate a mapping from the sliced grid map to the original grid map

        Returns
        -------
        mapping : Callable(int, int)
            mapping from the sliced grid map to the original grid map
        potential_start_location : Union[np.ndarray, None]
            potential start location for the agent
        """
        # slice the grid map into a 2D numpy array
        self._new_world = np.ones((int(2 * self.view_size), int(2 * self.view_size)))

        # new taget locations is of shape (2*view_size, 2*view_size) with all values
        # as self.MAX_VAL_UINT and dtype as np.uint32
        self._new_target_locations = np.full(
            shape=(int(2 * self.view_size), int(2 * self.view_size)),
            fill_value=self.MAX_VAL_UINT,
            dtype=np.uint32,
        )

        y_min = int(max(self._new_world_center[0] - self.view_size, 0))
        y_max = int(min(self._new_world_center[0] + self.view_size, self.world_size[0]))
        x_min = int(max(self._new_world_center[1] - self.view_size, 0))
        x_max = int(min(self._new_world_center[1] + self.view_size, self.world_size[1]))

        y_min_new = int(self.view_size - self._new_world_center[0] + y_min)
        y_max_new = int(self.view_size + y_max - self._new_world_center[0])
        x_min_new = int(self.view_size - self._new_world_center[1] + x_min)
        x_max_new = int(self.view_size + x_max - self._new_world_center[1])

        self._new_world[y_min_new:y_max_new, x_min_new:x_max_new] = self.grid_map[y_min:y_max, x_min:x_max]
        self._new_target_locations[y_min_new:y_max_new, x_min_new:x_max_new] = self.target_locations[
            y_min:y_max, x_min:x_max
        ]

        # set the edges to 1
        self._new_world[0, :-1] = self._new_world[:-1, -1] = self._new_world[-1, 1:] = self._new_world[1:, 0] = 1

        # set the edges to self.MAX_VAL_UINT in the new target locations
        self._new_target_locations[0, :-1] = self._new_target_locations[:-1, -1] = self._new_target_locations[
            -1, 1:
        ] = self._new_target_locations[1:, 0] = self.MAX_VAL_UINT
        # create the new astar grid
        self._make_astar_matrix()

        # define the mapping from the sliced grid map to the original grid map
        def _mapping(x: int, y: int) -> Tuple[int, int]:
            return x + x_min - x_min_new, y + y_min - y_min_new

        # find the start location in the sliced grid map
        # shuffle the possible_start_locations
        self.np_random.shuffle(self.possible_start_locations)
        # iterate over possible_start_locations
        potential_start_location = None
        for start_location in self.possible_start_locations:
            # check if the start location along with spacing = 1 is free of obstacles
            spacing = 1
            if np.all(
                self._new_world[
                    start_location[1] - spacing : start_location[1] + (spacing + 1),
                    start_location[0] - spacing : start_location[0] + (spacing + 1),
                ]
                == 0
            ):
                # if free of obstacles, set the agent location to the start location
                potential_start_location = start_location
                break

        return _mapping, potential_start_location

    def _find_target(self) -> np.ndarray:
        """
        Finds a target location for the agent to move to

        Returns
        -------
        target_location : np.ndarray
            target location

        Raises
        ------
        RuntimeError
            If a target location cannot be found
        """

        # 10% of the time find easy target (close to agent)
        easy_target = self.np_random.random() < 0.10
        # if not easy path, we don't want to sample within 8 cells of the agent
        # increase the count of cells in self._new_target_locations to self.MAX_VAL_UINT
        # for cells within 8 cells of the agent else 2 to have nothing close to the agent
        spread = 2 if easy_target else 8
        count = 0
        while True:
            self._new_target_locations[
                self._agent_location[0] - spread : self._agent_location[0] + (spread + 1),
                self._agent_location[1] - spread : self._agent_location[1] + (spread + 1),
            ] = self.MAX_VAL_UINT
            # get a list of all the target locations with the count less than self.MAX_VAL_UINT
            possible_target_locations = np.argwhere(self._new_target_locations < self.MAX_VAL_UINT)[:, ::-1]
            # shuffle the possible_target_locations
            self.random_gen.shuffle(possible_target_locations)
            # iterate while possible_target_locations
            for target_location in possible_target_locations:
                if not easy_target:
                    # check if its a straight path
                    rr, cc = draw.line(
                        self._agent_location[0],
                        self._agent_location[1],
                        target_location[0],
                        target_location[1],
                    )
                    straight_pixels = self._new_world[cc, rr]
                    straight_path = not np.any(straight_pixels == 1)

                    if straight_path:
                        continue

                # Astar search to find the shortest path to the target
                self.astarPath, self.cost_astar, self.astar_runs = self._run_astar(target_location)
                if len(self.astarPath) > 0:
                    if easy_target:
                        return target_location
                    if self.astar_runs > 60:
                        return target_location

            self.logger.info("   ---Target not set, soft reset---")
            # increase the count of agent location in the self.state_counts to self.MAX_VAL_UINT
            # so that the agent location is not sampled again
            mapped_start = self._mapping(self._agent_location[0], self._agent_location[1])
            self.start_locations[mapped_start[::-1]] = self.MAX_VAL_UINT
            self._soft_reset()
            count += 1
            if count > 100000:
                raise RuntimeError("Cannot find a target location")

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
            "obs with astar path": self.obs_world_astar,
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

        # mark the astar path in the self.obs_world_astar
        self.obs_world_astar = np.copy(self.obs_world)
        for pt in self.astarPath:
            self.obs_world_astar[:, pt[1], pt[0]] = 0.6 * 255.0
        self.obs_world_astar = self.obs_world_astar.astype(dtype=np.uint8)

        if self.show_path and (len(self._path) != 0):
            for pt in self._path:
                if self.obs_world[2, pt[1], pt[0]] == 0:
                    self.obs_world[:, pt[1], pt[0]] = 0.6 * 255.0

        # make channel last
        obs_world_cv: Union[np.ndarray, MatLike] = np.moveaxis(self.obs_world, 0, 2)
        obs_world_cv = cv2.resize(
            obs_world_cv,
            dsize=(self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        self.obs_world = np.moveaxis(obs_world_cv, -1, 0).astype(dtype=np.uint8)

        if not self.multi_output:
            return self.obs_world
        return OrderedDict(
            {
                # normalize delta
                "delta": ((self._target_location - self._agent_location) / (2 * self.view_size - 1)).astype(np.float32),
                "world": self.obs_world,
            }
        )

    def _get_new_index_from_counts(self, counts_mat: np.ndarray, alpha_p: float = 1.0) -> Tuple[int, int]:
        """
        Returns a new index sampled from the counts matrix

        Parameters
        ----------
        counts_mat : np.ndarray
            counts matrix from which is used to sample the new index
        alpha_p : float
            parameter to control the sampling probability

        Returns
        -------
        sampled_index : Tuple[int, int]
            sampled index from the counts matrix in the form (y, x)
        """
        flattened_counts = counts_mat.flatten()
        # higher the count, lower the probability of sampling that cell
        probabilities = np.exp(-alpha_p * flattened_counts)
        probabilities /= np.sum(probabilities)

        # sample a cell based on the probabilities
        sampled_index = self.np_random.choice(np.arange(len(flattened_counts)), p=probabilities)
        # convert the sampled index to 2D index
        sampled_index = np.unravel_index(sampled_index, counts_mat.shape)
        return sampled_index  # (y, x)

    # set the world center based on sampling from current counts
    # tries to set the world center to cells with lower counts
    def _soft_reset(self) -> None:
        """Moves the agent to the center of the map and resets the target"""
        sampled_index = self._get_new_index_from_counts(self.start_locations)
        self._new_world_center = np.array([sampled_index[0], sampled_index[1]], dtype=np.int32)
        self._mapping, potential_start_location = self._slice_grid_map()
        if potential_start_location is None:
            potential_start_location = np.array([self.view_size, self.view_size], dtype=np.int32)
        self._agent_location = potential_start_location
        self.target_num = 0

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[Dict, None] = None,
    ) -> Tuple[Union[np.ndarray, OrderedDict], Dict]:
        """
        Resets the environment to the initial state and returns the initial observation and info

        Parameters
        ----------
        seed : Union[int, None]
            seed to use for the environment
        options : Union[Dict, None]
            options to use for the environment

        Returns
        -------
        obs : np.ndarray or OrderedDict
            observation from manystep environment
        info : Dict
            info dictionary of the last step in the stack
        """

        # Initialize the RNG if the seed is manually passed
        super().reset(seed=seed)

        # seed the other random number generator
        if seed is not None:
            self.random_gen, _ = seeding.np_random(seed)

        self.target_num += 1
        if (self.target_num % self.soft_reset_freq) == 0:
            self._soft_reset()
        else:
            self._new_world_center = self._next_new_world_center

        self.current_step = 0
        self.num_collisions = 0
        self._mapping, potential_start_location = self._slice_grid_map()
        if potential_start_location is None:
            potential_start_location = np.array([self.view_size, self.view_size], dtype=np.int32)
        self._agent_location = potential_start_location
        # We will sample the target's location randomly until it does
        # not coincide with the agent's location
        self._target_location = self._find_target()

        # increase the count of the start and target locations
        mapped_start = self._mapping(self._agent_location[0], self._agent_location[1])
        self.start_locations[mapped_start[::-1]] += 1
        mapped_target = self._mapping(self._target_location[0], self._target_location[1])
        self.target_locations[mapped_target[::-1]] += 1

        # set the next new world center at the mapped target location
        self._next_new_world_center = mapped_target[::-1]
        self.ini_distance_to_target = self.min_distance_to_target = float(
            np.linalg.norm(self._target_location - self._agent_location)
        )
        self.ini_astarPath = self.astarPath
        self.ini_cost_astar = self.min_cost_astar = self.cost_astar
        self._path = []
        self.action = [-1, -1]
        self.action_cost = 0.0

        self._create_base_obs()
        observation = self._get_obs()
        return (observation, self._get_info())

    def _compute_reward(self, completion_reward: bool = False):
        """
        Computes the reward for the current step of the episode
        """
        raise NotImplementedError

    def _take_action(self, action: np.ndarray) -> Tuple[List, bool]:
        """
        Takes the action and returns the new agent location
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Closes all open matplotlib figures
        """
        plt.close("all")

    def step(self, action: np.ndarray) -> Tuple[Union[np.ndarray, OrderedDict], float, bool, bool, Dict]:
        """
        Takes a step in the environment and returns the observation, reward, terminated, truncated and info

        Parameters
        ----------
        action : np.ndarray
            the action to take

        Returns
        -------
        observation : np.ndarray or OrderedDict
            observation
        reward : float
            reward
        terminated : bool
            whether the episode terminated
        truncated : bool
            whether the episode was truncated
        info : Dict
            info dictionary
        """
        terminated = False
        truncated = False
        valid_action_path, collision = self._take_action(action)
        self.current_step += 1

        self._agent_location = np.array(valid_action_path[-1], dtype=np.int32)

        reward = 0.0
        self._path = []

        if collision:
            self.num_collisions += 1
            reward = -1.0
        elif len(valid_action_path) > 1:  # only do it if the agent moves
            self._path = valid_action_path[:-1]  # remove the agent location
            if np.array_equal(self._target_location, self._agent_location):
                terminated = True
                reward += self._compute_reward(completion_reward=True)
                self.num_targets_reached += 1
                self.logger.info(
                    "%s Target reached in %s steps, Collisions : %s",
                    self.ordinal(self.num_targets_reached),
                    self.current_step,
                    self.num_collisions,
                )
            else:
                reward += self._compute_reward(completion_reward=False)
        else:
            self.logger.warning("No movement caused by action: %s!", action)

        self.min_distance_to_target = float(np.linalg.norm(self._target_location - self._agent_location))
        self.min_cost_astar = self.cost_astar

        if self.num_collisions > self.max_collisions:
            terminated = True
            if not self.partial_reward:
                reward += -self.cost_astar / 100
            self.logger.info(
                " --Max Collisions! Collisions: %s/%s",
                self.num_collisions,
                self.max_collisions + 1,
            )
        elif (not terminated) and (self.current_step > self.max_steps):
            terminated = True
            if not self.partial_reward:
                reward += -self.cost_astar / 100
            self.logger.info(
                " --Max Steps! Steps: %s/%s",
                self.current_step,
                self.max_steps + 1,
            )

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Renders the environment

        Returns
        -------
        None
        """
        plt.subplot(1, 2, 1)
        plt.title("Agent View")
        plt.imshow(np.moveaxis(self.obs_world, 0, 2))
        plt.subplot(1, 2, 2)
        plt.title("Current Astar Path")
        plt.imshow(np.moveaxis(self.obs_world_astar, 0, 2))

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
