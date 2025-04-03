
################################################################################################################
"""
Environment based on the popular Runscape MMORPG.
https://mapgenie.io/old-school-runescape/maps/runescape-surface
"""

# Built-in/Generic Imports
import sys
import os

# Libs
import cv2
import numpy as np
from pathlib import Path

# Own modules
from MAF.MAF_array_environment import MAF_array_environment
from Environments.Tools.action import Action
from Environments.Tools.tile_type import TileType
from Environments.RS.Grids_loaders.gen_obstacle_grid import gen_obstacle_grid
from Environments.RS.Grids_loaders.gen_terminal_fail_grid import gen_terminal_fail_grid
from Environments.RS.Grids_loaders.gen_paths_grid import gen_paths_grid

# from src.Visualiser.Visualiser import Visualiser
# from src.Visualiser.Visualiser_tools import Visualiser_tools


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '31/01/2020'

################################################################################################################

SUPPORTED_ACTIONS = [
    Action.WAIT,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    ]


class Environment(MAF_array_environment):
    def __init__(self,
                 namespace: str,
                 map_reference: str,
                 simulation_origin: tuple = (0, 0),
                 start_state_world: tuple = None,
                 start_state_sim: tuple = None,
                 goal_world: tuple = None,
                 goal_sim: tuple = None,
                 scale_factor: int = 1,
                 cache_grids: bool = False
                 ):
        """
        RS environment class, used to generate RS environments
        """
        environment_type = "Empty"

        # ---- Store all inputs (! Important for instance cloning)
        self.namespace = namespace
        self.map_reference = map_reference
        self.simulation_origin = simulation_origin
        self.start_state_world = start_state_world
        self.start_state_sim = start_state_sim
        self.goal_world = goal_world
        self.goal_sim = goal_sim
        self.scale_factor = scale_factor
        self.cache_grids = cache_grids

        # -> Get environment map path
        world_image_path = f"{str(os.path.dirname(__file__))}/Assets/{map_reference}.png"


        if not Path(world_image_path).is_file():
            print("!!!!! Environment image is not provided !!!!!")
            print(f" Path set: {world_image_path}")
            sys.exit()

        # -> Load world image
        world_image = cv2.imread(world_image_path, cv2.IMREAD_UNCHANGED)

        # -> Get environment root path
        root = str(os.getcwd())
        folder_path = f"{root}/Environment/{environment_type}/{map_reference}"

        # -> Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # ---- Generate grids
        grids = []

        # -> Create empty 1000 x 1000 array
        empty_grid = np.zeros(shape=(1000, 1000))

        grids.append({
            "ref": "base",
            "category": [],
            "tile_type": TileType.EMPTY,
            "array": empty_grid
        })

        # -> Initialise MAF environment
        MAF_array_environment.__init__(self,
                                       namespace=namespace,
                                       map_reference=map_reference,
                                       environment_type=environment_type,

                                       world_image=world_image,
                                       grids=grids,

                                       simulation_origin=(0, 0),
                                       start_state_world=start_state_world,
                                       start_state_sim=start_state_sim,
                                       goal_world=goal_world,
                                       goal_sim=goal_sim,
                                       scale_factor=scale_factor,
                                       cache_grids=cache_grids
                                       )
