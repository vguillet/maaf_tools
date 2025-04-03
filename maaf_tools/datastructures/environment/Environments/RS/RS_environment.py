
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
from pathlib import Path

# Own modules
from MAF.MAF_array_environment import MAF_array_environment
from Environments.Tools.action import Action
from Environments.Tools.tile_type import TileType
from Environments.RS.Grids_loaders.gen_obstacle_grid import gen_obstacle_grid
from Environments.RS.Grids_loaders.gen_terminal_fail_grid import gen_terminal_fail_grid
from Environments.RS.Grids_loaders.gen_paths_grid import gen_paths_grid
from Environments.RS.Grids_loaders.gen_relief_grid import gen_relief_grid

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
                 simulation_origin: tuple = (3136, 3136),
                 scale_factor: int = 1,
                 cache_grids: bool = False,
                 cache_path: str = None
                 ):
        """
        RS environment class, used to generate RS environments
        """
        environment_type = "RS"

        # ---- Store all inputs (! Important for instance cloning)
        self.namespace = namespace
        self.map_reference = map_reference 
        self.simulation_origin = simulation_origin
        self.scale_factor = scale_factor
        self.cache_grids = cache_grids
        self.cache_path = cache_path

        # -> Get environment map path
        world_image_path = f"{str(os.path.dirname(__file__))}/Assets/{map_reference}.png"

        if not Path(world_image_path).is_file():
            print("!!!!! Environment image is not provided !!!!!")
            print(f" Path set: {world_image_path}")
            sys.exit()

        # -> Load world image
        world_image = cv2.imread(world_image_path, cv2.IMREAD_UNCHANGED)

        # -> Get cache folder
        if cache_path is None:
            root = str(os.getcwd())
            folder_path = f"{root}/Environment/{environment_type}/{map_reference}"
        else:
            folder_path = f"{cache_path}/{map_reference}"

        # -> Create folder if it doesn't exist
        if not os.path.exists(folder_path) and cache_grids is True:
            os.makedirs(folder_path)

        # ---- Generate grids
        grids = []

        # -> Setup obstacle grid
        obstacle_grid = gen_obstacle_grid(world_image_path=world_image_path,
                                          obstacle_image_path=folder_path + "/obstacle.png",
                                          cache_grid=cache_grids)

        grids.append({
            "ref": "obstacle",
            "category": ["blocking", "comms"],
            "tile_type": TileType.OBSTACLE,
            "array": obstacle_grid,
            "comms_permeability": 0.0
        })

        # -> Setup terminal fail grid
        terminal_fail_grid = gen_terminal_fail_grid(world_image_path=world_image_path,
                                                    terminal_fail_image_path=folder_path + "/terminal_fail.png",
                                                    cache_grid=cache_grids)

        grids.append({
            "ref": "terminal_fail",
            "category": ["blocking"],
            "tile_type": TileType.TERMINAL_FAIL,
            "array": terminal_fail_grid
        })

        # -> Setup path grid
        path_grid = gen_paths_grid(world_image_path=world_image_path,
                                   path_image_path=folder_path + "/path.png",
                                   cache_grid=cache_grids
                                   )

        grids.append({
            "ref": "path",
            "category": ["passable"],
            "tile_type": TileType.PATHS,
            "array": path_grid
        })

        # -> Setup terrain relief
        try:
            relief_grid = gen_relief_grid(world_image_path=world_image_path,
                                        relief_image_path=folder_path + "/height_map.png",
                                        cache_grid=cache_grids)

            grids.append({
                "ref": "height",
                "category": [],
                "tile_type": TileType.EMPTY,
                "array": relief_grid
            })
            
        except:
            pass

        # -> Initialise MAF environment
        MAF_array_environment.__init__(self,
                                       namespace=namespace,
                                       map_reference=map_reference,
                                       environment_type=environment_type,

                                       world_image=world_image,
                                       grids=grids,

                                       simulation_origin=simulation_origin,
                                       scale_factor=scale_factor,
                                       cache_grids=cache_grids,
                                       cache_path=cache_path
                                       )
