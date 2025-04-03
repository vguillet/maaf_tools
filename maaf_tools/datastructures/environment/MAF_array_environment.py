
################################################################################################################
"""
Environment based on the popular Runscape MMORPG.
https://mapgenie.io/old-school-runescape/maps/runescape-surface
"""

# Built-in/Generic Imports
from copy import deepcopy
import random
import os

# Libs
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.best_first import BestFirst

# Own modules
from MAF_Environments.MAF_environment import MAF_environment
from MAF_Environments.Tools.Grid_tools import reduce_grid_scale
from MAF_Environments.Tools.tile_type import *
from MAF_Environments.Tools.action import Action
from MAF_Environments.Tools.Grid_tools import convert_coordinates

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


class MAF_array_environment(MAF_environment):
    def __init__(self,
                 namespace: str,         # Name of the environment instance
                 map_reference: str,     # Reference to the map used
                 environment_type: str,  # Type of environment

                 world_image,
                 grids: list,

                 simulation_origin: tuple = (3136, 3136),
                 scale_factor: int = 1,
                 cache_grids: bool = False,
                 cache_path: str = None
                 ):
        """
        RS environment class, used to generate RS environments
        """

        # -> Initialise parent class
        MAF_environment.__init__(
            self,
            namespace=namespace,
            environment_type=environment_type,
            map_reference=map_reference
        )

        # ----- Setup reference properties
        self.origin = simulation_origin        # Origin coordinates of grid on the exploded map

        self.shape = grids[0]["array"].shape
        self.world_image = world_image

        # ----- Setup environment grids
        # -> Save generated array
        if cache_grids:
            # -> Create folder
            if cache_path is None:
                root = str(os.getcwd())
                folder_path = f"{root}/Environment/{environment_type}/{map_reference}"
            else:
                folder_path = f"{cache_path}/{map_reference}"

            for grid in grids:
                np.save(folder_path + f"/grid_{grid['ref']}", grid['array'])

        # -> Scale grids
        if scale_factor != 1:
            for grid in grids:
                grid['array'] = reduce_grid_scale(grid=grid['array'], scale_factor=scale_factor)

        # ----- Create combined grids
        # -> Maze grid
        maze_array = np.zeros_like(grids[0]['array'])

        for grid in grids:
            if grid["tile_type"] is not None and grid["tile_type"] != TileType.EMPTY:
                for coordinates in np.argwhere(grid['array'] == 1):
                    maze_array[coordinates[0], coordinates[1]] = grid["tile_type"].value

        # -> Blocked tiles grid
        blocked_tiles_grid = np.zeros_like(grids[0]['array'])

        for grid in grids:
            if "blocking" in grid["category"]:
                for coordinates in np.argwhere(grid['array'] == 1):
                    blocked_tiles_grid[coordinates[0], coordinates[1]] = 1

        # -> Comms grid
        comms_grid = np.ones_like(grids[0]['array'], dtype=np.float32)

        for grid in grids:
            if "comms" in grid["category"]:
                for coordinates in np.argwhere(grid['array'] == 1):
                    # -> Set comms permeability to lowest value
                    if comms_grid[coordinates[0], coordinates[1]] > grid["comms_permeability"]:
                        comms_grid[coordinates[0], coordinates[1]] = grid["comms_permeability"]

        self.grids_definition = grids

        # -> Setup grids dictionary
        self.grids_dict = {
            "maze": maze_array,
            "blocked_tiles": blocked_tiles_grid,
            "comms": comms_grid
        }

        # -> Add all grids to dictionary
        for grid in grids:
            self.grids_dict[grid["ref"]] = grid["array"]

        # -> Add placeholder height map if not provided
        if "height" not in self.grids_dict.keys():
            self.grids_dict["height"] = None

    # ===================================== Properties
    @property
    def maze_array(self):
        return self.grids_dict["maze"]

    @property
    def valid_start_positions(self) -> list:
        return np.argwhere(self.grids_dict["blocked_tiles"] == 0).tolist()

    # ===================================== Prints
    def render_terrain(self,
                       paths: list = [], 
                       positions: list = [],
                       POIs: list = [],
                       show: bool = True,
                       flat: bool = False, 
                       ):

        # ----- Create background image
        shape = self.grids_dict["maze"].shape
        image_array = np.full((shape[0], shape[1], 3), fill_value=TileColor.EMPTY.value, dtype=np.uint8)

        # > Tiles
        for tile_type in TileType:
            for coordinates in np.argwhere(self.grids_dict["maze"] == tile_type.value):
                image_array[coordinates[0], coordinates[1]] = TileColor[tile_type.name].value

        # > Comms
        # Adjust shade of tiles to reflect comms permeability
        for coordinates in np.argwhere(self.grids_dict["comms"] != 1):
            image_array[coordinates[0], coordinates[1]] = self.get_shade(
                rgb_color=image_array[coordinates[0], coordinates[1]],
                shade_factor=self.grids_dict["comms"][coordinates[0], coordinates[1]]
            )

        # > Obstacles
        # Go over obstacles tiles and set them to obstacle color
        # (to remove shading from comms as obstacles are always opaque
        if "obstacle" in self.grids_dict.keys():
            for coordinates in np.argwhere(self.grids_dict["obstacle"] == 1):
                image_array[coordinates[0], coordinates[1]] = TileColor.OBSTACLE.value

        # # > Paths
        # agents_path_color = self.gen_rainbow_color_list(length=len(paths))

        # for i, path in enumerate(paths):
        #     for step in path:
        #         image_array[step[0], step[1]] = agents_path_color[i]

        # # > Positions
        # position_colors = self.gen_rainbow_color_list(length=len(positions))

        # for i, position in enumerate(positions):
        #     image_array[position[0], position[1]] = position_colors[i]

        # ----- Create matplotlib plot
        fig, ax, image_array = self._render_state(
            image_array=image_array,
            height_map=self.grids_dict["height"],
            paths=paths,
            positions=positions,
            POIs=POIs,
            show=show,
            flat=flat
        )

        return fig, ax, image_array

    def render_comms(self,
                     paths: list = [], 
                     positions: list = [],
                     POIs: list = [],
                     show: bool = True,
                     flat: bool = True
                     ):

        shape = self.grids_dict["maze"].shape
        image_array = np.full((shape[0], shape[1], 3), fill_value=(255, 255, 255), dtype=np.uint8)

        # > Comms
        # Adjust shade of tiles to reflect comms permeability
        for coordinates in np.argwhere(self.grids_dict["comms"] != 1):
            image_array[coordinates[0], coordinates[1]] = self.get_shade(
                rgb_color=image_array[coordinates[0], coordinates[1]],
                shade_factor=self.grids_dict["comms"][coordinates[0], coordinates[1]]
            )

        # # > Paths
        # agents_path_color = self.gen_rainbow_color_list(length=len(paths))

        # for i, path in enumerate(paths):
        #     for step in path:
        #         image_array[step[0], step[1]] = agents_path_color[i]

        # # > Positions
        # position_colors = self.gen_rainbow_color_list(length=len(positions))

        # for i, position in enumerate(positions):
        #     image_array[position[0], position[1]] = position_colors[i]

        # # -> Create image
        # img = Image.fromarray(image_array, 'RGB')

        # if show:
        #     img.show()

        # ----- Create matplotlib plot
        fig, ax, image_array = self._render_state(
            image_array=image_array,
            height_map=self.grids_dict["height"],
            paths=paths,
            positions=positions,
            POIs=POIs,
            show=show,
            flat=flat
        )

        return fig, ax, image_array

    def animate(self,
                background: str = "terrain",    # -> "terrain" or "comms"
                paths: list = [],
                plot_paths: bool = False,
                POIs: list = [],
                duration: int = 200,
                save_path: str = None
                ):

        print(">>> Creating animation")

        # -> Create base env image
        if background == "terrain":
            fig, ax, img_array = self.render_terrain(show=False)
        elif background == "comms":
            fig, ax, img_array = self.render_comms(show=False)
        else:
            raise ValueError(f"Invalid background type: {background}, must be 'terrain' or 'comms'")

        # -> Create agent colors
        agent_colors = self.gen_rainbow_color_list(length=len(paths))

        # -> Get max path length
        max_path_length = 0

        for path in paths:
            if len(path) > max_path_length:
                max_path_length = len(path)

        # -> Create frames
        frames = []

        for frame_index in range(max_path_length):
            print(f"> Frame {frame_index+1}/{max_path_length}")
        
            new_frame = deepcopy(img_array)

            # -> Color agents positions
            for agent_index, path in enumerate(paths):
                # -> Plot paths
                if plot_paths:
                    try:
                        # -> Agent path
                        for step in range(len(path[:frame_index])):
                            new_frame[path[step][0], path[step][1]] = (255, 255, 0)
                    except IndexError:
                        pass

                # -> Plot agents positions
                try:
                    new_frame[path[frame_index][0], path[frame_index][1]] = agent_colors[agent_index]
                except IndexError:
                    pass

            for POI in POIs:
                new_frame[POI][frame_index][0], POI[frame_index][1] = TileColor.POIs

            # -> Create image
            new_frame = Image.fromarray(new_frame, 'RGB')

            # -> Save new frame
            frames.append(new_frame)

        # -> Create gif
        # > Get root
        if save_path is None:
            root = str(os.getcwd()) + "/Results"

            # > Create folder if not exists
            if not os.path.exists(root):
                os.makedirs(root)

            # > Create save path
            save_path = f"{root}/{self.name}_{self.environment_type}_{self.map_reference}.gif"

        frames[0].save(save_path,
                       save_all=True,
                       append_images=frames[1:],
                       optimize=False,
                       duration=1/(duration/len(frames)),
                       fps= duration/len(frames),
                       loop=1)
                       
        print(f"- Animation saved to: {save_path}")

    def plt_animate(self,
                    background: str = "terrain",    # -> "terrain" or "comms"
                    paths: list = [],
                    POIs: list = [],
                    duration: int = 20,
                    save_path: str = None
                    ):

            print("----- Creating animation")

            # -> Create base env image
            if background == "terrain":
                fig, ax, img_array = self.render_terrain(show=False)
            elif background == "comms":
                fig, ax, img_array = self.render_comms(show=False)
            else:
                raise ValueError(f"Invalid background type: {background}, must be 'terrain' or 'comms'")

            # -> Create agent colors
            agent_colors = self.gen_rainbow_color_list(length=len(paths))

            # -> Create a scatter plot for each agent loc
            agent_positions = []

            # ... for every agent
            for agent in range(self.fleet.base_agent_count):
                agent_positions.append(ax.scatter([], [], color=agent_colors[agent], marker="D"))

            # -> Get max path length
            max_path_length = 0

            for path in paths:
                if len(path) > max_path_length:
                    max_path_length = len(path)

            # -> Create frames
            frames = []

            for frame_index in range(max_path_length):
                print(f"> Frame {frame_index+1}/{max_path_length}")
                
                

                # new_frame = deepcopy(img_array)

                # # -> Color agents positions
                # for agent_index, path in enumerate(paths):
                #     try:
                #         # -> Agent path
                #         for step in range(len(path[:frame_index])):
                #             new_frame[path[step][0], path[step][1]] = (255, 255, 0)

                #         # -> Agent pos
                #         new_frame[path[frame_index][0], path[frame_index][1]] = agent_colors[agent_index]
                #     except IndexError:
                #         new_frame[path[-1][0], path[-1][1]] = agent_colors[agent_index]

                # -> Create image
                new_frame = Image.fromarray(new_frame, 'RGB')

                # -> Save new frame
                frames.append(new_frame)

            # -> Create gif
            # > Get root
            if save_path is None:
                root = str(os.getcwd()) + "/Results"

                # > Create folder if not exists
                if not os.path.exists(root):
                    os.makedirs(root)

                # > Create save path
                save_path = f"{root}/{self.name}_{self.environment_type}_{self.map_reference}.gif"

            frames[0].save(save_path,
                        save_all=True,
                        append_images=frames[1:],
                        optimize=False,
                        duration=duration,
                        loop=0)
                        
            print(f"- Animation saved to: {save_path}")

    def render_path(self, path, positions: list = [], show: bool = True):
        shape = self.grids_dict["maze"].shape
        image_array = np.full((shape[0], shape[1], 3), fill_value=(102, 102, 102), dtype=np.uint8)

        # > Add blocking tiles
        for grid in self.grids_definition:
            if "blocking" in grid["category"]:
                for coordinates in np.argwhere(grid['array'] == 1):
                    image_array[coordinates[0], coordinates[1]] = TileColor[grid["tile_type"].name].value

        # > Path steps
        for step in path:
            image_array[step[0]][step[1]] = (255, 255, 0)

        # > Positions
        position_colors = self.gen_rainbow_color_list(length=len(positions))

        for i, position in enumerate(positions):
            image_array[position[0], position[1]] = position_colors[i]

        # -> Create image
        img = Image.fromarray(image_array, 'RGB')

        if show:
            img.show()

        return img, image_array

    # ===================================== Interfaces
    def compute_path_from_to(self, start, end):
        # > Set pathfinding grid
        grid = Grid(matrix=1-self.grids_dict["blocked_tiles"])

        # > Define starting point and goal as current state and current goal
        start = grid.node(start[-1], start[0])
        end = grid.node(end[-1], end[0])

        # > Set pathfinding algorithm
        # finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        finder = BestFirst(diagonal_movement=DiagonalMovement.only_when_no_obstacle)

        # -> Find path
        path, _ = finder.find_path(start, end, grid)

        if len(path) == 0:
            print(f"WARNING: Could not find path for {start} -> {end}")

        # -> Format path
        ordered_path = []
        for coordinate in path:
            ordered_path.append(tuple(reversed(coordinate)))

        return ordered_path

