
################################################################################################################
"""
Environment based on the popular Runscape MMORPG.
https://mapgenie.io/old-school-runescape/maps/runescape-surface
"""

# Built-in/Generic Imports
from copy import deepcopy
import random
import os
from abc import ABC, abstractmethod

# Libs
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# Own modules
from Environments.Tools.Grid_tools import reduce_grid_scale
from Environments.Tools.tile_type import *
from Environments.Tools.action import Action
from Environments.Tools.Grid_tools import convert_coordinates

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


class MAF_environment:
    def __init__(self,
                 namespace: str,         # Name of the environment instance
                 environment_type: str,  # Type of environment
                 map_reference: str = "Unknown",     # Reference to the map used
                 ):
        """
        RS environment class, used to generate RS environments
        """

        # ----- Setup reference properties
        self.id = id(self)
        self.name = namespace
        self.environment_type = environment_type
        self.map_reference = map_reference

        self._state = None
        self._goal = None
        self._path_to_goal = None
        self._weighted_path_to_goal = None

    def __str__(self):
        return f"{self.id} - {self.environment_type} environment - {self.map_reference}"

    def __repr__(self):
        return self.__str__()

    # ===================================== Properties
    # -------------------- Abstract
    @property
    @abstractmethod
    def valid_start_positions(self) -> list:
        pass

    # ===================================== Prints
    @staticmethod
    def gen_rainbow_color_list(length):
        colors_raw = list(cm.rainbow(np.linspace(0, 1, length)))

        colors_rgb = []

        for color in range(len(colors_raw)):
            colors_rgb.append(list(colors_raw[color]))

            for value in range(len(colors_rgb[color])):
                colors_rgb[color][value] = int(255 * colors_rgb[color][value])

            colors_rgb[color] = colors_rgb[color][:-1]

        return colors_raw, colors_rgb

    @staticmethod
    def get_shade(rgb_color: tuple, shade_factor: int):
        rgb_color[0] *= shade_factor
        rgb_color[1] *= shade_factor
        rgb_color[2] *= shade_factor

        return rgb_color

    @staticmethod
    def hex_to_rgb(hex: str):
        rgb = []
        for i in (0, 2, 4):
            decimal = int(hex[i:i + 2], 16)
            rgb.append(decimal)

        return tuple(rgb)

    @staticmethod
    def rgb_to_hex(r: int, g: int, bv):
        return '#{:02x}{:02x}{:02x}'.format(r, g, bv)

    # -------------------- Abstract
    @abstractmethod
    def render_terrain(self,
                       paths: list = [],
                       positions: list = [],
                       POIs: list = [],
                       show: bool = True,
                       flat: bool = False
                       ):
        pass

    @abstractmethod
    def render_comms(self,
                     paths: list = [],
                     positions: list = [],
                     POIs: list = [],
                     show: bool = True,
                     flat: bool = True
                     ):
        pass

    @abstractmethod
    def animate(self,
                background: str = "terrain",  # -> "terrain" or "comms"
                paths: list = [],
                plot_paths: bool = False,
                POIs: list = [],
                duration: int = 200,
                save_path: str = None
                ):
        pass

    @abstractmethod
    def render_path(self, path, positions: list = [], show: bool = True):
        pass

    # ===================================== Interfaces
    # -------------------- Abstract
    @abstractmethod
    def compute_path_from_to(self, start, end, weighted, *args, **kwargs):
        pass
