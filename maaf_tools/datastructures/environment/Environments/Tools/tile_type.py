
from enum import Enum


class TileType(Enum):
    EMPTY = 0
    PATHS = 1
    OBSTACLE = 2
    TERMINAL_FAIL = 3
    POIs = 4


class TileColor(Enum):
    EMPTY = (0, 102, 0)                 # Green
    PATHS = (153, 102, 51)              # Brown
    OBSTACLE = (255, 255, 255)          # White
    TERMINAL_FAIL = (255, 0, 0)         # Red
    POIs = (255, 153, 51)               # Orange