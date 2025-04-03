
from enum import Enum


class Action(Enum):
    EMPTY = -1
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DIAGONAL_UL = 5
    DIAGONAL_UR = 6
    DIAGONAL_DL = 7
    DIAGONAL_DR = 8
