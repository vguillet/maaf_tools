
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs
import cv2
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Environments.Tools.Grid_tools import reduce_grid_scale

# from pathfinding.core.diagonal_movement import DiagonalMovement
# from pathfinding.core.grid import Grid
# from pathfinding.finder.a_star import AStarFinder

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def gen_paths_grid(world_image_path, 
                   path_image_path=None,
                   cache_grid: bool = True
                   ):

    # -> Load world image
    world_image = cv2.imread(world_image_path, cv2.IMREAD_UNCHANGED)

    if Path(path_image_path).is_file():
        mask = cv2.imread(path_image_path, cv2.IMREAD_UNCHANGED)

    else:
        # -> Load image
        img_hsv = cv2.cvtColor(world_image, cv2.COLOR_BGR2HSV)

        # -> Filter image colors
        # > Create stone_road_1 mask
        stone_road_1_low = np.asarray([0, 25, 78])
        stone_road_1_high = np.asarray([5, 30, 85])

        stone_road_1_mask = cv2.inRange(img_hsv, stone_road_1_low, stone_road_1_high)

        # > Create stone_road_2 mask
        stone_road_2_low = np.asarray([15, 20, 95])
        stone_road_2_high = np.asarray([25, 35, 105])

        stone_road_2_mask = cv2.inRange(img_hsv, stone_road_2_low, stone_road_2_high)

        # > Create stone_road_3 mask
        stone_road_3_low = np.asarray([15, 65, 95])
        stone_road_3_high = np.asarray([25, 75, 105])

        stone_road_3_mask = cv2.inRange(img_hsv, stone_road_3_low, stone_road_3_high)

        # > Create desert_road mask
        desert_road_low = np.asarray([27, 103, 114])
        desert_road_high = np.asarray([27, 103, 114])

        desert_road_mask = cv2.inRange(img_hsv, desert_road_low, desert_road_high)

        # > Create interiors mask
        interiors_low = np.asarray([15, 60, 160])
        interiors_high = np.asarray([25, 75, 170])

        interiors_mask = cv2.inRange(img_hsv, interiors_low, interiors_high)

        # > Create dirt_road_1 mask
        dirt_road_1_low = np.asarray([15, 130, 75])
        dirt_road_1_high = np.asarray([25, 145, 90])

        dirt_road_1_mask = cv2.inRange(img_hsv, dirt_road_1_low, dirt_road_1_high)

        # # > Create dirt_road_2 mask
        # dirt_road_2_low = np.asarray([20, 102, 120])
        # dirt_road_2_high = np.asarray([20, 102, 120])

        # dirt_road_2_mask = cv2.inRange(img_hsv, dirt_road_2_low, dirt_road_2_high)

        # # > Create dirt_road_3 mask
        # dirt_road_3_low = np.asarray([20, 153, 80])
        # dirt_road_3_high = np.asarray([20, 153, 80])

        # dirt_road_3_mask = cv2.inRange(img_hsv, dirt_road_3_low, dirt_road_3_high)

        # # > Create dirt_road_4 mask
        # dirt_road_4_low = np.asarray([22, 95, 116])
        # dirt_road_4_high = np.asarray([22, 95, 116])

        # dirt_road_4_mask = cv2.inRange(img_hsv, dirt_road_4_low, dirt_road_4_high)

        # # > Create dirt_road_5 mask
        # dirt_road_5_low = np.asarray([21, 185, 84])
        # dirt_road_5_high = np.asarray([21, 185, 84])

        # dirt_road_5_mask = cv2.inRange(img_hsv, dirt_road_5_low, dirt_road_5_high)

        # # > Create dirt_road_6 mask
        # dirt_road_6_low = np.asarray([19, 209, 61])
        # dirt_road_6_high = np.asarray([19, 209, 61])

        # dirt_road_6_mask = cv2.inRange(img_hsv, dirt_road_6_low, dirt_road_6_high)

        # # -> Add masks
        mask = cv2.bitwise_or(stone_road_1_mask, stone_road_2_mask)
        mask = cv2.bitwise_or(stone_road_3_mask, mask)
        mask = cv2.bitwise_or(desert_road_mask, mask)
        mask = cv2.bitwise_or(interiors_mask, mask)
        mask = cv2.bitwise_or(dirt_road_1_mask, mask)
        # mask = cv2.bitwise_or(dirt_road_2_mask, mask)
        # mask = cv2.bitwise_or(dirt_road_3_mask, mask)
        # mask = cv2.bitwise_or(dirt_road_4_mask, mask)
        # mask = cv2.bitwise_or(dirt_road_5_mask, mask)
        # mask = cv2.bitwise_or(dirt_road_6_mask, mask)

        # -> Save path image if path provided
        if path_image_path is not None and cache_grid:
            cv2.imwrite(path_image_path, mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if Path(path_image_path[:-4] + ".npy").is_file():
        mask = np.load(path_image_path[:-4] + ".npy")

    else:
        # -> Set paths to 1 and rest to 0 on mask
        mask[mask == 0] = 0
        mask[mask == 255] = 1

        return mask.astype(int)


if __name__ == "__main__":
    path_obstacles_array = gen_path_grid(world_image=cv2.imread("src\Data\Assets\Environment\World_image_L.png", cv2.IMREAD_UNCHANGED),
                                         path_image_path="src\Data\Environment\Path_image.png")

    plt.imshow(path_obstacles_array)
    plt.colorbar()
    plt.show()

