
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs
import cv2
from pathlib import Path

import numpy as np

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def gen_obstacle_grid(world_image_path: str, 
                      obstacle_image_path: str,
                      cache_grid: bool = True
                      ):
    
    # -> Load world image
    world_image = cv2.imread(world_image_path, cv2.IMREAD_UNCHANGED)

    if Path(obstacle_image_path).is_file():
        mask = cv2.imread(obstacle_image_path, cv2.IMREAD_UNCHANGED)

    else:
        # -> Convert image to hsv colorspace
        img_hsv = cv2.cvtColor(world_image, cv2.COLOR_BGR2HSV)

        # -> Filter image colors
        # > Create Walls mask
        walls_low = np.asarray([0, 0, 255])
        walls_high = np.asarray([0, 0, 255])

        walls_mask = cv2.inRange(img_hsv, walls_low, walls_high)

        # -> Add masks
        mask = walls_mask

        # -> Save obstacle image if path provided
        if obstacle_image_path is not None and cache_grid:
            cv2.imwrite(obstacle_image_path, mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if Path(obstacle_image_path[:-4] + ".npy").is_file():
        mask = np.load(obstacle_image_path[:-4] + ".npy")

    else:
        # -> Set obstacles to 1 and rest to 0 on mask
        mask[mask == 0] = 0
        mask[mask == 255] = 1

    return mask.astype(int)
