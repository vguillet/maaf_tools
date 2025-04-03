
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

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def gen_terminal_fail_grid(world_image_path: str, 
                           terminal_fail_image_path: str,
                           cache_grid: bool = True
                           ):
    # -> Load world image
    world_image = cv2.imread(world_image_path, cv2.IMREAD_UNCHANGED)

    if Path(terminal_fail_image_path).is_file():
        mask = cv2.imread(terminal_fail_image_path, cv2.IMREAD_UNCHANGED)

    else:
        # -> Convert image to hsv colorspace
        img_hsv = cv2.cvtColor(world_image, cv2.COLOR_BGR2HSV)

        # -> Filter image colors
        # > Create Water mask
        water_low = np.asarray([107, 104, 162])
        water_high = np.asarray([112, 106, 166])

        water_mask = cv2.inRange(img_hsv, water_low, water_high)

        # -> Add masks
        mask = water_mask
        # mask = cv2.bitwise_or(walls_mask, water_mask)
        # mask = cv2.bitwise_or(icons_mask, mask)

        # -> Save terminal fail image if path provided
        if terminal_fail_image_path is not None and cache_grid:
            cv2.imwrite(terminal_fail_image_path, mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if Path(terminal_fail_image_path[:-4] + ".npy").is_file():
        mask = np.load(terminal_fail_image_path[:-4] + ".npy")

    else:
        # -> Set terminal fail to 1 and rest to 0 on mask
        mask[mask == 0] = 0
        mask[mask == 255] = 1

        return mask.astype(int)


if __name__ == "__main__":
    world_terminal_fail_array = gen_terminal_fail_grid(world_image_path="/home/vguillet/Documents/Repositories/CBAA_with_intercession/Environments/RS/Assets/RS_M.png",
                                                       obstacle_image_path="src\Data\Assets\Environment\Obstacle_image_L.png")

    plt.imshow(world_terminal_fail_array)
    plt.colorbar()
    plt.show()

