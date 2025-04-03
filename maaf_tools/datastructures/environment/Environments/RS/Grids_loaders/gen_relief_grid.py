
##################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs
import cv2
from pathlib import Path

from cv2 import split

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def gen_relief_grid(world_image_path: str, 
                    relief_image_path: str, 
                    cache_grid: bool = True
                    ):

    if Path(relief_image_path).is_file():
        img = cv2.imread(relief_image_path, cv2.IMREAD_UNCHANGED)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    else:
        # -> Get asset path
        asset_ref = f"/Height_image_{world_image_path.split('/')[-1].split('_')[-1]}"
        asset_path = "/".join(world_image_path.split('/')[:-1]) + asset_ref
        
        # -> Load image in
        mask = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)

        # -> Convert to array of ints
        gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # -> Save obstacle image if path provided
        if relief_image_path is not None and cache_grid:
            cv2.imwrite(relief_image_path, mask)

    return gray_img
