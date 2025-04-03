
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import sys

# Libs
import numpy as np
import cv2

# Own modules


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


def load_feature_map_array(path: str):
    # -> Load image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # -> Set transparent values as white
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))

    # -> Convert to binary image
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)

    return bw_img


def reduce_grid_scale(grid, scale_factor):
    # -> Skip if scale factor is 1
    if scale_factor == 1:
        return grid

    vertical_chunk_count = int(grid.shape[0] / scale_factor)
    horizontal_chunk_count = int(grid.shape[1] / scale_factor)

    downscaled_array = np.zeros((vertical_chunk_count, horizontal_chunk_count))

    # -> Iterate through chunks
    for chunk_y in range(vertical_chunk_count):
        for chunk_x in range(horizontal_chunk_count):
            feature_counter = 0

            # -> Iterate through tiles in chunk
            for tile_y in range(scale_factor):
                for tile_x in range(scale_factor):
                    if grid[(chunk_y * scale_factor) + tile_y][(chunk_x * scale_factor) + tile_x] == 1:
                        feature_counter += 1
                    else:
                        pass

            # -> Flag tile as obstacle if obstacle counter is too high
            if feature_counter >= 2:
                downscaled_array[chunk_y][chunk_x] = 1

    # -> Convert to int
    downscaled_array = downscaled_array.astype(int)

    return downscaled_array


def convert_coordinates(simulation_origin,
                        simulation_shape,
                        world_pos=None,         # (x, y)
                        simulation_pos=None):   # (x, y)
    """
    Used to convert from both from world coordinates to sim coordinates and back.
    Specify known coordinates.

    :param simulation_origin: Simulation grid origin in world coordinates
    :param simulation_shape: Size of simulation grid
    :param world_pos: Position in world coordinates (Optional)
    :param simulation_pos: Position in sim coordinates (Optional)
    :return: world_coordinates, simulation coordinates
    """

    if simulation_pos is None:
        translated_x_coordinate = world_pos[0] - simulation_origin[0]
        translated_y_coordinate = simulation_shape[0] - (world_pos[1] - simulation_origin[1])

        return world_pos, (translated_y_coordinate, translated_x_coordinate)

    elif world_pos is None:
        translated_x_coordinate = simulation_pos[0] + simulation_origin[0]
        translated_y_coordinate = simulation_origin[1] + (simulation_shape[0] - simulation_pos[0])

        return (translated_x_coordinate, translated_y_coordinate), simulation_pos

    else:
        print("!!!!! Pos need to be specified in at least one coordinate system")
        sys.exit()


def convert_route_coordinates(simulation_origin, simulation_size,
                              world_path=None,
                              simulation_path=None):
    """
    Used to convert from both from world coordinates to sim coordinates and back.
    Specify known path.

    :param simulation_origin: Simulation grid origin in world coordinates
    :param simulation_size: Size of simulation grid
    :param world_path: Position in world coordinates (Optional)
    :param simulation_path: Position in sim coordinates (Optional)
    :return: World path, simulation path
    """

    if simulation_path is None:
        # --> Create simulation path
        simulation_path = []

        # --> Iterate through world path to convert steps to simulation coordinates
        for step in world_path:
            translated_x_coordinate = step[0] - simulation_origin[0]
            translated_y_coordinate = simulation_size[0] - (step[1] - simulation_origin[1])
            simulation_path.append((translated_x_coordinate, translated_y_coordinate))

        return world_path, simulation_path

    elif world_path is None:
        # --> Create world path
        world_path = []

        # --> Iterate through simulation path to convert steps to world coordinates
        for step in simulation_path:
            translated_y_coordinate = (simulation_size[0] - step[1]) + simulation_origin[1]
            translated_x_coordinate = step[0] + simulation_origin[0]
            world_path.append((translated_x_coordinate, translated_y_coordinate))

        return world_path, simulation_path

    else:
        print("!!!!! Path needs to be specified in at least one coordinate system")
        sys.exit()
