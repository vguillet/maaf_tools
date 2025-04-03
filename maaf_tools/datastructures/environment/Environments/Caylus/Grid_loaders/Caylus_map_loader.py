
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from Environments.Tools.Grid_tools import load_feature_map_array


def load_maps(
    assets_path: str,
    hard_obstacles: bool = True,
    dense_vegetation: bool = True, 
    light_vegetation: bool = True, 
    paths: bool = True) -> dict:

    # -> Initialise grids dict
    grid_dict = {}

    # -> load all requested grids
    if hard_obstacles:
        grid_dict["obstacles"] = 1-load_feature_map_array(path=assets_path + "/Caylus_map_obstacles.png")

    if dense_vegetation:
        grid_dict["dense_vegetation"] = 1-load_feature_map_array(path=assets_path + "/Caylus_map_dense_vegetation.png")

    if light_vegetation:
        grid_dict["light_vegetation"] = 1-load_feature_map_array(path=assets_path + "/Caylus_map_light_vegetation.png")

    if paths:
        grid_dict["paths"] = load_feature_map_array(path=assets_path + "/Caylus_map_paths.png")

    return grid_dict
