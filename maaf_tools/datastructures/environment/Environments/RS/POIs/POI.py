
################################################################################################################
"""

"""

# Built-in/Generic Imports

# Libs
# from faker import Faker

# Own modules
from Environments.Tools.Grid_tools import convert_coordinates


__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '31/01/2020'

################################################################################################################


class POI:
    def __init__(self, 
                 name: str,     # Name of POI 
                 POI_class: str,     # POI class
                 simulation_origin, simulation_shape,
                 world_pos: tuple = None, sim_pos: tuple = None):
        
        # ----- Setup reference properties
        self.id = id(self)
        
        self.name = name
        self.POI_class = POI_class

        # --> Setup POI position
        self.world_pos, self.simulation_pos = convert_coordinates(simulation_origin=simulation_origin,
                                                                  simulation_shape=simulation_shape,
                                                                  world_pos=world_pos,
                                                                  simulation_pos=sim_pos)

    def __str__(self):
        return f"{self.id} - {self.name} ({self.POI_class})"

    def __repr__(self):
        return self.__str__()
