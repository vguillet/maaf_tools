
##################################################################################################################

import json
from pprint import pprint
import warnings

from bloom.generators.rpm.generate_cmd import description

#from typing import Self

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


class AllocationSpecification(dict, MaafItem):
    def __init__(self,
                 allocation_specification: dict or None = None,
                 functional_specification = None,
                 structural_specification = None,
                 deontic_specification = None
                 ):
        # If no specification is provided, use the default template.
        if allocation_specification is None:
            allocation_specification = {
                "allocation": []
            }

        elif isinstance(allocation_specification, AllocationSpecification):
            allocation_specification = allocation_specification.copy()

        elif isinstance(allocation_specification, dict):
            allocation_specification = allocation_specification.copy()

        elif not isinstance(allocation_specification, dict) or not isinstance(allocation_specification, AllocationSpecification):
            raise ValueError("The allocation specification must be a dictionary or AllocationSpecification object.")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(allocation_specification)

        # -> Initialize the structural, functional, and deontic specification
        self.__structural_specification = structural_specification
        self.__functional_specification = functional_specification
        self.__deontic_specification = deontic_specification

    # ============================================================== Properties
    @property
    def structural_specification(self):
        return self.__structural_specification

    @structural_specification.setter
    def structural_specification(self, value):
        self.__structural_specification = value

    @property
    def functional_specification(self):
        return self.__functional_specification

    @functional_specification.setter
    def functional_specification(self, value):
        self.__functional_specification = value

    @property
    def deontic_specification(self):
        return self.__deontic_specification

    @deontic_specification.setter
    def deontic_specification(self, value):
        self.__deontic_specification = value

    # ============================================================== Check

    # ============================================================== Get
    def get_group_ambassadors(self, group_id: str):
        """
        Get the ambassadors for a specific group ID.

        :param group_id: The ID of the group.
        :return: List of ambassadors.
        """
        pass

    def get_intercession_targets(self, goal_id, roles: list) -> list:
        """
        Get the intercession targets for a specific goal ID and roles.

        :param goal_id: The ID of the goal.
        :param roles: List of roles to check for intercession targets.
        :return: List of intercession targets (agent_ids).
        """
        pass

    def get_hierarchy_level(self, agent_id: str, group_id: str) -> str:
        """
        Get the hierarchy level of a specific agent within a group.

        :param agent_id: The ID of the agent.
        :param group_id: The ID of the group.
        :return: Hierarchy level (e.g., "P1", "P2", "Captain", etc...).
        """
        pass


    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self