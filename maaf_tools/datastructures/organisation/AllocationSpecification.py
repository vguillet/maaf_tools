
##################################################################################################################

import json
from pprint import pprint
import warnings

from bloom.generators.rpm.generate_cmd import description

#from typing import Self

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel

##################################################################################################################


class AllocationSpecification(dict, MaafItem):
    def __init__(self,
                 allocation_specification: dict or None = None,
                 moise_model: dict or None = None,
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
        self.__moise_model = None
        self.moise_model = moise_model

    # ============================================================== Properties
    @property
    def moise_model(self):
        """
        Get the MOISE model associated with this allocation specification.

        :return: The MOISE model.
        """
        return self.__moise_model

    @moise_model.setter
    def moise_model(self, moise_model: dict):
        """
        Set the MOISE model associated with this allocation specification.

        :param moise_model: The MOISE model to set.
        """
        self.__moise_model = moise_model

    @property
    def moise_model_available(self) -> bool:
        """
        Check if the MOISE model is available.

        :return: True if the MOISE model is available, False otherwise.
        """

        # Check if the MOISE model is available
        if self.__moise_model is None or not isinstance(self.__moise_model, MoiseModel):
            return False

        return True

    # ============================================================== Check

    @staticmethod
    def moise_model_check(func):
        def wrapper(self, *args, **kwargs):
            if not self.moise_model_available:
                raise ValueError("The MOISE model is not available.")
            return func(self, *args, **kwargs)
        return wrapper

    # ============================================================== Get

    @moise_model_check
    def get_group_ambassadors(self, group_id: str):
        """
        Get the ambassadors for a specific group ID.

        :param group_id: The ID of the group.
        :return: List of ambassadors.
        """

        pass

    @moise_model_check
    def get_intercession_targets(self, goal_id, roles: list) -> list:
        """
        Get the intercession targets for a specific goal ID and roles.

        :param goal_id: The ID of the goal.
        :param roles: List of roles to check for intercession targets.
        :return: List of intercession targets (agent_ids).
        """
        pass

    @moise_model_check
    def get_hierarchy_level(self, agent_id: str, group_id: str) -> str:
        """
        Get the hierarchy level of a specific agent within a group.

        :param agent_id: The ID of the agent.
        :param group_id: The ID of the group.
        :return: Hierarchy level (e.g., "P1", "P2", "Captain", etc...).
        """
        pass

    @moise_model_check
    def get_task_bidding_logic(self, task_id: str) -> callable:
        """
        Get the bidding logic for a specific task ID.

        :param task_id: The ID of the task.
        :return: Bidding logic (function or callable).
        """
        pass

    def get_property(self, agent_id: str, property_name: str):
        """
        Get a specific property from the allocation specification for a given agent ID.

        :param agent_id: The ID of the agent.
        :param property_name: The name of the property to retrieve.
        :return: The value of the specified property.
        """

        # -> Check if the property name is in global properties
        if property_name in self.get("global", {}):
            return self["global"][property_name]

        # -> Check if the property name is in the agent's properties
        if agent_id in self.get("team", {}):
            agent_properties = self["team"][agent_id]
            if property_name in agent_properties:
                return agent_properties[property_name]
            else:
                raise ValueError(f"Property '{property_name}' not found for agent ID '{agent_id}'.")
        else:
            raise ValueError(f"Agent ID '{agent_id}' not found in the allocation specification.")

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self