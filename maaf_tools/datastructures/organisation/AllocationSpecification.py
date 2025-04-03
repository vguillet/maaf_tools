
##################################################################################################################

import json
from pprint import pprint
import warnings

from bloom.generators.rpm.generate_cmd import description

#from typing import Self

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_allocation_node.bidding_logics.bidding_logics_dict import bidding_logics_dict

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_allocation_node.maaf_allocation_node.bidding_logics.bidding_logics_dict import bidding_logics_dict

##################################################################################################################

ABSTRACT_AUCTION_ROLES = {
    "Auction Participant": {
        "Announcer": {},
        "Ambassador": {},
        "Bidder": {}
    }
}


class AllocationSpecification(dict, MaafItem):
    def __init__(self,
                 allocation_specification: dict or None = None,
                 moise_model: dict or None = None,
                 role_allocation: dict or None = None,
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

        # -> Initialize the moise model and role allocation
        self.__moise_model = None
        self.moise_model = moise_model

        self.__role_allocation = None
        self.role_allocation = role_allocation


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

        # -> Check if the MOISE model is valid
        if isinstance(moise_model, MoiseModel):
            self.check_allocation_roles_structure()

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

    @property
    def role_allocation(self):
        """
        Get the role allocation associated with this allocation specification.

        :return: The role allocation.
        """
        return self.__role_allocation

    @role_allocation.setter
    def role_allocation(self, role_allocation: dict):
        """
        Set the role allocation associated with this allocation specification.

        :param role_allocation: The role allocation to set.
        """
        self.__role_allocation = role_allocation

    # ============================================================== Check
    @staticmethod
    def moise_model_check(func):
        def wrapper(self, *args, **kwargs):
            if not self.moise_model_available:
                raise ValueError("The MOISE model is not available.")
            return func(self, *args, **kwargs)
        return wrapper

    @moise_model_check
    def check_allocation_roles_structure(self):
        """
        Check the structure of the allocation roles.

        Ensure that the structural specification contains the basic key abstract roles structure:
        - Auction Participant (parent)
            - Announcer
            - Ambassador
            - Bidder

        :return: True if the structure is valid, False otherwise.
        """

        # -> Fetch roles from structural specification
        roles = self.moise_model.structural_specification.roles

        # -> Build a lookup dictionary for quick access by role name.
        role_lookup = {role["name"]: role for role in roles}

        def check_node(parent_name, children_dict):
            for child_name, grandchildren in children_dict.items():
                # Check that the child role exists.
                if child_name not in role_lookup:
                    return False
                # Check that the child's 'inherits' field is as expected.
                if role_lookup[child_name]["inherits"] != parent_name:
                    return False
                # Recursively check the children of this child.
                if not check_node(child_name, grandchildren):
                    return False
            return True

        # ... for each top-level role in the structure, verify the node and its descendants.
        for top_role, children in ABSTRACT_AUCTION_ROLES.items():
            if top_role not in role_lookup:
                return False
            if not check_node(top_role, children):
                return False

        return True

    # ============================================================== Get
    @moise_model_check
    def get_group_ambassadors(self, group_id: str):
        """
        Get the ambassadors for a specific group ID.

        :param group_id: The ID of the group.
        :return: List of ambassadors.
        """

        # -> Get ambassador roles
        ambassador_roles = self.moise_model.structural_specification.get_children_roles(parent_role="Ambassador")

        # -> Get agents in group
        agents = self.role_allocation.get_agents_in_group(group_id=group_id)

        # -> Get agents with ambassador role
        ambassadors = []
        for agent in agents:
            agent_roles = self.role_allocation.get_agent_roles(agent_id=agent, group_id=group_id)
            for role in agent_roles:
                if role in ambassador_roles:
                    ambassadors.append(agent)
                    break

        return ambassadors

    @moise_model_check
    def get_intercession_targets(self, goal_id, roles: list) -> list:
        """
        Get the intercession targets for a specific goal ID and roles.

        :param goal_id: The ID of the goal.
        :param roles: List of roles to check for intercession targets.
        :return: List of intercession targets (agent_ids).
        """
        raise NotImplementedError("This method is not implemented yet.")

    @moise_model_check
    def get_hierarchy_level(self, agent_id: str, group_id: str) -> str:
        """
        Get the hierarchy level of a specific agent within a group.

        :param agent_id: The ID of the agent.
        :param group_id: The ID of the group.
        :return: Hierarchy level (e.g., "P1", "P2", "Captain", etc...).
        """
        raise NotImplementedError("This method is not implemented yet.")

    @moise_model_check
    def get_task_bidding_logic(self, task_id: str) -> callable:
        """
        Get the bidding logic for a specific task ID.

        :param task_id: The ID of the task.
        :return: Bidding logic (function or callable).
        """
        # -> Get the task
        task = self.moise_model.functional_specification.get_goal(goal_id=task_id)

        if task is None:
            raise ValueError(f"Task ID '{task_id}' not found in the functional specification.")

        # -> Get the bidding logic
        bidding_logic = task.get("bidding_logic", None)

        if bidding_logic is None:
            raise ValueError(f"Bidding logic not found for task ID '{task_id}'.")

        elif bidding_logic not in bidding_logics_dict:
            raise ValueError(f"Bidding logic {bidding_logic} not found for task ID '{task_id}'.")

        # -> Return the bidding logic function
        return bidding_logics_dict[bidding_logic]

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
    def asdict(self, *args, **kwargs) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self
