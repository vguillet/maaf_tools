
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
                 allocation_specification: dict or "AllocationSpecification" = None,
                 moise_model: dict or None = None,
                 role_allocation: dict or None = None,
                 ):
        # If no specification is provided, use the default template.
        if allocation_specification is None:
            allocation_specification = {
                "allocation": []
            }

        elif isinstance(allocation_specification, AllocationSpecification) or isinstance(allocation_specification, dict):
            allocation_specification = allocation_specification.copy()

        else:
            raise ValueError("The allocation specification must be a dictionary or AllocationSpecification object.")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(allocation_specification)

        # -> Initialize the moise model and role allocation
        self.__moise_model = None
        self.moise_model = moise_model

        self.__role_allocation = None
        self.role_allocation = role_allocation

    def __repr__(self):
        """Returns a string representation of the AllocationSpecification object."""
        return f"AllocationSpecification with {len(self.role_allocation)} allocations and MOISE model: {self.moise_model_available}"

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
        warnings.warn("This method is not implemented yet. Returning 1 as placeholder") # TODO: Implement string based hierarchy

        return 1

    @moise_model_check
    def get_task_bidding_logic(self, task_type: str) -> callable:
        """
        Get the bidding logic for a specific task ID.

        :param task_type: The type of the task.
        :return: Bidding logic (function or callable).
        """
        # -> Get the task
        task = self.moise_model.functional_specification.get_goal(goal_type=task_type)

        if task is None:
            raise ValueError(f"Task type '{task_type}' not found in the functional specification.")

        # -> Get the bidding logic
        bidding_logic = task.get("bidding_logic", None)

        if bidding_logic is None:
            raise ValueError(f"Bidding logic not found for task type '{task_type}'.")

        elif bidding_logic not in bidding_logics_dict:
            raise ValueError(f"Bidding logic {bidding_logic} not found for task type '{task_type}'.")

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

    # ============================================================== Merge
    def merge(self, allocation_specification: "AllocationSpecification", prioritise_local: bool = True) -> bool:
        """
        Merges the current AllocationSpecification with another one.

        The merging is performed as follows:
          - For the allocation data (the list stored under the "allocation" key):
              • For each incoming allocation entry, if a matching entry exists in the local specification
                (matched by the "id" field when available or by equality), then:
                   - If prioritise_local is True, the local entry is kept.
                   - Otherwise (if prioritise_local is False) the local entry is replaced by the incoming one.
              • If no matching allocation is found, the incoming allocation entry is appended.
          - If both AllocationSpecifications have MOISE models and RoleAllocations defined, their own
            merge methods are called accordingly.

        :param allocation_specification: An AllocationSpecification instance to merge with.
        :param prioritise_local: If True, the local entries are kept on conflict;
                                  if False, incoming entries replace local ones.
        :return: True if the merge completes successfully.
        :raises ValueError: If the allocation_specification object is not an AllocationSpecification.
        """
        if not isinstance(allocation_specification, AllocationSpecification):
            raise ValueError("The object to merge must be an AllocationSpecification instance.")

        # Merge the underlying allocation data.
        # If the allocation specification dictionary does not have an "allocation" key,
        # default to an empty list.
        local_allocations = self.get("allocation", [])
        incoming_allocations = allocation_specification.get("allocation", [])

        # Create a helper lookup for local allocations.
        # We assume that if an allocation item has an "id" field, that uniquely identifies it.
        def find_local(allocation_item):
            if "id" in allocation_item:
                for idx, loc in enumerate(local_allocations):
                    if isinstance(loc, dict) and loc.get("id") == allocation_item["id"]:
                        return idx
            # Fallback: if no "id" exists, try to match by complete equality.
            for idx, loc in enumerate(local_allocations):
                if loc == allocation_item:
                    return idx
            return None

        for incoming in incoming_allocations:
            idx = find_local(incoming)
            if idx is not None:
                # A matching allocation is found.
                if not prioritise_local:
                    # Replace the local allocation with the incoming one.
                    local_allocations[idx] = incoming
                # If prioritising local, we keep the current entry—do nothing.
            else:
                # No matching allocation was found; add the incoming allocation.
                local_allocations.append(incoming)

        # Update the "allocation" key with the merged list.
        self["allocation"] = local_allocations

        # Merge sub-components if possible.
        # Merge the MOISE model subcomponent.
        if self.moise_model_available and allocation_specification.moise_model is not None:
            if hasattr(self.moise_model, "merge"):
                self.moise_model.merge(allocation_specification.moise_model, prioritise_local=prioritise_local)

        # Merge the role allocation subcomponent.
        if self.role_allocation is not None and allocation_specification.role_allocation is not None:
            if hasattr(self.role_allocation, "merge"):
                self.role_allocation.merge(allocation_specification.role_allocation, prioritise_local=prioritise_local)

        return True

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self, *args, **kwargs) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self


if __name__ == "__main__":
    # --- Test data for the allocation list ---

    # Local allocation specification contains two entries.
    local_alloc_spec_dict = {
        "allocation": [
            {"id": "a1", "detail": "Local detail 1"},
            {"id": "a2", "detail": "Local detail 2"}
        ]
    }

    # Incoming allocation specification contains two entries:
    # one having the same id ("a1") as an existing local allocation (but with different detail),
    # and one new entry ("a3").
    incoming_alloc_spec_dict = {
        "allocation": [
            {"id": "a1", "detail": "Incoming detail for a1"},
            {"id": "a3", "detail": "Incoming detail 3"}
        ]
    }

    # Create AllocationSpecification objects.
    local_spec = AllocationSpecification(local_alloc_spec_dict)
    incoming_spec = AllocationSpecification(incoming_alloc_spec_dict)

    #####################################################################
    # Test 1: Merge with prioritise_local = True
    #####################################################################
    print("== Test 1: Merge with prioritise_local=True ==")
    result = local_spec.merge(incoming_spec, prioritise_local=True)
    assert result is True, "Merge did not return True for prioritise_local=True."

    # After merge with prioritise_local=True:
    # - The "a1" entry should keep its local value.
    # - The "a2" entry remains.
    # - The new "a3" entry is appended.
    allocations = local_spec.get("allocation", [])
    alloc_by_id = {entry["id"]: entry for entry in allocations}

    assert alloc_by_id["a1"]["detail"] == "Local detail 1", (
        "Local entry for 'a1' was changed even though prioritise_local=True."
    )
    assert "a2" in alloc_by_id, "Entry 'a2' is missing after merge."
    assert "a3" in alloc_by_id, "New entry 'a3' was not added in the merge (prioritise_local=True)."
    print("Test 1 passed.")

    #####################################################################
    # Test 2: Merge with prioritise_local = False
    #####################################################################
    print("== Test 2: Merge with prioritise_local=False ==")
    # Reinitialize the local allocation specification.
    local_spec = AllocationSpecification(local_alloc_spec_dict)
    result = local_spec.merge(incoming_spec, prioritise_local=False)
    assert result is True, "Merge did not return True for prioritise_local=False."

    # Now, with prioritise_local=False:
    # - The "a1" local entry should be replaced by the incoming entry.
    # - "a2" remains unchanged, and "a3" is appended.
    allocations = local_spec.get("allocation", [])
    alloc_by_id = {entry["id"]: entry for entry in allocations}

    assert alloc_by_id["a1"]["detail"] == "Incoming detail for a1", (
        "Local entry for 'a1' was not replaced when prioritise_local=False."
    )
    assert "a2" in alloc_by_id, "Entry 'a2' is missing after merge with prioritise_local=False."
    assert "a3" in alloc_by_id, "New entry 'a3' was not added in the merge (prioritise_local=False)."
    print("Test 2 passed.")

    print("All AllocationSpecification merge tests passed.")
