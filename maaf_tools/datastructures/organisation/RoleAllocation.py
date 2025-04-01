
##################################################################################################################

import json
from pprint import pprint
import warnings
#from typing import Self

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem


##################################################################################################################


class RoleAllocation(dict, MaafItem):
    def __init__(self, role_allocation: dict or None = None):
        # If no specification is provided, use the default template.
        if role_allocation is None:
            role_allocation = {
                "group_instances": [],
                "team": []
            }

        elif isinstance(role_allocation, RoleAllocation):
            role_allocation = role_allocation.copy()

        elif isinstance(role_allocation, dict):
            role_allocation = role_allocation.copy()

        elif not isinstance(role_allocation, dict) or not isinstance(role_allocation, RoleAllocation):
            raise ValueError("The role allocation must be a dictionary or RoleAllocation object.")

        if not self.check_role_allocation_definition(role_allocation=role_allocation, verbose=1):
            raise ValueError("The provided role allocation is not valid.")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(role_allocation)

    # ============================================================== Properties

    # ============================================================== Check
    def validate_role_allocation(self, fleet, verbose: int = 1) -> bool:
        """
        Validates the role allocation definition.

        :return: True if the definition is valid, False otherwise.
        """

        valid_wrt_structure = self.check_role_allocation_definition(role_allocation=self, verbose=verbose)

        valid_wrt_fleet = True

        return valid_wrt_structure and valid_wrt_fleet

    @staticmethod
    def check_role_allocation_definition(role_allocation, verbose: int = 1) -> bool:
        """
        Validates the structure of a role allocation definition.

        :param role_allocation: Role allocation definition.
        :param verbose: The verbosity level of the validation output.
        :return: True if the definition is valid, False otherwise.
        """

        errors = []

        # Step 1: Basic structure check
        if not isinstance(role_allocation, dict):
            return ["self is not a dictionary"]

        # Ensure the required top-level keys are present
        required_keys = {"group_instances", "team"}
        if not required_keys.issubset(role_allocation):
            errors.append("Missing required top-level keys: " + str(required_keys - role_allocation.keys()))

        # Step 2: Validate group_instances structure
        group_instances = role_allocation.get("group_instances", [])
        instance_to_parent = {}  # Maps instance -> parent for hierarchy traversal
        instance_names = set()  # Set of all defined group instance names

        for group in group_instances:
            # Check required fields in each group
            if "instance" not in group or "group_type" not in group:
                errors.append(f"Group instance missing 'instance' or 'group_type': {group}")
                continue

            name = group["instance"]

            # Ensure unique instance names
            if name in instance_names:
                errors.append(f"Duplicate group instance name: {name}")
            instance_names.add(name)

            # Track parent relationships if specified
            if "parent" in group:
                instance_to_parent[name] = group["parent"]

        # Step 3: Check that all referenced parents exist
        for child, parent in instance_to_parent.items():
            if parent not in instance_names:
                errors.append(f"Parent group '{parent}' for '{child}' is not a valid group instance")

        # Step 4: Detect cyclic parenting using DFS
        def dfs(node, visited, stack):
            visited.add(node)
            stack.add(node)

            # Recursively follow the parent chain
            parent = instance_to_parent.get(node)
            if parent:
                if parent in stack:
                    # Cycle detected: current path revisits a node
                    return [f"Cyclic parenting detected: {' → '.join(list(stack) + [parent])}"]
                if parent not in visited:
                    # Continue DFS on the parent
                    result = dfs(parent, visited, stack)
                    if result:
                        return result

            # Done with this branch
            stack.remove(node)
            return []

        visited = set()
        for node in instance_names:
            if node not in visited:
                # Start DFS traversal for each node that hasn't been visited
                cycle_error = dfs(node, visited, set())
                if cycle_error:
                    errors.extend(cycle_error)

        # Step 5: Validate team member assignments
        for member in role_allocation.get("team", []):
            # Ensure required keys are present
            if not all(k in member for k in ("id", "name", "class", "assignments")):
                errors.append(f"Missing required keys in team member: {member}")
                continue

            # Check each assignment block
            for assignment in member["assignments"]:
                if "instance" not in assignment or "roles" not in assignment:
                    errors.append(f"Assignment missing 'instance' or 'roles': {assignment}")
                    continue

                instance = assignment["instance"]

                # Ensure assignment references a known group instance
                if instance not in instance_names:
                    errors.append(f"Assignment references unknown group instance: {instance}")

                roles = assignment["roles"]
                if not isinstance(roles, list) or not roles:
                    errors.append(f"Roles must be a non-empty list in assignment: {assignment}")
                    continue

                # Check for duplicate roles in the same assignment
                seen_roles = set()
                duplicates = set()
                for role in roles:
                    if role in seen_roles:
                        duplicates.add(role)
                    seen_roles.add(role)

                if duplicates:
                    errors.append(
                        f"Duplicate role(s) in assignment for agent '{member['id']}' in instance '{instance}': {', '.join(duplicates)}"
                    )

        # Step 6: Return result
        if errors:
            if verbose >= 1:
                print("Role allocation validation failed:")
                for error in errors:
                    print("  -", error)
            return False
        return True

    # ============================================================== Get
    def get_group_affiliations(self, agent_id: str) -> list:
        """
        Get the group affiliations for a specific agent ID.

        :param agent_id: The ID of the agent.
        :return: List of group affiliations.
        """
        group_affiliations = []
        for member in self.get("team", []):
            if member["id"] == agent_id:
                for assignment in member["assignments"]:
                    group_affiliations.append(assignment["instance"])
        return group_affiliations

    def get_roles(self, agent_id: str) -> list:
        """
        Get the roles for a specific agent ID.

        :param agent_id: The ID of the agent.
        :return: List of roles.
        """
        roles = []
        for member in self.get("team", []):
            if member["id"] == agent_id:
                for assignment in member["assignments"]:
                    roles.extend(assignment["roles"])
        return roles

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
