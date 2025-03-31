
##################################################################################################################

import json
from pprint import pprint
import warnings
from typing import Self

##################################################################################################################


class RoleAllocation(dict):
    def __init__(self, role_allocation: Self or dict or None = None):
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

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(role_allocation)

        if not self.validate_role_allocation():
            raise ValueError("The provided role allocation is not valid.")

    # ============================================================== Properties

    # ============================================================== Check
    def validate_role_allocation(self, verbose: int = 1) -> bool:
        """
        Validates the structure of a role allocation definition.

        :param verbose: The verbosity level of the validation output.
        :return: True if the definition is valid, False otherwise.
        """

        errors = []

        # Step 1: Basic structure check
        if not isinstance(self, dict):
            return ["self is not a dictionary"]

        # Ensure the required top-level keys are present
        required_keys = {"group_instances", "team"}
        if not required_keys.issubset(self):
            errors.append("Missing required top-level keys: " + str(required_keys - self.keys()))

        # Step 2: Validate group_instances structure
        group_instances = self.get("group_instances", [])
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
        for member in self.get("team", []):
            # Ensure required keys are present
            if not all(k in member for k in ("agent_id", "name", "class", "assignments")):
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
                        f"Duplicate role(s) in assignment for agent '{member['agent_id']}' in instance '{instance}': {', '.join(duplicates)}"
                    )

                # Check for exactly one priority role (P1–P4)
                priority_roles = [r for r in roles if r in {"P1", "P2", "P3", "P4"}]
                if len(priority_roles) != 1:
                    errors.append(f"Assignment must have exactly one priority role (P1–P4): {assignment}")

        # Step 6: Return result
        if errors:
            if verbose >= 1:
                print("Role allocation validation failed:")
                for error in errors:
                    print("  -", error)
            return False
        return True

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing

