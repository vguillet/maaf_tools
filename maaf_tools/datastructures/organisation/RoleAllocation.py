
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

    def __repr__(self):
        """
        Returns a string representation of the RoleAllocation object.
        """
        return f"RoleAllocation with {len(self.get('group_instances', []))} group instances and {len(self.get('team', []))} team members."

    # ============================================================== Properties

    # ============================================================== Check
    @staticmethod
    def check_role_allocation_definition(role_allocation, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        """
        Validates the structure of a role allocation definition.

        :param role_allocation: Role allocation definition.
        :param stop_at_first_error: If True, stop at the first error found.
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
            if stop_at_first_error:
                return False

        # Step 2: Validate group_instances structure
        group_instances = role_allocation.get("group_instances", [])
        instance_to_parent = {}  # Maps instance -> parent for hierarchy traversal
        instance_names = set()  # Set of all defined group instance names

        for group in group_instances:
            # Check required fields in each group
            if "instance" not in group or "group_type" not in group:
                errors.append(f"Group instance missing 'instance' or 'group_type': {group}")
                if stop_at_first_error:
                    return False

                continue

            name = group["instance"]

            # Ensure unique instance names
            if name in instance_names:
                errors.append(f"Duplicate group instance name: {name}")
                if stop_at_first_error:
                    return False
            instance_names.add(name)

            # Track parent relationships if specified
            if "parent" in group:
                instance_to_parent[name] = group["parent"]

        # Step 3: Check that all referenced parents exist
        for child, parent in instance_to_parent.items():
            if parent not in instance_names:
                errors.append(f"Parent group '{parent}' for '{child}' is not a valid group instance")
                if stop_at_first_error:
                    return False

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
                    if stop_at_first_error:
                        return False

        # Step 5: Validate team member assignments
        for member in role_allocation.get("team", []):
            # Ensure required keys are present
            if not all(k in member for k in ("id", "name", "class", "assignments")):
                errors.append(f"Missing required keys in team member: {member}")
                if stop_at_first_error:
                    return False
                continue

            # Check each assignment block
            for assignment in member["assignments"]:
                if "instance" not in assignment or "roles" not in assignment:
                    errors.append(f"Assignment missing 'instance' or 'roles': {assignment}")
                    if stop_at_first_error:
                        return False
                    continue

                instance = assignment["instance"]

                # Ensure assignment references a known group instance
                if instance not in instance_names:
                    errors.append(f"Assignment references unknown group instance: {instance}")
                    if stop_at_first_error:
                        return False

                roles = assignment["roles"]
                if not isinstance(roles, list) or not roles:
                    errors.append(f"Roles must be a non-empty list in assignment: {assignment}")
                    if stop_at_first_error:
                        return False
                    continue

                # Check for duplicate roles in the same assignment
                seen_roles = set()
                duplicates = set()
                for role in roles:
                    if role in seen_roles:
                        duplicates.add(role)
                    seen_roles.add(role)

                if duplicates:
                    errors.append(f"Duplicate role(s) in assignment for agent '{member['id']}' in instance '{instance}': {', '.join(duplicates)}")
                    if stop_at_first_error:
                        return False

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

    def get_agent_roles(self, agent_id: str, group_id: str = None) -> list:
        """
        Get the roles for a specific agent ID. If a group ID is provided, filter roles by that group.

        :param agent_id: The ID of the agent.
        :return: List of roles.
        """
        roles = []
        for member in self.get("team", []):
            if member["id"] == agent_id:
                for assignment in member["assignments"]:
                    if group_id is None or assignment["instance"] == group_id:
                        if assignment["instance"] == group_id:
                            roles.extend(assignment["roles"])
                        else:
                            # If no group ID is provided, get all roles
                            roles.extend(assignment["roles"])
                break
        return roles

    def get_agent_missions(self, agent_id: str, group_id: str = None) -> list:
        """
        Get the missions for a specific agent ID. If a group ID is provided, filter missions by that group.

        :param agent_id: The ID of the agent.
        :return: List of missions.
        """
        missions = []
        for member in self.get("team", []):
            if member["id"] == agent_id:
                for assignment in member["assignments"]:
                    if group_id is None or assignment["instance"] == group_id:
                        if assignment["instance"] == group_id:
                            missions.extend(assignment["missions"])
                        else:
                            # If no group ID is provided, get all missions
                            missions.extend(assignment["missions"])
        return missions

    def get_agents_in_group(self, group_id: str) -> list:
        """R
        Get the agents in a specific group ID.

        :param group_id: The ID of the group.
        :return: List of agents in the group.
        """
        agents = []
        for member in self.get("team", []):
            for assignment in member["assignments"]:
                if assignment["instance"] == group_id:
                    agents.append(member["id"])
        return agents

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self, other: "RoleAllocation", prioritise_local: bool = True) -> bool:
        """
        Merges the current RoleAllocation with another RoleAllocation.

        The merging process is as follows:
          1. **Group Instances**: For each incoming group instance (matched by its
             "instance" key), if a matching instance is found in the local RoleAllocation:
               - If prioritise_local is False, the local entry is replaced.
               - If prioritise_local is True, the local entry is kept.
             If no match is found, the incoming group instance is appended.

          2. **Team**: For each team member in the incoming RoleAllocation, identified by
             the "id" field:
             - If a team member with the same "id" exists locally, then for each of the incoming
               member’s assignments:
                 - If a local assignment exists (matched by "instance"):
                     • When prioritise_local is False, the entire assignment is replaced by the incoming one.
                     • When prioritise_local is True, the roles from the incoming assignment are added
                       to the local assignment (duplicates are removed).
                 - If no local assignment exists, the incoming assignment is appended.
             - If no team member with the incoming "id" exists, append the entire team member.

        :param other: Another RoleAllocation instance.
        :param prioritise_local: If True, keep local assignments on conflict; if False, override
                                  with the incoming ones.
        :return: True if the merge completes successfully.
        :raises ValueError: If the other object is not a RoleAllocation.
        """
        if not isinstance(other, RoleAllocation):
            raise ValueError("Can only merge with another RoleAllocation instance.")

        # --- Merge Group Instances ---
        local_groups = self.get("group_instances", [])
        incoming_groups = other.get("group_instances", [])

        # Helper: find index in local group_instances with the same "instance" key.
        def find_local_group(incoming_item):
            for idx, item in enumerate(local_groups):
                if item.get("instance") == incoming_item.get("instance"):
                    return idx
            return None

        for incoming in incoming_groups:
            idx = find_local_group(incoming)
            if idx is not None:
                if not prioritise_local:
                    # Replace the local group instance with the incoming one.
                    local_groups[idx] = incoming
                # If prioritising local, keep the local instance (do nothing).
            else:
                # Append new group instance.
                local_groups.append(incoming)
        self["group_instances"] = local_groups

        # --- Merge Team ---
        local_team = self.get("team", [])
        incoming_team = other.get("team", [])

        # Helper: find a team member in local_team by 'id'.
        def find_local_team_member(member_id):
            for member in local_team:
                if member.get("id") == member_id:
                    return member
            return None

        # Helper: merge assignments within a team member.
        def merge_assignments(local_assignments, incoming_assignments):
            # For easier processing, build a lookup of local assignments by "instance".
            assignment_lookup = {a.get("instance"): a for a in local_assignments}
            for inc_a in incoming_assignments:
                instance = inc_a.get("instance")
                if instance in assignment_lookup:
                    if not prioritise_local:
                        # Replace the entire assignment.
                        # (Also update the lookup so that future incoming assignments for the same instance get replaced.)
                        for idx, a in enumerate(local_assignments):
                            if a.get("instance") == instance:
                                local_assignments[idx] = inc_a
                                assignment_lookup[instance] = inc_a
                                break
                    else:
                        # Union the roles: add incoming roles that are not already present.
                        local_roles = set(assignment_lookup[instance].get("roles", []))
                        incoming_roles = set(inc_a.get("roles", []))
                        # Take the union.
                        merged_roles = list(local_roles.union(incoming_roles))
                        # Update the local assignment.
                        assignment_lookup[instance]["roles"] = merged_roles
                        # Also update in the local_assignments list.
                        for idx, a in enumerate(local_assignments):
                            if a.get("instance") == instance:
                                local_assignments[idx]["roles"] = merged_roles
                                break
                else:
                    # No assignment for this instance exists locally; append the incoming one.
                    local_assignments.append(inc_a)
                    assignment_lookup[instance] = inc_a
            return local_assignments

        for inc_member in incoming_team:
            member_id = inc_member.get("id")
            local_member = find_local_team_member(member_id)
            if local_member is None:
                # Append the entire team member if not found locally.
                local_team.append(inc_member)
            else:
                # Team member exists: merge assignments.
                local_assignments = local_member.get("assignments", [])
                incoming_assignments = inc_member.get("assignments", [])
                merged_assignments = merge_assignments(local_assignments, incoming_assignments)
                local_member["assignments"] = merged_assignments
                # Optionally, merge other member-level attributes (e.g., name, class)
                # Here, we assume local member attributes are preferred if prioritise_local is True.
                if not prioritise_local:
                    # Override name and class from incoming member.
                    local_member["name"] = inc_member.get("name", local_member.get("name"))
                    local_member["class"] = inc_member.get("class", local_member.get("class"))
        self["team"] = local_team

        return True

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self, include_local: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the dataclass instance with their current values.

        :param include_local: Whether to include the local field in the dictionary.
        :return: Dictionary representation of the instance.
        """
        return self


if __name__ == "__main__":
    from pprint import pprint

    # Define a local RoleAllocation dictionary.
    local_role_alloc_dict = {
        "group_instances": [
            {"instance": "group1", "group_type": "TypeA"},
            {"instance": "group2", "group_type": "TypeB"}
        ],
        "team": [
            {
                "id": "agent1",
                "name": "Alice",
                "class": "X",
                "assignments": [
                    {"instance": "group1", "roles": ["role1", "role2"]}
                ]
            },
            {
                "id": "agent2",
                "name": "Bob",
                "class": "Y",
                "assignments": [
                    {"instance": "group2", "roles": ["role3"]}
                ]
            }
        ]
    }

    # Define an incoming RoleAllocation dictionary.
    incoming_role_alloc_dict = {
        "group_instances": [
            # Same group1 but with an extra field; this will conflict with the local group1.
            {"instance": "group1", "group_type": "TypeA", "extra": "incoming"},
            # New group instance not present in local.
            {"instance": "group3", "group_type": "TypeC"}
        ],
        "team": [
            {
                "id": "agent1",
                "name": "Alice A.",
                "class": "X-new",
                "assignments": [
                    {"instance": "group1", "roles": ["role2", "role4"]}
                ]
            },
            {
                "id": "agent3",
                "name": "Charlie",
                "class": "Z",
                "assignments": [
                    {"instance": "group3", "roles": ["role5"]}
                ]
            }
        ]
    }

    ###################################################################
    # Test 1: prioritise_local = True (keep local values)
    ###################################################################
    print("== Test 1: Merge with prioritise_local=True ==")
    local_alloc = RoleAllocation(local_role_alloc_dict)
    incoming_alloc = RoleAllocation(incoming_role_alloc_dict)

    # Merge incoming into local while preserving local values.
    result = local_alloc.merge(incoming_alloc, prioritise_local=True)
    assert result is True, "Merge did not return True for prioritise_local=True."

    # --- Check group_instances ---
    group_instances = local_alloc.get("group_instances", [])
    group_lookup = {g["instance"]: g for g in group_instances}
    # Local group1 should be kept (i.e. should NOT have the extra field from incoming).
    assert "group1" in group_lookup, "group1 missing after merge."
    assert "group2" in group_lookup, "group2 missing after merge."
    assert "group3" in group_lookup, "group3 missing after merge."
    assert "extra" not in group_lookup["group1"], (
        "Local group1 should be preserved when prioritise_local=True."
    )

    # --- Check team members ---
    team = local_alloc.get("team", [])
    # Create a lookup by agent id.
    team_lookup = {member["id"]: member for member in team}
    # For agent1, assignments for group1 should be merged as a union.
    # Local assignment for group1 had roles: ["role1", "role2"]
    # Incoming assignment for group1 had roles: ["role2", "role4"]
    # Expected union: {"role1", "role2", "role4"}
    assert "agent1" in team_lookup, "agent1 missing after merge."
    agent1 = team_lookup["agent1"]
    assignment = next((a for a in agent1["assignments"] if a.get("instance") == "group1"), None)
    assert assignment is not None, "agent1 assignment for group1 is missing."
    roles_set = set(assignment.get("roles", []))
    expected_roles = {"role1", "role2", "role4"}
    assert roles_set == expected_roles, f"agent1 roles expected {expected_roles}, got {roles_set}"
    # Agent2 should remain unchanged.
    assert "agent2" in team_lookup, "agent2 missing after merge."
    # Incoming team member agent3 should be appended.
    assert "agent3" in team_lookup, "agent3 not added when prioritise_local=True."
    print("Test 1 passed.")

    ###################################################################
    # Test 2: prioritise_local = False (incoming overrides local)
    ###################################################################
    print("== Test 2: Merge with prioritise_local=False ==")
    # Reinitialize local allocation.
    local_alloc = RoleAllocation(local_role_alloc_dict)
    result = local_alloc.merge(incoming_alloc, prioritise_local=False)
    assert result is True, "Merge did not return True for prioritise_local=False."

    # --- Check group_instances ---
    group_instances = local_alloc.get("group_instances", [])
    group_lookup = {g["instance"]: g for g in group_instances}
    # With prioritise_local False, group1 should be replaced by the incoming version.
    assert "group1" in group_lookup, "group1 missing after merge with prioritise_local=False."
    assert group_lookup["group1"].get("extra") == "incoming", (
        "Incoming group1 should replace local when prioritise_local=False."
    )
    # group2 should remain and group3 should be appended.
    assert "group2" in group_lookup, "group2 missing after merge with prioritise_local=False."
    assert "group3" in group_lookup, "group3 missing after merge with prioritise_local=False."

    # --- Check team members ---
    team = local_alloc.get("team", [])
    team_lookup = {member["id"]: member for member in team}
    # For agent1, with prioritise_local False, the incoming assignment replaces the local one.
    agent1 = team_lookup["agent1"]
    assignment = next((a for a in agent1["assignments"] if a.get("instance") == "group1"), None)
    assert assignment is not None, "agent1 assignment for group1 missing after merge."
    # Expected roles exactly as in incoming: {"role2", "role4"}
    roles_set = set(assignment.get("roles", []))
    expected_roles = {"role2", "role4"}
    assert roles_set == expected_roles, f"agent1 roles expected {expected_roles}, got {roles_set}"
    # Additionally, for agent1, other member attributes should be updated (since local is not preserved).
    assert agent1["name"] == "Alice A.", "agent1 name not updated when prioritise_local=False."
    assert agent1["class"] == "X-new", "agent1 class not updated when prioritise_local=False."
    # Agent2 remains unchanged.
    assert "agent2" in team_lookup, "agent2 missing after merge with prioritise_local=False."
    # Incoming team member agent3 should be added.
    assert "agent3" in team_lookup, "agent3 not added when prioritise_local=False."

    print("Test 2 passed.")
    print("All RoleAllocation merge tests passed.")
