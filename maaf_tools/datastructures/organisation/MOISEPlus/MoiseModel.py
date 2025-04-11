
##################################################################################################################

import json
from pprint import pprint
import warnings

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.organisation.MOISEPlus.StructuralSpecification import StructuralSpecification
    from maaf_tools.datastructures.organisation.MOISEPlus.FunctionalSpecification import FunctionalSpecification
    from maaf_tools.datastructures.organisation.MOISEPlus.DeonticSpecification import DeonticSpecification

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.StructuralSpecification import StructuralSpecification
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.FunctionalSpecification import FunctionalSpecification
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.DeonticSpecification import DeonticSpecification

##################################################################################################################


class MoiseModel(MaafItem):
    """
    A class representing a MOISEPlus model for specifying multiagent system organizations.

    The model is divided into three main specifications:
      - Structural Specification: Contains roles, role relations, and groups.
      - Functional Specification: Contains social schemes (including goals, plans, missions, and preferences).
      - Deontic Specification: Contains permissions and obligations.

    This class provides methods to build the model programmatically, as well as parsers for
    loading from and serializing to JSON, following the provided reference format.

    Naming convention:
    - item          - a generic term for any entity in the model. Contains all the information related to the item
    - item_name     - the identifier of the item "type"
    - item_id       - the unique identifier of an instance of the item
    """

    def __init__(self,
                 data: dict or None = None,
                 structural_specification: StructuralSpecification or dict or None = None,
                 functional_specification: FunctionalSpecification or dict or None = None,
                 deontic_specification: DeonticSpecification or dict or None = None,
                 ):
        """
        Initializes the MOISEPlus model.
        :param data: A dictionary containing the model data. If provided, it will override the other parameters.
        :param structural_specification: Structural specification of the model.
        :param functional_specification: Functional specification of the model.
        :param deontic_specification: Deontic specification of the model.
        """

        if data is not None:
            # TODO: Add check for data type
            if not isinstance(data, dict):
                raise ValueError("The data must be a dictionary.")
            structural_specification = data.get("structural_specification", None)
            functional_specification = data.get("functional_specification", None)
            deontic_specification = data.get("deontic_specification", None)

        # -> Initialize specifications
        self.structural_specification = StructuralSpecification(structural_specification)
        self.functional_specification = FunctionalSpecification(functional_specification)
        self.deontic_specification = DeonticSpecification(deontic_specification)

        # -> Setup cross-references between specifications
        self.__set_cross_references()

    def __set_cross_references(self):
        """
        Sets the cross-references between the different specifications.
        """
        # Structural Specification
        self.structural_specification.functional_specification = self.functional_specification
        self.structural_specification.deontic_specification = self.deontic_specification

        # Functional Specification
        self.functional_specification.structural_specification = self.structural_specification
        self.functional_specification.deontic_specification = self.deontic_specification

        # Deontic Specification
        self.deontic_specification.structural_specification = self.structural_specification
        self.deontic_specification.functional_specification = self.functional_specification

    def __repr__(self):
        """Returns a string representation of the MOISEPlus model."""

        string = f"MOISEPlus Model with {len(self.structural_specification['roles'])} roles, " \
               f"{len(self.structural_specification['role_relations'])} role relations, " \
               f"and {len(self.structural_specification['groups'])} groups"

        return string

    def __str__(self):
        """Returns a string representation of the MOISEPlus model."""
        return self.__repr__()

    # ============================================================== Properties

    # ============================================================== Check
    def check_model_definition(self, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        """
        Checks the validity of the MOISEPlus model. Check is performed on the following aspects:
          - Structural Specification: Checks roles, role relations, and groups.
          - Functional Specification: Checks social schemes (goals, plans, missions, and preferences).
          - Deontic Specification: Checks permissions and obligations.

        :param stop_at_first_error: If True, stops checking at the first error found.
        :param verbose: If 0, no output; if 1, prints errors; if 2, prints warnings.
        :return: True if the model is valid, False otherwise.
        """

        # Check each specification and collect errors.
        structural_spec_valid = self.structural_specification.check_specification_definition(
            structural_specification=self.structural_specification,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        functional_spec_valid = self.functional_specification.check_specification_definition(
            functional_specification=self.functional_specification,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )
        deontic_spec_valid = self.deontic_specification.check_specification_definition(
            deontic_specification=self.deontic_specification,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        # Combine errors from all specifications.
        return structural_spec_valid and functional_spec_valid and deontic_spec_valid

    @staticmethod
    def __has_required_skills(agent_skillset: list[str], required_skills: list[str]) -> bool:
        return set(required_skills).issubset(set(agent_skillset))

    def check_role_goal_compatibility(self, role_name: str, goal_name: str) -> bool:
        """
        Check if the role is responsible for the goal
        :param role_name: The name of the role.
        :param goal_name: The name of the goal.
        :return: True if the role is responsible for the goal, False otherwise.
        """

        # -> Get the goals associated with the role
        role_goals = self.get_goals_associated_with_role(role_name=role_name)

        # -> Check if the goal is in the role's goals
        return goal_name in role_goals

    def check_agent_skillset_goal_compatibility(self, agent_skillset: list[str], goal_name: str) -> bool:
        """
        Checks if the agent's skillset is compatible with the goal's skill requirements.

        :param agent_skillset: The skillset of the agent.
        :param goal_name: The name of the goal.
        :return: True if compatible, False otherwise.
        """
        # -> Get the goal's skill requirements
        goal_skill_requirements = self.get_goal_skill_requirements(goal_name=goal_name)

        # -> Check if the agent's skillset meets the goal's skill requirements
        return self.__has_required_skills(agent_skillset=agent_skillset, required_skills=goal_skill_requirements)

    def check_agent_mission_compatibility(self, agent_skillset: list[str], mission_name: str) -> bool:
        """
        Checks if the agent's skillset is compatible with all of the mission's goals.

        :param agent_skillset: The skillset of the agent.
        :param mission_name: The name of the mission.

        :return: True if compatible, False otherwise.
        """

        # -> Get the mission's skill requirements
        mission_skill_requirements = self.get_mission_skill_requirements(mission_name=mission_name)

        # -> Check if the agent's skillset meets the mission's skill requirements
        for goal_skill_requirements in mission_skill_requirements:
            if not self.__has_required_skills(agent_skillset=agent_skillset, required_skills=goal_skill_requirements):
                return False
        return True

    def check_agent_role_compatibility(self, agent_skillset: list[str], role_name: str) -> bool:
        """
        Check whether an agent skillset is compatible with a given role.
        Compatibility means the agent's skillset includes all required skills for the role and all its ancestors.

        The method looks up the role in the structural specification (self.structural_specification["roles"])
        and verifies that all skills listed in the role's "skill_requirements" (if any) are present in the
        agent_skillset. If the role inherits from another role, the parent's skill requirements are also checked recursively.

        :param agent_skillset: (list[str]): A list of skills the agent possesses
        :param role_name: (str) The role name to check compatibility for

        :return: (bool) True if the agent's skillset satisfies the role's (and its ancestors') skill requirements; otherwise False.
        """

        # Retrieve the role definition from the model.
        role_def = next((r for r in self.structural_specification["roles"] if r["name"] == role_name), None)

        if role_def is None:
            raise ValueError(f"Role '{role_name}' is not defined in the model.")

        # Get the role's required skills (treating None as an empty list).
        required_skills = self.get_role_skill_requirements(role_name=role_name)

        # Check if all required skills for the role are in the agent's killset.
        if not self.__has_required_skills(agent_skillset=agent_skillset, required_skills=required_skills):
            return False

        # If the role inherits from a parent role, recursively check parent's compatibility.
        parent_role = role_def.get("inherits")
        if parent_role:
            if not self.check_agent_role_compatibility(agent_skillset, parent_role):
                return False

        return True

    def check_missing_roles_in_group(self, role_allocation: dict, group_instance: str, verbose: int = 1) -> dict:
        """
        Checks missing role assignments for a specific group instance.

        This method aggregates the role counts from the specified group instance and all its descendant
        instances (as defined by the "parent" field in role_allocation["group_instances"]) and compares the
        totals with the minimal requirements (role_cardinality) defined in the model for that group type.

        Parameters:
          role_allocation (dict): The team allocation specification, which should include "group_instances" and "team".
          group_instance (str): The specific group instance identifier to check.
          verbose (int): Verbosity level (>=1 prints messages).

        Returns:
          dict: A dictionary mapping role names to the number of additional assignments needed to meet the minimal
                requirements. An empty dictionary indicates that the instance meets (or exceeds) its requirements.
        """
        # Ensure group_instances is provided.
        if "group_instances" not in role_allocation:
            if verbose >= 1:
                warnings.warn("No group_instances defined in the allocation.")
            return {}

        group_instances = role_allocation["group_instances"]

        # Build instance mapping and parent-to-children relationships.
        instance_map = {}
        parent_to_children = {}
        for gi in group_instances:
            inst = gi["instance"]
            gtype = gi["group_type"]
            parent = gi.get("parent")
            instance_map[inst] = {"group_type": gtype, "parent": parent}
            if parent:
                parent_to_children.setdefault(parent, []).append(inst)

        # Check if the specified group_instance is defined.
        if group_instance not in instance_map:
            if verbose >= 1:
                warnings.warn(f"Group instance '{group_instance}' is not defined in group_instances.")
            return {}

        # Retrieve the group type for the specified instance.
        group_type = instance_map[group_instance]["group_type"]

        # Build direct role counts from team assignments.
        direct_counts = {}  # instance_id -> { role: count }
        for agent in role_allocation.get("team", []):
            for assignment in agent.get("assignments", []):
                inst = assignment.get("instance")
                roles = assignment.get("roles", [])
                if inst in instance_map:
                    direct_counts.setdefault(inst, {})
                    for role in roles:
                        direct_counts[inst][role] = direct_counts[inst].get(role, 0) + 1

        # Helper: recursively aggregate role counts from an instance and all its descendants.
        def get_aggregated_counts(inst_id: str) -> dict:
            counts = dict(direct_counts.get(inst_id, {}))
            for child in parent_to_children.get(inst_id, []):
                child_counts = get_aggregated_counts(child)
                for r, cnt in child_counts.items():
                    counts[r] = counts.get(r, 0) + cnt

            return counts

        aggregated = get_aggregated_counts(group_instance)

        # Retrieve the model definition for this group type.
        group_model = next((g for g in self.structural_specification["groups"] if g["name"] == group_type), None)
        if group_model is None:
            if verbose >= 1:
                warnings.warn(f"Group type '{group_type}' is not defined in the model.")
            return {}

        role_cardinality = group_model.get("role_cardinality", {})

        # Determine missing roles by comparing aggregated counts with minimal requirements.
        missing = {}
        for role, card in role_cardinality.items():
            min_required = card.get("min", 0)
            current = aggregated.get(role, 0)
            if current < min_required:
                missing[role] = min_required - current

        if verbose >= 1:
            print(f"In group instance '{group_instance}' (type '{group_type}'): missing roles: {missing}")

        return missing

    def check_role_allocation(self, role_allocation: dict, fleet, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        # -> Check role allocation against agent skillsets
        valid_wrt_agents_skillsets = self.check_role_allocation_against_skillsets(
            role_allocation=role_allocation,
            fleet=fleet,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        # -> Check role allocation against model
        valid_wrt_structural_model = self.check_role_allocation_against_model(
            role_allocation=role_allocation,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        return valid_wrt_agents_skillsets and valid_wrt_structural_model

    def check_role_allocation_against_skillsets(self, role_allocation: dict, fleet, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        """
        Verify that a given role allocation respects the agents' skillsets.

        Checks include:
          - Every agent referenced in team assignments is defined in the fleet.
          - Every role in the assignments is compatible with the agent's skillset.

        :param role_allocation: The team allocation specification including "group_instances" and "team".
        :param fleet: The fleet specification.
        :param verbose: Verbosity level (>=1 prints messages).
        :param stop_at_first_error: If True, return immediately on the first error found.
        :return: True if the allocation satisfies all constraints, False otherwise.
        """
        errors = []

        # ... for every agent in the fleet
        for agent in fleet:
            # Get the agent's assigned roles
            assigned_roles = []

            for agent_assignment in role_allocation["team"]:
                if agent_assignment["id"] == agent.id:
                    for assignment in agent_assignment["assignments"]:
                        assigned_roles += assignment["roles"]

            assigned_roles = set(assigned_roles)

            # -> Check every assigned role skill requirements against the agent's skillset
            for role in assigned_roles:
                if not self.check_agent_role_compatibility(agent.skillset, role):
                    err = f"Agent '{agent.id}' is not compatible with role '{role}'."
                    errors.append(err)
                    if stop_at_first_error:
                        return False

        # --- Final reporting ---
        if errors:
            if verbose >= 1:
                print("Errors found in role allocation:")
                for err in errors:
                    print("  -", err)
            return False
        else:
            return True

    def check_role_allocation_against_model(self, role_allocation: dict, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        """
        Verify that a given role allocation respects the structural specification.
        This version accumulates all error messages (if stop_at_first_error is False) or stops at the first error found
        (if stop_at_first_error is True), printing all errors if verbose is enabled.

        Checks include:
          - Every group instance referenced in team assignments is defined in role_allocation["group_instances"].
          - If a "team" field is provided in an assignment, it must match the group_type of the referenced instance.
          - All roles in the assignments must be defined in the model.
          - Group composition constraints:
                • For each instance (of a group type with role_cardinality), the aggregated role counts
                  (from that instance and its descendant instances) must meet the minimal requirements defined
                  in self["groups"].
                • For each parent group instance that allows subgroups (via a "subgroups" mapping), the number
                  of direct child instances for each allowed subgroup type must be within the specified cardinality.
          - Role relation constraints:
                • For each role relation defined in the model, if both roles appear in the assignments, there must be
                  at least one pair of assignments that satisfy the relation's scope:
                      - "intra": Assignments are in the same instance or in instances sharing the same parent.
                      - "inter": Assignments are in different instances and not siblings.
                      - "omni": No grouping constraint.

        :param role_allocation: The team allocation specification including "group_instances" and "team".
        :param verbose: Verbosity level (>=1 prints messages).
        :param stop_at_first_error: If True, return immediately on the first error found.
        :return: True if the allocation satisfies all constraints, False otherwise.
        """
        errors = []

        # --- Step 0: Process group_instances ---
        if "group_instances" not in role_allocation:
            err = "No group_instances defined in the allocation."
            errors.append(err)
            if stop_at_first_error:
                return False

        group_instances = role_allocation.get("group_instances", [])
        instance_map = {}  # instance_id -> {"group_type": ..., "parent": ...}
        parent_to_children = {}  # parent instance_id -> list of child instance_ids
        for gi in group_instances:
            inst = gi["instance"]
            group_type = gi["group_type"]
            parent = gi.get("parent")  # May be None for top-level groups.
            instance_map[inst] = {"group_type": group_type, "parent": parent}
            if parent:
                parent_to_children.setdefault(parent, []).append(inst)

        # --- Step 1: Process team assignments ---
        roles_defined = {role["name"] for role in self.structural_specification.roles}
        assignments_list = []  # list of tuples: (agent_id, instance, role)

        for agent in role_allocation.get("team", []):
            agent_id = agent.get("id", "unknown")
            for assignment in agent.get("assignments", []):
                inst = assignment.get("instance")
                if inst not in instance_map:
                    err = f"Instance '{inst}' (agent {agent_id}) is not defined in group_instances."
                    errors.append(err)
                    # if verbose >= 1:
                    #     print(err)
                    if stop_at_first_error:
                        return False
                    continue  # Skip processing this assignment

                expected_group = instance_map[inst]["group_type"]
                if "team" in assignment and assignment["team"] != expected_group:
                    err = (f"Assignment for agent {agent_id} specifies team '{assignment['team']}' "
                           f"but instance '{inst}' is of type '{expected_group}'.")
                    errors.append(err)
                    if stop_at_first_error:
                        return False

                roles = assignment.get("roles", [])
                for role in roles:
                    if role not in roles_defined:
                        err = f"Role '{role}' is not defined in the model (agent {agent_id}, instance {inst})."
                        errors.append(err)
                        if stop_at_first_error:
                            return False
                    else:
                        assignments_list.append((agent_id, inst, role))

        # --- Step 2: Validate group composition constraints via check_missing_roles_in_group ---
        for group in self.structural_specification["groups"]:
            if "role_cardinality" in group and group["role_cardinality"]:
                group_type = group["name"]
                # For each instance that has this group_type:
                for inst, info in instance_map.items():
                    if info["group_type"] == group_type:
                        missing = self.check_missing_roles_in_group(role_allocation, inst, verbose=0)
                        if missing:
                            err = f"Instance '{inst}' of group type '{group_type}' is missing roles: {missing}"
                            errors.append(err)
                            if stop_at_first_error:
                                return False

        # --- Step 3: Validate subgroup cardinalities ---
        for group in self.structural_specification["groups"]:
            if "subgroups" in group and group["subgroups"]:
                group_type = group["name"]
                for inst, info in instance_map.items():
                    if info["group_type"] == group_type:
                        children = parent_to_children.get(inst, [])
                        subgroup_counts = {}
                        for child_inst in children:
                            child_type = instance_map[child_inst]["group_type"]
                            subgroup_counts[child_type] = subgroup_counts.get(child_type, 0) + 1
                        for subgroup_type, cardinality in group["subgroups"].items():
                            count = subgroup_counts.get(subgroup_type, 0)
                            min_required = cardinality.get("min", 0)
                            max_allowed = cardinality.get("max", None)
                            if count < min_required:
                                err = (f"In parent instance '{inst}' of group type '{group_type}', "
                                       f"subgroup '{subgroup_type}' count {count} is below the minimum required {min_required}.")
                                errors.append(err)
                                if stop_at_first_error:
                                    return False
                            if max_allowed is not None and count > max_allowed:
                                err = (f"In parent instance '{inst}' of group type '{group_type}', "
                                       f"subgroup '{subgroup_type}' count {count} exceeds the maximum allowed {max_allowed}.")
                                errors.append(err)
                                if stop_at_first_error:
                                    return False

        # --- Step 4: Validate role relation constraints ---
        for relation in self.structural_specification["role_relations"]:
            src_role = relation["source"]
            dst_role = relation["destination"]
            scope = relation["scope"]

            src_assignments = [(agent_id, inst) for agent_id, inst, role in assignments_list if role == src_role]
            dst_assignments = [(agent_id, inst) for agent_id, inst, role in assignments_list if role == dst_role]

            # If one of the roles isn't assigned anywhere, skip this relation.
            if not src_assignments or not dst_assignments:
                continue

            relation_satisfied = False
            for src_agent, src_inst in src_assignments:
                for dst_agent, dst_inst in dst_assignments:
                    if scope == "intra":
                        if src_inst == dst_inst:
                            relation_satisfied = True
                            break
                        parent_src = instance_map[src_inst].get("parent")
                        parent_dst = instance_map[dst_inst].get("parent")
                        if parent_src and parent_src == parent_dst:
                            relation_satisfied = True
                            break
                    elif scope == "inter":
                        if src_inst != dst_inst:
                            parent_src = instance_map[src_inst].get("parent")
                            parent_dst = instance_map[dst_inst].get("parent")
                            if not parent_src or not parent_dst or parent_src != parent_dst:
                                relation_satisfied = True
                                break
                    elif scope == "omni":
                        relation_satisfied = True
                        break
                if relation_satisfied:
                    break

            if not relation_satisfied:
                err = f"Role relation constraint not satisfied: {src_role} -> {dst_role} with scope '{scope}'."
                errors.append(err)
                if stop_at_first_error:
                    return False

        # --- Final reporting ---
        if errors:
            if verbose >= 1:
                print("Errors found in role allocation:")
                for err in errors:
                    print("  -", err)
            return False
        else:
            return True

    # ============================================================== Get
    # ----- Skill requirements
    def get_goal_skill_requirements(self, goal_name: str, verbose: int = 1) -> list[str] or None:
        """
        Returns the skill requirements for a given goal.

        :param goal_name: The name of the goal.
        :param verbose: Verbosity level (0: no output, 1: print warnings).
        :return: A list of skill requirements for the specified goal, or None if goal is not found.
        """
        goal = self.functional_specification.get_goal(goal_name)
        if goal is not None:
            return goal.get("skill_requirements", [])
        else:
            if verbose > 0: warnings.warn(f"Goal with name '{goal_name}' not found in the functional specification.")
            return None

    def get_mission_skill_requirements(self, mission_name: str, verbose: int = 1) -> list[str] or None:
        """
        Returns the skill requirements for a given mission.

        :param mission_name: The name of the mission.
        :param verbose: Verbosity level (0: no output, 1: print warnings).
        :return: A list of skill requirements for the specified mission, or None if mission is not found.
        """
        mission = self.functional_specification.get_mission(mission_name)
        if mission is not None:
            return [self.get_goal_skill_requirements(goal) for goal in mission.get("goals", [])]
        else:
            if verbose > 0: warnings.warn(f"Mission with name '{mission_name}' not found in the functional specification.")
            return None

    def get_role_skill_requirements(self, role_name: str) -> list[str] or None:
        """
        Returns the skill requirements for a given role_name. The skill requirements are determined based on the
        goal requirements associated with the missions the role_name is responsible for (permissions and obligations).

        :param role_name: The role_name for which to retrieve skill requirements.
        :return : A list of skill requirements for the specified role_name.
        """

        if self.functional_specification is None:
            warnings.warn("Functional specification is not set.")
            return None

        # -> Get goals associated with the role
        goals = self.get_goals_associated_with_role(role_name=role_name)

        # -> Get all skills associated with the goals
        skills = []
        for goal in goals:
            skills.extend(self.get_goal_skill_requirements(goal))

        skills = set(skills)

        return list(skills)

    def get_roles_compatible_with_skillset(self, skillset: list[str]) -> list[str]:
        """
        Returns a list of concrete role names from the structural specification that are compatible
        with a given skillset. A role is considered compatible if:
          - It is concrete (i.e. not abstract).
          - The skillset includes all required skills for that role and all its ancestors.

        :param skillset: A list of skills the agent possesses.
        :return: A list of compatible role names.
        """

        compatible_roles = []
        # Iterate over all roles in the model.
        for role in self.structural_specification["roles"]:
            role_name = role["name"]
            # Use the check_agent_role_compatibility method to determine if the agent is compatible.
            if self.check_agent_role_compatibility(skillset, role_name):
                compatible_roles.append(role_name)
        return compatible_roles

    # ----- Mappings query
    @staticmethod
    def __post_process_mappings_query(result: list[dict] or list, names_only: bool = True):
        seen = set()
        deduplicated_items = []
        for item in result:
            if isinstance(item, dict):
                name = item.get('name')
            else:
                name = item
            if name not in seen:
                seen.add(name)
                deduplicated_items.append(item)
        if names_only:
            processed_result = []
            for item in deduplicated_items:
                if isinstance(item, dict):
                    processed_result.append(item['name'])
                else:
                    processed_result.append(item)
            return processed_result
        else:
            return deduplicated_items

    # Goals to [...]
    def get_goals_associated_with_role(self, role_name: str, names_only: bool = True) -> list:
        """
        Gets the goals associated with a specific role_name.

        :param role_name: The role_name to check.
        :param names_only: If True, return only the names of the goals.
        :return: A list of goals associated with the specified role_name.
        """

        # -> Gather all missions associated with the role_name
        missions = self.get_missions_associated_with_role(role_name=role_name)

        # -> Get all goals associated with the missions
        goals = []
        for mission_name in missions:
            goals.extend(self.get_goals_associated_with_mission(mission_name=mission_name))

        return self.__post_process_mappings_query(
            result = goals,
            names_only = names_only
        )

    # ============================================================== Set

    def get_goals_associated_with_mission(self, mission_name: str, names_only: bool = True):
        """
        Returns a list of goals associated with a given mission name.

        :param mission_name: The name of the mission.
        :param names_only: If True, return only the names of the goals.
        :return: A list of goal dictionaries associated with the specified mission name.
        """
        goals = []
        for scheme in self.functional_specification["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["name"] == mission_name:
                    goals.extend(mission.get("goals", []))

        return self.__post_process_mappings_query(
            result = [self.functional_specification.get_goal(goal) for goal in goals],
            names_only = names_only
        )

    # Missions to [...]
    def get_missions_associated_with_role(self, role_name: str, names_only: bool = True) -> list:
        """
        Gets the missions associated with a specific role_name.

        :param role_name: The role_name to check.
        :param names_only: If True, return only the names of the missions.
        :return: A list of missions associated with the specified role_name.
        """

        missions = []
        missions.extend(self.deontic_specification.get_missions_permitted_to_role(role_name=role_name))
        missions.extend(self.deontic_specification.get_missions_obligated_to_role(role_name=role_name))

        return self.__post_process_mappings_query(
            result = missions,
            names_only = names_only
        )

    def get_missions_associated_with_goal(self, goal_name: str, names_only: bool = True):
        """
        Returns a list of missions associated with a given goal name.

        :param goal_name: The name of the goal.
        :param names_only: If True, return only the names of the missions.
        :return: A list of mission dictionaries associated with the specified goal name.
        """
        missions = []
        for scheme in self.functional_specification["social_schemes"]:
            for mission in scheme["missions"]:
                if goal_name in mission.get("goals", []):
                    missions.append(mission)

        return self.__post_process_mappings_query(
            result = missions,
            names_only = names_only
        )

    # Roles to [...]
    def get_roles_associated_with_mission(self, mission_name: str, names_only: bool = True) -> list:
        """
        Gets the roles associated with a specific mission type.

        :param mission_name: The mission type to check.
        :param names_only: If True, return only the names of the roles.
        :return: A list of roles associated with the specified mission type.
        """

        roles = []

        # Check permissions
        for permission in self.deontic_specification["permissions"]:
            if permission["mission_name"] == mission_name:
                roles.append(permission["role_name"])

        # Check obligations
        for obligation in self.deontic_specification["obligations"]:
            if obligation["mission_name"] == mission_name:
                roles.append(obligation["role_name"])

        return self.__post_process_mappings_query(
            result = roles,
            names_only = names_only
        )

    def get_roles_associated_with_goal(self, goal_name: str, names_only: bool = True) -> list:
        """
        Gets the roles associated with a specific goal.

        :param goal_name: The goal name to check.
        :param names_only: If True, return only the names of the roles.
        :return: A list of roles associated with the specified goal.
        """

        # -> Get missions associated with the goal
        missions = self.get_missions_associated_with_goal(goal_name=goal_name)

        # -> Get roles associated with the missions
        roles = []

        for mission in missions:
            roles.extend(self.get_roles_associated_with_mission(mission_name=mission["name"]))

        return self.__post_process_mappings_query(
            result = roles,
            names_only = names_only
        )

    # ============================================================== Merge

    def merge(self, moise_model: "MoiseModel", prioritise_local: bool = True) -> bool:
        """
        Merges the current MOISEPlus model with anmoise_model one.
        Merging is performed separately on each of the three specifications:
          - StructuralSpecification,
          - FunctionalSpecification, and
          - DeonticSpecification.

        For each sub-specification, if prioritise_local is True the local values take precedence;
        if False, the incoming model’s values override local values.

        Cross-references between specifications are re-established after merging.

        :param moise_model: A MoiseModel instance to merge into the current model.
        :param prioritise_local: If True, keep local entries on conflict; if False, override with incoming entries.
        :return: True if the merged model is valid.
        :raises ValueError: If the incoming object is not a MoiseModel.
        """
        if not isinstance(moise_model, MoiseModel):
            raise ValueError("The model to merge must be a MoiseModel object.")

        # Merge each specification using their own merge methods.
        self.structural_specification.merge(moise_model.structural_specification, prioritise_local=prioritise_local)
        self.functional_specification.merge(moise_model.functional_specification, prioritise_local=prioritise_local)
        self.deontic_specification.merge(moise_model.deontic_specification, prioritise_local=prioritise_local)

        # Re-establish the cross-references among the specifications.
        self.structural_specification.functional_specification = self.functional_specification
        self.structural_specification.deontic_specification = self.deontic_specification

        self.functional_specification.structural_specification = self.structural_specification
        self.functional_specification.deontic_specification = self.deontic_specification

        self.deontic_specification.structural_specification = self.structural_specification
        self.deontic_specification.functional_specification = self.functional_specification

        # Validate the entire merged model
        if not self.check_model_definition(stop_at_first_error=True, verbose=0):
            return False

        return True

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """Returns the MOISEPlus model as a dictionary."""
        return {
            "structural_specification": self.structural_specification,
            "functional_specification": self.functional_specification,
            "deontic_specification": self.deontic_specification
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> None:
        """
        Converts a dictionary to a MOISEPlus model.

        :param item_dict: The dictionary representation of the MOISEPlus model.
        """

        # -> Create a new instance of the class
        instance = cls(
            structural_specification=item_dict.get("structural_specification"),
            functional_specification=item_dict.get("functional_specification"),
            deontic_specification=item_dict.get("deontic_specification")
            )

        return instance

    def plot(self):
        """
        Plots an interactive view of the MOISEPlus model showing:
          - The roles hierarchy at the top, drawn as a top-down tree with reversed arrows
            (arrows from child to parent, empty arrowheads).
          - The groups hierarchy at the bottom, drawn as a bottom-up tree
            (arrows from parent to child, empty arrowheads).
        Nodes are represented as rounded boxes (abstract roles in light gray, concrete roles in white).

        Note: This method lays out roles and groups in the same plot. Horizontal spacing
        of roles has been increased to prevent overlapping.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # -------------------- Roles Hierarchy (Top) --------------------
        G_roles = nx.DiGraph()
        for role in self.structural_specification["roles"]:
            role_name = role["name"]
            G_roles.add_node(role_name, abstract=role.get("abstract", False))
            if "inherits" in role:
                if role["inherits"] is not None:
                    parent_role = role["inherits"]
                    G_roles.add_node(parent_role, abstract=False)
                    G_roles.add_edge(parent_role, role_name)

        # Try to use Graphviz for roles layout; if not available, use a custom top-down layout.
        try:
            pos_roles = nx.nx_agraph.graphviz_layout(G_roles, prog="dot")
        except Exception:
            def hierarchy_pos(G, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5):
                def _hierarchy_pos(G, root, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None):
                    if pos is None:
                        pos = {root: (xcenter, vert_loc)}
                    else:
                        pos[root] = (xcenter, vert_loc)
                    children = list(G.successors(root))
                    if children:
                        dx = width / len(children)
                        next_x = xcenter - width / 2 + dx / 2
                        for child in children:
                            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                                 vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos)
                            next_x += dx
                    return pos

                roots = [n for n, d in G.in_degree() if d == 0]
                pos = {}
                if len(roots) == 1:
                    pos = _hierarchy_pos(G, roots[0], width, vert_gap, vert_loc, xcenter)
                else:
                    dx = width / len(roots)
                    for i, root in enumerate(roots):
                        pos.update(_hierarchy_pos(G, root, width=dx, vert_gap=vert_gap, vert_loc=vert_loc,
                                                  xcenter=xcenter - width / 2 + dx / 2 + i * dx))
                return pos

            pos_roles = hierarchy_pos(G_roles)

        # Increase horizontal spacing by applying a separate horizontal scale factor.
        horizontal_scale_roles = 4.0  # Increased horizontal scale factor
        vertical_scale_roles = 1.5  # Vertical scale factor remains
        pos_roles = {node: (x * horizontal_scale_roles, y * vertical_scale_roles) for node, (x, y) in
                     pos_roles.items()}
        roles_offset = 3.0  # Shift roles upward so they appear at the top.
        pos_roles = {node: (x, y + roles_offset) for node, (x, y) in pos_roles.items()}

        # -------------------- Groups Hierarchy (Bottom) --------------------
        G_groups = nx.DiGraph()
        for group in self.structural_specification["groups"]:
            group_name = group["name"]
            G_groups.add_node(group_name)
            if "subgroups" in group:
                for subgroup in group["subgroups"]:
                    G_groups.add_node(subgroup)
                    G_groups.add_edge(group_name, subgroup)

        def hierarchy_pos_groups(G, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5):
            def _hierarchy_pos_groups(G, root, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None):
                if pos is None:
                    pos = {root: (xcenter, vert_loc)}
                else:
                    pos[root] = (xcenter, vert_loc)
                children = list(G.successors(root))
                if children:
                    dx = width / len(children)
                    next_x = xcenter - width / 2 + dx / 2
                    for child in children:
                        pos = _hierarchy_pos_groups(G, child, width=dx, vert_gap=vert_gap,
                                                    vert_loc=vert_loc + vert_gap, xcenter=next_x, pos=pos)
                        next_x += dx
                return pos

            roots = [n for n, d in G.in_degree() if d == 0]
            pos = {}
            if len(roots) == 1:
                pos = _hierarchy_pos_groups(G, roots[0], width, vert_gap, vert_loc, xcenter)
            else:
                dx = width / len(roots)
                for i, root in enumerate(roots):
                    pos.update(_hierarchy_pos_groups(G, root, width=dx, vert_gap=vert_gap,
                                                     vert_loc=vert_loc,
                                                     xcenter=xcenter - width / 2 + dx / 2 + i * dx))
            return pos

        pos_groups = hierarchy_pos_groups(G_groups, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5)
        horizontal_scale_groups = 1.5
        vertical_scale_groups = 1.5
        pos_groups = {node: (x * horizontal_scale_groups, y * vertical_scale_groups) for node, (x, y) in
                      pos_groups.items()}

        # -------------------- Plotting Both Hierarchies --------------------
        fig, ax = plt.subplots()

        # Plot roles (top) with reversed arrows (child -> parent, empty arrowhead)
        for (parent, child) in G_roles.edges():
            ax.annotate("",
                        xy=pos_roles[parent], xycoords='data',
                        xytext=pos_roles[child], textcoords='data',
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='black',
                            shrinkA=5, shrinkB=5,
                            connectionstyle="arc3,rad=0",
                            fill=False,
                            mutation_scale=20
                        ))
        # Draw roles nodes as rounded boxes.
        for node, (x, y) in pos_roles.items():
            color = "lightgray" if G_roles.nodes[node].get("abstract", False) else "white"
            ax.text(x, y, node, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="black", lw=1))

        # Plot groups (bottom) with normal arrows (parent -> child, empty arrowhead)
        for (parent, child) in G_groups.edges():
            ax.annotate("",
                        xy=pos_groups[child], xycoords='data',
                        xytext=pos_groups[parent], textcoords='data',
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='black',
                            shrinkA=5, shrinkB=5,
                            connectionstyle="arc3,rad=0",
                            fill=False,
                            mutation_scale=20
                        ))
        # Draw groups nodes as rounded boxes.
        for node, (x, y) in pos_groups.items():
            ax.text(x, y, node, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))

        # -------------------- (Optional) Connect Groups and Roles --------------------
        # If there are relationships connecting groups to roles, draw them here.
        # Example placeholder:
        # for connection in self.structural_specification.get("group_role_connections", []):
        #     role = connection["role"]
        #     group = connection["group"]
        #     if role in pos_roles and group in pos_groups:
        #         ax.annotate("",
        #                     xy=pos_roles[role], xycoords='data',
        #                     xytext=pos_groups[group], textcoords='data',
        #                     arrowprops=dict(arrowstyle='-|>', color='red',
        #                                     shrinkA=5, shrinkB=5,
        #                                     connectionstyle="arc3,rad=0", fill=False, mutation_scale=20))

        # -------------------- Adjust Axis Limits and Display --------------------
        all_x = [x for pos in [pos_roles, pos_groups] for x, y in pos.values()]
        all_y = [y for pos in [pos_roles, pos_groups] for x, y in pos.values()]
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
        ax.relim()
        ax.autoscale_view()

        plt.title("Roles Hierarchy (Top) and Groups Hierarchy (Bottom)")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    """
    # Create a new MOISEPlus model
    model = MoiseModel()

    # --- Structural Specification ---
    # ----- Add roles
    model.structural_specification.add_role("Agent", abstract=True)

    # Coordination
    model.structural_specification.add_role(name="Auction Participant", abstract=True, inherits="Agent", skill_requirements=None, description="Actor in the coordination auction")
    model.structural_specification.add_role(name="Announcer", abstract=True, inherits="Auction Participant", skill_requirements=None, description="Announce new tasks")
    model.structural_specification.add_role(name="Ambassador", abstract=True, inherits="Auction Participant", skill_requirements=None, description="Represent the team")
    model.structural_specification.add_role(name="P1", abstract=False, inherits="Ambassador", skill_requirements=None, description="Primary ambassador")
    model.structural_specification.add_role(name="P2", abstract=False, inherits="Ambassador", skill_requirements=None, description="Secondary ambassador")
    model.structural_specification.add_role(name="P3", abstract=False, inherits="Ambassador", skill_requirements=None, description="Tertiary ambassador")
    model.structural_specification.add_role(name="Bidder", abstract=True, inherits="Auction Participant", skill_requirements=None, description="Bid in the coordination auction")
    model.structural_specification.add_role(name="P4", abstract=False, inherits="Bidder", skill_requirements=None, description="Basic bidder")

    # Operation
    model.structural_specification.add_role(name="Operational Agent", abstract=True, inherits="Agent", skill_requirements=None, description="Operational agent")
    model.structural_specification.add_role(name="Scout", abstract=False, inherits="Operational Agent", skill_requirements=["goto"], description="Perform observation tasks")
    model.structural_specification.add_role(name="Monitor", abstract=False, inherits="Operational Agent", skill_requirements=["goto"], description="Monitor the environment")
    model.structural_specification.add_role(name="Patroller", abstract=False, inherits="Operational Agent", skill_requirements=["goto"], description="Patrol the environment")
    model.structural_specification.add_role(name="Obstructor", abstract=True, inherits="Operational Agent", skill_requirements=None, description="Block the path")
    model.structural_specification.add_role(name="Trapper", abstract=False, inherits="Operational Agent", skill_requirements=["trap"], description="Trap the target")
    model.structural_specification.add_role(name="Tracker", abstract=False, inherits="Operational Agent", skill_requirements=["track"], description="Track the target")
    model.structural_specification.add_role(name="Neutraliser", abstract=False, inherits="Operational Agent", skill_requirements=["neutralise"], description="Neutralise the target")

    # Operators
    model.structural_specification.add_role(name="Operator", abstract=True, inherits="Agent")
    model.structural_specification.add_role(name="Tactical Operator", abstract=False, inherits="Operator")
    model.structural_specification.add_role(name="Situation Operator", abstract=False, inherits="Operator")
    model.structural_specification.add_role(name="Robot Operator", abstract=False, inherits="Operator")

    # ----- Add role relations
    # Acquaintance
    model.structural_specification.add_role_relation(source="Operator", destination="Agent", relation_type="acquaintance",
                                             scope="intra")
    model.structural_specification.add_role_relation(source="Tactical Operator", destination="Agent",
                                             relation_type="acquaintance",
                                             scope="inter")

    # Communication
    model.structural_specification.add_role_relation(source="Agent", destination="Agent", relation_type="communication",
                                             scope="omni")

    # Authority
    model.structural_specification.add_role_relation(source="Operator", destination="Agent", relation_type="authority",
                                             scope="intra")
    model.structural_specification.add_role_relation(source="Tactical Operator", destination="Agent", relation_type="authority",
                                             scope="omni")

    # Compatibility
    model.structural_specification.add_role_relation(source="Operator", destination="Announcer", relation_type="compatible",
                                             scope="omni")
    model.structural_specification.add_role_relation(source="Operator", destination="Ambassador", relation_type="compatible",
                                             scope="intra")
    model.structural_specification.add_role_relation(source="Tactical Operator", destination="Ambassador",
                                             relation_type="compatible",
                                             scope="omni")

    model.structural_specification.add_role_relation(source="Tactical Operator", destination="Auction Participant",
                                             relation_type="compatible", scope="inter")
    model.structural_specification.add_role_relation(source="Operational Agent", destination="Operational Agent",
                                             relation_type="compatible",
                                             scope="intra")
    model.structural_specification.add_role_relation(source="Operational Agent", destination="Auction Participant",
                                             relation_type="compatible", scope="intra")
    model.structural_specification.add_role_relation(source="Auction Participant", destination="Auction Participant",
                                             relation_type="compatible", scope="intra")

    # Couple
    model.structural_specification.add_role_relation(source="Operational Agent", destination="Bidder", relation_type="couple",
                                             scope="inter")
    model.structural_specification.add_role_relation(source="Operational Agent", destination="Bidder", relation_type="couple",
                                             scope="intra")

    # ----- Add groups
    model.structural_specification.add_group(name="DroneSwarm",
                                     subgroups=None,
                                     role_cardinality={
                                         "Scout": {"min": 3, "max": 3}
                                     })

    model.structural_specification.add_group(name="SentinelTeam",
                                     subgroups={
                                         "DroneSwarm": {"min": 1, "max": 1},
                                     },
                                     role_cardinality={
                                         # Auction Participants
                                         "P1": {"min": 1, "max": 1},
                                         "P2": {"min": 1, "max": 1},
                                         "P3": {"min": 1, "max": 1},

                                         # Operational Agents
                                         "Scout": {"min": 3, "max": 4},
                                         "Monitor": {"min": 1, "max": 1},
                                         "Patroller": {"min": 1, "max": 3},

                                         # Operators
                                         "Situation Operator": {"min": 1, "max": 1},
                                     })

    model.structural_specification.add_group(name="DefenceTeam",
                                     subgroups=None,
                                     role_cardinality={
                                         # Auction Participants
                                         "P1": {"min": 1, "max": 1},
                                         "P2": {"min": 1, "max": 1},
                                         "P3": {"min": 1, "max": 1},

                                         # Operational Agents
                                         "Patroller": {"min": 2, "max": None},
                                         "Obstructor": {"min": 1, "max": None},
                                         "Trapper": {"min": 1, "max": None},
                                         "Tracker": {"min": 1, "max": None},
                                         "Neutraliser": {"min": 1, "max": None},

                                         # Operators
                                         "Robot Operator": {"min": 1, "max": 1},
                                     })

    model.structural_specification.add_group(name="ScoutingTeam",
                                     role_cardinality={
                                         # Auction Participants
                                         "P1": {"min": 1, "max": 1},
                                         "P2": {"min": 1, "max": 1},
                                         "P3": {"min": 1, "max": 1},

                                         # Operational Agents
                                         "Scout": {"min": 2, "max": None},
                                         "Tracker": {"min": 1, "max": None},
                                         "Neutraliser": {"min": 1, "max": None},

                                         # Operators
                                         "Robot Operator": {"min": 1, "max": 1},
                                     })

    model.structural_specification.add_group(name="Team",
                                     subgroups={
                                         "SentinelTeam": {"min": 1, "max": 1},
                                         "DefenceTeam": {"min": 1, "max": None},
                                         "ScoutingTeam": {"min": 1, "max": None}
                                     },
                                     role_cardinality={
                                         "Tactical Operator": {"min": 1, "max": 1},
                                     }
                                     )

    # --- Functional Specification ---
    # social_scheme = {
    #     "goals": [
    #         {"id": "g0", "description": "Score a goal"},
    #         {"id": "g2", "description": "Ball in midfield"},
    #         {"id": "g3", "description": "Ball in attack field"},
    #         {"id": "g4", "description": "Shoot at goal"}
    #     ],
    #     "plans": [
    #         {"goal": "g0", "subgoals": ["g2", "g3", "g4"], "operator": "sequence"}
    #     ],
    #     "missions": [
    #         {"id": "m1", "goals": ["g2", "g3", "g4"], "min_agents": 1, "max_agents": 4}
    #     ],
    #     "preferences": [
    #         {"higher": "m1", "lower": "m2"}
    #     ]
    # }
    # model.add_social_scheme("score_goal",
    #                         goals=social_scheme["goals"],
    #                         plans=social_scheme["plans"],
    #                         missions=social_scheme["missions"],
    #                         preferences=social_scheme["preferences"])
    # --- Deontic Specification ---    def plot(self):
    # model.add_permission("goalkeeper", "m7")
    # model.add_obligation("defender", "m1")
    # model.add_obligation("coach", "m6")
    # model.add_obligation("midfielder", "m2")
    # model.add_obligation("midfielder", "m3")
    # model.add_obligation("attacker", "m4")
    # model.add_obligation("attacker", "m5")

    # Print out the JSON representation of the model
    # print(model)

    # source = "Tactical Operator"
    # destination = "Bidder"

    # print(f"Relations between '{source}' and '{destination}':\n")
    # pprint(model.get_roles_relations_state(source=source, destination=destination), indent=2)

    # print(json.dumps(model.get_role_relations(role=source), indent=2))

    # model.plot()

    #pprint(model.structural_specification.get_group_specification("DefenceTeam"), indent=2)
    # pprint(model.to_dict())
    # model.save_to_file(filename="CoHoMa_moise_model_v1.json")
    """

    with open("icare_alloc_config/icare_alloc_config/__cache/__CoHoMa_organisation_model_v1_moise+_model.json", "r") as file:
        model = json.load(file)


    model = MoiseModel.from_dict(item_dict=model)

    print(model)

    print(model.check_agent_role_compatibility(["goto"], "Scout"))



    # =============================== Unit tests ===============================


