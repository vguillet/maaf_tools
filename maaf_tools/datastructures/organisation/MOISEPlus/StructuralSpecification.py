
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

RELATIONS = ["acquaintance", "communication", "authority", "compatible", "couple"]
SCOPES = ["inter", "intra", "omni"]


class StructuralSpecification(dict, MaafItem):
    """
    A class representing the structural specification of a MOISE+ model.
    The structural specification defines the roles, role relations, and groups within the model.
    """
    def __init__(self,
                 structural_specification: dict or "StructuralSpecification" = None,
                 functional_specification = None,
                 deontic_specification = None
                 ):
        # If no specification is provided, use the default template.
        if structural_specification is None:
            structural_specification = {
                "roles": [],
                "role_relations": [],
                "groups": []
            }

        elif isinstance(structural_specification, StructuralSpecification):
            structural_specification = structural_specification.copy()

        elif isinstance(structural_specification, dict):
            structural_specification = structural_specification.copy()

        elif not isinstance(structural_specification, dict) or not isinstance(structural_specification, StructuralSpecification):
            raise ValueError("Structural specification must be a dictionary or StructuralSpecification object.")

        if not self.check_specification_definition(structural_specification=structural_specification, verbose=1):
            raise ValueError("The provided structural specification is not valid.")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(structural_specification)

        # -> Initialize the functional and deontic specifications if provided.
        self.__functional_specification = functional_specification
        self.__deontic_specification = deontic_specification

    # ============================================================== Properties
    @property
    def functional_specification(self):
        if self.__functional_specification is None:
            warnings.warn("Functional specification is not set.")
            return None
        return self.__functional_specification

    @functional_specification.setter
    def functional_specification(self, functional_specification):
        self.__functional_specification = functional_specification

    @property
    def deontic_specification(self):
        if self.__deontic_specification is None:
            warnings.warn("Deontic specification is not set.")
            return None
        return self.__deontic_specification

    @deontic_specification.setter
    def deontic_specification(self, deontic_specification):
        self.__deontic_specification = deontic_specification

    @property
    def roles(self) -> list:
        """Returns the list of roles in the structural specification."""
        return self["roles"]

    @property
    def roles_names(self) -> list:
        """Returns the list of role names in the structural specification."""
        return [role["name"] for role in self.roles]

    @property
    def abstract_roles(self) -> list:
        """Returns the list of abstract roles in the structural specification."""
        return [role for role in self.roles if role["abstract"]]

    @property
    def concrete_roles(self) -> list:
        """Returns the list of concrete roles in the structural specification."""
        return [role for role in self.roles if not role["abstract"]]

    @property
    def groups(self) -> list:
        """Returns the list of groups in the structural specification."""
        return self["groups"]

    # ============================================================== Check
    @staticmethod
    def check_specification_definition(structural_specification, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        """
        Validates the structure of the structural specification.

        :param verbose: The verbosity level of the validation output.
        :return: True if the definition is valid, False otherwise.
        """
        errors = []

        # Step 1: Check top-level keys and their types
        roles = structural_specification.get("roles")
        role_relations = structural_specification.get("role_relations")
        groups = structural_specification.get("groups")

        if not isinstance(roles, list):
            if stop_at_first_error: return False
            errors.append("Missing or invalid top-level field 'roles': expected a list.")
            roles = []

        if not isinstance(role_relations, list):
            if stop_at_first_error: return False
            errors.append("Missing or invalid top-level field 'role_relations': expected a list.")
            role_relations = []

        if not isinstance(groups, list):
            if stop_at_first_error: return False
            errors.append("Missing or invalid top-level field 'groups': expected a list.")
            groups = []

        # Step 2: Validate roles
        role_names = []
        for i, role in enumerate(roles):
            if not isinstance(role, dict):
                if stop_at_first_error: return False
                errors.append(f"Invalid role at index {i}: expected a dictionary.")
                continue

            role_id = role.get("name", f"<unnamed role #{i}>")
            if "name" not in role:
                if stop_at_first_error: return False
                errors.append(f"Role at index {i} is missing required field 'name'.")
            elif not isinstance(role["name"], str):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'name' for role at index {i}: expected a string.")
            else:
                role_names.append(role["name"])

            if "abstract" not in role:
                if stop_at_first_error: return False
                errors.append(f"Role '{role_id}' is missing required field 'abstract'.")
            elif not isinstance(role["abstract"], bool):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'abstract' for role '{role_id}': expected a boolean.")

            if "inherits" not in role:
                if stop_at_first_error: return False
                errors.append(f"Role '{role_id}' is missing required field 'inherits'.")
            elif not (role["inherits"] is None or isinstance(role["inherits"], str)):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'inherits' for role '{role_id}': expected a string or None.")
            elif isinstance(role["inherits"], str) and role["inherits"] not in role_names:
                if stop_at_first_error: return False
                errors.append(f"Role '{role_id}' inherits from undefined role '{role['inherits']}'.")

            if "description" not in role:
                if stop_at_first_error: return False
                errors.append(f"Role '{role_id}' is missing required field 'description'.")
            elif not isinstance(role["description"], str):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'description' for role '{role_id}': expected a string.")

        # Check duplicate role names
        if len(role_names) != len(set(role_names)):
            duplicates = set(name for name in role_names if role_names.count(name) > 1)
            if stop_at_first_error:
                return False
            errors.append(f"Duplicate role names detected: {duplicates}.")

        # Step 3: Validate role relations
        for i, relation in enumerate(role_relations):
            if not isinstance(relation, dict):
                if stop_at_first_error:
                    return False
                errors.append(f"Invalid role relation at index {i}: expected a dictionary.")
                continue

            source = relation.get("source", "<unknown>")
            destination = relation.get("destination", "<unknown>")

            if "source" not in relation:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation at index {i} is missing required field 'source'.")
            elif source not in role_names:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation at index {i} has undefined source '{source}'.")

            if "destination" not in relation:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation at index {i} is missing required field 'destination'.")
            elif destination not in role_names:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation from '{source}' has undefined destination '{destination}'.")

            if "type" not in relation:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation from '{source}' to '{destination}' is missing required field 'type'.")
            elif relation["type"] not in RELATIONS:
                if stop_at_first_error:
                    return False
                errors.append(
                    f"Invalid 'type' in role relation from '{source}' to '{destination}': expected one of {RELATIONS}.")

            if "scope" not in relation:
                if stop_at_first_error:
                    return False
                errors.append(f"Role relation from '{source}' to '{destination}' is missing required field 'scope'.")
            elif relation["scope"] not in SCOPES:
                if stop_at_first_error:
                    return False
                errors.append(
                    f"Invalid 'scope' in role relation from '{source}' to '{destination}': expected one of {SCOPES}.")

        # Step 4: Validate groups
        group_names = []
        for i, group in enumerate(groups):
            if not isinstance(group, dict):
                if stop_at_first_error: return False
                errors.append(f"Invalid group at index {i}: expected a dictionary.")
                continue

            group_id = group.get("name", f"<unnamed group #{i}>")
            if "name" not in group:
                if stop_at_first_error: return False
                errors.append(f"Group at index {i} is missing required field 'name'.")
            elif not isinstance(group["name"], str):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'name' for group at index {i}: expected a string.")
            else:
                group_names.append(group["name"])

            if "role_cardinality" not in group:
                if stop_at_first_error: return False
                errors.append(f"Group '{group_id}' is missing required field 'role_cardinality'.")
            elif not isinstance(group["role_cardinality"], dict):
                if stop_at_first_error: return False
                errors.append(f"Invalid 'role_cardinality' for group '{group_id}': expected a dictionary.")
            else:
                for role_key, cardinality in group["role_cardinality"].items():
                    if role_key not in role_names:
                        if stop_at_first_error: return False
                        errors.append(f"Group '{group_id}' references undefined role '{role_key}' in 'role_cardinality'.")
                    if not isinstance(cardinality, dict):
                        if stop_at_first_error: return False
                        errors.append(f"Invalid cardinality for role '{role_key}' in group '{group_id}': expected a dictionary.")
                    else:
                        min_val = cardinality.get("min")
                        max_val = cardinality.get("max")
                        if not isinstance(min_val, int) or min_val < 0:
                            if stop_at_first_error: return False
                            errors.append(f"Invalid minimum cardinality for role '{role_key}' in group '{group_id}': {min_val}.")
                        if max_val is not None:
                            if not isinstance(max_val, int) or max_val < 0 or max_val < min_val:
                                if stop_at_first_error: return False
                                errors.append(f"Invalid maximum cardinality for role '{role_key}' in group '{group_id}': {max_val}.")

            if "subgroups" in group:
                if not isinstance(group["subgroups"], dict):
                    if stop_at_first_error: return False
                    errors.append(f"Invalid 'subgroups' for group '{group_id}': expected a dictionary.")
                else:
                    for subgroup_key, cardinality in group["subgroups"].items():
                        if subgroup_key not in group_names:
                            if stop_at_first_error: return False
                            errors.append(f"Group '{group_id}' references undefined subgroup '{subgroup_key}'.")
                        if not isinstance(cardinality, dict):
                            if stop_at_first_error: return False
                            errors.append(f"Invalid cardinality for subgroup '{subgroup_key}' in group '{group_id}': expected a dictionary.")
                        else:
                            min_val = cardinality.get("min")
                            max_val = cardinality.get("max")
                            if not isinstance(min_val, int) or min_val < 0:
                                if stop_at_first_error: return False
                                errors.append(f"Invalid minimum cardinality for subgroup '{subgroup_key}' in group '{group_id}': {min_val}.")
                            if max_val is not None:
                                if not isinstance(max_val, int) or max_val < 0 or max_val < min_val:
                                    if stop_at_first_error: return False
                                    errors.append(f"Invalid maximum cardinality for subgroup '{subgroup_key}' in group '{group_id}': {max_val}.")

        # Check duplicate group names
        if len(group_names) != len(set(group_names)):
            duplicates = set(name for name in group_names if group_names.count(name) > 1)
            if stop_at_first_error: return False
            errors.append(f"Duplicate group names detected: {duplicates}.")

        # Report errors
        if errors and verbose >= 1:
            print("Errors found in structural specification:")
            for error in errors:
                print(f"  - {error}")

        return not errors

    # ============================================================== Get
    def get_group(self, name: str):
        """
        Returns the group specification by name.

        :param name: The name of the group.
        :return: A dictionary with the group specification.
        """
        for group in self["groups"]:
            if group["name"] == name:
                return group
        return None

    def get_role(self, name: str):
        """
        Returns the role specification by name.

        :param name: The ID of the role.
        :return: A dictionary with the role specification.
        """
        for role in self["roles"]:
            if role["name"] == name:
                return role
        return None

    def get_role_relations(self, role: str) -> dict:
        """
        Returns the role relations for a given role. This method checks for all relations involving the role.

        :param role: role name
        :return: dictionary with the relations and scopes for the given role
        """

        # -> Check that the role is defined in the model
        if role not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{role}' is not defined in the model.")

        # -> Check relations
        relations = {}

        for relation in RELATIONS:
            relations[relation] = []

            for role_relation in self["role_relations"]:
                if role_relation["source"] == role and role_relation["type"] == relation:
                    relations[relation].append(role_relation)
        return relations

    def get_roles_relation_state(self, source: str, destination: str, relation: str) -> dict:
        """
        Returns the role relations between two roles. This method checks for a specific relation, and returns both
        if the relation is True/False, and the scope (inter/intra) if True. Since the model also defines relations
        through inheritance, the method will recursively check the ancestors of the source role if no direct relation
        is found

        :param source: source role
        :param destination: destination role
        :param relation: relation type
        :return: dictionary with the relation status and scope
        """

        # -> Check that the relation is in the list of valid relations
        if relation not in RELATIONS:
            raise ValueError(f"Invalid relation type '{relation}'. "
                             f"Valid types are: {', '.join(RELATIONS)}")

        # -> Check that both roles exist in the model
        if source not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{source}' is not defined in the model.")

        if destination not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{destination}' is not defined in the model.")

        def check_ancestors(source: str, destination: str, relation: str):
            # Retrieve the role object for the given source
            role_obj = next((role for role in self.roles if role["name"] == source), None)
            if role_obj is None:
                return False, None

            # Check for a direct relation between the source and destination
            for role_relation in self["role_relations"]:
                if (role_relation["source"] == source and
                        role_relation["destination"] == destination and
                        role_relation["type"] == relation):
                    print(f"Role '{source}' -> Role '{destination}' ({relation}, {role_relation['scope']})")
                    return True, role_relation["scope"]

            # First, check for a relation via the destination's ancestors
            destination_obj = next((role for role in self.roles if role["name"] == destination), None)
            if destination_obj:
                inherited_target = destination_obj.get("inherits")
                if inherited_target:
                    found, scope = check_ancestors(source, inherited_target, relation)
                    if found:
                        return found, scope

            # Then, check for a relation via the source's ancestors
            inherited_source = role_obj.get("inherits")
            if inherited_source:
                found, scope = check_ancestors(inherited_source, destination, relation)
                if found:
                    return found, scope

            return False, None

        # -> Check relation state
        status, scope = check_ancestors(source, destination, relation)

        return {"status": status, "scope": scope}

    def get_roles_relations_state(self, source: str, destination: str) -> dict:
        """
        Returns the role relations between two roles. This method checks for all relations between the source and
        the destination.

        :param source: source role
        :param destination: destination role
        :return: dictionary with the relations status and scopes between the source and the destination
        """

        # -> Check relations
        relations_status = {}

        for relation in RELATIONS:
            relations_status[relation] = self.get_roles_relation_state(source, destination, relation)

        return relations_status

    def get_group_specification(self, group_name: str):
        """
        Returns the specification of a group by name.

        :param group_name: The name of the group.
        :return: A dictionary with the group specification.
        """
        for group in self["groups"]:
            if group["name"] == group_name:
                return group
        return None

    def get_minimal_team_composition(self):
        """
        Returns the minimal team composition based on the group specifications.
        :return:
        """

        minimal_viable_team_composition = []

        # -> Create minimal group structure
        for group in self["groups"]:
            # > Add the minimal number of subgroups
            if group.get("subgroups"):
                for subgroup_name, subgroup_cardinality in group["subgroups"].items():
                    for i in range(subgroup_cardinality["min"]):
                        minimal_viable_team_composition.append(
                            {"group_type": subgroup_name,
                             "roles": {}
                             }
                        )

        # -> Add minimal role assignments
        for group in minimal_viable_team_composition:
            # > Add the minimal number of roles
            group_type = group["group_type"]

            # Get the role cardinality for the group
            role_cardinality = next((group["role_cardinality"] for group in self["groups"]
                                     if group["name"] == group_type), None)
            if role_cardinality is None:
                pass

            for role_name, role_cardinality in role_cardinality.items():
                group["roles"][role_name] = role_cardinality["min"]

        return minimal_viable_team_composition

    def get_children_roles(self, parent_role: str, include_abstract_roles=False) -> list:
        """
        Recursive returns the children roles of a given parent role.

        :param parent_role: The name of the parent role.
        :return: A list of child roles.
        """
        children = []
        for role in self.roles:
            if role.get("inherits") == parent_role:
                if include_abstract_roles and not role["abstract"]:
                    children.append(role["name"])
                else:
                    children.append(role["name"])
                children.extend(self.get_children_roles(role["name"]))
        return children

    # ============================================================== Set
    # ============================================================== Merge

    def merge(self,
              structural_specification: "StructuralSpecification",
              prioritise_local: bool = True,
              ) -> bool:
        """
        Merges the current structural specification with another one.
        If prioritise_local is True, the local specification takes precedence over the other one.
        """
        # Validate incoming specification.
        if not isinstance(structural_specification, StructuralSpecification):
            raise ValueError("The structural specification to merge is not of type StructuralSpecification.")

        if not self.check_specification_definition(structural_specification=structural_specification,
                                                   stop_at_first_error=True):
            raise ValueError("The structural specification to merge is not valid.")

        # ----- Merge "roles"
        for role in structural_specification["roles"]:
            role_name = role["name"]
            exists = any(r["name"] == role_name for r in self["roles"])
            if exists:
                if not prioritise_local:
                    # Remove the existing role and add the new one
                    self.remove_role(role_name)
                    # Use the add_role helper to insert the new role.
                    # (It will perform internal validations, e.g. for the "inherits" field.)
                    self.add_role(
                        name=role["name"],
                        abstract=role["abstract"],
                        inherits=role["inherits"],
                        skill_requirements=role.get("skill_requirements", []),
                        description=role.get("description", "")
                    )
                # Else, if prioritise_local, skip the duplicate.
            else:
                self.add_role(
                    name=role["name"],
                    abstract=role["abstract"],
                    inherits=role["inherits"],
                    skill_requirements=role.get("skill_requirements", []),
                    description=role.get("description", "")
                )

        # ----- Merge "role_relations"
        for rel in structural_specification["role_relations"]:
            key = (rel["source"], rel["destination"], rel["type"], rel["scope"])
            exists = any(
                (
                existing_rel["source"], existing_rel["destination"], existing_rel["type"], existing_rel["scope"]) == key
                for existing_rel in self["role_relations"]
            )
            if exists:
                if not prioritise_local:
                    # Remove existing relation and add new
                    self.remove_role_relation(rel["source"], rel["destination"], rel["type"], rel["scope"])
                    self.add_role_relation(rel["source"], rel["destination"], rel["type"], rel["scope"])
                # Else, if prioritise_local, skip.
            else:
                self.add_role_relation(rel["source"], rel["destination"], rel["type"], rel["scope"])

        # ----- Merge "groups"
        for group in structural_specification["groups"]:
            group_name = group["name"]
            exists = any(g["name"] == group_name for g in self["groups"])
            if exists:
                if not prioritise_local:
                    # Remove the current group and add the new one.
                    self.remove_group(group_name)
                    self.add_group(
                        name=group["name"],
                        subgroups=group.get("subgroups", None),
                        role_cardinality=group.get("role_cardinality", None)
                    )
                # Else, if prioritise_local, leave the current group as is.
            else:
                self.add_group(
                    name=group["name"],
                    subgroups=group.get("subgroups", None),
                    role_cardinality=group.get("role_cardinality", None)
                )

        # Run a final validation check to ensure the merged specification is valid.
        if not self.check_specification_definition(structural_specification=self, stop_at_first_error=True):
            raise ValueError("The merged structural specification is not valid.")

        return True

    # ============================================================== Add
    # ------------------------ Structural Specification Methods
    def add_role(self,
                 name: str,
                 abstract: bool = False,
                 inherits: str = None,
                 skill_requirements: list or None = [],
                 description: str = ""
                 ) -> None:
        """
        Adds a role to the structural specification.

        :param name: The role name.
        :param abstract: Whether the role is abstract (default False).
        :param inherits: Name of the role this one inherits from.
        :param skill_requirements: List of required skills for the role.
        :param description: Description of the role.
        """
        role = {
            "name": name,
            "abstract": abstract,
            "inherits": None,
            "skill_requirements": [] if None else skill_requirements,
            "description": "" if None else description
        }

        if inherits is not None:
            # Check if the inherited role is defined in the model
            if inherits not in [role["name"] for role in self.roles]:
                raise ValueError(f"Role '{inherits}' is not defined in the model.")

            role["inherits"] = inherits

        self["roles"].append(role)

    def add_role_relation(self, source: str, destination: str, relation_type: str, scope: str) -> None:
        """
        Adds a role relation.

        Args:
            source (str): The source role.
            destination (str): The destination role.
            relation_type (str): The type of relation (e.g., "authority", "communication").
            scope (str): The scope (e.g., "inter" or "intra").
        """

        # -> Check if the relation type is valid
        if relation_type not in RELATIONS:
            raise ValueError(f"Invalid relation type '{relation_type}'. "
                             f"Valid types are: {', '.join(RELATIONS)}")

        # -> Check if the scope is valid
        if scope not in SCOPES:
            raise ValueError(f"Invalid scope '{scope}'. "
                             f"Valid scopes are: {', '.join(SCOPES)}")

        # -> Check if the source and target roles are defined in the model
        if source not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{source}' is not defined in the model.")

        if destination not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{destination}' is not defined in the model.")

        # -> Check if the relation already exists
        for relation in self["role_relations"]:
            if relation["source"] == source and relation["destination"] == destination and \
                    relation["type"] == relation_type and relation["scope"] == scope:
                raise ValueError(
                    f"Relation '{source}' -> '{destination}' ({relation_type}, {scope}) already exists.")

        # -> Add the relation
        relation = {
            "source": source,
            "destination": destination,
            "type": relation_type,
            "scope": scope
        }
        self["role_relations"].append(relation)

    def add_group(self, name: str, subgroups: dict or None = None, role_cardinality: dict or None = None) -> None:
        """
        Adds a group specification.

        Args:
            name (str): The group name.
            subgroups (dict, optional): A mapping from subgroup names to their role cardinality.
            role_cardinality (dict, optional): A mapping from role names to their cardinality.
                                                Example: {"coach": {"min": 1, "max": 2}, "player": {"min": 3, "max": 11}}
        """
        group = {"name": name}
        if subgroups is not None:
            # -> For every subgroup, check if it is defined in the model
            for subgroup in subgroups:
                if subgroup not in [group["name"] for group in self["groups"]]:
                    raise ValueError(f"Subgroup '{subgroup}' is not defined in the model.")
            group["subgroups"] = subgroups

        if role_cardinality is not None:
            group["role_cardinality"] = role_cardinality

        self["groups"].append(group)

    # ============================================================== Remove
    # ------------------------ Structural Specification Methods
    def remove_role(self, name):
        """Removes a role by name. Return True if the role was removed, False otherwise."""

        for role in self["roles"]:
            if role["name"] == name:
                self["roles"].remove(role)
                return True
        return False

    def remove_role_relation(self, source, destination, relation_type, scope):
        """Removes a role relation. Return True if the relation was removed, False otherwise."""

        for relation in self["role_relations"]:
            if relation["source"] == source and relation["destination"] == destination and \
                    relation["type"] == relation_type and relation["scope"] == scope:
                self["role_relations"].remove(relation)
                return True
        return False

    def remove_group(self, name):
        """Removes a group by name. Return True if the group was removed, False otherwise."""

        for group in self["groups"]:
            if group["name"] == name:
                self["groups"].remove(group)
                return True
        return False

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self


if __name__ == "__main__":
    # Example dictionaries for two structural specifications.
    # Local specification.
    local_spec_dict = {
        "roles": [
            {
                "name": "Leader",
                "abstract": False,
                "inherits": None,
                "skill_requirements": ["manage"],
                "description": "Team leader."
            },
            {
                "name": "Member",
                "abstract": False,
                "inherits": None,
                "skill_requirements": [],
                "description": "Team member."
            }
        ],
        "role_relations": [
            {
                "source": "Leader",
                "destination": "Member",
                "type": "authority",
                "scope": "intra"
            }
        ],
        "groups": [
            {
                "name": "Team",
                "role_cardinality": {
                    "Leader": {"min": 1, "max": 1},
                    "Member": {"min": 2, "max": 5}
                }
            }
        ]
    }

    # Incoming specification to merge.
    merge_spec_dict = {
        "roles": [
            {
                "name": "Leader",
                "abstract": False,
                "inherits": None,
                "skill_requirements": ["strategize"],
                "description": "Updated team leader."
            },
            {
                "name": "Member",
                "abstract": False,
                "inherits": None,
                "skill_requirements": [],
                "description": "Team member."
            },
            {
                "name": "Support",
                "abstract": False,
                "inherits": None,
                "skill_requirements": ["assist"],
                "description": "Team support."
            }
        ],
        "role_relations": [
            {
                "source": "Leader",
                "destination": "Support",
                "type": "communication",
                "scope": "intra"
            },
            {
                "source": "Leader",
                "destination": "Member",
                "type": "authority",
                "scope": "intra"
            }
        ],
        "groups": [
            {
                "name": "Team",
                "role_cardinality": {
                    "Leader": {"min": 1, "max": 1},
                    "Member": {"min": 3, "max": 5},
                    "Support": {"min": 1, "max": 3}
                }
            }
        ]
    }


    # Helper function to find a role by name in a specification.
    def find_role(spec, role_name):
        return next((role for role in spec["roles"] if role["name"] == role_name), None)


    # Assuming the StructuralSpecification class (with the merge method implemented)
    # is already defined and imported.
    # We create two instances of it.

    local_spec = StructuralSpecification(local_spec_dict)
    merge_spec_obj = StructuralSpecification(merge_spec_dict)


    def print_separator(title):
        print("\n" + "=" * 20)
        print(title)
        print("=" * 20)


    # ---------------------- Test 1: prioritise_local = True ----------------------
    print_separator("Test 1: Merge with prioritise_local=True")
    print("\nLocal Specification BEFORE merge:")
    pprint(local_spec)
    print("\nMerge Specification:")
    pprint(merge_spec_obj)

    # Merge into the local specification; local takes precedence.
    result = local_spec.merge(merge_spec_obj, prioritise_local=True)
    assert result is True, "Merge returned False unexpectedly (Test 1)."

    # Expected Outcome with prioritise_local=True:
    # - Role "Leader" should remain unchanged (description "Team leader.", skill_requirements ["manage"]).
    # - Role "Member" remains.
    # - Role "Support" (new role) is added.
    # - For role_relations, the existing relation (Leader->Member) remains, and the new relation (Leader->Support) is added.
    # - Group "Team" remains unchanged, keeping role_cardinality with "Member": {"min": 2, "max": 5}.

    # Role checks.
    leader_role = find_role(local_spec, "Leader")
    assert leader_role is not None, "Leader role missing (Test 1)."
    assert leader_role["description"] == "Team leader.", "Leader role description changed (Test 1)."
    assert leader_role["skill_requirements"] == ["manage"], "Leader role skill_requirements changed (Test 1)."

    member_role = find_role(local_spec, "Member")
    assert member_role is not None, "Member role missing (Test 1)."

    support_role = find_role(local_spec, "Support")
    assert support_role is not None, "Support role was not added (Test 1)."
    assert support_role["description"] == "Team support.", "Support role description is incorrect (Test 1)."

    assert len(local_spec["roles"]) == 3, f"Expected 3 roles, got {len(local_spec['roles'])} (Test 1)."

    # Role relation checks.
    # Create set of relation keys for easier checking.
    relation_keys = {(rel["source"], rel["destination"], rel["type"], rel["scope"])
                     for rel in local_spec["role_relations"]}
    assert ("Leader", "Member", "authority",
            "intra") in relation_keys, "Expected relation Leader->Member missing (Test 1)."
    assert ("Leader", "Support", "communication",
            "intra") in relation_keys, "Expected relation Leader->Support missing (Test 1)."
    assert len(local_spec[
                   "role_relations"]) == 2, f"Expected 2 role relations, got {len(local_spec['role_relations'])} (Test 1)."

    # Group checks.
    assert len(local_spec["groups"]) == 1, f"Expected 1 group, got {len(local_spec['groups'])} (Test 1)."
    team_group = local_spec["groups"][0]
    # Expect group "Team" to be unchanged from local_spec_dict.
    assert team_group["role_cardinality"].get("Member", {}).get("min") == 2, \
        "Group 'Team' role_cardinality for 'Member' changed (Test 1)."

    print("\nTest 1 passed.")

    # ---------------------- Test 2: prioritise_local = False ----------------------
    # For the second test, reinitialize local_spec to the original definition.
    local_spec = StructuralSpecification(local_spec_dict)

    print_separator("Test 2: Merge with prioritise_local=False")
    print("\nLocal Specification BEFORE merge:")
    pprint(local_spec)
    print("\nMerge Specification:")
    pprint(merge_spec_obj)

    # Merge into the local specification; incoming specification overrides conflicting entries.
    result = local_spec.merge(merge_spec_obj, prioritise_local=False)
    assert result is True, "Merge returned False unexpectedly (Test 2)."

    # Expected Outcome with prioritise_local=False:
    # - Role "Leader" is replaced by the incoming one (description "Updated team leader.", skill_requirements ["strategize"]).
    # - Role "Member" remains as in local_spec.
    # - Role "Support" is added.
    # - For role_relations, the incoming relation "Leader->Member" replaces the local one (though they may be equal),
    #   and the new relation "Leader->Support" is added.
    # - Group "Team" is replaced by the incoming one, so the role_cardinality for "Member" becomes {"min": 3, ...}.

    # Role checks.
    leader_role = find_role(local_spec, "Leader")
    assert leader_role is not None, "Leader role missing (Test 2)."
    assert leader_role["description"] == "Updated team leader.", "Leader role description not updated (Test 2)."
    assert leader_role["skill_requirements"] == ["strategize"], "Leader role skill_requirements not updated (Test 2)."

    member_role = find_role(local_spec, "Member")
    assert member_role is not None, "Member role missing (Test 2)."

    support_role = find_role(local_spec, "Support")
    assert support_role is not None, "Support role was not added (Test 2)."
    assert support_role["description"] == "Team support.", "Support role description is incorrect (Test 2)."

    assert len(local_spec["roles"]) == 3, f"Expected 3 roles, got {len(local_spec['roles'])} (Test 2)."

    # Role relation checks.
    relation_keys = {(rel["source"], rel["destination"], rel["type"], rel["scope"])
                     for rel in local_spec["role_relations"]}
    assert ("Leader", "Member", "authority",
            "intra") in relation_keys, "Expected relation Leader->Member missing (Test 2)."
    assert ("Leader", "Support", "communication",
            "intra") in relation_keys, "Expected relation Leader->Support missing (Test 2)."
    assert len(local_spec[
                   "role_relations"]) == 2, f"Expected 2 role relations, got {len(local_spec['role_relations'])} (Test 2)."

    # Group checks.
    assert len(local_spec["groups"]) == 1, f"Expected 1 group, got {len(local_spec['groups'])} (Test 2)."
    team_group = local_spec["groups"][0]
    # For prioritise_local=False, group "Team" is replaced with the incoming one.
    assert team_group["role_cardinality"].get("Member", {}).get("min") == 3, \
        "Group 'Team' role_cardinality for 'Member' not updated (Test 2)."

    print("\nTest 2 passed.")
    print_separator("All tests passed")