
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


class DeonticSpecification(dict, MaafItem):
    """
    A class representing the functional specification of a MOISE+ model.
    The deontic specification provides the mapping between the roles and the missions they are allowed to perform.
    """
    def __init__(self,
                 deontic_specification: dict or None = None,
                 structural_specification = None,
                 functional_specification = None
                 ):        # If no specification is provided, use the default template.
        if deontic_specification is None:
            deontic_specification = {
                "permissions": [],
                "obligations": []
            }

        elif isinstance(deontic_specification, DeonticSpecification):
            deontic_specification = deontic_specification.copy()

        elif isinstance(deontic_specification, dict):
            deontic_specification = deontic_specification.copy()

        elif not isinstance(deontic_specification, dict) or not isinstance(deontic_specification, DeonticSpecification):
            raise ValueError("The deontic specification must be a dictionary or DeonticSpecification object.")

        if not self.check_specification_definition(deontic_specification=deontic_specification, verbose=1):
            raise ValueError("The provided deontic specification is not valid")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(deontic_specification)

        # -> Initialize the structural and functional specifications
        self.__structural_specification = structural_specification
        self.__functional_specification = functional_specification

    # ============================================================== Properties
    @property
    def structural_specification(self):
        if self.__structural_specification is None:
            warnings.warn("Structural specification is not set.")
            return None
        return self.__structural_specification

    @structural_specification.setter
    def structural_specification(self, structural_specification):
        self.__structural_specification = structural_specification

    @property
    def functional_specification(self):
        if self.__functional_specification is None:
            warnings.warn("Functional specification is not set.")
            return None
        return self.__functional_specification

    @functional_specification.setter
    def functional_specification(self, functional_specification):
        self.__functional_specification = functional_specification

    # ============================================================== Check
    @staticmethod
    def check_specification_definition(deontic_specification, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        warnings.warn("Checking deontic specification definition is not implemented yet.")
        return True

    def check_agent_goal_compatibility(self, agent_roles: list[str], goal_name: str) -> bool:
        """
        Checks if the agent roles are compatible with the goal type.

        :param agent_roles: List of roles associated with the agent.
        :param goal_name: The goal type to check compatibility with.

        :return: True if compatible, False otherwise.
        """

        # -> Get missions associated with the goal type
        missions = self.functional_specification.get_missions_associated_with_goal(goal_name=goal_name, names_only=True)

        # -> Get roles associated with the missions
        missions_roles = []

        for mission in missions:
            missions_roles.extend(self.get_role_associated_with_mission(mission_name=mission))

        # -> Check if the agent has at least one role_name associated with the goal type
        for role_name in agent_roles:
            if role_name in missions_roles:
                return True

        return False

    # ============================================================== Get
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

        # -> Get all missions associated with the role_name
        missions = []
        for permission in self["permissions"]:
            if permission["role_name"] == role_name:
                missions.append(permission["mission_name"])

        for obligation in self["obligations"]:
            if obligation["role_name"] == role_name:
                missions.append(obligation["mission_name"])

        # -> Get all goals associated with the missions
        goals = []
        for mission_name in missions:
            mission_spec = self.functional_specification.get_mission(mission_name)
            if mission_spec is not None:
                if "goals" in mission_spec:
                    goals.extend(mission_spec["goals"])

        goals = set(goals)

        # -> Get all skills associated with the goals
        skills = []
        for goal in goals:
            skills.extend(self.functional_specification.get_goal_skill_requirements(goal))

        skills = set(skills)

        return list(skills)

    def get_role_associate_with_mission(self, mission_name: str) -> list:
        """
        Gets the roles associated with a specific mission type.

        :param mission_name: The mission type to check.
        :return: A list of roles associated with the specified mission type.
        """

        roles = []

        # Check permissions
        for permission in self["permissions"]:
            if permission["mission_name"] == mission_name:
                roles.append(permission["role_name"])

        # Check obligations
        for obligation in self["obligations"]:
            if obligation["mission_name"] == mission_name:
                roles.append(obligation["role_name"])

        return list(set(roles))

    def get_missions_associated_with_role(self, role_name: str) -> list:
        """
        Gets the missions associated with a specific role_name.

        :param role_name: The role_name to check.
        :return: A list of missions associated with the specified role_name.
        """

        missions = []

        # Check permissions
        for permission in self["permissions"]:
            if permission["role_name"] == role_name:
                missions.append(permission["mission_name"])

        # Check obligations
        for obligation in self["obligations"]:
            if obligation["role_name"] == role_name:
                missions.append(obligation["mission_name"])

        return list(set(missions))

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self,
              deontic_specification: "DeonticSpecification",
              prioritise_local: bool = True) -> bool:
        """
        Merges the current deontic specification with another one.
        If prioritise_local is True, the local definition of a permission or obligation takes precedence.
        Otherwise, the incoming specification will override local entries in case of conflict.

        :param deontic_specification: A DeonticSpecification object to merge from.
        :param prioritise_local: Boolean flag indicating if local values should be kept on conflict.
        :return: True if the merged specification passes validation.
        :raises ValueError: If the incoming specification is not of the correct type or fails validation.
        """
        # Validate the type of the incoming specification.
        if not isinstance(deontic_specification, DeonticSpecification):
            raise ValueError("The deontic specification must be a DeonticSpecification object.")

        # Validate the incoming specification.
        if not self.check_specification_definition(deontic_specification=deontic_specification,
                                                   stop_at_first_error=True):
            raise ValueError("The provided deontic specification is not valid.")

        # ----- Merge Permissions -----
        # Iterate over each incoming permission.
        for incoming_permission in deontic_specification.get("permissions", []):
            role_name = incoming_permission.get("role_name")
            mission_name = incoming_permission.get("mission_name")
            # Search for an existing permission with the same role_name and mission_name.
            existing_permission = next(
                (perm for perm in self.get("permissions", [])
                 if perm.get("role_name") == role_name and perm.get("mission_name") == mission_name), None)

            if existing_permission:
                # If local is prioritized, do nothing (preserve the local permission).
                # Otherwise, replace the local permission with the incoming one.
                if not prioritise_local:
                    self.remove_permission(role_name, mission_name)
                    self.add_permission(role_name, mission_name,
                                        time_constraint=incoming_permission.get("time_constraint", "Any"))
            else:
                # Add new permission from incoming specification.
                self.add_permission(role_name, mission_name,
                                    time_constraint=incoming_permission.get("time_constraint", "Any"))

        # ----- Merge Obligations -----
        # Iterate over each incoming obligation.
        for incoming_obligation in deontic_specification.get("obligations", []):
            role_name = incoming_obligation.get("role_name")
            mission_name = incoming_obligation.get("mission_name")
            # Find an existing obligation with the same role_name and mission_name.
            existing_obligation = next(
                (obl for obl in self.get("obligations", [])
                 if obl.get("role_name") == role_name and obl.get("mission_name") == mission_name), None)

            if existing_obligation:
                if not prioritise_local:
                    self.remove_obligation(role_name, mission_name)
                    self.add_obligation(role_name, mission_name,
                                        time_constraint=incoming_obligation.get("time_constraint", "Any"))
            else:
                self.add_obligation(role_name, mission_name,
                                    time_constraint=incoming_obligation.get("time_constraint", "Any"))

        # Final validation of the merged specification.
        if not self.check_specification_definition(deontic_specification=self, stop_at_first_error=True):
            raise ValueError("The merged deontic specification is not valid.")

        return True

    # ============================================================== Add
    def add_permission(self, role_name: str, mission_name: str, time_constraint="Any"):
        """
        Adds a permission to the deontic specification.

        Args:
            role_name (str): The role_name associated with the permission.
            mission_name (str): The mission_name to which the permission applies.
            time_constraint (str): The time constraint (default "Any").
        """
        permission = {"role_name": role_name, "mission_name": mission_name, "time_constraint": time_constraint}
        self["permissions"].append(permission)

    def add_obligation(self, role_name, mission_name, time_constraint="Any"):
        """
        Adds an obligation to the deontic specification.

        Args:
            role_name (str): The role_name associated with the obligation.
            mission_name (str): The mission_name to which the obligation applies.
            time_constraint (str): The time constraint (default "Any").
        """
        obligation = {"role_name": role_name, "mission_name": mission_name, "time_constraint": time_constraint}
        self["obligations"].append(obligation)

    # ============================================================== Remove
    def remove_permission(self, role_name, mission_name: str):
        """Removes a permission. Return True if the permission was removed, False otherwise."""

        for permission in self["permissions"]:
            if permission["role_name"] == role_name and permission["mission_name"] == mission_name:
                self["permissions"].remove(permission)
                return True
        return False

    def remove_obligation(self, role_name, mission_name):
        """Removes an obligation. Return True if the obligation was removed, False otherwise."""

        for obligation in self["obligations"]:
            if obligation["role_name"] == role_name and obligation["mission_name"] == mission_name:
                self["obligations"].remove(obligation)
                return True
        return False

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self

if __name__ == "__main__":

    deontic_spec = DeonticSpecification()
    deontic_spec.add_obligation(role_name="Scout", mission_name="m_scouting")
    deontic_spec.add_obligation(role_name="Monitor", mission_name="m_monitoring")
    deontic_spec.add_obligation(role_name="Patroller", mission_name="m_patrolling")
    deontic_spec.add_obligation(role_name="Obstructor", mission_name="m_interdiction")
    deontic_spec.add_obligation(role_name="Obstructor", mission_name="m_obstructing")
    deontic_spec.add_obligation(role_name="Trapper", mission_name="m_interdiction")
    deontic_spec.add_obligation(role_name="Trapper", mission_name="m_trapping")
    deontic_spec.add_obligation(role_name="Tracker", mission_name="m_tracking")
    deontic_spec.add_obligation(role_name="Neutraliser", mission_name="m_neutralising")

    deontic_spec.get_role_skill_requirements("Scout")

    deontic_spec.save_to_file("icare_alloc_config/icare_alloc_config/moise_deontic_specification.json")

    # ========================================================== Tests
    # Define a local deontic specification dictionary.
    local_deontic_spec_dict = {
        "permissions": [
            {"role_name": "Scout", "mission_name": "m_scouting", "time_constraint": "Immediate"},
            {"role_name": "Monitor", "mission_name": "m_monitoring", "time_constraint": "Any"}
        ],
        "obligations": [
            {"role_name": "Patroller", "mission_name": "m_patrolling", "time_constraint": "Scheduled"},
            {"role_name": "Obstructor", "mission_name": "m_interdiction", "time_constraint": "Any"}
        ]
    }

    # Define an incoming deontic specification dictionary.
    incoming_deontic_spec_dict = {
        "permissions": [
            # New permission not in local.
            {"role_name": "Scout", "mission_name": "m_recon", "time_constraint": "Urgent"},
            # Duplicate permission: same role_name and mission_name as local, but different time_constraint.
            {"role_name": "Scout", "mission_name": "m_scouting", "time_constraint": "Delayed"}
        ],
        "obligations": [
            # New obligation.
            {"role_name": "Monitor", "mission_name": "m_analysis", "time_constraint": "Any"},
            # Duplicate obligation: same role_name and mission_name as local, but different time_constraint.
            {"role_name": "Patroller", "mission_name": "m_patrolling", "time_constraint": "Immediate"}
        ]
    }

    ########################################################################
    # Test 1: Merge with prioritise_local=True (i.e. local values are preserved)
    ########################################################################
    print("== Test 1: Merge with prioritise_local=True ==")
    local_ds = DeonticSpecification(local_deontic_spec_dict)
    incoming_ds = DeonticSpecification(incoming_deontic_spec_dict)

    result = local_ds.merge(incoming_ds, prioritise_local=True)
    assert result is True, "Merge did not return True for prioritise_local=True."

    # Permission ("Scout", "m_scouting") should remain unchanged.
    scout_permission = next(
        (p for p in local_ds["permissions"] if p["role_name"] == "Scout" and p["mission_name"] == "m_scouting"), None)
    assert scout_permission is not None, "Scout permission for m_scouting not found."
    assert scout_permission["time_constraint"] == "Immediate", \
        "Local permission not preserved when prioritise_local=True (expected 'Immediate')."

    # New permission ("Scout", "m_recon") should be added.
    recon_permission = next(
        (p for p in local_ds["permissions"] if p["role_name"] == "Scout" and p["mission_name"] == "m_recon"), None)
    assert recon_permission is not None, "New permission m_recon not added."

    # Obligation ("Patroller", "m_patrolling") should remain unchanged.
    patroller_obligation = next(
        (o for o in local_ds["obligations"] if o["role_name"] == "Patroller" and o["mission_name"] == "m_patrolling"), None)
    assert patroller_obligation is not None, "Patroller obligation for m_patrolling not found."
    assert patroller_obligation["time_constraint"] == "Scheduled", \
        "Local obligation not preserved when prioritise_local=True (expected 'Scheduled')."

    # New obligation ("Monitor", "m_analysis") should be added.
    monitor_obligation = next(
        (o for o in local_ds["obligations"] if o["role_name"] == "Monitor" and o["mission_name"] == "m_analysis"), None)
    assert monitor_obligation is not None, "New obligation m_analysis not added."

    print("Test 1 passed.")

    ########################################################################
    # Test 2: Merge with prioritise_local=False (i.e. incoming values override local)
    ########################################################################
    print("== Test 2: Merge with prioritise_local=False ==")
    # Reinitialize the local deontic specification.
    local_ds = DeonticSpecification(local_deontic_spec_dict)
    incoming_ds = DeonticSpecification(incoming_deontic_spec_dict)

    result = local_ds.merge(incoming_ds, prioritise_local=False)
    assert result is True, "Merge did not return True for prioritise_local=False."

    # In this case, the duplicate permission should be replaced.
    scout_permission = next(
        (p for p in local_ds["permissions"] if p["role_name"] == "Scout" and p["mission_name"] == "m_scouting"), None)
    assert scout_permission is not None, "Scout permission for m_scouting not found."
    assert scout_permission["time_constraint"] == "Delayed", \
        "Permission not replaced when prioritise_local=False (expected 'Delayed')."

    # New permission ("Scout", "m_recon") should be added.
    recon_permission = next(
        (p for p in local_ds["permissions"] if p["role_name"] == "Scout" and p["mission_name"] == "m_recon"), None)
    assert recon_permission is not None, "New permission m_recon not added."

    # The duplicate obligation for "Patroller" should be replaced.
    patroller_obligation = next(
        (o for o in local_ds["obligations"] if o["role_name"] == "Patroller" and o["mission_name"] == "m_patrolling"), None)
    assert patroller_obligation is not None, "Patroller obligation for m_patrolling not found."
    assert patroller_obligation["time_constraint"] == "Immediate", \
        "Obligation not replaced when prioritise_local=False (expected 'Immediate')."

    # New obligation ("Monitor", "m_analysis") should be added.
    monitor_obligation = next(
        (o for o in local_ds["obligations"] if o["role_name"] == "Monitor" and o["mission_name"] == "m_analysis"), None)
    assert monitor_obligation is not None, "New obligation m_analysis not added."

    print("Test 2 passed.")

    print("All tests passed.")
