
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

    # ============================================================== Get
    def get_role_skill_requirements(self, role: str) -> list:
        """
        Returns the skill requirements for a given role. The skill requirements are determined based on the
        goal requirements associated with the missions the role is responsible for (permissions and obligations).

        :param role: The role for which to retrieve skill requirements.
        :return : A list of skill requirements for the specified role.
        """

        if self.functional_specification is None:
            warnings.warn("Functional specification is not set.")
            return []

        # -> Get all missions associated with the role
        missions = []
        for permission in self["permissions"]:
            if permission["role"] == role:
                missions.append(permission["mission"])

        for obligation in self["obligations"]:
            if obligation["role"] == role:
                missions.append(obligation["mission"])

        # -> Get all goals associated with the missions
        goals = []
        for mission in missions:
            mission_spec = self.functional_specification.get_mission(mission)
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
            role = incoming_permission.get("role")
            mission = incoming_permission.get("mission")
            # Search for an existing permission with the same role and mission.
            existing_permission = next(
                (perm for perm in self.get("permissions", [])
                 if perm.get("role") == role and perm.get("mission") == mission), None)

            if existing_permission:
                # If local is prioritized, do nothing (preserve the local permission).
                # Otherwise, replace the local permission with the incoming one.
                if not prioritise_local:
                    self.remove_permission(role, mission)
                    self.add_permission(role, mission,
                                        time_constraint=incoming_permission.get("time_constraint", "Any"))
            else:
                # Add new permission from incoming specification.
                self.add_permission(role, mission,
                                    time_constraint=incoming_permission.get("time_constraint", "Any"))

        # ----- Merge Obligations -----
        # Iterate over each incoming obligation.
        for incoming_obligation in deontic_specification.get("obligations", []):
            role = incoming_obligation.get("role")
            mission = incoming_obligation.get("mission")
            # Find an existing obligation with the same role and mission.
            existing_obligation = next(
                (obl for obl in self.get("obligations", [])
                 if obl.get("role") == role and obl.get("mission") == mission), None)

            if existing_obligation:
                if not prioritise_local:
                    self.remove_obligation(role, mission)
                    self.add_obligation(role, mission,
                                        time_constraint=incoming_obligation.get("time_constraint", "Any"))
            else:
                self.add_obligation(role, mission,
                                    time_constraint=incoming_obligation.get("time_constraint", "Any"))

        # Final validation of the merged specification.
        if not self.check_specification_definition(deontic_specification=self, stop_at_first_error=True):
            raise ValueError("The merged deontic specification is not valid.")

        return True

    # ============================================================== Add
    def add_permission(self, role, mission, time_constraint="Any"):
        """
        Adds a permission to the deontic specification.

        Args:
            role (str): The role associated with the permission.
            mission (str): The mission to which the permission applies.
            time_constraint (str): The time constraint (default "Any").
        """
        permission = {"role": role, "mission": mission, "time_constraint": time_constraint}
        self["permissions"].append(permission)

    def add_obligation(self, role, mission, time_constraint="Any"):
        """
        Adds an obligation to the deontic specification.

        Args:
            role (str): The role associated with the obligation.
            mission (str): The mission to which the obligation applies.
            time_constraint (str): The time constraint (default "Any").
        """
        obligation = {"role": role, "mission": mission, "time_constraint": time_constraint}
        self["obligations"].append(obligation)

    # ============================================================== Remove
    def remove_permission(self, role, mission):
        """Removes a permission. Return True if the permission was removed, False otherwise."""

        for permission in self["permissions"]:
            if permission["role"] == role and permission["mission"] == mission:
                self["permissions"].remove(permission)
                return True
        return False

    def remove_obligation(self, role, mission):
        """Removes an obligation. Return True if the obligation was removed, False otherwise."""

        for obligation in self["obligations"]:
            if obligation["role"] == role and obligation["mission"] == mission:
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
    # Define a local deontic specification dictionary.
    local_deontic_spec_dict = {
        "permissions": [
            {"role": "Scout", "mission": "m_scouting", "time_constraint": "Immediate"},
            {"role": "Monitor", "mission": "m_monitoring", "time_constraint": "Any"}
        ],
        "obligations": [
            {"role": "Patroller", "mission": "m_patrolling", "time_constraint": "Scheduled"},
            {"role": "Obstructor", "mission": "m_interdiction", "time_constraint": "Any"}
        ]
    }

    # Define an incoming deontic specification dictionary.
    incoming_deontic_spec_dict = {
        "permissions": [
            # New permission not in local.
            {"role": "Scout", "mission": "m_recon", "time_constraint": "Urgent"},
            # Duplicate permission: same role and mission as local, but different time_constraint.
            {"role": "Scout", "mission": "m_scouting", "time_constraint": "Delayed"}
        ],
        "obligations": [
            # New obligation.
            {"role": "Monitor", "mission": "m_analysis", "time_constraint": "Any"},
            # Duplicate obligation: same role and mission as local, but different time_constraint.
            {"role": "Patroller", "mission": "m_patrolling", "time_constraint": "Immediate"}
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
        (p for p in local_ds["permissions"] if p["role"] == "Scout" and p["mission"] == "m_scouting"), None)
    assert scout_permission is not None, "Scout permission for m_scouting not found."
    assert scout_permission["time_constraint"] == "Immediate", \
        "Local permission not preserved when prioritise_local=True (expected 'Immediate')."

    # New permission ("Scout", "m_recon") should be added.
    recon_permission = next(
        (p for p in local_ds["permissions"] if p["role"] == "Scout" and p["mission"] == "m_recon"), None)
    assert recon_permission is not None, "New permission m_recon not added."

    # Obligation ("Patroller", "m_patrolling") should remain unchanged.
    patroller_obligation = next(
        (o for o in local_ds["obligations"] if o["role"] == "Patroller" and o["mission"] == "m_patrolling"), None)
    assert patroller_obligation is not None, "Patroller obligation for m_patrolling not found."
    assert patroller_obligation["time_constraint"] == "Scheduled", \
        "Local obligation not preserved when prioritise_local=True (expected 'Scheduled')."

    # New obligation ("Monitor", "m_analysis") should be added.
    monitor_obligation = next(
        (o for o in local_ds["obligations"] if o["role"] == "Monitor" and o["mission"] == "m_analysis"), None)
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
        (p for p in local_ds["permissions"] if p["role"] == "Scout" and p["mission"] == "m_scouting"), None)
    assert scout_permission is not None, "Scout permission for m_scouting not found."
    assert scout_permission["time_constraint"] == "Delayed", \
        "Permission not replaced when prioritise_local=False (expected 'Delayed')."

    # New permission ("Scout", "m_recon") should be added.
    recon_permission = next(
        (p for p in local_ds["permissions"] if p["role"] == "Scout" and p["mission"] == "m_recon"), None)
    assert recon_permission is not None, "New permission m_recon not added."

    # The duplicate obligation for "Patroller" should be replaced.
    patroller_obligation = next(
        (o for o in local_ds["obligations"] if o["role"] == "Patroller" and o["mission"] == "m_patrolling"), None)
    assert patroller_obligation is not None, "Patroller obligation for m_patrolling not found."
    assert patroller_obligation["time_constraint"] == "Immediate", \
        "Obligation not replaced when prioritise_local=False (expected 'Immediate')."

    # New obligation ("Monitor", "m_analysis") should be added.
    monitor_obligation = next(
        (o for o in local_ds["obligations"] if o["role"] == "Monitor" and o["mission"] == "m_analysis"), None)
    assert monitor_obligation is not None, "New obligation m_analysis not added."

    print("Test 2 passed.")

    print("All tests passed.")


# if __name__ == "__main__":
#
#     deontic_spec = DeonticSpecification()
#     deontic_spec.add_obligation(role="Scout", mission="m_scouting")
#     deontic_spec.add_obligation(role="Monitor", mission="m_monitoring")
#     deontic_spec.add_obligation(role="Patroller", mission="m_patrolling")
#     deontic_spec.add_obligation(role="Obstructor", mission="m_interdiction")
#     deontic_spec.add_obligation(role="Obstructor", mission="m_obstructing")
#     deontic_spec.add_obligation(role="Trapper", mission="m_interdiction")
#     deontic_spec.add_obligation(role="Trapper", mission="m_trapping")
#     deontic_spec.add_obligation(role="Tracker", mission="m_tracking")
#     deontic_spec.add_obligation(role="Neutraliser", mission="m_neutralising")
#
#     #pprint(deontic_spec.asdict())
#
#     deontic_spec.get_role_skill_requirements("Scout")
#
#     deontic_spec.save_to_file("icare_alloc_config/icare_alloc_config/moise_deontic_specification.json")