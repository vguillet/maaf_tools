
##################################################################################################################

import json
from pprint import pprint
import warnings
#from typing import Self

try:
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseComponent import MoiseComponent

except:
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseComponent import MoiseComponent

##################################################################################################################


class DeonticSpecification(dict, MoiseComponent):
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
    def check_specification_definition(deontic_specification, verbose: int = 1) -> bool:
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
            goal_spec = self.functional_specification.get_goal(goal)
            if goal_spec is not None:
                if "skills" in goal_spec:
                    skills.extend(goal_spec["skills"])

        skills = set(skills)

        return list(skills)

    # ============================================================== Set

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

    deontic_spec = DeonticSpecification()
    deontic_spec.add_obligation(role="Scout", mission="m_scouting")
    deontic_spec.add_obligation(role="Monitor", mission="m_monitoring")
    deontic_spec.add_obligation(role="Patroller", mission="m_patrolling")
    deontic_spec.add_obligation(role="Obstructor", mission="m_interdiction")
    deontic_spec.add_obligation(role="Obstructor", mission="m_obstructing")
    deontic_spec.add_obligation(role="Trapper", mission="m_interdiction")
    deontic_spec.add_obligation(role="Trapper", mission="m_trapping")
    deontic_spec.add_obligation(role="Tracker", mission="m_tracking")
    deontic_spec.add_obligation(role="Neutraliser", mission="m_neutralising")

    #pprint(deontic_spec.asdict())

    deontic_spec.get_role_skill_requirements("Scout")

    deontic_spec.save_to_file("icare_alloc_config/icare_alloc_config/moise_deontic_specification.json")