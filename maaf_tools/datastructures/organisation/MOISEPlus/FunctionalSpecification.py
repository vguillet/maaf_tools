
##################################################################################################################

import json
from pprint import pprint
import warnings

from bloom.generators.rpm.generate_cmd import description

#from typing import Self

try:
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseComponent import MoiseComponent

except:
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseComponent import MoiseComponent

##################################################################################################################


class FunctionalSpecification(dict, MoiseComponent):
    def __init__(self,
                 functional_specification: dict or None = None,
                 structural_specification = None,
                 deontic_specification = None
                 ):
        # If no specification is provided, use the default template.
        if functional_specification is None:
            functional_specification = {
                "social_schemes": []
            }

        elif isinstance(functional_specification, FunctionalSpecification):
            functional_specification = functional_specification.copy()

        elif isinstance(functional_specification, dict):
            functional_specification = functional_specification.copy()

        elif not isinstance(functional_specification, dict) or not isinstance(functional_specification, FunctionalSpecification):
            raise ValueError("The functional specification must be a dictionary or FunctionalSpecification object.")

        if not self.check_specification_definition(functional_specification=functional_specification, verbose=1):
            raise ValueError("The provided functional specification is not valid")

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(functional_specification)

        # -> Initialize the structural and functional specifications
        self.__structural_specification = structural_specification
        self.__deontic_specification = deontic_specification

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
    def deontic_specification(self):
        if self.__deontic_specification is None:
            warnings.warn("Deontic specification is not set.")
            return None
        return self.__deontic_specification

    @deontic_specification.setter
    def deontic_specification(self, deontic_specification):
        self.__deontic_specification = deontic_specification

    # ============================================================== Check
    @staticmethod
    def check_specification_definition(functional_specification, verbose: int = 1) -> bool:
        return True

    # ============================================================== Get
    def get_social_scheme(self, name: str):
        """
        Returns the social scheme with the given name, or None if not found.
        """
        for scheme in self["social_schemes"]:
            if scheme["name"] == name:
                return scheme
        return None

    def get_mission(self, name: str):
        """
        Returns the mission with the given name, or None if not found.

        :param name: The name of the mission.
        :return: The mission dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["id"] == name:
                    return mission
        return None

    def get_goal(self, name: str):
        """
        Returns the goal with the given name, or None if not found.

        :param name: The name of the goal.
        :return: The goal dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                if goal["id"] == name:
                    return goal
        return None

    # ============================================================== Set

    # ============================================================== Add
    def add_social_scheme(self,
                          name: str,
                          description: str = None,
                          goals=None,
                          plans=None,
                          missions=None,
                          preferences=None
                          ):
        """
        Adds a social scheme to the functional specification.

        Args:
            name (str): The name of the social scheme.
            description (str, optional): A description of the social scheme.
            goals (list, optional): A list of goal dictionaries.
            plans (list, optional): A list of plan dictionaries.
            missions (list, optional): A list of mission dictionaries.
            preferences (list, optional): A list of preference dictionaries.
        """
        scheme = {
            "name": name,
            "description": description,
            "goals": goals if goals is not None else [],
            "plans": plans if plans is not None else [],
            "missions": missions if missions is not None else [],
            "preferences": preferences if preferences is not None else []
        }

        self["social_schemes"].append(scheme)

    # ============================================================== Remove
    def remove_social_scheme(self, name):
        """Removes a social scheme by name. Return True if the scheme was removed, False otherwise."""

        for scheme in self["social_schemes"]:
            if scheme["name"] == name:
                self["social_schemes"].remove(scheme)
                return True
        return False

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self


if __name__ == "__main__":
    functional_spec = FunctionalSpecification()
    functional_spec.add_social_scheme(
        name="Operational Agents: Surveillance and Interdiction",
        description="A comprehensive set of surveillance and interdiction objectives",
        goals=[
            {"id": "g_point_obs", "description": "A point was observed", "skill_requirements": ["goto"]},
            {"id": "g_axis_obs", "description": "An axis was observed", "skill_requirements": ["goto"]},
            {"id": "g_zone_obs", "description": "A zone was observed", "skill_requirements": ["goto"]},
            {"id": "g_point_mon", "description": "A point is monitored", "skill_requirements": ["goto"]},
            {"id": "g_axis_mon", "description": "An axis is monitored", "skill_requirements": ["goto"]},
            {"id": "g_axis_patrol", "description": "An axis is patrolled", "skill_requirements": ["goto"]},
            {"id": "g_zone_patrol", "description": "A zone is patrolled", "skill_requirements": ["goto"]},
            {"id": "g_path_interdict", "description": "A path is interdicted (Obstructed or Trapped)", "skill_requirements": ["goto"]},
            {"id": "g_path_obstruct", "description": "A path is obstructed", "skill_requirements": ["goto"]},
            {"id": "g_path_trap", "description": "A path is trapped", "skill_requirements": ["goto", "trap"]},
            {"id": "g_target_track", "description": "A target is tracked", "skill_requirements": ["track"]},
            {"id": "g_target_neutral", "description": "A target is neutralised", "skill_requirements": ["neutralise"]}
        ],
        plans=[],
        missions=[
            {"id": "m_scouting", "description": "Scouting mission", "goals": ["g_point_obs", "g_axis_obs", "g_zone_obs"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_monitoring", "description": "Monitoring mission", "goals": ["g_point_mon", "g_axis_mon"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_patrolling", "description": "Patrolling mission", "goals": ["g_axis_patrol", "g_zone_patrol"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_interdiction", "description": "Interdiction mission", "goals": ["g_path_interdict"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_obstructing", "description": "Obstructing mission", "goals": ["g_path_obstruct"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_trapping", "description": "Trapping mission", "goals": ["g_path_trap"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_tracking", "description": "Tracking mission", "goals": ["g_target_track"], "assignment_cardinality": {"min": 1, "max": None}},
            {"id": "m_neutralising", "description": "Neutralising mission", "goals": ["g_target_neutral"], "assignment_cardinality": {"min": 1, "max": None}}
        ],
    )

    print(functional_spec.asdict())

    functional_spec.save_to_file("icare_alloc_config/icare_alloc_config/moise_functional_specification.json")