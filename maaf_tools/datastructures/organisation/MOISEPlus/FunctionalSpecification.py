
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
    def get_social_scheme(self, social_scheme_id: str):
        """
        Returns the social scheme with the given name, or None if not found.

        :param social_scheme_id: The name of the social scheme.
        """
        for scheme in self["social_schemes"]:
            if scheme["id"] == social_scheme_id:
                return scheme
        return None

    def get_mission(self, mission_id: str):
        """
        Returns the mission with the given name, or None if not found.

        :param mission_id: The name of the mission.
        :return: The mission dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["id"] == mission_id:
                    return mission
        return None

    def get_goal(self, goal_id: str):
        """
        Returns the goal with the given name, or None if not found.

        :param goal_id: The name of the goal.
        :return: The goal dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                if goal["id"] == goal_id:
                    return goal
        return None

    def get_plan(self, plan_id: str):
        """
        Returns the plan with the given name, or None if not found.

        :param plan_id: The name of the plan.
        :return: The plan dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for plan in scheme["plans"]:
                if plan["id"] == plan_id:
                    return plan
        return None

    # ============================================================== Set

    # ============================================================== Add
    def add_social_scheme(self,
                          social_scheme_id: str,
                          description: str = None,
                          goals: list = None,
                          plans: list = None,
                          missions: list = None,
                          **kwargs
                          ):
        """
        Adds a social scheme to the functional specification.

        :param social_scheme_id: The name of the social scheme.
        :param description: A description of the social scheme.
        :param goals: A list of goal dictionaries.
        :param plans: A list of plan dictionaries.
        :param missions: A list of mission dictionaries.
        """

        # -> Check social scheme format
        # Check if the social scheme already exists
        existing_scheme = self.get_social_scheme(social_scheme_id)
        if existing_scheme is not None:
            raise ValueError(f"Social scheme with ID '{social_scheme_id}' already exists in the functional specification.")

        # -> Construct the social scheme
        scheme = {
            "id": social_scheme_id,
            "description": description,
            "goals": [],
            "plans": [],
            "missions": [],
            "args": kwargs
        }

        # -> Add the social scheme to the functional specification
        self["social_schemes"].append(scheme)

        # -> Add goals
        if goals is not None:
            for goal in goals:
                # Separate **kwargs from the goal dictionary
                goal_kwargs = {k: v for k, v in goal.items() if k not in ["id", "description", "skill_requirements"]}

                self.add_goal(
                    social_scheme_id=social_scheme_id,
                    goal_id=goal["id"],
                    description=goal["description"],
                    skill_requirements=goal["skill_requirements"],
                    **goal_kwargs
                )

        # -> Add plans
        if plans is not None:
            for plan in plans:
                # Separate **kwargs from the plan dictionary
                plan_kwargs = {k: v for k, v in plan.items() if k not in ["id", "goal_sequence"]}

                self.add_plan(
                    social_scheme_id=social_scheme_id,
                    plan_id=plan["id"],
                    goal_sequence=plan["goal_sequence"],
                    **plan_kwargs
                )

        # -> Add missions
        if missions is not None:
            for mission in missions:
                # Separate **kwargs from the mission dictionary
                mission_kwargs = {k: v for k, v in mission.items() if k not in ["id", "description", "goals", "assignment_cardinality"]}

                self.add_mission(
                    social_scheme_id=social_scheme_id,
                    mission_id=mission["id"],
                    description=mission["description"],
                    goals=mission["goals"],
                    assignment_cardinality=mission["assignment_cardinality"],
                    **mission_kwargs
                )


    def add_goal(self,
                 social_scheme_id: str,
                 goal_id: str,
                 description: str,
                 skill_requirements: list,
                 **kwargs
                 ):
        """
        Adds a goal under the specified social scheme.

        :param social_scheme_id: The name of the social scheme to which the goal belongs.
        :param goal_id: The ID of the goal.
        :param description: The description of the goal.
        :param skill_requirements: The skill requirements for the goal.
        :param kwargs: Additional arguments for the goal.
        """

        # -> Check goal format
        # Check if the goal already exists
        existing_goal = self.get_goal(goal_id)
        if existing_goal is not None:
            raise ValueError(f"Goal with ID '{goal_id}' already exists in the functional specification.")

        # -> Construct the goal
        goal = {
            "id": goal_id,
            "description": description,
            "skill_requirements": skill_requirements,
            **kwargs
        }

        # -> Add the goal to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_id)
        if scheme is not None:
            scheme["goals"].append(goal)
        else:
            raise ValueError(f"Social scheme '{social_scheme_id}' not found.")

    def add_mission(self,
                    social_scheme_id: str,
                    mission_id: str,
                    description: str,
                    goals: list,
                    assignment_cardinality: dict,
                    **kwargs
                    ):
        """
        Adds a mission under the specified social scheme.

        :param social_scheme_id: The name of the social scheme to which the mission belongs.
        :param mission_id: The ID of the mission.
        :param description: The description of the mission.
        :param goals: The goals associated with the mission.
        :param assignment_cardinality: The assignment cardinality for the mission.
        :param kwargs: Additional arguments for the mission.
        """

        # -> Check mission format
        # Check if the mission already exists
        existing_mission = self.get_mission(mission_id)
        if existing_mission is not None:
            raise ValueError(f"Mission with ID '{mission_id}' already exists in the functional specification.")

        # Check if the assignment_cardinality has the required keys
        required_keys = ["min", "max"]
        for key in required_keys:
            if key not in assignment_cardinality:
                raise ValueError(f"Mission assignment_cardinality must contain the key '{key}'.")

        # Check if the min is an integer
        if not isinstance(assignment_cardinality["min"], int):
            raise ValueError("Mission assignment_cardinality min must be an integer.")
        # Check if the max is an integer or None
        if not isinstance(assignment_cardinality["max"], (int, type(None))):
            raise ValueError("Mission assignment_cardinality max must be an integer or None.")

        # Check if the min is greater than or equal to 0
        if assignment_cardinality["min"] < 0:
            raise ValueError("Mission assignment_cardinality min must be greater than or equal to 0.")
        # Check if the max is greater than or equal to the min
        if assignment_cardinality["max"] is not None and assignment_cardinality["max"] < assignment_cardinality["min"]:
            raise ValueError("Mission assignment_cardinality max must be greater than or equal to min.")

        # Check if the goals are valid
        for goal_id in goals:
            goal = self.get_goal(goal_id)
            if goal is None:
                raise ValueError(f"Goal with ID '{goal_id}' not found in the functional specification.")

        # -> Construct the mission
        mission = {
            "id": mission_id,
            "description": description,
            "goals": goals,
            "assignment_cardinality": assignment_cardinality,
            **kwargs
        }

        # -> Add the mission to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_id)
        if scheme is not None:
            scheme["missions"].append(mission)
        else:
            raise ValueError(f"Social scheme '{social_scheme_id}' not found.")

    def add_plan(self,
                 social_scheme_id: str,
                 plan_id: str,
                 goal_sequence: list,
                 **kwargs
                 ):
        """
        Adds a plan under the specified social scheme.

        :param social_scheme_id: The name of the social scheme to which the plan belongs.
        :param plan_id: The ID of the plan.
        :param goal_sequence: The goal sequence for the plan.
        :param kwargs: Additional arguments for the plan.
        """

        # -> Check plan format
        # Check if a plan already exists with the same ID
        existing_plan = self.get_plan(plan_id)
        if existing_plan is not None:
            raise ValueError(f"Plan with ID '{plan_id}' already exists in the functional specification.")

        # Check if the keys in the plan are valid goal IDs
        for goal in goal_sequence:
            if not isinstance(goal, str):
                raise ValueError(f"Plan goal ID must be a string. Found: {goal}")
            goal_spec = self.get_goal(goal)
            if goal_spec is None:
                raise ValueError(f"Goal with ID '{goal}' not found in the functional specification.")

        # -> Construct the plan
        plan = {
            "id": plan_id,
            "goal_sequence": goal_sequence,
            **kwargs
        }

        # -> Add the plan to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_id)
        if scheme is not None:
            scheme["plans"].append(plan)
        else:
            raise ValueError(f"Social scheme '{social_scheme_id}' not found.")

    # ============================================================== Remove
    def remove_social_scheme(self, social_scheme_id: str) -> bool:
        """
        Removes a social scheme by name. Return True if the scheme was removed, False otherwise.

        :param social_scheme_id: The name of the social scheme to remove.
        :return: True if the scheme was removed, False otherwise.
        """

        for scheme in self["social_schemes"]:
            if scheme["id"] == social_scheme_id:
                self["social_schemes"].remove(scheme)
                return True
        return False

    def remove_goal(self, goal_id: str) -> bool:
        """
        Removes a goal. Return True if the goal was removed, False otherwise.

        :param goal_id: The ID of the goal to remove.
        :return: True if the goal was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                if goal["id"] == goal_id:
                    scheme["goals"].remove(goal)
                    return True
        return False

    def remove_mission(self, mission_id: str) -> bool:
        """
        Removes a mission. Return True if the mission was removed, False otherwise.

        :param mission_id: The ID of the mission to remove.
        :return: True if the mission was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["id"] == mission_id:
                    scheme["missions"].remove(mission)
                    return True
        return False

    def remove_plan(self, plan_id: str) -> bool:
        """
        Removes a plan. Return True if the plan was removed, False otherwise.

        :param plan_id: The ID of the plan to remove.
        :return: True if the plan was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for plan in scheme["plans"]:
                if plan["id"] == plan_id:
                    scheme["plans"].remove(plan)
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
        social_scheme_id="Operational Agents: Surveillance and Interdiction",
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
        plans=[
            #{"id": "g_point_obs", "goal_sequence": ["g_point_obs"]},
        ],
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