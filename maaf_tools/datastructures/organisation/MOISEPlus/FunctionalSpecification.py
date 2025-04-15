
##################################################################################################################

import json
from pprint import pprint
import warnings

# from typing import Self

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


class FunctionalSpecification(dict, MaafItem):
    """
    A class representing the functional specification of a MOISE+ model.
    The functional specification provides the mapping between the system's goals, plans, and missions.
    """

    def __init__(self,
                 functional_specification: dict or "FunctionalSpecification" = None,
                 structural_specification=None,
                 deontic_specification=None
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
            raise ValueError(
                "The functional specification must be a dictionary or FunctionalSpecification object.")

        if not self.check_specification_definition(functional_specification=functional_specification, verbose=1):
            raise ValueError(
                "The provided functional specification is not valid")

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

    @property
    def goals(self) -> list[str]:
        """
        Returns the list of goals types in the functional specification.
        """
        goals = []
        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                goals.append(goal["name"])
        return goals

    # ============================================================== Check
    @staticmethod
    def check_specification_definition(functional_specification, stop_at_first_error: bool = False, verbose: int = 1) -> bool:
        warnings.warn(
            "Checking functional specification definition is not implemented yet.")
        return True

    # ============================================================== Get
    def get_plan(self, plan_type: str):
        """
        Returns the plan with the given name, or None if not found.

        :param plan_type: The name of the plan.
        :return: The plan dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for plan in scheme["plans"]:
                if plan["name"] == plan_type:
                    return plan
        return None

    def get_social_scheme(self, social_scheme_type: str):
        """
        Returns the social scheme with the given name, or None if not found.

        :param social_scheme_type: The name of the social scheme.
        """
        for scheme in self["social_schemes"]:
            if scheme["name"] == social_scheme_type:
                return scheme
        return None

    def get_goal(self, goal_name: str):
        """
        Returns the goal with the given name, or None if not found.

        :param goal_name: The name of the goal.
        :return: The goal dictionary if found, None otherwise.
        """

        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                if goal["name"] == goal_name:
                    return goal
        return None

    def get_mission(self, mission_type: str):
        """
        Returns the mission with the given name, or None if not found.

        :param mission_type: The name of the mission.
        :return: The mission dictionary if found, None otherwise.
        """
        for scheme in self["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["name"] == mission_type:
                    return mission
        return None

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self,
              functional_specification: "FunctionalSpecification",
              prioritise_local: bool = True) -> bool:
        """
        Merges the current functional specification with another one.
        If prioritise_local is True, the local specification takes precedence.
        """
        # Validate incoming specification
        if not isinstance(functional_specification, FunctionalSpecification):
            raise ValueError(
                "The functional specification must be a FunctionalSpecification object.")
        if not self.check_specification_definition(functional_specification=functional_specification, stop_at_first_error=True):
            raise ValueError(
                "The provided functional specification is not valid")

        # Iterate over incoming social schemes
        for incoming_scheme in functional_specification.get("social_schemes", []):
            scheme_type = incoming_scheme.get("name")
            # Check if the scheme already exists
            existing_scheme = self.get_social_scheme(scheme_type)
            if existing_scheme:
                if not prioritise_local:
                    # Replace the existing scheme entirely:
                    self.remove_social_scheme(scheme_type)
                    # Use add_social_scheme to add the new scheme,
                    # which will call add_goal, add_plan, add_mission internally.
                    self.add_social_scheme(
                        social_scheme_type=scheme_type,
                        description=incoming_scheme.get("description"),
                        goals=incoming_scheme.get("goals", []),
                        plans=incoming_scheme.get("plans", []),
                        missions=incoming_scheme.get("missions", []),
                        **incoming_scheme.get("args", {})
                    )
                else:
                    # Merge the sub-elements into the existing scheme:
                    # For goals, plans, and missions, check each incoming item.
                    for key, add_method in (("goals", self.add_goal),
                                            ("plans", self.add_plan),
                                            ("missions", self.add_mission)):
                        for item in incoming_scheme.get(key, []):
                            # Here, you might call the corresponding getter (get_goal, get_plan, get_mission)
                            # and if the item does not exist in the local scheme, then add it.
                            if key == "goals":
                                if self.get_goal(item["name"]) is None:
                                    add_method(
                                        social_scheme_type=scheme_type,
                                        goal_name=item["name"],
                                        description=item["description"],
                                        skill_requirements=item["skill_requirements"],
                                        **{k: v for k, v in item.items() if
                                           k not in ["name", "description", "skill_requirements"]}
                                    )
                            elif key == "plans":
                                if self.get_plan(item["name"]) is None:
                                    add_method(
                                        social_scheme_type=scheme_type,
                                        plan_type=item["name"],
                                        goal_sequence=item["goal_sequence"],
                                        **{k: v for k, v in item.items() if k not in ["name", "goal_sequence"]}
                                    )
                            elif key == "missions":
                                if self.get_mission(item["name"]) is None:
                                    add_method(
                                        social_scheme_type=scheme_type,
                                        mission_type=item["name"],
                                        description=item["description"],
                                        goals=item["goals"],
                                        assignment_cardinality=item["assignment_cardinality"],
                                        **{k: v for k, v in item.items() if
                                           k not in ["name", "description", "goals", "assignment_cardinality"]}
                                    )
                    # Optionally, update the description if it is missing locally.
                    local_scheme = self.get_social_scheme(scheme_type)
                    if not local_scheme.get("description") and incoming_scheme.get("description"):
                        local_scheme["description"] = incoming_scheme["description"]
            else:
                # Add a completely new social scheme using the add_social_scheme method.
                self.add_social_scheme(
                    social_scheme_type=scheme_type,
                    description=incoming_scheme.get("description"),
                    goals=incoming_scheme.get("goals", []),
                    plans=incoming_scheme.get("plans", []),
                    missions=incoming_scheme.get("missions", []),
                    **incoming_scheme.get("args", {})
                )

        # Validate the merged specification
        if not self.check_specification_definition(functional_specification=self, stop_at_first_error=True):
            raise ValueError(
                "The merged functional specification is not valid")

        return True

    # ============================================================== Add
    def add_social_scheme(self,
                          social_scheme_type: str,
                          description: str = None,
                          goals: list = None,
                          plans: list = None,
                          missions: list = None,
                          **kwargs
                          ):
        """
        Adds a social scheme to the functional specification.

        :param social_scheme_type: The name of the social scheme.
        :param description: A description of the social scheme.
        :param goals: A list of goal dictionaries.
        :param plans: A list of plan dictionaries.
        :param missions: A list of mission dictionaries.
        """

        # -> Check social scheme format
        # Check if the social scheme already exists
        existing_scheme = self.get_social_scheme(social_scheme_type)
        if existing_scheme is not None:
            raise ValueError(
                f"Social scheme with name '{social_scheme_type}' already exists in the functional specification.")

        # -> Construct the social scheme
        scheme = {
            "name": social_scheme_type,
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
                goal_kwargs = {k: v for k, v in goal.items() if k not in [
                    "name", "description", "skill_requirements"]}

                self.add_goal(
                    social_scheme_type=social_scheme_type,
                    goal_name=goal["name"],
                    description=goal["description"],
                    skill_requirements=goal["skill_requirements"],
                    **goal_kwargs
                )

        # -> Add plans
        if plans is not None:
            for plan in plans:
                # Separate **kwargs from the plan dictionary
                plan_kwargs = {k: v for k, v in plan.items() if k not in [
                    "name", "goal_sequence"]}

                self.add_plan(
                    social_scheme_type=social_scheme_type,
                    plan_type=plan["name"],
                    goal_sequence=plan["goal_sequence"],
                    **plan_kwargs
                )

        # -> Add missions
        if missions is not None:
            for mission in missions:
                # Separate **kwargs from the mission dictionary
                mission_kwargs = {k: v for k, v in mission.items() if k not in [
                    "name", "description", "goals", "assignment_cardinality"]}

                self.add_mission(
                    social_scheme_type=social_scheme_type,
                    mission_type=mission["name"],
                    description=mission["description"],
                    goals=mission["goals"],
                    assignment_cardinality=mission["assignment_cardinality"],
                    **mission_kwargs
                )

    def add_goal(self,
                 social_scheme_type: str,
                 goal_name: str,
                 description: str,
                 skill_requirements: list,
                 **kwargs
                 ):
        """
        Adds a goal under the specified social scheme.

        :param social_scheme_type: The name of the social scheme to which the goal belongs.
        :param goal_name: The name of the goal.
        :param description: The description of the goal.
        :param skill_requirements: The skill requirements for the goal.
        :param kwargs: Additional arguments for the goal.
        """

        # -> Check goal format
        # Check if the goal already exists
        existing_goal = self.get_goal(goal_name)
        if existing_goal is not None:
            raise ValueError(
                f"Goal with name '{goal_name}' already exists in the functional specification.")

        # -> Construct the goal
        goal = {
            "name": goal_name,
            "description": description,
            "skill_requirements": skill_requirements,
            **kwargs
        }

        # -> Add the goal to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_type)
        if scheme is not None:
            scheme["goals"].append(goal)
        else:
            raise ValueError(
                f"Social scheme '{social_scheme_type}' not found.")

    def add_mission(self,
                    social_scheme_type: str,
                    mission_type: str,
                    description: str,
                    goals: list,
                    assignment_cardinality: dict,
                    **kwargs
                    ):
        """
        Adds a mission under the specified social scheme.

        :param social_scheme_type: The name of the social scheme to which the mission belongs.
        :param mission_type: The name of the mission.
        :param description: The description of the mission.
        :param goals: The goals associated with the mission.
        :param assignment_cardinality: The assignment cardinality for the mission.
        :param kwargs: Additional arguments for the mission.
        """

        # -> Check mission format
        # Check if the mission already exists
        existing_mission = self.get_mission(mission_type)
        if existing_mission is not None:
            raise ValueError(
                f"Mission with name '{mission_type}' already exists in the functional specification.")

        # Check if the assignment_cardinality has the required keys
        required_keys = ["min", "max"]
        for key in required_keys:
            if key not in assignment_cardinality:
                raise ValueError(
                    f"Mission assignment_cardinality must contain the key '{key}'.")

        # Check if the min is an integer
        if not isinstance(assignment_cardinality["min"], int):
            raise ValueError(
                "Mission assignment_cardinality min must be an integer.")
        # Check if the max is an integer or None
        if not isinstance(assignment_cardinality["max"], (int, type(None))):
            raise ValueError(
                "Mission assignment_cardinality max must be an integer or None.")

        # Check if the min is greater than or equal to 0
        if assignment_cardinality["min"] < 0:
            raise ValueError(
                "Mission assignment_cardinality min must be greater than or equal to 0.")
        # Check if the max is greater than or equal to the min
        if assignment_cardinality["max"] is not None and assignment_cardinality["max"] < assignment_cardinality["min"]:
            raise ValueError(
                "Mission assignment_cardinality max must be greater than or equal to min.")

        # Check if the goals are valid
        for goal_name in goals:
            goal = self.get_goal(goal_name)
            if goal is None:
                raise ValueError(
                    f"Goal with name '{goal_name}' not found in the functional specification.")

        # -> Construct the mission
        mission = {
            "name": mission_type,
            "description": description,
            "goals": goals,
            "assignment_cardinality": assignment_cardinality,
            **kwargs
        }

        # -> Add the mission to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_type)
        if scheme is not None:
            scheme["missions"].append(mission)
        else:
            raise ValueError(
                f"Social scheme '{social_scheme_type}' not found.")

    def add_plan(self,
                 social_scheme_type: str,
                 plan_type: str,
                 goal_sequence: list,
                 **kwargs
                 ):
        """
        Adds a plan under the specified social scheme.

        :param social_scheme_type: The name of the social scheme to which the plan belongs.
        :param plan_type: The name of the plan.
        :param goal_sequence: The goal sequence for the plan.
        :param kwargs: Additional arguments for the plan.
        """

        # -> Check plan format
        # Check if a plan already exists with the same name
        existing_plan = self.get_plan(plan_type)
        if existing_plan is not None:
            raise ValueError(
                f"Plan with name '{plan_type}' already exists in the functional specification.")

        # Check if the keys in the plan are valid goal IDs
        for goal in goal_sequence:
            if not isinstance(goal, str):
                raise ValueError(
                    f"Plan goal name must be a string. Found: {goal}")
            goal_spec = self.get_goal(goal)
            if goal_spec is None:
                raise ValueError(
                    f"Goal with name '{goal}' not found in the functional specification.")

        # -> Construct the plan
        plan = {
            "name": plan_type,
            "goal_sequence": goal_sequence,
            **kwargs
        }

        # -> Add the plan to the specified social scheme
        scheme = self.get_social_scheme(social_scheme_type)
        if scheme is not None:
            scheme["plans"].append(plan)
        else:
            raise ValueError(
                f"Social scheme '{social_scheme_type}' not found.")

    # ============================================================== Remove
    def remove_social_scheme(self, social_scheme_type: str) -> bool:
        """
        Removes a social scheme by name. Return True if the scheme was removed, False otherwise.

        :param social_scheme_type: The name of the social scheme to remove.
        :return: True if the scheme was removed, False otherwise.
        """

        for scheme in self["social_schemes"]:
            if scheme["name"] == social_scheme_type:
                self["social_schemes"].remove(scheme)
                return True
        return False

    def remove_goal(self, goal_name: str) -> bool:
        """
        Removes a goal. Return True if the goal was removed, False otherwise.

        :param goal_name: The name of the goal to remove.
        :return: True if the goal was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for goal in scheme["goals"]:
                if goal["name"] == goal_name:
                    scheme["goals"].remove(goal)
                    return True
        return False

    def remove_mission(self, mission_type: str) -> bool:
        """
        Removes a mission. Return True if the mission was removed, False otherwise.

        :param mission_type: The name of the mission to remove.
        :return: True if the mission was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for mission in scheme["missions"]:
                if mission["name"] == mission_type:
                    scheme["missions"].remove(mission)
                    return True
        return False

    def remove_plan(self, plan_type: str) -> bool:
        """
        Removes a plan. Return True if the plan was removed, False otherwise.

        :param plan_type: The name of the plan to remove.
        :return: True if the plan was removed, False otherwise.
        """
        for scheme in self["social_schemes"]:
            for plan in scheme["plans"]:
                if plan["name"] == plan_type:
                    scheme["plans"].remove(plan)
                    return True
        return False

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Returns the structural specification as a dictionary.
        """
        return self


# Test Script with Asserts to verify the merge method for FunctionalSpecification
if __name__ == "__main__":
    functional_spec = FunctionalSpecification()
    functional_spec.add_social_scheme(
        social_scheme_type="Operational Agents: Surveillance and Interdiction",
        description="A comprehensive set of surveillance and interdiction objectives",
        goals=[
            {"name": "g_point_obs", "description": "A point was observed", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_axis_obs", "description": "An axis was observed", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_zone_obs", "description": "A zone was observed", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_point_mon", "description": "A point is monitored", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_axis_mon", "description": "An axis is monitored", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_axis_patrol", "description": "An axis is patrolled", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_zone_patrol", "description": "A zone is patrolled", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_path_interdict", "description": "A path is interdicted (Obstructed or Trapped)", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_path_obstruct", "description": "A path is obstructed", "skill_requirements": [
                "goto"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_path_trap", "description": "A path is trapped", "skill_requirements": [
                "goto", "trap"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_target_track", "description": "A target is tracked", "skill_requirements": [
                "track"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"},
            {"name": "g_target_neutral", "description": "A target is neutralised", "skill_requirements": [
                "neutralise"], "bidding_logic": "GraphWeightedManhattanDistanceBundleBid"}
        ],
        plans=[
            # {"name": "g_point_obs", "goal_sequence": ["g_point_obs"]},
        ],
        missions=[
            {"name": "m_scouting", "description": "Scouting mission", "goals": [
                "g_point_obs", "g_axis_obs", "g_zone_obs"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_monitoring", "description": "Monitoring mission", "goals": [
                "g_point_mon", "g_axis_mon"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_patrolling", "description": "Patrolling mission", "goals": [
                "g_axis_patrol", "g_zone_patrol"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_interdiction", "description": "Interdiction mission", "goals": [
                "g_path_interdict"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_obstructing", "description": "Obstructing mission", "goals": [
                "g_path_obstruct"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_trapping", "description": "Trapping mission", "goals": [
                "g_path_trap"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_tracking", "description": "Tracking mission", "goals": [
                "g_target_track"], "assignment_cardinality": {"min": 1, "max": None}},
            {"name": "m_neutralising", "description": "Neutralising mission", "goals": [
                "g_target_neutral"], "assignment_cardinality": {"min": 1, "max": None}}
        ],
    )

    functional_spec.save_to_file(
        "icare_alloc_config/icare_alloc_config/moise_functional_specification.json")
    # print(functional_spec.goals)

    # Create a local functional specification with one social scheme.
    local_func_spec_dict = {
        "social_schemes": [
            {
                "name": "SchemeA",
                "description": "Local scheme A description.",
                "goals": [
                    {"name": "goal1", "description": "Local goal 1",
                        "skill_requirements": ["a"]},
                ],
                "plans": [
                    {"name": "plan1", "goal_sequence": ["goal1"]},
                ],
                "missions": [
                    {"name": "mission1", "description": "Local mission 1", "goals": ["goal1"],
                     "assignment_cardinality": {"min": 1, "max": 2}}
                ],
                "args": {}
            }
        ]
    }

    # Create an incoming functional specification with:
    # - A scheme with the same name "SchemeA" but with a new goal, plan, and mission.
    # - A new scheme "SchemeB".
    incoming_func_spec_dict = {
        "social_schemes": [
            {
                "name": "SchemeA",
                # Only used if prioritise_local is False.
                "description": "Incoming scheme A description.",
                "goals": [
                    {"name": "goal2", "description": "Incoming goal 2",
                        "skill_requirements": ["b"]},
                ],
                "plans": [
                    {"name": "plan2", "goal_sequence": ["goal2"]},
                ],
                "missions": [
                    {"name": "mission2", "description": "Incoming mission 2", "goals": ["goal2"],
                     "assignment_cardinality": {"min": 1, "max": None}}
                ],
                "args": {}
            },
            {
                "name": "SchemeB",
                "description": "Incoming scheme B description.",
                "goals": [
                    {"name": "goal3", "description": "Incoming goal 3",
                        "skill_requirements": ["c"]},
                ],
                "plans": [],
                "missions": [],
                "args": {}
            }
        ]
    }

    # Create FunctionalSpecification objects.
    local_fs = FunctionalSpecification(local_func_spec_dict)
    incoming_fs = FunctionalSpecification(incoming_func_spec_dict)

    # ---------------------- Test 1: prioritise_local = True ----------------------
    print("=== Test 1: Merge with prioritise_local=True ===")
    result = local_fs.merge(incoming_fs, prioritise_local=True)
    assert result is True, "Merge did not return True (Test 1)"

    # For SchemeA, local values should be kept.
    schemeA = local_fs.get_social_scheme("SchemeA")
    assert schemeA is not None, "SchemeA missing (Test 1)"
    # The description should remain as in local (since local is prioritized).
    assert schemeA[
        "description"] == "Local scheme A description.", "SchemeA description should not be replaced (Test 1)"
    # Existing goal "goal1" should be there.
    goal1 = next((g for g in schemeA["goals"] if g["name"] == "goal1"), None)
    assert goal1 is not None, "Local goal1 missing (Test 1)"
    # New goal "goal2" should have been merged.
    goal2 = next((g for g in schemeA["goals"] if g["name"] == "goal2"), None)
    assert goal2 is not None, "Incoming goal2 not merged (Test 1)"

    # For SchemeB, it should simply be added.
    schemeB = local_fs.get_social_scheme("SchemeB")
    assert schemeB is not None, "SchemeB not added (Test 1)"
    assert schemeB[
        "description"] == "Incoming scheme B description.", "SchemeB description incorrect (Test 1)"

    # ---------------------- Test 2: prioritise_local = False ----------------------
    # Reinitialize the local functional specification.
    local_fs = FunctionalSpecification(local_func_spec_dict)
    print("=== Test 2: Merge with prioritise_local=False ===")
    result = local_fs.merge(incoming_fs, prioritise_local=False)
    assert result is True, "Merge did not return True (Test 2)"

    # For SchemeA, the incoming scheme should replace the local one.
    schemeA = local_fs.get_social_scheme("SchemeA")
    assert schemeA is not None, "SchemeA missing (Test 2)"
    # Now the description should be replaced.
    assert schemeA[
        "description"] == "Incoming scheme A description.", "SchemeA description not updated (Test 2)"
    # The local goal1 may be replaced or overwritten by the merge.
    goal1 = next((g for g in schemeA["goals"] if g["name"] == "goal1"), None)
    # In our implementation for prioritise_local=False, we fully replace the scheme so goal1 should no longer be present.
    assert goal1 is None, "Local goal1 should be replaced (Test 2)"
    # And goal2 must be there.
    goal2 = next((g for g in schemeA["goals"] if g["name"] == "goal2"), None)
    assert goal2 is not None, "Incoming goal2 missing (Test 2)"

    # For SchemeB, it should be added as before.
    schemeB = local_fs.get_social_scheme("SchemeB")
    assert schemeB is not None, "SchemeB missing (Test 2)"
    assert schemeB[
        "description"] == "Incoming scheme B description.", "SchemeB description incorrect (Test 2)"

    print("All tests passed.")
