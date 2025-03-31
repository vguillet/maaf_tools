
##################################################################################################################

import json
from pprint import pprint
import warnings
from typing import Self

##################################################################################################################


class FunctionalSpecification(dict):
    def __init__(self, functional_specification: Self or dict or None = None):
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

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(functional_specification)

    # ============================================================== Properties

    # ============================================================== Check

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Add
    def add_social_scheme(self, name, goals=None, plans=None, missions=None, preferences=None):
        """
        Adds a social scheme to the functional specification.

        Args:
            name (str): The name of the social scheme.
            goals (list, optional): A list of goal dictionaries.
            plans (list, optional): A list of plan dictionaries.
            missions (list, optional): A list of mission dictionaries.
            preferences (list, optional): A list of preference dictionaries.
        """
        scheme = {"name": name}
        if goals is not None:
            scheme["goals"] = goals
        if plans is not None:
            scheme["plans"] = plans
        if missions is not None:
            scheme["missions"] = missions
        if preferences is not None:
            scheme["preferences"] = preferences
        self.functional_specification["social_schemes"].append(scheme)

    # ============================================================== Remove
    def remove_social_scheme(self, name):
        """Removes a social scheme by name. Return True if the scheme was removed, False otherwise."""

        for scheme in self.functional_specification["social_schemes"]:
            if scheme["name"] == name:
                self.functional_specification["social_schemes"].remove(scheme)
                return True
        return False
