
##################################################################################################################

import json
from pprint import pprint
import warnings
#from typing import Self

##################################################################################################################


class DeonticSpecification(dict):
    def __init__(self, deontic_specification: dict or None = None):        # If no specification is provided, use the default template.
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

        # Initialize the underlying dict with the provided or default dictionary.
        super().__init__(deontic_specification)

    # ============================================================== Properties

    # ============================================================== Check

    # ============================================================== Get

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
        self.deontic_specification["permissions"].append(permission)

    def add_obligation(self, role, mission, time_constraint="Any"):
        """
        Adds an obligation to the deontic specification.

        Args:
            role (str): The role associated with the obligation.
            mission (str): The mission to which the obligation applies.
            time_constraint (str): The time constraint (default "Any").
        """
        obligation = {"role": role, "mission": mission, "time_constraint": time_constraint}
        self.deontic_specification["obligations"].append(obligation)

    # ============================================================== Remove
    def remove_permission(self, role, mission):
        """Removes a permission. Return True if the permission was removed, False otherwise."""

        for permission in self.deontic_specification["permissions"]:
            if permission["role"] == role and permission["mission"] == mission:
                self.deontic_specification["permissions"].remove(permission)
                return True
        return False

    def remove_obligation(self, role, mission):
        """Removes an obligation. Return True if the obligation was removed, False otherwise."""

        for obligation in self.deontic_specification["obligations"]:
            if obligation["role"] == role and obligation["mission"] == mission:
                self.deontic_specification["obligations"].remove(obligation)
                return True
        return False
