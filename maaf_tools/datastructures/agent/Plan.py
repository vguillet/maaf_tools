
##################################################################################################################

from dataclasses import dataclass, fields, field
from maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


@dataclass
class Plan(MaafItem):
    recompute: bool = False
    task_bundle: list[str] = field(default_factory=list)  # List of tasks to be executed
    path: list[str] = field(default_factory=list)         # List of waypoints to be visited

    def __repr__(self) -> str:
        return f"Plan with {len(self.task_bundle)} tasks and {len(self.path)} waypoints"

    def __str__(self) -> str:
        return self.__repr__()

    def has_task(self, task_id: str) -> bool:
        """
        Check if the plan has a specific task
        """
        return task_id in self.task_bundle

    def has_waypoint(self, waypoint_id: str) -> bool:
        """
        Check if the plan has a specific waypoint
        """
        return waypoint_id in self.path

    # ============================================================== To
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the Plan data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """
        # -> Get the fields of the Plan class
        plan_fields = fields(self)

        # -> Create a dictionary with the field names as keys and the current values as values
        plan_dict = {field.name: getattr(self, field.name) for field in plan_fields}

        return plan_dict

    # ============================================================== From
    @classmethod
    def from_dict(cls, plan_dict: dict, partial: bool = False) -> "Plan":
        """
        Create a dictionary containing the fields of the Plan data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """
        plan_fields = fields(cls)

        # -> Extract field names from the fields
        field_names = {f.name for f in plan_fields}

        if not partial:
            # -> Check if all required fields are present in the dictionary
            if not field_names.issubset(plan_dict.keys()):
                raise ValueError(f"!!! Plan creation from dictionary failed: Plan dictionary is missing required fields: {plan_dict.keys() - field_names} !!!")

        else:
            # > Remove all fields not present in the dictionary
            plan_fields = [f for f in plan_fields if f.name in plan_dict.keys()]

        # -> Extract values from the dictionary for the fields present in the class
        field_values = {f.name: plan_dict[f.name] for f in plan_fields}

        return cls(**field_values)
