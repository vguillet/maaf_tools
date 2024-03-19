
##################################################################################################################

# Built-in/Generic Imports
from dataclasses import dataclass, fields, field
from typing import List, Optional
from datetime import datetime

# Libs
import networkx as nx
from networkx import MultiGraph

# Local Imports
from maaf_tools.datastructures.MaafItem import MaafItem

# from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################

DEBUG = True


@dataclass
class Task(MaafItem):
    """
    Dataclass to represent a task in the maaf allocation framework.
    """
    # > Metadata
    id: str
    type: str
    creator: str
    affiliations: List[str]

    # > Task data
    priority: int
    instructions: dict[str]     # [(skill_ref, task_details_for skill), ...]

    # > Task status
    creation_timestamp: float
    termination_timestamp: Optional[datetime] = None
    termination_source_id: Optional[str] = None
    status: str = "pending"     # pending, completed, cancelled

    # > Wildcard fields
    shared: dict = field(default_factory=dict)  # Shared data of the agent, gets serialized and passed around
    local: dict = field(default_factory=dict)  # Local data of the agent, does not get serialized and passed around

    def __repr__(self) -> str:
        return f"Task {self.id} ({self.type}) - Creation timestamp: {self.creation_timestamp} - Status: {self.status} - Priority: {self.priority}"

    def __str__(self) -> str:
        return self.__repr__()

    def has_affiliation(self, affiliation: str) -> bool:
        """
        Check if the task has a specific affiliation.
        """
        return affiliation in self.affiliations

    # ============================================================== To
    def asdict(self, include_local: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the Task data class instance with their current values.

        :param include_local: Whether to include the local field in the dictionary.

        :return: A dictionary with field names as keys and current values.
        """
        # -> Get the fields of the Task class
        task_fields = fields(self)

        if not include_local:
            # > Exclude the local field
            task_fields = [f for f in task_fields if f.name != "local"]

        # -> Create a dictionary with field names as keys and their current values
        fields_dict = {f.name: getattr(self, f.name) for f in task_fields}

        return fields_dict

    # ============================================================== From
    @classmethod
    def from_dict(cls, task_dict: dict, partial: bool = False) -> "Task":
        """
        Convert a dictionary to a task.

        :param task_dict: The dictionary representation of the task
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A task object
        """
        # -> Get the fields of the Task class
        task_fields = fields(cls)

        # -> Exclude the local field if not provided
        if "local" not in task_dict.keys():
            task_fields = [f for f in task_fields if f.name != "local"]

        # -> Extract field names from the fields
        field_names = {f.name for f in task_fields}

        if not partial:
            # -> Check if all required fields are present in the dictionary
            if not field_names.issubset(task_dict.keys()):
                raise ValueError(f"!!! Task creation from dictionary failed: Task dictionary is missing required fields: {task_dict.keys() - field_names} !!!")

        else:
            # > Remove all fields not present in the dictionary
            task_fields = [f for f in task_fields if f.name in task_dict.keys()]

        # -> Extract values from the dictionary for the fields present in the class
        field_values = {f.name: task_dict[f.name] for f in task_fields}

        # -> Create and return a Task object
        return cls(**field_values)
