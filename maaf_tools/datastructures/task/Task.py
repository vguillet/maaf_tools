
##################################################################################################################

# Built-in/Generic Imports
from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict
from datetime import datetime

# Libs
import networkx as nx
from networkx import MultiGraph

# Local Imports
# try:
#     from maaf_tools.datastructures.MaafItem import MaafItem
#     from maaf_tools.datastructures.serialisation import *
#
# except:
#     from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
#     from maaf_tools.maaf_tools.datastructures.serialisation import *
try:
    from maaf_tools.datastructures.MaafItem import MaafItem
except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################

DEBUG = True


@dataclass
class Task(MaafItem):
    """
    Dataclass to represent a task in the maaf allocation framework.
    """
    # ----- Fixed
    # > Metadata
    id: str
    type: str
    creator: str
    instructions: dict[str: str]     # [(skill_ref, task_details_for skill), ...]

    # ----- Variable
    # > Task data
    affiliations: list[str]
    priority: int

    # > Task status
    creation_timestamp: float
    termination_timestamp: Optional[datetime] = None
    termination_source_id: Optional[str] = None
    status: str = "pending"     # pending, completed, cancelled

    # > Wildcard fields
    shared: dict = field(default_factory=dict)  # Shared data of the agent, gets serialized and passed around
    local: dict = field(default_factory=dict)   # Local data of the agent, does not get serialized and passed around

    def __repr__(self) -> str:
        return f"Task {self.id} ({self.type}) - Creation timestamp: {self.creation_timestamp} - Status: {self.status} - Priority: {self.priority}"

    def __str__(self) -> str:
        return self.__repr__()

    # def __eq__(self, other):
    #     if not isinstance(other, Task):
    #         return False
    #
    #     # -> Compare all fields except the excluded ones
    #     fields_exclusion = ["local"]
    #
    #     for f in fields(self):
    #         if f in fields_exclusion:
    #             continue
    #         if getattr(self, f.name) != getattr(other, f.name):
    #             return False
    #     return True

    @property
    def signature(self) -> dict:
        """
        Get the signature of the task. Made up of the task's fixed properties
        """
        return {
            "id": self.id,
            "type": self.type,
            # "creator": self.creator,
            "instructions": self.instructions,
        }

    def has_affiliation(self, affiliation: str) -> bool:
        """
        Check if the task has a specific affiliation.
        """
        return affiliation in self.affiliations

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self,
              task: "Task",
              prioritise_local: bool = False,
              *args, **kwargs
              ) -> (bool, bool):
        """
        Merge the current task with another task.

        :param task: The task to merge with.
        :param prioritise_local: Whether to prioritise the local fields when merging (add only).

        :return: A tuple containing the success of the merge and whether the task has been terminated.
        """

        # -> Verify if task is of type Task
        if not isinstance(task, Task):
            raise ValueError(f"!!! Task merge failed: Task is not of type Task: {type(task)} !!!")

        # -> Verify that the tasks signatures match
        if task.signature != self.signature:
            raise ValueError(f"!!! Task merge failed: Task signatures do not match: {self.signature} != {task.signature} !!!")

        # -> Check if the task is the same as the current task
        # if task is self:
        #     return False, False

        # -> Setup a flag to track if the task status has changed
        task_state_change = False
        task_terminated = False

        # -> If prioritise_local is True, only add new information from the received task shared field
        if prioritise_local:
            # -> Add new information from received task shared field
            for key, value in task.shared.items():
                if key not in self.shared.keys():
                    self.shared[key] = value

        else:
            # ---- Merge general fields
            # -> Get the fields of the Task class
            task_fields = fields(self)

            # > Create fields to exclude
            field_exclude = ["local", "shared",
                             "status", "termination_timestamp", "termination_source_id",
                             *self.signature.keys()]

            # > Exclude fields
            task_fields = [f for f in task_fields if f.name not in field_exclude]

            # -> Update the fields
            for field in task_fields:
                # > Get the field value from the received task
                field_value = getattr(task, field.name)

                if field_value != getattr(self, field.name):
                    # > Update the field value
                    setattr(self, field.name, field_value)
                    task_state_change = True

            # -> Add new information from received task shared field
            for key, value in task.shared.items():
                self.shared[key] = value

            # ---- Merge state
            # -> Check if the task status has changed
            if task.status != "pending" and self.status == "pending":
                self.status = task.status
                self.termination_timestamp = task.termination_timestamp
                self.termination_source_id = task.termination_source_id

                task_state_change = True
                task_terminated = True

        return task_state_change, task_terminated

    # # ============================================================== To
    # def asdict(self, include_local: bool = False) -> dict:
    #     """
    #     Create a dictionary containing the fields of the Task data class instance with their current values.
    #
    #     :param include_local: Whether to include the local field in the dictionary.
    #
    #     :return: A dictionary with field names as keys and current values.
    #     """
    #     try:
    #         from maaf_tools.datastructures.serialisation import asdict
    #
    #     except ImportError:
    #         from maaf_tools.maaf_tools.datastructures.serialisation import asdict
    #
    #     if not include_local:
    #         fields_exclusion_lst = ["local"]
    #     else:
    #         fields_exclusion_lst = []
    #
    #     fields_dict = asdict(item=self, fields_exclusion_lst=fields_exclusion_lst)
    #
    #     return fields_dict
    #
    # # ============================================================== From
    # @classmethod
    # def from_dict(cls, task_dict: dict, partial: bool = False) -> "Task":
    #     """
    #     Convert a dictionary to a task.
    #
    #     :param task_dict: The dictionary representation of the task
    #     :param partial: Whether to allow creation from a dictionary with missing fields.
    #
    #     :return: A task object
    #     """
    #     try:
    #         from maaf_tools.datastructures.serialisation import from_dict
    #
    #     except ImportError:
    #         from maaf_tools.maaf_tools.datastructures.serialisation import from_dict
    #
    #     item = from_dict(cls=cls, item_dict=task_dict, partial=partial)
    #
    #     return item


if __name__ == "__main__":
    from pprint import pprint
    import pandas as pd

    task = Task(
        id="task_1",
        type="task",
        creator="agent_1",
        instructions={"skill_1": "task_1"},

        affiliations=["affiliation_1"],
        priority=1,

        creation_timestamp=0.0,
        termination_timestamp=None,
        termination_source_id=None,

        status="pending",

        shared={
            "c_matrix": pd.DataFrame({
                "agent_1": [0, 1, 2],
                "agent_2": [1, 0, 3],
                "agent_3": [2, 3, 0]
            }),
        },
        local={}
    )

    task_dict = task.asdict()
    pprint(task_dict)
    task2 = Task.from_dict(task_dict, partial=True)
    pprint(task2.asdict())
