
##################################################################################################################

from typing import Optional
from dataclasses import dataclass, fields, field
import networkx as nx

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.datastructures.task.Task import Task

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.maaf_tools.datastructures.task.Task import Task

##################################################################################################################


@dataclass
class Plan(MaafItem):
    task_bundle: list[str] = field(default_factory=list)  # Ordered list of task ids to be executed
    paths: dict[str] = field(default_factory=dict)         # Dict of paths corresponding to each task

    def __repr__(self) -> str:
        return f"Plan with {len(self.task_bundle)} tasks and {len(self.path)} waypoints"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def path(self):
        """
        Get the full path of the plan.

        :return: A list of waypoints corresponding to the full path of the plan excluding the starting loc.
        """

        full_path = []
        missing_paths = []

        first_step = True

        for task_id in self.task_bundle:
            if task_id not in self.paths.keys():
                missing_paths.append(task_id)

            elif self.paths[task_id] is None:
                missing_paths.append(task_id)

            else:
                if first_step:
                    full_path += self.paths[task_id]["path"]
                    first_step = False
                else:
                    full_path += self.paths[task_id]["path"][1:]

        if missing_paths:
            print(f"!!! Plan path is missing for tasks: {missing_paths} !!!")
            return missing_paths

        return full_path

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

    # ============================================================== Add
    def add_task(self, task: Task or str, position: Optional[int] = None) -> bool:
        """
        Add a task to the plan. The task is added to the end of the task list by default if no position is specified.

        :param task: The task to be added to the plan.
        :param position: The position at which to add the task.

        :return: True if the task was added successfully, False otherwise.
        """

        # -> Check if the task is already in the plan
        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if self.has_task(task_id):
            print(f"!!! Task {task_id} is already in the plan !!!")
            return False

        # -> Add the task to the plan
        if position is not None:
            self.task_bundle.insert(position, task_id)
        else:
            self.task_bundle.append(task_id)

        # -> Add key in paths
        self.paths[task_id] = None

        return True

    # ============================================================== Remove
    def remove_task(self, task: Task or str) -> bool:
        """
        Remove a task from the plan.

        :param task: The task to be removed from the plan.

        :return: True if the task was removed successfully, False otherwise.
        """

        # -> Check if the task is in the plan
        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if not self.has_task(task_id):
            print(f"!!! Task {task_id} is not in the plan !!!")
            return False

        # -> Remove the task from the plan
        self.task_bundle.remove(task_id)
        del self.paths[task_id]

        return True

    # ============================================================== Get
    def update_path(self,
                    tasklog: TaskLog,
                    selection: str = "shortest"  # "shortest", "longest", "random"
                    ) -> bool:
        """
        Get the path of the plan as a list of waypoints. The path is obtained from the tasks graph, in which
        the path between two nodes is stored in the corresponding edge.

        :param tasklog: The graph containing the tasks and the paths between them.
        :param selection: The selection method for the path. Options: "shortest", "longest", "random"
        """

        node_bundle = ["agent"] + [task_id for task_id in self.task_bundle]

        sequence_paths = tasklog.get_sequence_path(
            node_sequence=node_bundle,
            requirement=None,        # TODO: Fix once agents have requirements
            selection=selection
        )

        # -> Check if a path exists for each step of the sequence
        complete_path = True

        for path in sequence_paths:
            if not path:
                complete_path = False

        self.paths = {}

        # -> Update local path (store path corresponding to each task)
        for i, task_id in enumerate(self.task_bundle):
            self.paths[task_id] = sequence_paths[i]

        return complete_path

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
