
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
    task_bundle: list[str] = field(default_factory=list)    # List of tasks in order of addition
    task_sequence: list[str] = field(default_factory=list)   # List of tasks in order of execution

    def __repr__(self) -> str:
        plan_str = f"{len(self.task_sequence)} tasks: "

        for task in self.task_sequence:
            plan_str += f"{task} -> "

        if len(self.task_sequence) > 0:
            return plan_str[:-4]
        else:
            return "Empty plan"

    def __str__(self) -> str:
        return self.__repr__()

    def __contains__(self, item):
        return item in self.task_sequence

    def __len__(self):
        return len(self.task_sequence)

    def __bool__(self):
        return bool(self.task_sequence)

    def __getitem__(self, index):
        return self.task_sequence[index]

    def __iter__(self):
        return iter(self.task_sequence)

    @property
    def current_task_id(self) -> str:
        """
        Get the id of the current task in the plan.

        :return: The id of the current task.
        """
        return self.task_sequence[0]

    def has_task(self, task_id: str) -> bool:
        """
        Check if the plan has a specific task
        """
        return task_id in self.task_sequence

    # def has_waypoint(self, waypoint_id: str) -> bool:
    #     """
    #     Check if the plan has a specific waypoint
    #     """
    #     return waypoint_id in self.path

    # ============================================================== Add
    def add_task(self, task: Task or str, position: Optional[int] = None) -> bool:
        """
        Add a task to the plan. The task is added to the end of the task list by default if no position is specified.

        :param task: The task to be added to the plan.
        :param position: The position in the plan sequence at which to add the task. Add to the end of the plan if None.

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
        # > Add to task sequence
        if position is not None:
            self.task_sequence.insert(position, task_id)
        else:
            self.task_sequence.append(task_id)

        # > Add to task bundle
        self.task_bundle.append(task)

        # # -> Add key in paths
        # self.paths[task_id] = None

        return True

    # ============================================================== Remove
    def remove_task(self, task: Task or str, forward: bool = False) -> bool:
        """
        Remove a task from the plan.

        :param task: The task to be removed from the plan.
        :param forward: Whether to remove the tasks after the specified task.

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

        if forward:
            # > For all tasks after the specified task...
            for follow_up_task_id in self.task_bundle[self.task_bundle.index(task_id):]:
                # > Remove the task from the plan
                self.task_sequence.remove(follow_up_task_id)
                # del self.paths[follow_up_task_id]

            # > Remove the task and all following tasks from the task bundle
            self.task_bundle = self.task_bundle[:self.task_bundle.index(task_id)]

        else:
            # -> Remove the task from the plan
            self.task_sequence.remove(task_id)
            # del self.paths[task_id]

            # > Remove the task from the task bundle
            self.task_bundle.remove(task_id)

        return True

    # ============================================================== Get
    def get_path(self,
                 agent_id: str,
                 tasklog: TaskLog,
                 requirement: Optional[str] = None,
                 selection: str = "shortest"  # "shortest", "longest", "random"
                 ) -> list:
        """
        Get the full path of the plan.

        :param tasklog: The graph containing the tasks and the paths between them.
        :param requirement: The requirement for the path. Options: "ground", "air", "water", "space"
        :param selection: The selection method for the path. Options: "shortest", "longest", "random"

        :return: A list of waypoints corresponding to the full path of the plan excluding the starting loc.
        """

        paths = self.get_paths(
            agent_id=agent_id,
            tasklog=tasklog,
            requirement=requirement,
            selection=selection
        )

        full_path = []
        missing_paths = []

        first_step = True

        for task_id in self.task_sequence:
            if task_id not in paths.keys():
                missing_paths.append(task_id)

            elif paths[task_id] is None:
                missing_paths.append(task_id)

            else:
                if first_step:
                    full_path += paths[task_id]["path"]
                    first_step = False
                else:
                    full_path += paths[task_id]["path"][1:]

        if missing_paths:
            print(f"!!! Plan path is missing for tasks: {missing_paths} !!!")
            return full_path

        return full_path

    def get_paths(self,
                  agent_id: str,
                  tasklog: TaskLog,
                  requirement: Optional[str] = None,
                  selection: str = "shortest"  # "shortest", "longest", "random"
                  ) -> dict:

        """
        Get the path of the plan as a list of waypoints. The path is obtained from the tasks graph, in which
        the path between two nodes is stored in the corresponding edge.

        :param tasklog: The graph containing the tasks and the paths between them.
        :param requirement: The requirement for the path. Options: "ground", "air", "water", "space"
        :param selection: The selection method for the path. Options: "shortest", "longest", "random"

        :return: A dictionary containing the task ids as keys and the corresponding paths as values.
        """

        node_bundle = [agent_id] + [task_id for task_id in self.task_sequence]

        sequence_paths = tasklog.get_sequence_paths(
            node_sequence=node_bundle,
            requirement=requirement,
            selection=selection
        )

        paths = {}

        for i, task_id in enumerate(self.task_sequence):
            paths[task_id] = sequence_paths[i]

        return paths

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
