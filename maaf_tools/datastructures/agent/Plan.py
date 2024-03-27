
##################################################################################################################

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
    task_bundle: list[Task] = field(default_factory=list)  # List of tasks to be executed
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

        node_bundle = ["agent"] + [task.id for task in self.task_bundle]

        sequence_paths_list, sequence_path_exist = tasklog.get_sequence_path(
                node_sequence=node_bundle,
                requirement=None,        # TODO: Fix once agents have requirements
                selection=selection
                )

        # -> Check if a path exists for each step of the sequence
        for path in sequence_path_exist:
            assert path, f"!!! Plan path update failed: No path found for bundle {node_bundle} !!!"

        # -> Check if path exists for every step of the bundle
        for path in sequence_paths_list:
            self.path = path["path"][1:]

        return True

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
