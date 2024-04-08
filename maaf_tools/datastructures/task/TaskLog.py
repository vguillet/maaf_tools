
##################################################################################################################

# Built-in/Generic Imports
from dataclasses import dataclass, fields, field
from typing import List, Optional
from datetime import datetime

# Libs
import networkx as nx
from networkx import MultiGraph

# Local Imports
try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.datastructures.task.Task import Task
    from maaf_tools.datastructures.task.TaskGraph import TaskGraph

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.maaf_tools.datastructures.task.Task import Task
    from maaf_tools.maaf_tools.datastructures.task.TaskGraph import TaskGraph


##################################################################################################################


@dataclass
class TaskLog(MaafList):
    task_graph: TaskGraph = field(default_factory=TaskGraph)
    item_class = Task
    __on_status_change_listeners: list[callable] = field(default_factory=list)

    def init_tasklog(self, agent_id: str):
        """
        Initialise the task log. Must be called after the task log has been created.
        """

        # -> Add agent to task graph
        self.task_graph.set_main_agent_node(agent_id=agent_id)

    def __repr__(self):
        return f"Task log: {len(self.items)} tasks ({len(self.tasks_completed)} completed, {len(self.tasks_pending)} pending, {len(self.tasks_cancelled)} cancelled)"

    # ============================================================== Listeners
    # ------------------------------ Status
    def add_on_status_change_listener(self, listener):
        """
        Add a listener to the task log that listens for changes in the status of tasks.
        """
        self.__on_status_change_listeners.append(listener)

    def call_on_status_change_listeners(self, task: Task):
        """
        Call all listeners that are listening for changes in the status of tasks.
        """
        for listener in self.__on_status_change_listeners:
            listener(task)

    # ============================================================== Properties
    # ------------------------------ IDs
    @property
    def ids_pending(self) -> list[int]:
        """
        Get a list of pending task ids in the task log.
        """
        return [task.id for task in self.tasks_pending]

    @property
    def ids_cancelled(self) -> list[int]:
        """
        Get a list of cancelled task ids in the task log.
        """
        return [task.id for task in self.tasks_cancelled]

    @property
    def ids_completed(self) -> list[int]:
        """
        Get a list of completed task ids in the task log.
        """
        return [task.id for task in self.tasks_completed]

    @property
    def ids_terminated(self) -> list[int]:
        """
        Get a list of terminated task ids in the task log.
        """
        return [task.id for task in self.tasks_terminated]

    # ------------------------------ Tasks
    @property
    def tasks_pending(self) -> list[Task]:
        """
        Get a list of pending tasks in the task log.
        """
        return [task for task in self.items if task.status == "pending"]

    @property
    def tasks_cancelled(self) -> list[Task]:
        """
        Get a list of cancelled tasks in the task log.
        """
        return [task for task in self.items if task.status == "cancelled"]

    @property
    def tasks_completed(self) -> list[Task]:
        """
        Get a list of completed tasks in the task log.
        """
        return [task for task in self.items if task.status == "completed"]

    @property
    def tasks_terminated(self) -> list[Task]:
        """
        Get a list of terminated tasks in the task log.
        """
        return [task for task in self.items if task.status in ["cancelled", "completed"]]

    # ============================================================== Get
    def query(
            self,
            task_type: Optional[str] = None,
            priority: Optional[int] = None,
            affiliation: Optional[List[str]] = None,
            status: Optional[str] = None
            ) -> List[Task]:
        """
        Query the task log for tasks that match the given criteria.

        :param task_type: The type of task to filter for.
        :param priority: The priority of the task to filter for.
        :param affiliation: The affiliation of the task to filter for.
        :param status: The status of the task to filter for.

        :return: A list of tasks that match the given criteria.
        """

        filtered_tasks = self.items

        if task_type:
            filtered_tasks = [task for task in filtered_tasks if task.type == task_type]
        if priority is not None:
            filtered_tasks = [task for task in filtered_tasks if task.priority == priority]
        if affiliation:
            filtered_tasks = [task for task in filtered_tasks if affiliation in task.affiliations]
        if status:
            filtered_tasks = [task for task in filtered_tasks if task.status == status]

        return filtered_tasks

    def get_sequence_paths(self,
                           node_sequence: List[str],
                           requirement: Optional[List[str]] = None,
                           selection: str = "shortest"   # "shortest", "longest", "random", "all"
                           ) -> (list[dict], list):
        """
        Get a path from the graph.

        :param node_sequence: The sequence of nodes to get the path for.
        :param requirement: The acceptable requirements for the path.
        :param selection: The selection method for the path if multiple meet the requirements. "shortest", "longest", "random", "all"

        :return: The path between the nodes.
        """

        return self.task_graph.get_sequence_paths(
            node_sequence=node_sequence,
            requirement=requirement,
            selection=selection
        )

    def get_path(self,
                 source: str,
                 target: str,
                 requirement: Optional[List[str]] = None,
                 selection: str = "shortest"   # "shortest", "longest", "random", "all
                 ) -> List[dict] or dict or None:
        """
        Get a path from the graph. Return all the existing paths between two nodes meeting the requirements.

        :param source: The source node of the path.
        :param target: The target node of the path.
        :param requirement: The acceptable requirements for the path.
        :param selection: The selection method for the path if multiple meet the requirements. "shortest", "longest", "random", "all"

        :return: The path(s) between the nodes.
        """

        return self.task_graph.get_path(
            source=source,
            target=target,
            requirement=requirement,
            selection=selection
        )

    # ============================================================== Set
    def set_task_status(self,
                        task: int or str or item_class or List[int or str or item_class],
                        termination_status: str,
                        termination_source_id,
                        termination_timestamp
                        ) -> None:
        """
        Set the status of a task with given task_id.

        :param task: The task to set the status of. Can be a task id, task object, or a list of task ids or task objects.
        :param termination_status: The status to set the task to. Must be a list if task is a list.
        :param termination_source_id: The source id of the task termination. Must be a list if task is a list.
        :param termination_timestamp: The timestamp of the task termination. Must be a list if task is a list.
        """

        # -> If the task_id is a list, flag each task as completed individually recursively
        if isinstance(task, list):
            for i in range(len(task)):
                self.set_task_status(
                    task=task[i],
                    termination_status=termination_status[i],
                    termination_source_id=termination_source_id[i],
                    termination_timestamp=termination_timestamp[i]
                )
            return

        # -> Update the task status to 'completed'
        self.update_item_fields(
            item=task,
            field_value_pair={
                "status": termination_status,
                "termination_source_id": termination_source_id,
                "termination_timestamp": termination_timestamp
            }
        )

        # -> Call the on_status_change listeners
        self.call_on_status_change_listeners(task)

    def flag_task_completed(self,
                            task: int or str or item_class or List[int or str or item_class],
                            termination_source_id,
                            termination_timestamp
                            ) -> None:
        """
        Mark a task as completed with given task_id.

        :param task: The task to flag as completed. Can be a task id, task object, or a list of task ids or task objects.
        :param termination_source_id: The source id of the task termination. Must be a list if task is a list.
        :param termination_timestamp: The timestamp of the task termination. Must be a list if task is a list.
        """

        self.set_task_status(
            task=task,
            termination_status="completed",
            termination_source_id=termination_source_id,
            termination_timestamp=termination_timestamp
        )

    def flag_task_cancelled(self,
                            task: int or str or item_class or List[int or str or item_class],
                            termination_source_id,
                            termination_timestamp
                            ) -> None:
        """
        Mark a task as cancelled with given task_id.

        :param task: The task to flag as cancelled. Can be a task id, task object, or a list of task ids or task objects.
        :param termination_source_id: The source id of the task termination. Must be a list if task is a list.
        :param termination_timestamp: The timestamp of the task termination. Must be a list if task is a list.
        """

        self.set_task_status(
            task=task,
            termination_status="cancelled",
            termination_source_id=termination_source_id,
            termination_timestamp=termination_timestamp
        )

    # ============================================================== Merge
    def merge(self,
              tasklog: "TaskLog",
              add_task_callback: Optional[callable] = None,
              terminate_task_callback: Optional[callable] = None,
              tasklog_state_change_callback: Optional[callable] = None,
              prioritise_local: bool = False,
              *args, **kwargs
              ) -> bool:
        """
        Merge the tasks from another task log into this task log. If a task with the same id exists in both task logs,

        :param tasklog: The task log to merge with this task log.
        :param add_task_callback: A callback function to call when a task is added to the task log.
        :param terminate_task_callback: A callback function to call when a task is terminated in the task log.
        :param tasklog_state_change_callback: A callback function to call at the end of the merge if the task log state has changed.
        :param prioritise_local: Whether to prioritise the local fields when merging (add only).

        :return: A boolean indicating whether the tasklog state has changed
        """

        # -> If the task log is None, return False
        if tasklog is None:
            return False

        # -> Verify if tasklog is of type TaskLog
        if not isinstance(tasklog, TaskLog):
            raise ValueError(f"!!! Task log merge failed: Task log is not of type TaskLog: {type(tasklog)} !!!")

        # -> Check if the task log is the same as the current task log
        if tasklog is self:
            return False

        tasklog_state_change = 0

        # ----- Merge tasks
        # -> For all tasks in the task log to merge ...
        for task in tasklog:
            # -> If the task is not in the task log ...
            if task.id not in self.ids:
                # -> If the task is pending, add to task log and extend local states with new rows
                if task.status == "pending":
                    self.add_task(task=task)

                    tasklog_state_change += 1

                    if add_task_callback is not None:
                        add_task_callback(task=task)

                # -> If the task is completed, only add to the task log (do not recompute bids)
                else:
                    self.add_task(task=task)

            # -> Else, merge the task with the task in the task log
            else:
                task_state_change, task_terminated = self[task.id].merge(
                    task=task,
                    prioritise_local=prioritise_local
                    *args, **kwargs
                )

                if task_state_change:
                    tasklog_state_change += 1

                if terminate_task_callback is not None and task_terminated:
                    terminate_task_callback(task=task)

        # ----- Merge task graph
        self.task_graph.merge(
            task_graph=tasklog.task_graph,
            prioritise_local=False   # TODO: Refactor to specify how to prioritise paths in merge
        )

        # -> Call the task state change callback if the task state has changed
        if tasklog_state_change_callback is not None and tasklog_state_change > 0:
            tasklog_state_change_callback()

        return tasklog_state_change > 0

    # ============================================================== Add
    def add_task(self, task: dict or item_class or List[dict or item_class]) -> None:
        """
        Add a task to the task log. If the task is a list, add each task to the task log individually recursively.

        :param task: The task(s) to add to the task log. Can be a task object, a task dictionary, or a list of task objects or task dictionaries.
        """
        # -> If the task is a list, add each task to the task log individually recursively
        if isinstance(task, list):
            for t in task:
                self.add_task(t)
            return

        # -> Add the task to the task log
        if isinstance(task, self.item_class):
            success = self.add_item(item=task)
        else:
            success = self.add_item_by_dict(item_data=task)

        if success:
            # -> Add node in task graph
            self.task_graph.add_node(
                node_for_adding=task.id,
                node_type="Task"
            )

    def add_path(self,
                 source_node: str,
                 target_node: str,
                 path: dict or List[dict],
                 two_way: bool = True,
                 selection: str = "latest",   # "shortest", "longest", "random", "latest", "all"
                 ) -> None:
        """
        Add a path to the graph.

        :param source_node: The source node of the path.
        :param target_node: The target node of the path.
        :param path: The path to add.
        :param two_way: Whether to add the path in both directions.
        :param selection: The selection method for the path if multiple meet the requirements.
            "shortest" : Replace the current shortest path with the new path if the new path is shorter.
            "longest"  : Replace the current longest path with the new path if the new path is longer.
            "random"   : Randomly replace a path with the new path.
            "latest"   : Replace all current paths with the new path.
            "all"      : Keep all paths. Add the new path to the list of paths.
        """

        self.task_graph.add_path(
            source_node=source_node,
            target_node=target_node,
            path=path,
            two_way=two_way,
            selection=selection
        )

    # ============================================================== Remove
    def remove_task(self, task: int or str or item_class or List[int or str or item_class]) -> None:
        """
        Remove a task from the task log. If the task is a list, remove each task from the task log individually recursively.

        :param task: The task(s) to remove from the task log. Can be a task id, task object, or a list of task ids or task objects.
        """
        # -> If the task is a list, remove each task from the task log individually recursively
        if isinstance(task, list):
            for t in task:
                self.remove_task(t)
            return

        # -> Remove the task from the task log
        if isinstance(task, self.item_class):
            success = self.remove_item(item=task)
        else:
            success = self.remove_item_by_id(item_id=task)

        if success:
            # -> Remove node in task graph
            self.task_graph.remove_node(task.id)


if __name__ == "__main__":
    from pprint import pprint

    # -> Create tasks
    task1 = Task(
        id="task_1",
        type="type_1",
        creator="creator_1",
        affiliations=["affiliation_1"],
        priority=1,
        instructions={"skill_1": "instruction_1"},
        creation_timestamp=0
    )

    task2 = Task(
        id="task_2",
        type="type_2",
        creator="creator_2",
        affiliations=["affiliation_2"],
        priority=2,
        instructions={"skill_2": "instruction_2"},
        creation_timestamp=1
    )

    task3 = Task(
        id="task_3",
        type="type_3",
        creator="creator_2",
        affiliations=["affiliation_2"],
        priority=2,
        instructions={"skill_2": "instruction_2"},
        creation_timestamp=1
    )

    # -> Create task log
    tasklog = TaskLog()

    # -> Add tasks to task log
    tasklog.add_task([task1, task2, task3])

    # -> Print task log
    print(tasklog)

    print(tasklog.task_graph)

    # -> Add path between tasks
    tasklog.task_graph.add_path(
        source_node=task1.id,
        target_node=task2.id,
        path={
            "id": "path_1",
            "requirements": ["skill_1"],
            "path": [task1.id, task2.id]
        }
    )

    tasklog.add_path(
        source_node=task2.id,
        target_node=task3.id,
        path={
            "id": "path_2",
            "requirements": ["skill_2"],
            "path": [task2.id, task3.id]
        }
    )

    tasklog.add_path(
        source_node=task1.id,
        target_node=task3.id,
        path={
            "id": "path_3",
            "requirements": ["skill_1"],
            "path": [task1.id, task3.id]
        }
    )

    # -> Serialise task log
    tasklog_serialised = tasklog.asdict()

    pprint(tasklog_serialised)

    # -> Deserialise task log
    tasklog_deserialised = TaskLog.from_dict(tasklog_serialised)

    print(tasklog_deserialised)

    print(tasklog_deserialised.task_graph)

    print(tasklog.asdf().to_string())

    new_tasklog = tasklog.clone()

    print(new_tasklog.task_graph)

    print(new_tasklog.asdict())