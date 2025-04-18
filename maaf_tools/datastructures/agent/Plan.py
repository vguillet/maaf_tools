
##################################################################################################################

from typing import Optional, List
from dataclasses import dataclass, fields, field

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.datastructures.task.Task import Task

    from maaf_tools.Singleton import SLogger

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.maaf_tools.datastructures.task.Task import Task

    from maaf_tools.maaf_tools.Singleton import SLogger

##################################################################################################################


@dataclass
class Plan(MaafItem):
    task_bundle: list[str] = field(default_factory=list)          # List of tasks ids in order of addition
    task_sequence: list[str] = field(default_factory=list)        # List of tasks ids in order of execution

    def __repr__(self) -> str:
        plan_str = ""

        for task in self.task_sequence:
            plan_str += f"{task} -> "

        if len(self.task_sequence) > 0:
            plan_str = plan_str[:-4]

        return plan_str

    def __str__(self) -> str:
        return f"Plan with {len(self.task_sequence)} tasks: {self.__repr__()}"

    def __contains__(self, item: str) -> bool:
        return item in self.task_sequence

    def __len__(self) -> int:
        return len(self.task_sequence)

    def __bool__(self) -> bool:
        return bool(self.task_sequence)

    def __getitem__(self, index) -> str:
        return self.task_sequence[index]

    def __iter__(self) -> iter:
        return iter(self.task_sequence)

    @property
    def current_task_id(self) -> Optional[str]:
        """
        Get the id of the current task in the plan.

        :return: The id of the current task.
        """
        try:
            return self.task_sequence[0]
        except IndexError:
            return None

    def has_task(self, task_id: str) -> bool:
        """
        Check if the plan has a specific task
        """
        return task_id in self.task_sequence

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
            SLogger().info(f"!!! Task {task_id} is already in the plan !!!")
            return False

        # -> Add the task to the plan
        # > Add to task sequence
        if position is not None:
            self.task_sequence.insert(position, task_id)
        else:
            self.task_sequence.append(task_id)

        # > Add to task bundle
        self.task_bundle.append(task_id)

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
            (f"!!! Task {task_id} is not in the plan !!!")
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
    # TODO: Delete is not used anymore (likley outdated since env refactor)
    # def get_node_pairs(self, agent_id: Optional[str]=None) -> list[tuple[str, str]]:
    #     """
    #     Get a list of node pairs for the plan. The pairs represent the source and target nodes for the tasks in the plan.
    #
    #     :param agent_id: The id of the agent if part of the plan
    #     """
    #
    #     if agent_id is not None:
    #         return [(agent_id, self.task_sequence[0])] + [(self.task_sequence[i], self.task_sequence[i+1]) for i in range(len(self.task_sequence) - 1)]
    #
    #     else:
    #         return [(self.task_sequence[i], self.task_sequence[i+1]) for i in range(len(self.task_sequence) - 1)]

    # ============================================================== Serialization / Parsing


if __name__ == "__main__":
    plan = Plan()

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
        creator="creator_3",
        affiliations=["affiliation_3"],
        priority=3,
        instructions={"skill_3": "instruction_3"},
        creation_timestamp=2
    )

    # -> Create task log
    tasklog = TaskLog()

    # -> Add tasks to task log
    tasklog.add_task([task1, task2, task3])

    tasklog.task_graph.add_path(
        source_node=task1.id,
        target_node=task2.id,
        path={
            "id": "path_1",
            "requirements": ["skill_1"],
            "path": [task1.id, "a", task2.id]
        }
    )

    tasklog.add_path(
        source_node=task2.id,
        target_node=task3.id,
        path={
            "id": "path_2",
            "requirements": ["skill_2"],
            "path": [task2.id, "b", task3.id]
        }
    )

    tasklog.add_path(
        source_node=task1.id,
        target_node=task3.id,
        path={
            "id": "path_3",
            "requirements": ["skill_1"],
            "path": [task1.id, "c", task3.id]
        }
    )

    plan.add_task(task1)
    print(plan)

    plan.add_task(task2)
    print(plan)

    plan.add_task(task3, position=1)
    print(plan)

    print("task_bundle:", plan.task_bundle)
    print("task_sequence:", plan.task_sequence)
    print("-------------------------")
    paths = tasklog.get_sequence_paths(node_sequence=plan.task_sequence, requirement=None, selection="shortest")
    print(paths)
    # plan.remove_task(task2, forward=True)
    # print(plan)

    print(plan.get_paths(tasklog=tasklog))
    print(plan.get_path(tasklog=tasklog))
    # print(plan.get_node_pairs())
    # print(plan.get_node_pairs(agent_id="agent_1"))

    # -> Clone and modify plan to ensure plans instance are disconnected
    print(plan)

    plan_clone = plan.clone()
    plan_clone.remove_task(task1)

    print(plan)

    print(plan_clone)

    print(plan.get_node_pairs())