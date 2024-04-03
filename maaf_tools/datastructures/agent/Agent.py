
##################################################################################################################
from dataclasses import dataclass, fields, field
from typing import List, Optional

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.datastructures.task.Task import Task

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.maaf_tools.datastructures.task.Task import Task

##################################################################################################################

DEBUG = True


@dataclass
class Agent(MaafItem):
    # ----- Fixed
    id: str                                     # ID of the agent
    name: str                                   # Name of the agent
    agent_class: str                            # Class of the agent

    # ----- Variable
    hierarchy_level: int                        # Hierarchy level of the agent
    affiliations: List[str]                     # Affiliations of the agent
    specs: dict                                 # Specifications of the agent
    skillset: List[str]                         # Skillset of the agent

    state: AgentState                           # State of the agent, state object
    plan: Plan                                  # Plan of the agent, plan object

    # shared: NestedDict = field(default_factory=NestedDict)  # Shared data of the agent, gets serialized and passed around
    # local: NestedDict = field(default_factory=NestedDict)   # Local data of the agent, does not get serialized and passed around

    shared: dict = field(default_factory=dict)  # Shared data of the agent, gets serialized and passed around
    local: dict = field(default_factory=dict)   # Local data of the agent, does not get serialized and passed around

    def __repr__(self) -> str:
        return f"Agent {self.name} ({self.id}) of class {self.agent_class} - Status: {self.state.status}"

    def __str__(self) -> str:
        return self.__repr__()

    # def __eq__(self, other):
    #     if not isinstance(other, Agent):
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
        Get the signature of the agent.
        """

        return {
            "id": self.id,
            "name": self.name,
            "agent_class": self.agent_class
        }

    def has_state(self, state: AgentState) -> bool:
        """
        Check if all the fields of the state match the fields of the agent's state
        """

        return self.state == state

    def has_skill(self, skill: str) -> bool:
        """
        Check if the agent has a given skill
        """
        return skill in self.skillset

    def has_affiliation(self, affiliation: str) -> bool:
        """
        Check if the agent has a given affiliation
        """
        return affiliation in self.affiliations

    def add_task_to_plan(self,
                         tasklog: TaskLog,
                         task: Task or str,
                         logger=None
                         ) -> (bool, bool):
        """
        Add a task to the plan of the agent. The task is added to the task bundle and the path is updated.

        :param tasklog: The task log containing the tasks and the paths between them.
        :param task: The task to be added to the plan.
        :param logger: The logger to log messages to (optional).

        :return: A tuple containing the success of adding the task to the task bundle and updating the plan.
        """

        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if not self.plan.has_task(task_id):
            # -> Add the task to the task bundle
            add_task_success = self.plan.add_task(task=task_id)

            if logger and add_task_success:
                logger.info(
                    f"{self.id} + Assigning task {task_id} to self - Pending task count: {len(tasklog.ids_pending)})")
            elif logger and not add_task_success:
                logger.info(f"!!! Task {task_id} could not be added to the plan of {self.id} !!!")

        else:
            add_task_success = True

        # -> Update the plan with the path from the task log
        update_plan_success = self.update_plan(tasklog=tasklog)

        return add_task_success, update_plan_success

    def remove_task_from_plan(self,
                              tasklog: TaskLog,
                              task: Task or str,
                              logger=None
                              ) -> (bool, bool):
        """
        Remove a task from the plan of the agent. The task is removed from the task bundle and the path is updated.

        :param tasklog: The task log containing the tasks and the paths between them.
        :param task: The task to be removed from the plan.
        :param logger: The logger to log messages to (optional).

        :return: A tuple containing the success of removing the task from the task bundle and updating the plan.
        """

        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if self.plan.has_task(task_id):
            # -> Remove the task from the task bundle
            remove_task_success = self.plan.remove_task(task=task_id)

            if logger and remove_task_success:
                logger.info(
                    f"{self.id} - Dropping task {task_id} from task list - Pending task count: {len(tasklog.ids_pending)}")
            elif logger and not remove_task_success:
                logger.warning(f"{self.id} - Task {task_id} not in task list")

        else:
            remove_task_success = True

        # -> Update the plan with the path from the task log
        update_plan_success = self.update_plan(tasklog=tasklog)

        return remove_task_success, update_plan_success

    def update_plan(self,
                    tasklog: TaskLog,
                    selection: str = "shortest"     # "shortest", "longest", "random"
                    ) -> bool:
        """
        Update the plan of the agent with the path obtained from a tasklog.

        :param tasklog: The tasklog containing the tasks and the paths between them.
        :param selection: The selection method for the path between tasks.
        """

        return self.plan.update_path(
            tasklog=tasklog,
            selection=selection
        )
    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self,
              agent: "Agent",
              prioritise_local: bool = False,
              *args, **kwargs
              ) -> (bool, bool, bool, bool):
        """
        Merge the current agent with another agent.

        :param agent: The agent to merge with.
        :param prioritise_local: Whether to prioritise the local fields when merging (add only).

        :return: A tuple containing the success of the merge and whether the agent has been enabled or disabled.
        """

        # -> Verify if agent is of type Agent
        if not isinstance(agent, Agent):
            raise ValueError(f"!!! Agent merge failed: Agent is not of type Agent: {type(agent)} !!!")

        # -> Verify that the agents signatures match
        if agent.signature != self.signature:
            raise ValueError(f"!!! Agent merge failed: Agent signatures do not match: {self.signature} != {agent.signature} !!!")

        # -> Check if the agent is the same as the current agent
        if agent is self:
            return False, False, False, False

        # -> Setup flags to track if the agent state and plan have changed
        agent_state_change = False
        agent_plan_change = False
        agent_enabled = False
        agent_disabled = False

        # -> If prioritise_local is True, only add new information from the received agent shared field
        if prioritise_local:
            # -> Add new information from received agent shared field
            for key, value in agent.shared.items():
                if key not in self.shared.keys():
                    self.shared[key] = value

        elif agent.state.timestamp > self.state.timestamp:
            # ---- Merge general fields
            # -> Get the fields of the Agent class
            agent_fields = fields(self)

            # > Create fields to exclude
            field_exclude = ["local", "state", "plan", *self.signature.keys()]

            # > Exclude fields
            agent_fields = [f for f in agent_fields if f.name not in field_exclude]

            # -> Update the fields
            for field in agent_fields:
                # > Get the field value from the received agent
                field_value = getattr(agent, field.name)

                if field_value != getattr(self, field.name):
                    # > Update the field value
                    setattr(self, field.name, field_value)
                    agent_state_change = True

            # ---- Merge state
            if self.state != agent.state:
                if agent.state.status == "enabled":
                    agent_enabled = True
                elif agent.state.status == "disabled":
                    agent_disabled = True

                self.state = agent.state

            # ---- Merge plan
            if self.plan != agent.plan:
                self.plan = agent.plan

                agent_state_change = True
                agent_plan_change = True

        return agent_state_change, agent_plan_change, agent_enabled, agent_disabled


    # ============================================================== To
    def asdict(self, include_local: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the Agent data class instance with their current values.

        :param include_local: Whether to include the local data in the dictionary

        :return: A dictionary with field names as keys and current values.
        """
        # -> Get the fields of the Agent class
        agent_fields = fields(self)

        if not include_local:
            # > Exclude the local field
            agent_fields = [f for f in agent_fields if f.name != "local"]

        # -> Create a dictionary with field names as keys and their current values
        fields_dict = {f.name: getattr(self, f.name) for f in agent_fields}

        # -> Convert state to dict
        fields_dict["state"] = self.state.asdict()

        # -> Convert plan to dict
        fields_dict["plan"] = self.plan.asdict()

        return fields_dict

    # ============================================================== From
    @classmethod
    def from_dict(cls, agent_dict: dict, partial: bool = False) -> "Agent":
        """
        Convert a dictionary to an agent.

        :param agent_dict: The dictionary representation of the agent

        :return: An agent object
        """
        # -> Get the fields of the Agent class
        agent_fields = fields(cls)

        # > Exclude the local field if not provided
        if "local" not in agent_dict.keys():
            agent_fields = [f for f in agent_fields if f.name != "local"]

        # -> Extract field names from the fields
        field_names = {f.name for f in agent_fields}

        if not partial:
            # -> Check if all required fields are present in the dictionary
            if not field_names.issubset(agent_dict.keys()):
                raise ValueError(f"!!! Agent creation from dictionary failed: Agent dictionary is missing required fields: {agent_dict.keys() - field_names} !!!")

        else:
            # > Remove all fields not present in the dictionary
            agent_fields = [f for f in agent_fields if f.name in agent_dict]

        # -> Extract values from the dictionary for the fields present in the class
        field_values = {f.name: agent_dict[f.name] for f in agent_fields}

        # -> Convert state from dict
        field_values["state"] = AgentState.from_dict(agent_dict["state"])

        # -> Convert plan from dict
        field_values["plan"] = Plan.from_dict(agent_dict["plan"])

        # -> Create and return an Agent object
        return cls(**field_values)