
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

    from maaf_tools.tools import deep_compare
    from maaf_tools.Singleton import SLogger

except ImportError:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.maaf_tools.datastructures.task.TaskLog import TaskLog
    from maaf_tools.maaf_tools.datastructures.task.Task import Task

    from maaf_tools.maaf_tools.tools import deep_compare
    from maaf_tools.maaf_tools.Singleton import SLogger

##################################################################################################################

DEBUG = True


@dataclass
class Agent(MaafItem):
    # ----- Fixed
    id: str                                     # ID of the agent
    name: str                                   # Name of the agent
    agent_class: str                            # Class of the agent

    # ----- Variable
    #hierarchy_level: int                        # Hierarchy level of the agent
    specs: dict                                 # Specifications of the agent
    skillset: list[str]                         # Skillset of the agent

    state: AgentState                           # State of the agent, state object
    plan: Plan                                  # Plan of the agent, plan object

    # local: NestedDict = field(default_factory=NestedDict)   # Local data of the agent, does not get serialized and passed around
    # shared: NestedDict = field(default_factory=NestedDict)  # Shared data of the agent, gets serialized and passed around

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
                         position: Optional[int] = None,
                         bid: Optional[float] = None,
                         logger=None
                         ) -> (bool, bool):
        """
        Add a task to the plan of the agent. The task is added to the task bundle and the path is updated.

        :param tasklog: The task log containing the tasks and the paths between them.
        :param task: The task to be added to the plan.
        :param position: The position in the plan sequence at which to add the task. Add to the end of the plan if None.
        :param logger: The logger to log messages to (optional).

        :return: A tuple containing the success of adding the task to the task bundle and updating the plan.
        """

        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if not self.plan.has_task(task_id):
            # -> Add the task to the task bundle
            add_task_success = self.plan.add_task(
                task=task_id,
                position=position
            )

            if logger and add_task_success:
                if bid is not None:
                    bid_text = f"(bid: {bid}) "
                else:
                    bid_text = ""

                if position is not None:
                    position_text = f"at position {position} "
                else:
                    position_text = ""

                logger.info(f"{self.id} + Task {task_id} assigned to self {position_text}{bid_text}- Pending task count: {len(tasklog.ids_pending)})")

            elif logger and not add_task_success:
                logger.info(f"!!! Task {task_id} could not be added to the plan of {self.id} !!!")

        else:
            add_task_success = True

        # # -> Update the plan with the path from the task log
        # update_plan_success = self.update_plan(tasklog=tasklog)

        return add_task_success

    def remove_task_from_plan(self,
                              tasklog: TaskLog,
                              task: Task or str,
                              forward: bool = False,
                              motive: Optional[str] = None,
                              logger=None
                              ) -> (bool, bool):
        """
        Remove a task from the plan of the agent. The task is removed from the task bundle and the path is updated.

        :param tasklog: The task log containing the tasks and the paths between them.
        :param task: The task to be removed from the plan.
        :param forward: Whether to remove the tasks after the specified task.
        :param motive: The motive for removing the task.
        :param logger: The logger to log messages to (optional).

        :return: A tuple containing the success of removing the task from the task bundle and updating the plan.
        """

        if isinstance(task, Task):
            task_id = task.id
        else:
            task_id = task

        if self.plan.has_task(task_id):
            # -> Remove the task from the task bundle
            remove_task_success = self.plan.remove_task(
                task=task_id,
                forward=forward
            )

            if logger and remove_task_success:
                if motive is not None:
                    motive_text = f"(motive: {motive}) "
                else:
                    motive_text = ""

                logger.info(f"{self.id} - Task {task_id} dropped from task list {motive_text}- Pending task count: {len(tasklog.ids_pending)}")

            elif logger and not remove_task_success:
                logger.warning(f"{self.id} - Task {task_id} not in task list")

        else:
            remove_task_success = True

        return remove_task_success

    # ============================================================== Get

    # ============================================================== Set
    def set_online_state(self, online: bool = True) -> None:
        """
        Set the online state of the agent.

        :param online: Whether the agent is online or not.
        """

        self.state.online = online

    # ============================================================== Merge
    def merge(self,
              agent: "Agent",
              prioritise_local: bool = False,
              robust_merge: bool = False,
              *args, **kwargs
              ) -> (bool, bool, bool, bool):
        """
        Merge the current agent with another agent.

        :param agent: The agent to merge with.
        :param prioritise_local: Whether to prioritise the local fields when merging (add only).
        :param robust_merge: Whether to perform a robust merge (a merge that will not fail even if the signatures do not match).

        :return: A tuple containing the success of the merge and whether the agent has been enabled or disabled.
            - agent_state_change: Whether the agent state has changed.
            - agent_plan_change: Whether the agent plan has changed.
            - agent_enabled: Whether the agent has been enabled.
            - agent_disabled: Whether the agent has been disabled.
            Note: The agent_enabled and agent_disabled flags reflect if the agent has been enabled or disabled, not the current state.
        """

        # -> Verify if agent is of type Agent
        if not isinstance(agent, Agent):
            raise ValueError(f"!!! Agent merge failed: Agent is not of type Agent: {type(agent)} !!!")

        # -> Verify that the agents signatures match
        if agent.signature != self.signature and not robust_merge:
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

                # > If field_value != getattr(self, field.name):
                if not deep_compare(field_value, getattr(self, field.name)):
                    # > Update the field value
                    setattr(self, field.name, field_value)
                    agent_state_change = True

            # ---- Merge state
            if self.state != agent.state:
                if agent.state.status == "active" and self.state.status == "inactive":
                    agent_enabled = True
                elif agent.state.status == "inactive" and self.state.status == "active":
                    agent_disabled = True

                self.state = agent.state

            # ---- Merge plan
            if self.plan != agent.plan:
                self.plan = agent.plan

                agent_state_change = True
                agent_plan_change = True

        return (
                agent_state_change, # Has the agent state changed?
                agent_plan_change,  # Has the agent plan changed?
                agent_enabled,      # Has the agent been enabled?
                agent_disabled      # Has the agent been disabled?
                )

    # ============================================================== Serialization / Parsing



if __name__ == "__main__":
    from pprint import pprint
    import pandas as pd

    agent = Agent(
        id="agent_0",
        name="Agent 0",
        agent_class="Drone",
        hierarchy_level=0,
        affiliations=["affiliation_0"],
        specs={"speed": 10},
        skillset=["skill_0", "skill_1"],
        state=AgentState(
            x=0,
            y=0,
            z=0,
            _timestamp=1,
            agent_id="agent_0",
        ),
        plan=Plan(),
        shared={
            "c_matrix": pd.DataFrame({
                "agent_1": [0, 1, 2],
                "agent_2": [1, 0, 3],
                "agent_3": [2, 3, 0]
            }),
        }
    )

    def print_agent():
        return None

    agent.state.set_get_timestamp(print_agent)

    print(agent.state.timestamp)

    agent_dict = agent.asdict()
    pprint(agent_dict)

    print("\n------------------------------\n")
    agent2 = Agent.from_dict(agent_dict, partial=True)

    # pprint(agent2.asdict())

    print(agent2.state.timestamp)

    # agent.merge(agent2)
