
##################################################################################################################
from dataclasses import dataclass, fields, field
from typing import List, Optional

from maaf_tools.datastructures.MaafItem import MaafItem
from maaf_tools.datastructures.MaafList import MaafList
from maaf_tools.datastructures.agent.AgentState import AgentState
from maaf_tools.datastructures.agent.Plan import Plan

##################################################################################################################

DEBUG = True


@dataclass
class Agent(MaafItem):
    id: str                                 # ID of the agent
    name: str                               # Name of the agent
    agent_class: str                        # Class of the agent
    hierarchy_level: int                    # Hierarchy level of the agent
    affiliations: List[str]                 # Affiliations of the agent
    specs: dict                             # Specifications of the agent
    skillset: List[str]                     # Skillset of the agent
    state: AgentState                       # State of the agent, state object

    # shared: NestedDict = field(default_factory=NestedDict)  # Shared data of the agent, gets serialized and passed around
    # local: NestedDict = field(default_factory=NestedDict)   # Local data of the agent, does not get serialized and passed around

    plan: Plan                              # Plan of the agent, plan object

    shared: dict = field(default_factory=dict)  # Shared data of the agent, gets serialized and passed around
    local: dict = field(default_factory=dict)   # Local data of the agent, does not get serialized and passed around

    def __repr__(self) -> str:
        return f"Agent {self.name} ({self.id}) of class {self.agent_class} - Status: {self.state.status}"

    def __str__(self) -> str:
        return self.__repr__()

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
        field_names = {field.name for field in agent_fields}

        if not partial:
            # -> Check if all required fields are present in the dictionary
            if not field_names.issubset(agent_dict.keys()):
                raise ValueError(f"!!! Agent creation from dictionary failed: Agent dictionary is missing required fields: {agent_dict.keys() - field_names} !!!")

        else:
            # > Remove all fields not present in the dictionary
            agent_fields = [field for field in agent_fields if field.name in agent_dict]

        # -> Extract values from the dictionary for the fields present in the class
        field_values = {field.name: agent_dict[field.name] for field in agent_fields}

        # -> Convert state from dict
        field_values["state"] = AgentState.from_dict(agent_dict["state"])

        # -> Convert plan from dict
        field_values["plan"] = Plan.from_dict(agent_dict["plan"])

        # -> Create and return an Agent object
        return cls(**field_values)