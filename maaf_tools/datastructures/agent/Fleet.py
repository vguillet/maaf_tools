
##################################################################################################################

from dataclasses import dataclass, fields, field
from typing import List, Optional
from datetime import datetime
from copy import deepcopy

from maaf_tools.datastructures.MaafList import MaafList

from maaf_tools.datastructures.agent.AgentState import AgentState
from maaf_tools.datastructures.agent.Plan import Plan
from maaf_tools.datastructures.agent.Agent import Agent

##################################################################################################################

DEBUG = True


@dataclass
class Fleet(MaafList):
    item_class = Agent
    __on_state_change_listeners: list[callable] = field(default_factory=list)

    def __repr__(self):
        return f"Fleet: {len(self.items)} agents ({len(self.agents_active)} active, {len(self.agents_inactive)} inactive)"

    # ============================================================== Listeners
    # ------------------------------ State
    def add_on_state_change_listener(self, listener: callable) -> None:
        """
        Add a listener to the list of listeners for the state change event.

        :param listener: The listener to add to the list of listeners.
        """
        self.__on_state_change_listeners.append(listener)

    def call_on_state_change_listeners(self, agent: Agent) -> None:
        """
        Call the state change listeners with the agent and the new state.

        :param agent: The agent that changed state.
        """
        for listener in self.__on_state_change_listeners:
            listener(agent)

    # ============================================================== Properties
    # ------------------------------ IDs
    @property
    def ids_active(self) -> List[str or int]:
        """
        Get the list of IDs of the active agents in the fleet
        """
        return [agent.id for agent in self.items if agent.state.status == "active"]

    @property
    def ids_inactive(self) -> List[str or int]:
        """
        Get the list of IDs of the inactive agents in the fleet
        """
        return [agent.id for agent in self.items if agent.state.status == "inactive"]

    # ------------------------------ Agents
    @property
    def agents_active(self) -> List[Agent]:
        """
        Get the list of active agents in the fleet
        """
        return [agent for agent in self.items if agent.state.status == "active"]

    @property
    def agents_inactive(self) -> List[Agent]:
        """
        Get the list of inactive agents in the fleet
        """
        return [agent for agent in self.items if agent.state.status == "inactive"]

    # ============================================================== Get
    def query(self,
              agent_class: str = None,
              hierarchy_level: int = None,
              affiliation: str = None,
              specs: List[str] = None,
              skillset: List[str] = None,
              status: str = None) -> List[Agent]:
        """
        Query the fleet for agents with specific characteristics

        :param agent_class: The class of the agents to query
        :param hierarchy_level: The hierarchy level of the agents to query
        :param affiliation: The affiliation of the agents to query
        :param specs: The specifications of the agents to query
        :param skillset: The skillset of the agents to query
        :param status: The status of the agents to query

        :return: A list of agents that match the query
        """

        # -> Create a list of agents to return
        filtered_agents = self.items

        # -> Filter the agents by class
        if agent_class is not None:
            filtered_agents = [agent for agent in filtered_agents if agent.agent_class == agent_class]

        # -> Filter the agents by hierarchy level
        if hierarchy_level is not None:
            filtered_agents = [agent for agent in filtered_agents if agent.hierarchy_level == hierarchy_level]

        # -> Filter the agents by affiliation
        if affiliation is not None:
            filtered_agents = [agent for agent in filtered_agents if affiliation in agent.affiliations]

        # -> Filter the agents by specs
        if specs is not None:
            filtered_agents = [agent for agent in filtered_agents if all([agent.specs[spec] for spec in specs])]

        # -> Filter the agents by skillset
        if skillset is not None:
            filtered_agents = [agent for agent in filtered_agents if all([skill in agent.skillset for skill in skillset])]

        # -> Filter the agents by status
        if status is not None:
            filtered_agents = [agent for agent in filtered_agents if agent.state.status == status]

        return filtered_agents

    # ============================================================== Set
    def set_agent_state(self, agent: str or int or item_class or List[int or str or item_class], state: dict or AgentState) -> None:
        """
        Set the state of an agent in the fleet. State can be a "active" or "inactive" string

        :param agent: The agent to set the state for. Can be the agent object, the agent ID, or a list of agent IDs.
        :param state: The state to set for the agent. Can be a dictionary or an Agent_state object.
        """
        # -> If the state is a dictionary, convert it to an Agent_state object
        if isinstance(state, dict):
            state = AgentState.from_dict(state)

        prev_state = deepcopy(agent.state)

        # -> Update the state of the agent
        self.update_item_fields(
            item=agent,
            field_value_pair={"state": state}
        )

        # -> Call on state change listeners
        if prev_state != agent.state:
            self.call_on_state_change_listeners(agent)

    def set_agent_plan(self, agent: str or int or item_class or List[int or str or item_class], plan: dict or Plan) -> None:
        """
        Set the plan of an agent in the fleet.

        :param agent: The agent to set the plan for. Can be the agent object, the agent ID, or a list of agent IDs.
        :param plan: The plan to set for the agent. Can be a dictionary or a Plan object.
        """
        # -> If the plan is a dictionary, convert it to a Plan object
        if isinstance(plan, dict):
            plan = Plan.from_dict(plan)

        # -> Update the plan of the agent
        self.update_item_fields(
            item=agent,
            field_value_pair={"plan": plan}
        )

    # ============================================================== Add
    def add_agent(self, agent: dict or Agent or List[dict or Agent]) -> None:
        """
        Add an agent to the fleet. If the agent is a list, add each agent to the fleet individually recursively.

        :param agent: The agent to add to the fleet.
        """
        # -> If the agent is a list, add each agent to the fleet individually recursively
        if isinstance(agent, list):
            for a in agent:
                self.add_agent(agent=a)
            return

        # > Add the agent to the fleet
        if isinstance(agent, self.item_class):
            self.add_item(item=agent)
        else:
            self.add_item_by_dict(item_data=agent)

    # ============================================================== Remove
    def remove_agent(self, agent: str or int or item_class or List[int or str or item_class]) -> None:
        """
        Remove an agent from the fleet.

        :param agent: The agent to remove from the fleet. Can be the agent object, the agent ID, or a list of agent IDs.
        """
        # -> If the input is a list, remove each agent in the list from the fleet individually recursively
        if isinstance(agent, list):
            for a in agent:
                self.remove_agent(agent=a)
            return

        # > Remove the agent from the fleet
        if isinstance(agent, self.item_class):
            self.remove_item(item=agent)
        else:
            self.remove_item_by_id(item_id=agent)


if "__main__" == __name__:
    # Test Agent data class
    agent1 = Agent(
        id=1,
        name="Agent 1",
        agent_class="class 1",
        hierarchy_level=1,
        affiliations=["affiliation 1", "affiliation 2"],
        specs={"spec1": "value1", "spec2": "value2"},
        skillset=["skill1", "skill2"],
        state=AgentState(
            agent_id=1,
            timestamp=1.0,
            battery_level=100,
            stuck=False,
            x=0,
            y=0,
            z=0,
            u=0,
            v=0,
            w=0
        )
    )
    print(agent1)
    print(agent1.asdict())

    agent2 = Agent.from_dict(agent1.asdict())
    print(agent2)

    # Test Fleet data class
    print("\nTest Fleet dataclass\n")
    fleet = Fleet()

    print("\nAdd agent to fleet:")
    fleet.add_agent(agent1)
    print(fleet)
    print(fleet.asdict())

    print("\nMark agent as inactive:")
    fleet.flag_agent_inactive(agent1)
    print(fleet)

    print("\nMark agent as active:")
    fleet.flag_agent_active(agent1)
    print(fleet)

    print("\nRemove agent from fleet:")
    fleet.remove_agent(agent1)
    print(fleet)




