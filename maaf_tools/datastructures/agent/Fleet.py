
##################################################################################################################

from dataclasses import dataclass, fields, field
from typing import List, Optional
from copy import deepcopy

try:
    from maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.datastructures.organisation.Organisation import Organisation

    from maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.datastructures.agent.Agent import Agent
    from maaf_tools.Singleton import SLogger

except:
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.maaf_tools.datastructures.organisation.Organisation import Organisation

    from maaf_tools.maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.maaf_tools.datastructures.agent.Agent import Agent
    from maaf_tools.maaf_tools.Singleton import SLogger

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

    # ============================================================== Merge
    def merge(self,
              fleet: "Fleet",
              add_agent_callback: Optional[callable] = None,
              remove_agent_callback: Optional[callable] = None,
              fleet_state_change_callback: Optional[callable] = None,
              prioritise_local: bool = False,
              *args, **kwargs
              ) -> bool:
        """
        Merge the fleet with another fleet. The local fleet can be prioritised over the new fleet.

        :param fleet: The fleet to merge with the local fleet.
        :param add_agent_callback: A callback function to call when an agent is added to the fleet.
        :param remove_agent_callback: A callback function to call when an agent is removed from the fleet.
        :param fleet_state_change_callback: A callback function to call at the end of the merge if the fleet state has changed.
        :param prioritise_local: Whether to prioritise the local fleet over the new fleet.

        :return: A boolean indicating whether the fleet state has changed.
        """

        # -> If the fleet is None, return False
        if fleet is None:
            return False

        # -> Verify if the fleet is of type Fleet
        if not isinstance(fleet, Fleet):
            raise ValueError(f"!!! Fleet merge failed: Fleet is not of type Fleet: {type(fleet)} !!!")

        # -> Check if the fleet is the same as the current fleet
        if fleet is self:
            return False

        fleet_state_change = 0

        # ----- Merge agents
        # -> For all agents in the fleet to merge ...
        for agent in fleet:
            # -> If the agent is not in the local fleet ...
            if agent.id not in self.ids:
                # -> If the agent is active, add to fleet and extend local states with new columns
                if agent.state.status == "active":
                    self.add_agent(agent=agent)             # Add agent to the fleet

                    fleet_state_change += 1                 # Set the fleet state change flag to True

                    if add_agent_callback is not None:
                        add_agent_callback(agent=agent)     # Call the add agent callback

                # -> If the agent is inactive, only add to the local fleet
                else:
                    self.add_agent(agent=agent)             # Add agent to the fleet

            else:
                agent_state_change, agent_plan_change, agent_enabled, agent_disabled = self[agent.id].merge(
                    agent=agent,
                    prioritise_local=prioritise_local,
                    *args, **kwargs
                )

                if agent_state_change:
                    fleet_state_change += 1

                if add_agent_callback is not None and agent_enabled:
                    add_agent_callback(agent=agent)

                elif remove_agent_callback is not None and agent_disabled:
                    remove_agent_callback(agent=agent)

        # -> Call the fleet state change callback if the fleet state has changed
        if fleet_state_change_callback is not None and fleet_state_change > 0:
            fleet_state_change_callback()

        return fleet_state_change > 0

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

    # ============================================================== Serialization / Parsing
    @classmethod
    def from_config_files(
            cls,
            fleet_agents: list,
            agent_classes: dict,
            organisation_model: Organisation or dict = None,
        ) -> "Fleet":
        """
        Create a Fleet object from configuration files.

        :param fleet_agents: The fleet agents configuration file.
        :param agent_classes: The agent classes configuration file.
        :param organisation_model: The organisation model configuration file. If None, a new Organisation object will be created.

        :return: A Fleet object.
        """

        # TODO: Add logic for checking if the config files are correctly formatted

        if organisation_model is None:
            organisation_model = Organisation()

        elif isinstance(organisation_model, dict):
            organisation_model = Organisation.from_dict(data=organisation_model)

        # -> If the organisation model contains a fleet, remove it
        if organisation_model is not None and organisation_model.fleet is not None:
            organisation_model.fleet = None

        fleet = cls()

        # -> For all agents in the fleet ...
        for agent in fleet_agents:
            # -> Create an agent object from the configuration file
            agent = Agent(
                id=agent["id"],
                name=agent["name"],
                agent_class=agent["agent_class"],

                # Determined from organisational structure
                specs=agent_classes[agent["agent_class"]]["specs"],
                skillset=agent_classes[agent["agent_class"]]["skillset"],

                organisation_model=organisation_model,

                # Default values
                state=AgentState(
                    agent_id=agent["id"],
                    _timestamp=0,
                    battery_level=100,
                    stuck=False,
                    x=0,
                    y=0,
                    z=0,
                    u=0,
                    v=0,
                    w=0
                ),
                plan=Plan(),
            )

            # -> Add the agent to the fleet
            fleet.add_agent(agent=agent)

        return fleet

if "__main__" == __name__:
    import pandas as pd
    from pprint import pprint

    # Test Agent data class
    agent1 = Agent(
        id="1",
        name="Agent 1",
        agent_class="class 1",
        hierarchy_level=1,
        affiliations=["affiliation 1", "affiliation 2"],
        specs={"spec1": "value1", "spec2": "value2"},
        skillset=["skill1", "skill2"],
        state=AgentState(
            agent_id=1,
            _timestamp=1.0,
            battery_level=100,
            stuck=False,
            x=0,
            y=0,
            z=0,
            u=0,
            v=0,
            w=0,
        ),
        plan=Plan(),
        local={"local1": "value1"},
        shared={
            "c_matrix": pd.DataFrame({
                "agent_1": [0, 1, 2],
                "agent_2": [1, 0, 3],
                "agent_3": [2, 3, 0]
            }),
        },
    )

    def print_agent():
        return None

    agent1.state.set_get_timestamp(print_agent)

    agent2 = Agent.from_dict(agent1.asdict())
    agent2.name = "Agent 2"
    agent2.id = "2"
    agent2.local["local1"] = "value2"

    # print(agent1.id, agent1.name, agent1.local)
    # print(agent2.id, agent2.name, agent2.local)

    # Test Fleet data class
    fleet_1 = Fleet()
    fleet_1.add_agent(agent1)
    fleet_1.add_agent(agent2)

    fleet_2 = Fleet()
    fleet_2.add_agent(agent2)
    # print(fleet_1)
    # print(fleet_2)
    #
    # print(fleet_1["1"].name, fleet_1["1"].local)
    fleet_1.merge(fleet_2, prioritise_local=False)
    # print(fleet_1["1"].name, fleet_1["1"].local)

    for agent in fleet_1:
        print(agent.state.timestamp)

    # print(fleet_1.asdf().to_string())
    # print("====================================================================")
    # # print(fleet_1)
    # pprint(fleet_1.asdict())
    #
    # print("------------------------", fleet_1, fleet_1.clone())
    # pprint(type(fleet_1.asdict()["items"][0]["shared"]["c_matrix"]))
    # pprint(Fleet.from_dict(fleet_1.asdict()))
    #
    # pprint(fleet_1.clone().asdict())

