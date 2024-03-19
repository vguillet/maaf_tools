
##################################################################################################################

from dataclasses import dataclass, fields, field

from maaf_tools.datastructures.MaafItem import MaafItem
from maaf_tools.datastructures.state.State import State

##################################################################################################################


@dataclass
class AgentState(State):
    # --- Metadata ---
    agent_id: str or int                # The ID of the agent the state is for

    # --- Status ---
    battery_level: float = None         # The battery level of the target agent in percentage
    __min_battery_level: float = 20     # The minimum battery level at which the agent is considered to have low battery

    stuck: bool = None                  # Whether the target agent is stuck or not

    # --- Position ---
    x: float = None                     # The x-coordinate of the target agent
    y: float = None                     # The y-coordinate of the target agent
    z: float = None                     # The z-coordinate of the target agent

    # --- Orientation ---
    u: float = None                     # The x-component of the velocity of the target agent
    v: float = None                     # The y-component of the velocity of the target agent
    w: float = None                     # The z-component of the velocity of the target agent

    def __repr__(self) -> str:
        return f"State of agent {self.agent_id} at {self.timestamp}"

    @property
    def status(self) -> str:
        """

        """

        active = True

        # -> Check if battery level is low
        active *= self.battery_level > self.__min_battery_level

        # -> Check if the agent is stuck
        active *= not self.stuck

        return "active" if active else "inactive"

    # ============================================================== To
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the AgentState data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """
        # -> Get the fields of the Agent class
        state_fields = fields(self)

        # -> Create a dictionary with field names as keys and their current values
        fields_dict = {f.name: getattr(self, f.name) for f in state_fields}

        return fields_dict

    # ============================================================== From
    @classmethod
    def from_dict(cls, agent_dict: dict, partial: bool = False) -> "AgentState":
        """
        Convert a dictionary to an AgentState.

        :param agent_dict: The dictionary representation of the AgentState
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: An agent object
        """

        # -> Get the fields of the AgentState class
        agent_fields = fields(cls)

        # -> Extract field names from the fields
        field_names = {field.name for field in agent_fields}

        if not partial:
            # -> Check if all required fields are present in the dictionary
            if not field_names.issubset(agent_dict.keys()):
                raise ValueError(f"!!! AgentState creation from dictionary failed: AgentState dictionary is missing required fields: {agent_dict.keys() - field_names} !!!")

        else:
            # > Remove fields not present in the dictionary
            agent_fields = [field for field in agent_fields if field.name in agent_dict]

        # -> Extract values from the dictionary for the fields present in the class
        field_values = {field.name: agent_dict[field.name] for field in agent_fields}

        # -> Create and return an Agent object
        return cls(**field_values)
