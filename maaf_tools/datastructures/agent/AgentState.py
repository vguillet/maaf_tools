
##################################################################################################################

from dataclasses import dataclass, fields, field
import pandas as pd

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.datastructures.state.State import State
    from maaf_tools.datastructures.agent.Plan import Plan

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    from maaf_tools.maaf_tools.datastructures.state.State import State
    from maaf_tools.maaf_tools.datastructures.agent.Plan import Plan


##################################################################################################################


@dataclass
class AgentState(State):
    # --- Metadata ---
    agent_id: str or int                # The ID of the agent the state is for

    # --- Status ---
    online: bool = False                # Whether the target agent is online or not
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
        return (f"State of agent {self.agent_id} at {self.timestamp}"
                f"\n    > Online: {self.online}"
                f"\n    > Battery level: {self.battery_level}%"
                f"\n    > Position: ({self.x}, {self.y}, {self.z})"
                )

    @property
    def pos(self) -> list:
        """
        Get the location of the agent as a tuple.

        :return: A tuple containing the x and y coordinates of the agent.
        """
        return [self.x, self.y]

    @property
    def status(self) -> str:
        """

        """

        active = True

        # -> Check if battery level is low
        active *= self.battery_level > self.__min_battery_level

        # -> Check if the agent is stuck
        active *= not self.stuck

        return "active" if active and self.online else "inactive"

    # ============================================================== Serialization / Parsing
    def asdf(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the AgentState data class instance.
        """

        # -> Get the fields of the AgentState class
        state_fields = fields(self)

        # -> Create a dictionary with field names as keys and their current values
        fields_dict = {f.name: getattr(self, f.name) for f in state_fields}

        # -> Create a DataFrame from the dictionary
        state_df = pd.DataFrame([fields_dict])

        return state_df
