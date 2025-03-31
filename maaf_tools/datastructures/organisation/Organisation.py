
##################################################################################################################

import json
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import warnings

from MoiseModel import MoiseModel
from RoleAllocation import RoleAllocation
# from agent.Fleet import Fleet

##################################################################################################################


class Organisation:
    def __init__(self,
                 # fleet: Fleet = None,
                 moise_model: (MoiseModel, dict) = None,
                 role_allocation: dict = None
                 ):
        # -> Setup fleet
        self.__fleet = None
        # self.fleet = fleet

        # -> Setup Moise Model
        self.__moise_model = None
        self.moise_model = moise_model

        # -> Setup Role Allocation
        self.__role_allocation = None
        self.role_allocation = role_allocation

    def __repr__(self):
        """Returns a string representation of the organisation."""
        string = f"MOISE+ Model:\n  > {self.moise_model}" \
                 f"Role allocation:\n  {self.role_allocation}"

        return string

    def __str__(self):
        """Returns a string representation of the organisation."""
        return self.__repr__()

    # ============================================================== Properties
    @property
    def fleet(self):
        return self.__fleet

    # @fleet.setter
    # def fleet(self, fleet):
    #     if fleet is None:
    #         self.__fleet = Fleet()
    #         return
    #     elif isinstance(fleet, Fleet):
    #         self.__fleet = fleet
    #
    #     elif isinstance(fleet, dict):
    #         self.__fleet = Fleet().from_dict(item_dict=fleet)
    #
    #     else:
    #         raise ValueError("Invalid Fleet format. Must be either a Fleet instance or a dictionary.")

    @property
    def moise_model(self):
        return self.__moise_model

    @moise_model.setter
    def moise_model(self, moise_model):
        if moise_model is None:
            self.__moise_model = MoiseModel()
            return

        elif moise_model is not None:
            if isinstance(moise_model, MoiseModel):
                self.__moise_model = moise_model

            elif isinstance(moise_model, dict):
                self.__moise_model = MoiseModel().from_dict(moise_model)

            else:
                raise ValueError("Invalid Moise Model format. Must be either a MoiseModel instance or a dictionary.")

            return

    @property
    def role_allocation(self):
        return self.__role_allocation

    @role_allocation.setter
    def role_allocation(self, role_allocation):
        if role_allocation is None:
            self.__role_allocation = RoleAllocation()
            return

        self.__role_allocation = RoleAllocation(role_allocation)

        if not self.role_allocation_valid:
            warnings.warn("Invalid role allocation, does not meet structural specification requirements.")

    @property
    def role_allocation_valid(self) -> bool:
        """
        Check if the role allocation is valid with respect to the agents' skillsets and structural specification.
        Pre-parameterised call to check_role_allocation_validity optimised for speed
        """
        # return self.check_role_allocation_validity(stop_at_first_error=True, verbose=0)
        return self.check_role_allocation_validity(stop_at_first_error=False, verbose=1)


    # ============================================================== Check
    def check_role_allocation_validity(self, stop_at_first_error = False, verbose = 1):
        if self.role_allocation is None:
            raise ValueError("Role allocation not set.")

        # -> Checking the role allocation against the agents skillsets
        valid_wrt_agents_skillsets = True

        # -> Checking the role allocation against the structural specification
        valid_wrt_structural_model = self.moise_model.structural_specification.check_role_allocation_against_model(
            role_allocation=self.role_allocation,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        return valid_wrt_agents_skillsets and valid_wrt_structural_model

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Plot
    def plot_team_structure(self) -> None:
        """
        Plots the group instance hierarchy and overlays role assignments from the team allocation.
        Now, each agent is represented as a smaller node attached to the team (instance) they belong to.
        Each node label displays the name first, the id below in brackets, and the roles below that (if applicable).
        """
        if self.role_allocation is None:
            raise ValueError("Role allocation not set.")

        group_instances = self.role_allocation.get("group_instances", [])
        G = nx.DiGraph()

        # Create team instance nodes with label: group type on top, then instance id in brackets.
        for gi in group_instances:
            instance_id = gi["instance"]
            label = f"{gi['group_type']}\n({instance_id})"
            G.add_node(instance_id, label=label, node_type="instance")
            parent = gi.get("parent")
            if parent:
                G.add_edge(parent, instance_id)

        # Create agent nodes with label: name, then id in brackets, then roles.
        for agent in self.role_allocation.get("team", []):
            agent_id = agent.get("agent_id", "unknown")
            name = agent.get("name", "unknown")
            for i, assignment in enumerate(agent.get("assignments", [])):
                inst = assignment.get("instance")
                roles = "\n".join(assignment.get("roles", []))
                classes = agent.get("class", "unknown")
                agent_node_id = f"{agent_id}_{inst}_{i}"

                label = f"{name}\n{classes}\n({agent_id})\n{roles}"
                G.add_node(agent_node_id, label=label, node_type="agent")
                G.add_edge(inst, agent_node_id)

        # Compute layout with increased spacing.
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args='-Gnodesep=2.0 -Granksep=1.5')
        except Exception:
            pos = nx.spring_layout(G)

        plt.figure(figsize=(12, 8))

        # Separate nodes based on type for different styling.
        instance_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "instance"]
        agent_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "agent"]

        # Draw team instance nodes (larger).
        nx.draw_networkx_nodes(G, pos, nodelist=instance_nodes, node_size=3000)
        # Draw agent nodes (smaller and in a different color).
        nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_size=1000, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, arrows=True)

        # Draw labels.
        labels = {node: attr.get("label", node) for node, attr in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

        # Adjust plot limits to ensure no nodes are clipped.
        x_values, y_values = zip(*pos.values())
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)

        plt.title("Group Instance Hierarchy and Role Assignments")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ============================================================== Serialization / Parsing

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the team."""
        return {
            "moise_model": self.moise_model.to_dict(),
            "role_allocation": self.role_allocation
        }

    def to_json(self, indent=2):
        """
        Serializes the team to a JSON string.

        :return:
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Initializes the team from a dictionary.

        :return: A new instance of Team.
        """

        return cls(
            moise_model=MoiseModel().from_dict(data["moise_model"]),
            role_allocation=data["role_allocation"]
        )

    @classmethod
    def from_json(cls, json_str: str):
        """
        Initializes the team from a JSON string.

        :return: A new instance of Team.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, filename: str):
        """
        Saves the team to a file.

        :param filename: The name of the file.
        """
        with open(filename, "w") as file:
            file.write(self.to_json())

    @classmethod
    def load_from_file(cls, filename: str):
        """
        Loads the team from a file.

        :param filename: The name of the file to load.
        :return: A new instance of Team.
        """
        with open(filename, "r") as file:
            return cls.from_json(file.read())

if __name__ == "__main__":
    # Load model json
    with open("CoHoMa_moise_model_v1.json", "r") as file:
        model = json.load(file)

    # Load team compo json
    with open("CoHoMa_role_composition.json", "r") as file:
        role_allocation = json.load(file)

    organisation_model = Organisation(
        moise_model=model,
        role_allocation=role_allocation
    )

    # -------------------------- Save to file
    organisation_model.save_to_file("CoHoMa_organisation_model_v1.json")

    with open("CoHoMa_organisation_model_v1.json", "r") as file:
        model = json.load(file)

    organisation_model = Organisation().from_dict(data=model)
    organisation_model.plot_team_structure()

    # print(organisation_model)

    # organisation_model.plot_team_structure()

    # print("Missing roles:", organisation_model.moise_model.check_missing_roles_in_group(role_allocation=role_allocation, group_instance="SentinelTeam_1", verbose=0))
    # print(organisation_model.role_allocation_valid)