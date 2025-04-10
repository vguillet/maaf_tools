
##################################################################################################################

import json
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import warnings

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.agent.Fleet import Fleet
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_tools.datastructures.organisation.AllocationSpecification import AllocationSpecification

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.agent.Fleet import Fleet
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_tools.maaf_tools.datastructures.organisation.AllocationSpecification import AllocationSpecification

##################################################################################################################


class Organisation(MaafItem):
    def __init__(self,
                 fleet = None,
                 moise_model: MoiseModel or dict = None,
                 role_allocation: RoleAllocation or dict = None,
                 allocation_specification: dict = None,
                 ):
        # -> Setup fleet
        self.__fleet = None
        self.fleet = fleet

        # -> Setup Moise Model
        self.__moise_model = None
        self.moise_model = moise_model

        # -> Setup Role Allocation
        self.__role_allocation = None
        self.role_allocation = role_allocation

        # -> Setup Allocation Specification
        self.__allocation_specification = None
        self.allocation_specification = allocation_specification

    def __repr__(self):
        """Returns a string representation of the organisation."""
        string = (f"Organisation with: "
                  f"\n  > {self.fleet}"
                  f"\n  > {self.moise_model}"
                  f"\n  > {self.role_allocation}"
                  f"\n  > {self.allocation_specification}"
                 )

        return string

    def __str__(self):
        """Returns a string representation of the organisation."""
        return self.__repr__()

    # ============================================================== Properties
    # ------------------------- Fleet
    @property
    def fleet(self):
        return self.__fleet

    @fleet.setter
    def fleet(self, fleet):
        if fleet is None:
            self.__fleet = Fleet()

        elif isinstance(fleet, dict):
             self.__fleet = Fleet().from_dict(item_dict=fleet)

        elif isinstance(fleet, Fleet):
            self.__fleet = fleet

        else:
            raise ValueError("Invalid Fleet format. Must be either a Fleet instance or a dictionary.")

    # ------------------------- Moise Model
    @property
    def moise_model(self):
        return self.__moise_model

    @moise_model.setter
    def moise_model(self, moise_model):
        if moise_model is None:
            self.__moise_model = MoiseModel()

        elif isinstance(moise_model, dict):
            self.__moise_model = MoiseModel(moise_model)

        elif isinstance(moise_model, MoiseModel):
            self.__moise_model = moise_model

        else:
            raise ValueError("Invalid Moise Model format. Must be either a MoiseModel instance or a dictionary.")

    # ------------------------- Role Allocation
    @property
    def role_allocation(self):
        return self.__role_allocation

    @role_allocation.setter
    def role_allocation(self, role_allocation):
        if role_allocation is None:
            self.__role_allocation = RoleAllocation()

        elif isinstance(role_allocation, dict):
            self.__role_allocation = RoleAllocation(role_allocation)

        elif isinstance(role_allocation, RoleAllocation):
            self.__role_allocation = role_allocation

        else:
            raise ValueError("Invalid Role Allocation format. Must be either a RoleAllocation instance or a dictionary.")

        # -> If fleet has already been set, check the role allocation validity
        if self.fleet is not None:
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

    # ------------------------- Allocation Specification
    @property
    def allocation_specification(self):
        return self.__allocation_specification

    @allocation_specification.setter
    def allocation_specification(self, allocation_specification):
        if allocation_specification is None:
            self.__allocation_specification = AllocationSpecification()

        elif isinstance(allocation_specification, dict):
            self.__allocation_specification = AllocationSpecification(allocation_specification)

        elif isinstance(allocation_specification, AllocationSpecification):
            self.__allocation_specification = allocation_specification

        else:
            raise ValueError("Invalid Allocation Specification format. Must be either an AllocationSpecification instance or a dictionary.")

        # -> If Moise Model has already been set, set pointer in the allocation specification
        if self.moise_model is not None:
            self.__allocation_specification.moise_model = self.moise_model

        if self.role_allocation is not None:
            self.__allocation_specification.role_allocation = self.role_allocation

    # ============================================================== Check
    def check_role_allocation_validity(self, stop_at_first_error = False, verbose = 1):
        if self.role_allocation is None:
            raise ValueError("Role allocation not set.")

        if self.fleet is None:
            raise ValueError("Fleet not set.")

        # -> Checking the role allocation against the structural specification
        #     - Check if the role allocation is valid with respect to the structural specification
        #     - Check if the role allocation is valid with respect to the agents' skillsets and role skill requirements
        valid_wrt_structural_model = self.moise_model.structural_specification.check_role_allocation(
            role_allocation=self.role_allocation,
            fleet=self.fleet,
            stop_at_first_error=stop_at_first_error,
            verbose=verbose
        )

        return valid_wrt_structural_model

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Merge
    def merge(self, organisation: "Organisation", prioritise_local: bool = True) -> bool:
        """
        Merges the current Organisation with another Organisation instance.

        For each sub-component (fleet, moise_model, role_allocation, allocation_specification),
        the corresponding merge method is called with the given prioritise_local flag. After merging,
        cross-references between the MOISE model and allocation specification are re‑established.

        :param organisation: Another Organisation instance to merge into this one.
        :param prioritise_local: If True, existing (local) entries take precedence in case of conflicts;
                                  otherwise, the incoming values will override local entries.
        :return: True if merging succeeds.
        :raises ValueError: If organisation is not an Organisation instance.
        """
        if not isinstance(organisation, Organisation):
            raise ValueError("The model to merge must be an Organisation instance.")

        # Merge each sub-component using its own merge method.
        self.fleet.merge(organisation.fleet, prioritise_local=prioritise_local)
        self.moise_model.merge(organisation.moise_model, prioritise_local=prioritise_local)
        self.role_allocation.merge(organisation.role_allocation, prioritise_local=prioritise_local)
        self.allocation_specification.merge(organisation.allocation_specification, prioritise_local=prioritise_local)

        # Re-establish cross-references between the MOISE model and its sub-specifications.
        self.moise_model.structural_specification.functional_specification = self.moise_model.functional_specification
        self.moise_model.structural_specification.deontic_specification = self.moise_model.deontic_specification

        self.moise_model.functional_specification.structural_specification = self.moise_model.structural_specification
        self.moise_model.functional_specification.deontic_specification = self.moise_model.deontic_specification

        self.moise_model.deontic_specification.structural_specification = self.moise_model.structural_specification
        self.moise_model.deontic_specification.functional_specification = self.moise_model.functional_specification

        # Update allocation specification pointers.
        self.allocation_specification.moise_model = self.moise_model
        self.allocation_specification.role_allocation = self.role_allocation

        return True

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

    def asdict(self, include_local: bool = False) -> dict:
         """Returns a dictionary representation of the team."""
         return {
             "fleet": self.fleet.asdict(),
             "moise_model": self.moise_model.asdict(),
             "role_allocation": self.role_allocation,
             "allocation_specification": self.allocation_specification
         }

    @classmethod
    def from_dict(cls, item_dict: dict, partial: bool = False) -> object:
         """
         Initializes the team from a dictionary.

         :return: A new instance of Team.
         """

         return cls(
             fleet=item_dict["fleet"],
             moise_model=item_dict["moise_model"],
             role_allocation=item_dict["role_allocation"],
             allocation_specification=item_dict["allocation_specification"]
         )

    def save_to_file(self,
                     filename: str,
                     organisation: bool = True,
                     role_allocation: bool = False,
                     model: bool = False,
                     structural_specification: bool = False,
                     functional_specification: bool = False,
                     deontic_specification: bool = False,
                     allocation_specification: bool = False
                     ):
        """
        Saves the team to a file. Also saves the role allocation and model separately if specified.

        :param filename: The name of the file.
        :param organisation: If True, saves the entire organisation as a single JSON file.
        :param role_allocation: If True, saves the role allocation separately.
        :param model: If True, saves the model separately.
        :param structural_specification: If True, saves the structural specification separately.
        :param functional_specification: If True, saves the functional specification separately.
        :param deontic_specification: If True, saves the deontic specification separately.
        :param allocation_specification: If True, saves the allocation specification separately.
        """

        if not organisation and not role_allocation and not model:
            raise ValueError("At least one of 'combined', 'role_allocation', or 'model' must be True.")

        if organisation:
            with open(filename, "w") as f:
                f.write(self.to_json(indent=2))

        if role_allocation:
            role_allocation_filename = filename.replace(".json", "_role_allocation.json")
            self.role_allocation.save_to_file(role_allocation_filename)

        if model:
            model_filename = filename.replace(".json", "_moise+_model.json")
            self.moise_model.save_to_file(model_filename)

        if structural_specification:
            structural_specification_filename = filename.replace(".json", "_structural_specification.json")
            self.moise_model.structural_specification.save_to_file(structural_specification_filename)

        if functional_specification:
            functional_specification_filename = filename.replace(".json", "_functional_specification.json")
            self.moise_model.functional_specification.save_to_file(functional_specification_filename)

        if deontic_specification:
            deontic_specification_filename = filename.replace(".json", "_deontic_specification.json")
            self.moise_model.deontic_specification.save_to_file(deontic_specification_filename)

        if allocation_specification:
            allocation_specification_filename = filename.replace(".json", "_allocation_specification.json")
            self.allocation_specification.save_to_file(allocation_specification_filename)

if __name__ == "__main__":
    from maaf_tools.maaf_tools.datastructures.agent.Fleet import Fleet

    # -> Load moise+ model json
    #with open("icare_alloc_config/icare_alloc_config/CoHoMa_moise+_model.json", "r") as file:
    #    model = json.load(file)

    # -> Load Moise+ model components
    with open("icare_alloc_config/icare_alloc_config/moise_deontic_specification.json", "r") as file:
        deontic_specification = json.load(file)

    with open("icare_alloc_config/icare_alloc_config/moise_functional_specification.json", "r") as file:
        functional_specification = json.load(file)

    with open("icare_alloc_config/icare_alloc_config/moise_structural_specification.json", "r") as file:
        structural_specification = json.load(file)

    model = MoiseModel(
        deontic_specification=deontic_specification,
        functional_specification=functional_specification,
        structural_specification=structural_specification
    )

    # -> Load team compo json
    with open("icare_alloc_config/icare_alloc_config/fleet_role_allocation.json", "r") as file:
        role_allocation = json.load(file)

    role_allocation = RoleAllocation(role_allocation=role_allocation)

    # -> Construct fleet
    # Load agent classes json and fleet agents json
    with open("icare_alloc_config/icare_alloc_config/fleet_agent_classes.json", "r") as file:
        agent_classes = json.load(file)

    with open("icare_alloc_config/icare_alloc_config/fleet_agents.json", "r") as file:
        fleet_agents = json.load(file)

    # Construct fleet instance
    fleet = Fleet.from_config_files(
        fleet_agents=fleet_agents,
        agent_classes=agent_classes
    )

    with open("icare_alloc_config/icare_alloc_config/fleet_allocation_specification.json", "r") as file:
        allocation_specification = json.load(file)

    organisation_model = Organisation(
        fleet=fleet,
        moise_model=model,
        role_allocation=role_allocation,
        allocation_specification=allocation_specification
    )

    # -------------------------- Save to file
    organisation_model.save_to_file(
        filename="icare_alloc_config/icare_alloc_config/__cache/__CoHoMa_organisation_model_v1.json",
        organisation=True,
        role_allocation=True,
        model=True,
        structural_specification=True,
        functional_specification=True,
        deontic_specification=True,
        allocation_specification=True
    )

    with open("icare_alloc_config/icare_alloc_config/__cache/__CoHoMa_organisation_model_v1.json", "r") as file:
        model = json.load(file)

    organisation_model = Organisation.from_dict(item_dict=model)

    print(organisation_model)

    # -> Check what tasks can be handled by D_2
    agent_id = "D_2"



    #print(organisation_model.allocation_specification.get_group_ambassadors("ScoutingTeam_1"))
    #organisation_model.plot_team_structure()

    # print(organisation_model)

    # organisation_model.plot_team_structure()

    # print("Missing roles:", organisation_model.moise_model.check_missing_roles_in_group(role_allocation=role_allocation, group_instance="SentinelTeam_1", verbose=0))
    # print(organisation_model.role_allocation_valid)