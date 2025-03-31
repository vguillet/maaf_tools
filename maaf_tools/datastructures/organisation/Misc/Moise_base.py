
##################################################################################################################

import json
from pprint import pprint

##################################################################################################################

RELATIONS = ["acquaintance", "communication", "authority", "compatible", "couple"]
SCOPES = ["inter", "intra", "omni"]


class MoiseModel:
    """
    A class representing a MOISEPlus model for specifying multiagent system organizations.

    The model is divided into three main specifications:
      - Structural Specification: Contains roles, role relations, and groups.
      - Functional Specification: Contains social schemes (including goals, plans, missions, and preferences).
      - Deontic Specification: Contains permissions and obligations.

    This class provides methods to build the model programmatically, as well as parsers for
    loading from and serializing to JSON, following the provided reference format.
    """

    def __init__(self, structural_specification=None, functional_specification=None, deontic_specification=None):
        # Initialize with default empty specifications if none provided
        self.structural_specification = structural_specification or {
            "roles": [],
            "role_relations": [],
            "groups": []
        }
        self.functional_specification = functional_specification or {
            "social_schemes": []
        }
        self.deontic_specification = deontic_specification or {
            "permissions": [],
            "obligations": []
        }

    def __str__(self):
        """Returns a string representation (JSON format) of the MOISEPlus model."""
        return self.to_json(indent=2)

    # ============================================================== Properties
    @property
    def roles(self) -> list:
        """Returns the list of roles in the structural specification."""
        return self.structural_specification["roles"]

    @property
    def abstract_roles(self) -> list:
        """Returns the list of abstract roles in the structural specification."""
        return [role for role in self.roles if role["abstract"]]

    @property
    def concrete_roles(self) -> list:
        """Returns the list of concrete roles in the structural specification."""
        return [role for role in self.roles if not role["abstract"]]

    # ============================================================== Check

    # ============================================================== Get
    def get_role_relations(self, role: str) -> dict:
        """
        Returns the role relations for a given role. This method checks for all relations involving the role.

        :param role: role name
        :return: dictionary with the relations and scopes for the given role
        """

        # -> Check that the role is defined in the model
        if role not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{role}' is not defined in the model.")

        # -> Check relations
        relations = {}

        for relation in RELATIONS:
            relations[relation] = []

            for role_relation in self.structural_specification["role_relations"]:
                if role_relation["source"] == role and role_relation["type"] == relation:
                    relations[relation].append(role_relation)
        return relations

    def get_roles_relations_state(self, source: str, destination: str) -> dict:
        """
        Returns the role relations between two roles. This method checks for all relations between the source and
        the destination.

        :param source: source role
        :param destination: destination role
        :return: dictionary with the relations status and scopes between the source and the destination
        """

        # -> Check relations
        relations_status = {}

        for relation in RELATIONS:
            relations_status[relation] = self.get_roles_relation_state(source, destination, relation)

        return relations_status

    def get_roles_relation_state(self, source: str, destination: str, relation: str) -> dict:
        """
        Returns the role relations between two roles. This method checks for a specific relation, and returns both
        if the relation is True/False, and the scope (inter/intra) if True. Since the model also defines relations
        through inheritance, the method will recursively check the ancestors of the source role if no direct relation
        is found

        :param source: source role
        :param destination: destination role
        :param relation: relation type
        :return: dictionary with the relation status and scope
        """

        # -> Check that the relation is in the list of valid relations
        if relation not in RELATIONS:
            raise ValueError(f"Invalid relation type '{relation}'. "
                             f"Valid types are: {', '.join(RELATIONS)}")

        # -> Check that both roles exist in the model
        if source not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{source}' is not defined in the model.")

        if destination not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{destination}' is not defined in the model.")

        def check_ancestors(source: str, destination: str, relation: str):
            # Retrieve the role object for the given source
            role_obj = next((role for role in self.roles if role["name"] == source), None)
            if role_obj is None:
                return False, None

            # Check for a direct relation between the source and destination
            for role_relation in self.structural_specification["role_relations"]:
                if (role_relation["source"] == source and
                        role_relation["destination"] == destination and
                        role_relation["type"] == relation):
                    print(f"Role '{source}' -> Role '{destination}' ({relation}, {role_relation['scope']})")
                    return True, role_relation["scope"]

            # First, check for a relation via the destination's ancestors
            destination_obj = next((role for role in self.roles if role["name"] == destination), None)
            if destination_obj:
                inherited_target = destination_obj.get("inherits")
                if inherited_target:
                    found, scope = check_ancestors(source, inherited_target, relation)
                    if found:
                        return found, scope

            # Then, check for a relation via the source's ancestors
            inherited_source = role_obj.get("inherits")
            if inherited_source:
                found, scope = check_ancestors(inherited_source, destination, relation)
                if found:
                    return found, scope

            return False, None

        # -> Check relation state
        status, scope = check_ancestors(source, destination, relation)

        return {"status": status, "scope": scope}

    def get_group_specification(self, group_name: str):
        """
        Returns the specification of a group by name.

        :param group_name: The name of the group.
        :return: A dictionary with the group specification.
        """
        for group in self.structural_specification["groups"]:
            if group["name"] == group_name:
                return group
        return None

    # ============================================================== Set

    # ============================================================== Add
    # ------------------------ Structural Specification Methods
    def add_role(self, name: str, abstract: bool = False, inherits: str = None) -> None:
        """
        Adds a role to the structural specification.

        Args:
            name (str): The role name.
            abstract (bool): Whether the role is abstract (default False).
            inherits (str, optional): Name of the role this one inherits from.
        """
        role = {"name": name, "abstract": abstract}
        if inherits is not None:
            # Check if the inherited role is defined in the model
            if inherits not in [role["name"] for role in self.roles]:
                raise ValueError(f"Role '{inherits}' is not defined in the model.")

            role["inherits"] = inherits

        self.structural_specification["roles"].append(role)

    def add_role_relation(self, source: str, destination: str, relation_type: str, scope: str) -> None:
        """
        Adds a role relation.

        Args:
            source (str): The source role.
            destination (str): The destination role.
            relation_type (str): The type of relation (e.g., "authority", "communication").
            scope (str): The scope (e.g., "inter" or "intra").
        """

        # -> Check if the relation type is valid
        if relation_type not in RELATIONS:
            raise ValueError(f"Invalid relation type '{relation_type}'. "
                             f"Valid types are: {', '.join(RELATIONS)}")

        # -> Check if the scope is valid
        if scope not in SCOPES:
            raise ValueError(f"Invalid scope '{scope}'. "
                             f"Valid scopes are: {', '.join(SCOPES)}")

        # -> Check if the source and target roles are defined in the model
        if source not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{source}' is not defined in the model.")

        if destination not in [role["name"] for role in self.roles]:
            raise ValueError(f"Role '{destination}' is not defined in the model.")

        # -> Check if the relation already exists
        for relation in self.structural_specification["role_relations"]:
            if relation["source"] == source and relation["destination"] == destination and \
               relation["type"] == relation_type and relation["scope"] == scope:
                raise ValueError(f"Relation '{source}' -> '{destination}' ({relation_type}, {scope}) already exists.")

        # -> Add the relation
        relation = {
            "source": source,
            "destination": destination,
            "type": relation_type,
            "scope": scope
        }
        self.structural_specification["role_relations"].append(relation)

    def add_group(self, name: str, subgroups: dict or None = None, role_cardinality: dict or None = None) -> None:
        """
        Adds a group specification.

        Args:
            name (str): The group name.
            subgroups (dict, optional): A mapping from subgroup names to their role cardinality.
            role_cardinality (dict, optional): A mapping from role names to their cardinality.
                                                Example: {"coach": {"min": 1, "max": 2}, "player": {"min": 3, "max": 11}}
        """
        group = {"name": name}
        if subgroups is not None:
            # -> For every subgroup, check if it is defined in the model
            for subgroup in subgroups:
                if subgroup not in [group["name"] for group in self.structural_specification["groups"]]:
                    raise ValueError(f"Subgroup '{subgroup}' is not defined in the model.")
            group["subgroups"] = subgroups

        if role_cardinality is not None:
            group["role_cardinality"] = role_cardinality

        self.structural_specification["groups"].append(group)

    # ------------------------ Functional Specification Methods
    def add_social_scheme(self, name, goals=None, plans=None, missions=None, preferences=None):
        """
        Adds a social scheme to the functional specification.

        Args:
            name (str): The name of the social scheme.
            goals (list, optional): A list of goal dictionaries.
            plans (list, optional): A list of plan dictionaries.
            missions (list, optional): A list of mission dictionaries.
            preferences (list, optional): A list of preference dictionaries.
        """
        scheme = {"name": name}
        if goals is not None:
            scheme["goals"] = goals
        if plans is not None:
            scheme["plans"] = plans
        if missions is not None:
            scheme["missions"] = missions
        if preferences is not None:
            scheme["preferences"] = preferences
        self.functional_specification["social_schemes"].append(scheme)

    # ------------------------ Deontic Specification Methods
    def add_permission(self, role, mission, time_constraint="Any"):
        """
        Adds a permission to the deontic specification.

        Args:
            role (str): The role associated with the permission.
            mission (str): The mission to which the permission applies.
            time_constraint (str): The time constraint (default "Any").
        """
        permission = {"role": role, "mission": mission, "time_constraint": time_constraint}
        self.deontic_specification["permissions"].append(permission)

    def add_obligation(self, role, mission, time_constraint="Any"):
        """
        Adds an obligation to the deontic specification.

        Args:
            role (str): The role associated with the obligation.
            mission (str): The mission to which the obligation applies.
            time_constraint (str): The time constraint (default "Any").
        """
        obligation = {"role": role, "mission": mission, "time_constraint": time_constraint}
        self.deontic_specification["obligations"].append(obligation)

    # ============================================================== Remove
    # ------------------------ Structural Specification Methods
    def remove_role(self, name):
        """Removes a role by name. Return True if the role was removed, False otherwise."""

        for role in self.structural_specification["roles"]:
            if role["name"] == name:
                self.structural_specification["roles"].remove(role)
                return True
        return False

    def remove_role_relation(self, source, destination, relation_type, scope):
        """Removes a role relation. Return True if the relation was removed, False otherwise."""

        for relation in self.structural_specification["role_relations"]:
            if relation["source"] == source and relation["destination"] == destination and \
               relation["type"] == relation_type and relation["scope"] == scope:
                self.structural_specification["role_relations"].remove(relation)
                return True
        return False

    def remove_group(self, name):
        """Removes a group by name. Return True if the group was removed, False otherwise."""

        for group in self.structural_specification["groups"]:
            if group["name"] == name:
                self.structural_specification["groups"].remove(group)
                return True
        return False

    # ------------------------ Functional Specification Methods
    def remove_social_scheme(self, name):
        """Removes a social scheme by name. Return True if the scheme was removed, False otherwise."""

        for scheme in self.functional_specification["social_schemes"]:
            if scheme["name"] == name:
                self.functional_specification["social_schemes"].remove(scheme)
                return True
        return False

    # ------------------------ Deontic Specification Methods
    def remove_permission(self, role, mission):
        """Removes a permission. Return True if the permission was removed, False otherwise."""

        for permission in self.deontic_specification["permissions"]:
            if permission["role"] == role and permission["mission"] == mission:
                self.deontic_specification["permissions"].remove(permission)
                return True
        return False

    def remove_obligation(self, role, mission):
        """Removes an obligation. Return True if the obligation was removed, False otherwise."""

        for obligation in self.deontic_specification["obligations"]:
            if obligation["role"] == role and obligation["mission"] == mission:
                self.deontic_specification["obligations"].remove(obligation)
                return True
        return False

    # ============================================================== Serialization / Parsing

    def to_dict(self):
        """Returns the MOISEPlus model as a dictionary."""
        return {
            "structural_specification": self.structural_specification,
            "functional_specification": self.functional_specification,
            "deontic_specification": self.deontic_specification
        }

    def to_json(self, indent=2):
        """
        Serializes the MOISEPlus model to a JSON string.

        Args:
            indent (int): The indent level for pretty-printing.

        Returns:
            str: A JSON-formatted string representation of the model.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data):
        """
        Creates a MoiseModel instance from a dictionary.

        Args:
            data (dict): A dictionary containing the model.

        Returns:
            MoiseModel: A new instance of MoiseModel.
        """
        return cls(
            structural_specification=data.get("structural_specification",
                                              {"roles": [], "role_relations": [], "groups": []}),
            functional_specification=data.get("functional_specification", {"social_schemes": []}),
            deontic_specification=data.get("deontic_specification", {"permissions": [], "obligations": []})
        )

    @classmethod
    def from_json(cls, json_str):
        """
        Creates a MoiseModel instance from a JSON string.

        Args:
            json_str (str): A JSON string containing the model.

        Returns:
            MoiseModel: A new instance of MoiseModel.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, filename):
        """
        Saves the MOISEPlus model to a file in JSON format.

        Args:
            filename (str): The name of the file to save the model.
        """
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads a MOISEPlus model from a JSON file.

        Args:
            filename (str): The name of the file to load the model from.

        Returns:
            MoiseModel: A new instance of MoiseModel.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def plot(self):
        """
        Plots an interactive view of the MOISEPlus model showing:
          - The roles hierarchy at the top, drawn as a top-down tree with reversed arrows
            (arrows from child to parent, empty arrowheads).
          - The groups hierarchy at the bottom, drawn as a bottom-up tree
            (arrows from parent to child, empty arrowheads).
        Nodes are represented as rounded boxes (abstract roles in light gray, concrete roles in white).

        Note: This method lays out roles and groups in the same plot. Horizontal spacing
        of roles has been increased to prevent overlapping.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # -------------------- Roles Hierarchy (Top) --------------------
        G_roles = nx.DiGraph()
        for role in self.structural_specification["roles"]:
            role_name = role["name"]
            G_roles.add_node(role_name, abstract=role.get("abstract", False))
            if "inherits" in role:
                parent_role = role["inherits"]
                G_roles.add_node(parent_role, abstract=False)
                G_roles.add_edge(parent_role, role_name)

        # Try to use Graphviz for roles layout; if not available, use a custom top-down layout.
        try:
            pos_roles = nx.nx_agraph.graphviz_layout(G_roles, prog="dot")
        except Exception:
            def hierarchy_pos(G, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5):
                def _hierarchy_pos(G, root, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None):
                    if pos is None:
                        pos = {root: (xcenter, vert_loc)}
                    else:
                        pos[root] = (xcenter, vert_loc)
                    children = list(G.successors(root))
                    if children:
                        dx = width / len(children)
                        next_x = xcenter - width / 2 + dx / 2
                        for child in children:
                            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                                 vert_loc=vert_loc - vert_gap, xcenter=next_x, pos=pos)
                            next_x += dx
                    return pos

                roots = [n for n, d in G.in_degree() if d == 0]
                pos = {}
                if len(roots) == 1:
                    pos = _hierarchy_pos(G, roots[0], width, vert_gap, vert_loc, xcenter)
                else:
                    dx = width / len(roots)
                    for i, root in enumerate(roots):
                        pos.update(_hierarchy_pos(G, root, width=dx, vert_gap=vert_gap, vert_loc=vert_loc,
                                                  xcenter=xcenter - width / 2 + dx / 2 + i * dx))
                return pos

            pos_roles = hierarchy_pos(G_roles)

        # Increase horizontal spacing by applying a separate horizontal scale factor.
        horizontal_scale_roles = 4.0  # Increased horizontal scale factor
        vertical_scale_roles = 1.5  # Vertical scale factor remains
        pos_roles = {node: (x * horizontal_scale_roles, y * vertical_scale_roles) for node, (x, y) in pos_roles.items()}
        roles_offset = 3.0  # Shift roles upward so they appear at the top.
        pos_roles = {node: (x, y + roles_offset) for node, (x, y) in pos_roles.items()}

        # -------------------- Groups Hierarchy (Bottom) --------------------
        G_groups = nx.DiGraph()
        for group in self.structural_specification["groups"]:
            group_name = group["name"]
            G_groups.add_node(group_name)
            if "subgroups" in group:
                for subgroup in group["subgroups"]:
                    G_groups.add_node(subgroup)
                    G_groups.add_edge(group_name, subgroup)

        def hierarchy_pos_groups(G, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5):
            def _hierarchy_pos_groups(G, root, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None):
                if pos is None:
                    pos = {root: (xcenter, vert_loc)}
                else:
                    pos[root] = (xcenter, vert_loc)
                children = list(G.successors(root))
                if children:
                    dx = width / len(children)
                    next_x = xcenter - width / 2 + dx / 2
                    for child in children:
                        pos = _hierarchy_pos_groups(G, child, width=dx, vert_gap=vert_gap,
                                                    vert_loc=vert_loc + vert_gap, xcenter=next_x, pos=pos)
                        next_x += dx
                return pos

            roots = [n for n, d in G.in_degree() if d == 0]
            pos = {}
            if len(roots) == 1:
                pos = _hierarchy_pos_groups(G, roots[0], width, vert_gap, vert_loc, xcenter)
            else:
                dx = width / len(roots)
                for i, root in enumerate(roots):
                    pos.update(_hierarchy_pos_groups(G, root, width=dx, vert_gap=vert_gap,
                                                     vert_loc=vert_loc, xcenter=xcenter - width / 2 + dx / 2 + i * dx))
            return pos

        pos_groups = hierarchy_pos_groups(G_groups, width=1.0, vert_gap=0.5, vert_loc=0, xcenter=0.5)
        horizontal_scale_groups = 1.5
        vertical_scale_groups = 1.5
        pos_groups = {node: (x * horizontal_scale_groups, y * vertical_scale_groups) for node, (x, y) in
                      pos_groups.items()}

        # -------------------- Plotting Both Hierarchies --------------------
        fig, ax = plt.subplots()

        # Plot roles (top) with reversed arrows (child -> parent, empty arrowhead)
        for (parent, child) in G_roles.edges():
            ax.annotate("",
                        xy=pos_roles[parent], xycoords='data',
                        xytext=pos_roles[child], textcoords='data',
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='black',
                            shrinkA=5, shrinkB=5,
                            connectionstyle="arc3,rad=0",
                            fill=False,
                            mutation_scale=20
                        ))
        # Draw roles nodes as rounded boxes.
        for node, (x, y) in pos_roles.items():
            color = "lightgray" if G_roles.nodes[node].get("abstract", False) else "white"
            ax.text(x, y, node, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="black", lw=1))

        # Plot groups (bottom) with normal arrows (parent -> child, empty arrowhead)
        for (parent, child) in G_groups.edges():
            ax.annotate("",
                        xy=pos_groups[child], xycoords='data',
                        xytext=pos_groups[parent], textcoords='data',
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='black',
                            shrinkA=5, shrinkB=5,
                            connectionstyle="arc3,rad=0",
                            fill=False,
                            mutation_scale=20
                        ))
        # Draw groups nodes as rounded boxes.
        for node, (x, y) in pos_groups.items():
            ax.text(x, y, node, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))

        # -------------------- (Optional) Connect Groups and Roles --------------------
        # If there are relationships connecting groups to roles, draw them here.
        # Example placeholder:
        # for connection in self.structural_specification.get("group_role_connections", []):
        #     role = connection["role"]
        #     group = connection["group"]
        #     if role in pos_roles and group in pos_groups:
        #         ax.annotate("",
        #                     xy=pos_roles[role], xycoords='data',
        #                     xytext=pos_groups[group], textcoords='data',
        #                     arrowprops=dict(arrowstyle='-|>', color='red',
        #                                     shrinkA=5, shrinkB=5,
        #                                     connectionstyle="arc3,rad=0", fill=False, mutation_scale=20))

        # -------------------- Adjust Axis Limits and Display --------------------
        all_x = [x for pos in [pos_roles, pos_groups] for x, y in pos.values()]
        all_y = [y for pos in [pos_roles, pos_groups] for x, y in pos.values()]
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
        ax.relim()
        ax.autoscale_view()

        plt.title("Roles Hierarchy (Top) and Groups Hierarchy (Bottom)")
        plt.axis('off')
        plt.show()


# -------------------- Example Usage --------------------
if __name__ == "__main__":
    # Create a new MOISEPlus model
    model = MoiseModel()

    # --- Structural Specification ---
    # ----- Add roles
    model.add_role("Agent", abstract=True)

    # Coordination
    model.add_role(name="Auction Participant", abstract=True, inherits="Agent")
    model.add_role(name="Announcer", abstract=False, inherits="Auction Participant")
    model.add_role(name="Ambassador", abstract=True, inherits="Auction Participant")
    model.add_role(name="P1", abstract=False, inherits="Ambassador")
    model.add_role(name="P2", abstract=False, inherits="Ambassador")
    model.add_role(name="P3", abstract=False, inherits="Ambassador")
    model.add_role(name="Bidder", abstract=True, inherits="Auction Participant")
    model.add_role(name="P4", abstract=False, inherits="Bidder")

    # Operation
    model.add_role(name="Operational Agent", abstract=True, inherits="Agent")
    model.add_role(name="Scout", abstract=False, inherits="Operational Agent")
    model.add_role(name="Monitor", abstract=False, inherits="Operational Agent")
    model.add_role(name="Patroller", abstract=False, inherits="Operational Agent")
    model.add_role(name="Obstructor", abstract=False, inherits="Operational Agent")
    model.add_role(name="Trapper", abstract=False, inherits="Operational Agent")
    model.add_role(name="Tracker", abstract=False, inherits="Operational Agent")
    model.add_role(name="Neutraliser", abstract=False, inherits="Operational Agent")

    # Operators
    model.add_role(name="Operator", abstract=True, inherits="Agent")
    model.add_role(name="Tactical Operator", abstract=False, inherits="Operator")
    model.add_role(name="Situation Operator", abstract=False, inherits="Operator")
    model.add_role(name="Robot Operator", abstract=False, inherits="Operator")

    # ----- Add role relations
    # Acquaintance
    model.add_role_relation(source="Operator", destination="Agent", relation_type="acquaintance", scope="intra")
    model.add_role_relation(source="Tactical Operator", destination="Agent", relation_type="acquaintance", scope="inter")

    # Communication
    model.add_role_relation(source="Agent", destination="Agent", relation_type="communication", scope="omni")

    # Authority
    model.add_role_relation(source="Operator", destination="Agent", relation_type="authority", scope="intra")
    model.add_role_relation(source="Tactical Operator", destination="Agent", relation_type="authority", scope="omni")

    # Compatibility
    model.add_role_relation(source="Operator", destination="Announcer", relation_type="compatible", scope="omni")
    model.add_role_relation(source="Operator", destination="Ambassador", relation_type="compatible", scope="intra")
    model.add_role_relation(source="Tactical Operator", destination="Ambassador", relation_type="compatible", scope="omni")

    model.add_role_relation(source="Tactical Operator", destination="Auction Participant", relation_type="compatible", scope="inter")
    model.add_role_relation(source="Operational Agent", destination="Operational Agent", relation_type="compatible", scope="intra")
    model.add_role_relation(source="Operational Agent", destination="Auction Participant", relation_type="compatible", scope="intra")
    model.add_role_relation(source="Auction Participant", destination="Auction Participant", relation_type="compatible", scope="intra")

    # Couple
    model.add_role_relation(source="Operational Agent", destination="Bidder", relation_type="couple", scope="inter")
    model.add_role_relation(source="Operational Agent", destination="Bidder", relation_type="couple", scope="intra")

    # ----- Add groups
    model.add_group(name="Drone Swarm",
                    subgroups=None,
                    role_cardinality={
                        "Scout": {"min": 3, "max": 3}
                    })

    model.add_group(name="Situation Operator Team",
                    subgroups={
                        "Drone Swarm": {"min": 1, "max": 1},
                    },
                    role_cardinality={
                        # Auction Participants
                        "P1": {"min": 1, "max": 1},
                        "P2": {"min": 1, "max": 1},
                        "P3": {"min": 1, "max": 1},

                        # Operational Agents
                        "Scout": {"min": 3, "max": 4},
                        "Monitor": {"min": 1, "max": 1},
                        "Patroller": {"min": 1, "max": 2},

                        # Operators
                        "Situation Operator": {"min": 1, "max": 1},
                    })

    model.add_group(name="Defence Team",
                    subgroups=None,
                    role_cardinality={
                        # Auction Participants
                        "P1": {"min": 1, "max": 1},
                        "P2": {"min": 1, "max": 1},
                        "P3": {"min": 1, "max": 1},

                        # Operational Agents
                        "Patroller": {"min": 2, "max": None},
                        "Obstructor": {"min": 1, "max": None},
                        "Trapper": {"min": 1, "max": None},
                        "Tracker": {"min": 1, "max": None},
                        "Neutraliser": {"min": 1, "max": None},

                        # Operators
                        "Robot Operator": {"min": 1, "max": 1},
                    })

    model.add_group(name="Scouting Team",
                    role_cardinality={
                        # Auction Participants
                        "P1": {"min": 1, "max": 1},
                        "P2": {"min": 1, "max": 1},
                        "P3": {"min": 1, "max": 1},

                        # Operational Agents
                        "Scout": {"min": 2, "max": None},
                        "Tracker": {"min": 1, "max": None},
                        "Neutraliser": {"min": 1, "max": None},

                        # Operators
                        "Robot Operator": {"min": 1, "max": 1},
                    })

    model.add_group(name="Team",
                    subgroups={
                        "Situation Operator Team": {"min": 1, "max": 1},
                        "Defence Team": {"min": 1, "max": None},
                        "Scouting Team": {"min": 1, "max": None}
                    },
                    role_cardinality={}
                    )


    # --- Functional Specification ---
    # social_scheme = {
    #     "goals": [
    #         {"id": "g0", "description": "Score a goal"},
    #         {"id": "g2", "description": "Ball in midfield"},
    #         {"id": "g3", "description": "Ball in attack field"},
    #         {"id": "g4", "description": "Shoot at goal"}
    #     ],
    #     "plans": [
    #         {"goal": "g0", "subgoals": ["g2", "g3", "g4"], "operator": "sequence"}
    #     ],
    #     "missions": [
    #         {"id": "m1", "goals": ["g2", "g3", "g4"], "min_agents": 1, "max_agents": 4}
    #     ],
    #     "preferences": [
    #         {"higher": "m1", "lower": "m2"}
    #     ]
    # }
    # model.add_social_scheme("score_goal",
    #                         goals=social_scheme["goals"],
    #                         plans=social_scheme["plans"],
    #                         missions=social_scheme["missions"],
    #                         preferences=social_scheme["preferences"])
    # --- Deontic Specification ---    def plot(self):
    # model.add_permission("goalkeeper", "m7")
    # model.add_obligation("defender", "m1")
    # model.add_obligation("coach", "m6")
    # model.add_obligation("midfielder", "m2")
    # model.add_obligation("midfielder", "m3")
    # model.add_obligation("attacker", "m4")
    # model.add_obligation("attacker", "m5")

    # Print out the JSON representation of the model
    # print(model)

    source = "Tactical Operator"
    destination = "Bidder"

    print(f"Relations between '{source}' and '{destination}':\n")
    pprint(model.get_roles_relations_state(source=source, destination=destination), indent=2)

    # print(json.dumps(model.get_role_relations(role=source), indent=2))

    # model.plot()

    pprint(model.get_group_specification("Defence Team"), indent=2)

    model.save_to_file(filename="CoHoMa_v1")