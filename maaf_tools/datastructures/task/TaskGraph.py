
##################################################################################################################
"""

path: {
    "id": str,
    "source": str,
    "target": str,
    "path": list[str],
    "requirements": list[str], (ground, air, water, etc.)
}

"""
# Built-in/Generic Imports
from dataclasses import dataclass, fields, field
from typing import List, Optional
from datetime import datetime
from copy import deepcopy
from random import choice

# Libs
import networkx as nx
from networkx import MultiGraph, Graph, DiGraph

# Local Imports
try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


class Path(dict):
    def __init__(self):
        self["path"] = []
        self["requirements"] = []
        self["memo"] = []

        super().__init__(self)


@dataclass
class TaskGraph(MaafItem):
    graph: DiGraph = field(default_factory=DiGraph)

    def add_agent(self, **kwargs):
        # -> Add agent node
        if "agent" not in self.graph.nodes:
            self.add_node(node_for_adding="agent", **kwargs)

    def __str__(self):
        return self.graph.__str__()

    def __repr__(self):
        return self.graph.__repr__()

    def __getitem__(self, item):
        return self.graph[item]

    # ============================================================== Add
    def add_node(self, node_for_adding: str, **kwargs):
        """
        Add a node to the graph. Wrapper methods to ensure that the consequent edges are added with the correct properties.

        :param node_for_adding: The node to add. Task id
        :param path: The path to the node.
        """

        self.graph.add_node(node_for_adding, **kwargs)

        # > Add edge new task and all existing tasks
        for existing_task in self.graph.nodes:
            if existing_task != node_for_adding:
                # -> Add uv edge
                self.graph.add_edge(
                    u_of_edge=node_for_adding,
                    v_of_edge=existing_task,
                    path=[]
                )

                # -> Add vu edge
                self.graph.add_edge(
                    u_of_edge=existing_task,
                    v_of_edge=node_for_adding,
                    path=[]
                )

    def add_edge(self, u_for_edge, v_for_edge, path: dict or List[dict] = {}, key=None, **kwargs):
        """
        Add an edge to the graph. Wrapper methods to ensure that the edge is added with the correct properties.
        """
        # -> Ensure path is a list
        path = [path] if isinstance(path, dict) else path

        # -> Add edge
        self.graph.add_edge(
            u_for_edge,
            v_for_edge,
            path=path,
            key=key,
            **kwargs)

    def add_path(self,
                 source_node: str,
                 target_node: str,
                 path: dict or List[dict],
                 two_way: bool = True,
                 selection: str = "latest",   # "shortest", "longest", "random", "latest", "all"
                 ) -> None:
        """
        Add a path to the graph.

        :param source_node: The source node of the path.
        :param target_node: The target node of the path.
        :param path: The path to add.
        :param two_way: Whether to add the path in both directions.
        :param selection: The selection method for the path if multiple meet the requirements.
            "shortest" : Replace the current shortest path with the new path if the new path is shorter.
            "longest"  : Replace the current longest path with the new path if the new path is longer.
            "random"   : Randomly replace a path with the new path.
            "latest"   : Replace all current paths with the new path.
            "all"      : Keep all paths. Add the new path to the list of paths.
            !!! If either source or target nodes is agent, default to "latest" selection.
        """

        # -> Recursively add paths if path is a list
        if isinstance(path, list):
            for path_ in path:
                self.add_path(
                    source_node=source_node,
                    target_node=target_node,
                    path=path_,
                    two_way=two_way,
                    selection=selection
                )

        # -> List of edges to add the path to
        uv_edge = (source_node, target_node)
        uv_path = deepcopy(path)

        # > Add source and target keys
        uv_path["source"] = source_node
        uv_path["target"] = target_node

        vu_edge = (target_node, source_node)
        vu_path = deepcopy(path)

        # > Reverse path
        vu_path["path"] = vu_path["path"][::-1]

        # > Add source and target keys
        vu_path["source"] = target_node
        vu_path["target"] = source_node

        edges = [(uv_edge, uv_path)]

        # -> Add the path in both directions
        if two_way:
            edges += [(vu_edge, vu_path)]

        # -> If either source or target nodes is agent, default to latest selection
        if source_node == "agent" or target_node == "agent":
            selection = "latest"

        # -> For ever edge...
        for edge_ in edges:
            edge = edge_[0]
            path = edge_[1]

            # -> List paths with the same requirements
            comparable_paths = [
                existing_path for existing_path in self.graph[edge[0]][edge[1]]["path"]
                if existing_path["requirements"] == path["requirements"]
            ]

            # -> If multiple paths with the same requirements exist, select which to keep
            # > Shortest path
            if selection == "shortest":
                current_path = None

                for comparable_path in comparable_paths:
                    if current_path is None:
                        current_path = comparable_path
                    elif len(comparable_path["path"]) < len(current_path["path"]):
                        current_path = comparable_path

                # -> Compare the path lengths
                if current_path is not None:
                    # -> If the new path is shorter, replace the old path
                    if len(path) < len(current_path):
                        # -> Replace the path
                        self.graph[edge[0]][edge[1]]["path"].remove(current_path)
                        self.graph[edge[0]][edge[1]]["path"].append(path)

                    # -> Else, do not add the path
                    else:
                        pass

                # -> If no comparable path exists, add the path
                else:
                    self.graph[edge[0]][edge[1]]["path"].append(path)

            # > Longest path
            elif selection == "longest":
                # -> Get the longest comparable path
                current_path = None

                for comparable_path in comparable_paths:
                    if current_path is None:
                        current_path = comparable_path
                    elif len(comparable_path["path"]) > len(current_path["path"]):
                        current_path = comparable_path

                # -> Compare the path lengths
                if current_path is not None:
                    # -> If the new path is longer, replace the old path
                    if len(path) > len(current_path):
                        # -> Replace the path
                        self.graph[edge[0]][edge[1]]["path"].remove(current_path)
                        self.graph[edge[0]][edge[1]]["path"].append(path)

                    # -> Else, do not add the path
                    else:
                        pass

                # -> If no comparable path exists, add the path
                else:
                    self.graph[edge[0]][edge[1]]["path"].append(path)

            # > Random
            elif selection == "random":
                # -> Randomly select a path to replace
                current_path = choice(comparable_paths)

                # -> Replace the path
                self.graph[edge[0]][edge[1]]["path"].remove(current_path)
                self.graph[edge[0]][edge[1]]["path"].append(path)

            # > Latest
            elif selection == "latest":
                # -> Remove all comparable paths
                for comparable_path in comparable_paths:
                    self.graph[edge[0]][edge[1]]["path"].remove(comparable_path)

                # -> Add the path
                self.graph[edge[0]][edge[1]]["path"].append(path)

            # > All
            elif selection == "all":
                # -> Add the path
                self.graph[edge[0]][edge[1]]["path"].append(path)

            else:
                raise ValueError(f"Invalid path selection method for add_path: {selection}")

    # ============================================================== Remove
    def remove_node(self, node_for_removal: str):
        """
        Remove a node from the graph.
        """
        self.graph.remove_node(node_for_removal)

    def remove_edge(self, u_of_edge: str, v_of_edge: str):
        """
        Remove an edge from the graph.
        """
        self.graph.remove_edge(u_of_edge, v_of_edge)

    def remove_path(self, edge, path: dict):
        """
        Remove a path from the graph.
        """
        try:
            # -> Delete the path from the edge
            self.graph[edge[0]][edge[1]]["path"].remove(path)
        except:
            pass

    # ============================================================== Get
    def clone(self) -> "TaskGraph":
        """
        Clone the TaskGraph.
        """
        return self.from_dict(self.asdict())

    def get_sequence_path(self,
                          node_sequence: List[str],
                          requirement: Optional[List[str]] = None,
                          selection: str = "shortest"   # "shortest", "longest", "random", "all"
                          ) -> (List[dict], List):
        """
        Get a path from the graph.

        :param node_sequence: The sequence of nodes to get the path for.
        :param requirement: The acceptable requirements for the path.
        :param selection: The selection method for the path if multiple meet the requirements. "shortest", "longest", "random", "all"

        :return: The path between the nodes for each edge in the sequence ordered similarly to the sequence.
        """
        sequence_paths = []

        for i in range(len(node_sequence) - 1):
            paths = self.get_path(
                source=node_sequence[i],
                target=node_sequence[i + 1],
                requirement=requirement,
                selection=selection
            )

            # -> Add found paths to the list
            sequence_paths.append(paths)

        return sequence_paths

    def get_path(self,
                 source: str,
                 target: str,
                 requirement: Optional[List[str]] = None,
                 selection: str = "shortest"   # "shortest", "longest", "random", "all
                 ) -> List[dict] or dict or None:
        """
        Get a path from the graph. Return all the existing paths between two nodes meeting the requirements.

        :param source: The source node of the path.
        :param target: The target node of the path.
        :param requirement: The acceptable requirements for the path.
        :param selection: The selection method for the path if multiple meet the requirements. "shortest", "longest", "random", "all"

        :return: The path(s) between the nodes.
        """

        paths = []

        # -> Find the paths between the source and target nodes meeting the requirements
        for path in self.graph[source][target]["path"]:
            if requirement is not None:
                meet_requirements = True

                # -> Check if all requirements for the path are met
                for requirement_ in path["requirements"]:
                    if not requirement_ in requirement:
                        meet_requirements = False
                        break

                if meet_requirements:
                    paths.append(path)
            else:
                paths.append(path)

        if not paths:
            return None

        # -> Filter the paths based on the selection method
        # > Shortest path
        if selection == "shortest":
            shortest_path = None

            for path in paths:
                if shortest_path is None:
                    shortest_path = path

                else:
                    if len(path["path"]) < len(shortest_path["path"]):
                        shortest_path = path

            return shortest_path

        # > Longest path
        elif selection == "longest":
            longest_path = None

            for path in paths:
                if longest_path is None:
                    longest_path = path

                else:
                    if len(path["path"]) > len(longest_path["path"]):
                        longest_path = path

            return longest_path

        # > Random path
        elif selection == "random":
            return choice(paths)

        elif selection == "all":
            return paths

        else:
            raise ValueError(f"Invalid path selection method: {selection}")

    # ============================================================== Set

    # ============================================================== Check
    def has_node(self, node: str) -> bool:
        """
        Check if the graph has a node.
        """
        return node in self.graph.nodes

    def has_edge(self, u: str, v: str) -> bool:
        """
        Check if the graph has an edge.
        """
        return self.graph.has_edge(u, v)

    # ============================================================== Merge
    def merge(self, task_graph: "TaskGraph", prioritise_local: bool = False) -> None:
        """
        Merge the current TaskGraph with another TaskGraph.

        :param task_graph: The TaskGraph to merge with.
        :param prioritise_local: Whether to prioritise the local fields when merging (add only).
        """

        # -> Verify if task_graph is of type TaskGraph
        if not isinstance(task_graph, TaskGraph):
            raise ValueError(f"!!! TaskGraph merge failed: TaskGraph is not of type TaskGraph: {type(task_graph)} !!!")

        # -> Check if the graphs is the same as the current graph
        if task_graph is self:
            return

        # -> Merge the graphs
        if prioritise_local:
            self.graph = nx.compose(task_graph.graph, self.graph)
        else:
            self.graph = nx.compose(self.graph, task_graph.graph)

        return

    # ============================================================== To
    def asdict(self, include_agent: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the Task data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """

        if include_agent or "agent" not in self.graph.nodes:
            task_graph = self.graph

        else:
            # -> Create graph clone
            task_graph = deepcopy(self.graph)

            # -> Remove agent node
            task_graph.remove_node("agent")

        # -> Return the dictionary representation of the graph
        return {
            "graph": nx.node_link_data(task_graph),
        }

    # ============================================================== From
    @classmethod
    def from_dict(cls, task_dict: dict, partial: bool = False):
        """
        Convert a dictionary to a TaskGraph.

        :param task_dict: The dictionary representation of the graph
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A TaskGraph object
        """
        # -> Create a graph from the dictionary
        graph = nx.node_link_graph(task_dict["graph"])

        # -> Create and return a TaskGraph object
        return cls(graph=graph)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint

    # -> Create a graph
    graph = TaskGraph()
    graph.add_agent(pos=(0, 0))

    # -> Add nodes
    graph.add_node("A", pos=(1, 0))
    graph.add_node("B", pos=(0, 1))
    graph.add_node("C", pos=(1, 1))

    # -> Add path
    graph.add_path("agent", "A", path={"path": ["agent", "A"], "requirements": ["ground"]}, two_way=False)
    graph.add_path("agent", "A", path={"path": ["agent", "a"], "requirements": ["ground"]}, two_way=False)
    graph.add_path("A", "B", path={"path": ["A", "B"], "requirements": ["air"]}, two_way=False)
    graph.add_path("B", "C", path={"path": ["B", "C"], "requirements": ["ground"]}, two_way=True)
    graph.add_path("C", "A", path={"path": ["C", "A"], "requirements": ["ground"]}, two_way=True)

    # -> Get dictionary representation
    graph_dict = graph.asdict()

    pprint(graph_dict)

    # -> Plot graph with edge path labels
    nx.draw(
        graph.graph,
        with_labels=True,
        font_weight='bold',
        pos=nx.get_node_attributes(graph.graph, "pos"),
    )

    edge_labels = nx.get_edge_attributes(graph.graph, "path")

    for edge in edge_labels:
        try:
            edge_labels[edge] = f"Path: {edge_labels[edge][0]['path']}"
        except:
            edge_labels[edge] = f"Path: None"

    nx.draw_networkx_edge_labels(graph.graph, pos=nx.get_node_attributes(graph.graph, "pos"), edge_labels=edge_labels)

    # -> Show plot
    # plt.show()

    print('\n["agent", "A", "B", "C"]:')

    for path in graph.get_sequence_path(["agent", "A", "B", "C"], requirement=["ground"]):
        print("     >", path)

    print("\n['agent', 'C', 'B', 'A']:")

    for path in graph.get_sequence_path(["agent", "C", "B", "A"], requirement=["air"]):
        print("     >", path)

    print("\n['agent', 'A', 'C', 'B']:")

    for path in graph.get_sequence_path(["agent", "A", "C", "B"], requirement=["ground"]):
        print("     >", path)

    print("\n")
