
##################################################################################################################

# Built-in/Generic Imports
from dataclasses import dataclass, fields, field
from typing import List, Optional
from datetime import datetime

# Libs
import networkx as nx
from networkx import MultiGraph, Graph

# Local Imports
from maaf_tools.datastructures.MaafItem import MaafItem

# from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


class Path(dict):
    def __init__(self):
        self["path"] = []
        self["requirements"] = []
        self["memo"] = []

        super().__init__(self)


@dataclass
class TaskGraph(MaafItem):
    graph: Graph = field(default_factory=Graph)

    def __str__(self):
        return self.graph.__str__()

    def __repr__(self):
        return self.graph.__repr__()

    def __getitem__(self, item):
        return self.graph[item]

    def add_node(self, node_for_adding, **kwargs):
        """
        Add a node to the graph. Wrapper methods to ensure that the consequent edges are added with the correct properties.
        """

        self.graph.add_node(node_for_adding, **kwargs)

        # > Add edge new task and all existing tasks
        for existing_task in self.graph.nodes:
            if existing_task != node_for_adding:
                self.graph.add_edge(
                    u_of_edge=node_for_adding,
                    v_of_edge=existing_task,
                    paths=[]
                )

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        """
        Add an edge to the graph. Wrapper methods to ensure that the edge is added with the correct properties.
        """

        self.graph.add_edge(
            u_for_edge,
            v_for_edge,
            paths=[],
            key=key,
            **attr)

    def add_path(self,
                 source_node,
                 target_node,
                 path: Path,
                 ):
        """
        Add a path to the graph.
        """

        if path not in self.graph[source_node][target_node]["paths"]:
            self.graph[source_node][target_node]["paths"].append(path)

    def get_paths(self,
                 source_node,
                 target_node,
                 ):
        """
        Get a path from the graph.
        """
        return self.graph[source_node][target_node]["path"]

    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the Task data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """
        return {
            "graph": nx.node_link_data(self.graph),
        }

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
    # -> Create a graph
    graph = TaskGraph()

    # -> Add nodes
    graph.add_node("A", pos=(0, 0))
    graph.add_node("B", pos=(1, 1))

    # -> Add edges
    graph.add_edge("A", "B", weight=1)

    # -> Get dictionary representation
    graph_dict = graph.asdict()

    print(graph_dict)