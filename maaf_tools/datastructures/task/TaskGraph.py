
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
from copy import deepcopy
from random import choice

# Libs
import networkx as nx
from networkx import MultiGraph, Graph, DiGraph

# Local Imports
try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.Singleton import SLogger

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.Singleton import SLogger

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
    main_agent_id: str = None

    def __str__(self):
        string = self.graph.__str__()

        # > Count paths
        path_count = 0
        for edge in self.graph.edges:
            path_count += len(self.graph[edge[0]][edge[1]]["path"])

        string += f" ({path_count} paths)"

        return string

    def __repr__(self):
        return self.graph.__repr__()

    def __getitem__(self, item):
        return self.graph[item]

    def set_main_agent_node(self, agent_id: str):
        """
        Add a main agent node to the graph.
        """

        # -> If main agent id already set, remove the node
        if self.main_agent_id:
            self.graph.remove_node(self.main_agent_id)

        # -> Set the main agent id
        self.main_agent_id = agent_id

        # > Add the main agent node
        self.graph.add_node(agent_id, node_type="Agent")

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    # ============================================================== Add
    def add_node(self, node_for_adding: str, node_type: str = "Task", **kwargs):
        """
        Add a node to the graph. Wrapper methods to ensure that the consequent edges are added with the correct properties.

        :param node_for_adding: The node to add. Task id
        :param path: The path to the node.
        """

        self.graph.add_node(node_for_adding, node_type=node_type, **kwargs)

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
            !!! If either source or target nodes is type Agent, default to "latest" selection.
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

        edges = [(uv_edge, uv_path)]

        # -> Add the path in both directions
        if two_way:
            vu_edge = (target_node, source_node)
            vu_path = deepcopy(path)

            # > Reverse path
            vu_path["path"] = vu_path["path"][::-1]

            # > Add source and target keys
            vu_path["source"] = target_node
            vu_path["target"] = source_node

            edges += [(vu_edge, vu_path)]

        # -> If either source or target nodes is agent, default to latest selection
        if self.nodes[source_node]["node_type"] == "Agent" or self.nodes[target_node]["node_type"] == "Agent":
            selection = "latest"

        # -> For every edge...
        for edge, path in edges:
            # edge = edge_[0]
            # path = edge_[1]

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

    def get_sequence_paths(self,
                           node_sequence: List[str],
                           requirement: Optional[List[str]] = None,
                           selection: str = "shortest"   # "shortest", "longest", "random", "all"
                           ) -> (list[dict], list):
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

        # -> Verify that for each path in the sequence, the one that follows it starts with the same node if it is not None
        if len(sequence_paths) > 1:
            for i in range(len(sequence_paths) - 1):
                if sequence_paths[i] is not None and sequence_paths[i + 1] is not None:
                    if list(sequence_paths[i]["path"][-1]) != list(sequence_paths[i + 1]["path"][0]):

                        SLogger().info(f"!!! TaskGraph get_sequence_paths failed: Path sequence is not continuous: {list(sequence_paths[i]['path'][-1])} != {list(sequence_paths[i + 1]['path'][0])} !!!"
                                       f"\nNode sequence: {node_sequence}"
                                       f"\n{[path['path'] for path in sequence_paths]}\n\n"
                                       )

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

        :return: The path(s) between the nodes. If no path is found, return None.
        """

        # -> Verify if the source and target nodes exist
        if not self.has_node(source) or not self.has_node(target):
            if not self.has_node(source):
                print(f"!!! TaskGraph get_path failed: Source node does not exist: {source} !!!")
            else:
                print(f"!!! TaskGraph get_path failed: Target node does not exist: {target} !!!")
            return None

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
            # print(f"!!! TaskGraph get_path failed: No path found between {source} and {target} meeting the requirements: {requirement} !!!")
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

    # ============================================================== Visualise
    def draw(self, **kwargs):
        """
        Draw the graph.
        """
        # Filter edges based on path property
        edges_to_draw = [(u, v) for u, v in self.graph.edges() if self.graph[u][v]['path']]

        # Create a circular layout
        pos = nx.circular_layout(self.graph)

        # Draw the filtered edges
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges_to_draw)

        # Draw the nodes
        nx.draw_networkx_nodes(self.graph, pos)

        # Draw labels if needed
        # nx.draw_networkx_labels(self.graph, pos)

        # Show the plot
        plt.axis('off')
        plt.show()

        # # -> Draw nodes
        # nx.draw_networkx_nodes(self.graph)
        # # nx.draw_circular(self.graph, **kwargs)
        # plt.show()

    # ============================================================== Merge
    def merge(self, task_graph: "TaskGraph", prioritise_local: bool = False) -> None:
        """
        # TODO: Refactor to specify how to prioritise paths in merge
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

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the Task data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """

        graph_copy = deepcopy(self.graph)

        # -> Remove all agent nodes which are not the main agent
        for node in self.nodes:
            if self.nodes[node]["node_type"] == "Agent" and node != self.main_agent_id:
                graph_copy.remove_node(node)

        # -> Return the dictionary representation of the graph
        return {
            "graph": nx.node_link_data(graph_copy),
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
    import matplotlib.pyplot as plt
    from pprint import pprint

    # -> Create a graph
    graph = TaskGraph()

    # -> Add nodes
    graph.add_node("A", pos=(1, 0))
    graph.add_node("B", pos=(0, 1))
    graph.add_node("C", pos=(1, 1))

    # -> Add path
    # graph.add_path("agent", "A", path={"id": "gse", "path": ["agent", "A"], "requirements": ["ground"]}, two_way=False)
    # graph.add_path("agent", "A", path={"id": "srtghg", "path": ["agent", "a"], "requirements": ["ground"]}, two_way=False)
    graph.add_path("A", "B", path={"id": "szt", "path": ["A", "B"], "requirements": ["air"]}, two_way=False)
    # graph.add_path("B", "C", path={"id": "tduyj", "path": ["B", "C"], "requirements": ["ground"]}, two_way=True)
    # graph.add_path("C", "A", path={"id": "fghn", "path": ["C", "A"], "requirements": ["ground"]}, two_way=True)

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

    # print('\n["agent", "A", "B", "C"]:')
    #
    # for path in graph.get_sequence_paths(["agent", "A", "B", "C"], requirement=["ground"]):
    #     print("     >", path)
    #
    # print("\n['agent', 'C', 'B', 'A']:")
    #
    # for path in graph.get_sequence_paths(["agent", "C", "B", "A"], requirement=["air"]):
    #     print("     >", path)
    #
    # print("\n['agent', 'A', 'C', 'B']:")
    #
    # for path in graph.get_sequence_paths(["agent", "A", "C", "B"], requirement=["ground"]):
    #     print("     >", path)

    print("\n")

    print(graph)
    print(graph.asdict())
    # print(graph.clone())

    print(graph.nodes)

    # Print nodes with property node type
    for node in graph.nodes:
        print(f"{node}: {graph.nodes[node]}")