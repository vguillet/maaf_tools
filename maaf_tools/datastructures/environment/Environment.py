
##################################################################################################################

import json
from pprint import pprint
import warnings

import networkx as nx

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.tools import loads, dumps

except ImportError:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.tools import loads, dumps

##################################################################################################################


class Environment(MaafItem):
    """
    Environment class to hold environment variables and their values.
    """

    def __init__(self,
                 name: str,
                 graph: str or nx.Graph,
                 shortest_paths: dict = None,
                 compute_missing_paths: bool = True,
                 description: str = None):
        """
        Initialize the Environment object.

        :param name: Name of the environment.
        :param graph: Graph associated with the environment.
        :param shortest_paths: Shortest paths in the graph.
        :param compute_missing_paths: Whether to compute missing paths if they are not complete.
        :param description: Description of the environment.
        """
        self.name = name
        self.description = description

        # -> Check if the graph is a valid NetworkX graph
        if not isinstance(graph, nx.Graph) and not isinstance(graph, str):
            raise TypeError("Graph must be a NetworkX graph or a json string.")

        # -> If the graph is a string, parse it
        if isinstance(graph, str):
            try:
                self.graph = self.from_json(json_str=graph)
            except json.decoder.JSONDecodeError:
                raise ValueError("Graph string is not a valid JSON string.")

        else:
            self.graph = graph

        # -> If shortest_paths is not None:
        self.shortest_paths = shortest_paths
        self.check_shortest_paths(compute_missing_paths=compute_missing_paths)

    # ============================================================== Properties
    @property
    def nodes(self):
        """
        Get the nodes of the graph.

        :return: The nodes of the graph.
        """
        if isinstance(self.graph, nx.Graph):
            return self.graph.nodes
        else:
            return None

    @property
    def edges(self):
        """
        Get the edges of the graph.

        :return: The edges of the graph.
        """
        if isinstance(self.graph, nx.Graph):
            return self.graph.edges
        else:
            return None

    @property
    def pos(self):
        """
        Get the positions of the nodes in the graph.

        :return: The positions of the nodes in the graph.
        """
        if isinstance(self.graph, nx.Graph):
            return nx.get_node_attributes(self.graph, "pos")
        else:
            return None

    # ============================================================== Check
    def check_shortest_paths(self, compute_missing_paths: bool = True, verbose: int = 1) -> bool:
        """
        Check if shortest paths are:
        - in the right format
        - complete
            > If not, compute the missing paths if compute_missing_paths is True.

        :param compute_missing_paths: Whether to compute missing paths if they are not complete.
        :param verbose: Verbosity level.

        :return: bool indicating if the shortest paths are complete.
        """
        if self.shortest_paths is None:
            return False

        if not isinstance(self.shortest_paths, dict):
            raise TypeError("Shortest paths must be a dictionary.")

        # -> Check if the shortest paths are in the right format
        for key, value in self.shortest_paths.items():
            if not isinstance(key, str):
                raise TypeError("Shortest paths keys must be strings.")
            if not isinstance(value, dict):
                raise TypeError("Shortest paths values must be dictionaries.")

        # -> Check if the shortest paths refer to nodes in the graph
        for key, value in self.shortest_paths.items():
            if key not in self.graph.nodes:
                raise ValueError(f"Node {key} is not in the graph.")
            for key2, value2 in value.items():
                if key2 not in self.graph.nodes:
                    raise ValueError(f"Node {key2} is not in the graph.")

        # -> Check if the shortest paths are complete
        missing_paths = []

        for node in self.graph.nodes:
            for node2 in self.graph.nodes:
                shortest_path = self.get_shortest_path(source=node, target=node2, compute_missing_paths=compute_missing_paths)

                if shortest_path is None:
                    missing_paths.append(f"\n - No path between {node} and {node2}.")

        if verbose >= 1:
            if len(missing_paths) > 0:
                print(f"Missing paths: {missing_paths}")
                print(f"Computing missing paths...")

        return True

    # ============================================================== Get
    def get_shortest_path(self, source: str, target: str, compute_missing_paths: bool = True) -> list or None:
        """
        Get the shortest path between two nodes.

        :param source: Source node.
        :param target: Target node.
        :param compute_missing_paths: Whether to compute missing paths if they are not complete.

        :return: The shortest path between the source and target nodes.
        """
        if source not in self.shortest_paths.keys() or target not in self.shortest_paths[source].keys():
            if compute_missing_paths:
                self.compute_shortest_paths(source=source, target=target)
            else:
                return None

        return self.shortest_paths[source][target]

    def compute_shortest_paths(self, source: str, target: str) -> list or None:
        """
        Compute the shortest path between two nodes.

        :param source: Source node
        :param target: Target node
        :return: The shortest path between the source and target nodes.
        """

        if source not in self.graph.nodes or target not in self.graph.nodes:
            raise ValueError(f"Node {source} or {target} is not in the graph.")

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path

        except nx.NetworkXNoPath:
            warnings.warn(f"No path between {source} and {target}.")
            return None

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    def asdict(self, include_local: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the dataclass instance with their current values.

        :param include_local: Whether to include the local field in the dictionary.

        :return: A dictionary with field names as keys and current values.
        """
        return {
            "name": self.name,
            "description": self.description,
            "graph": self.graph,
            "shortest_paths": self.shortest_paths
        }

    @classmethod
    def  from_dict(cls, item_dict: dict, partial: bool = False) -> object:
        """
        Convert a dictionary to a dataclass instance.

        :param item_dict: The dictionary representation of the dataclass
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A task object
        """
        return cls(
            name=item_dict["name"],
            graph=item_dict["graph"],
            shortest_paths=item_dict["shortest_paths"],
            description=item_dict["description"]
        )

    def to_json(self, shortest_paths=None) -> dict:
        """
        Convert the graph and the positions to a JSON string.

        :return: The JSON string representation of the graph.
        """

        graph_json = {
            "env_type": "graph",
            "graph": nx.node_link_data(self.graph),
            "pos": {str(k): v for k, v in self.pos.items()}
        }

        if shortest_paths is not None:
            graph_json["shortest_paths"] = {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in
                                            shortest_paths.items()}

        return graph_json

    @staticmethod
    def from_json(json_str: str, *args, **kwargs) -> (nx.Graph, dict):
        """
        Convert a JSON string to a graph.

        :param json_str: The JSON string representation of the graph.

        :return: A graph object.
        :return: A dictionary containing the positions of the nodes.
        """

        data = loads(json_str)

        graph = nx.node_link_graph(data["graph"])
        pos = {eval(k): v for k, v in data["pos"].items()}
        env_type = data["env_type"]

        environment = {
            "env_type": env_type,
            "graph": graph,
            "pos": pos
        }

        if "shortest_paths" in data.keys():
            environment["shortest_paths"] = {eval(k): {eval(k2): v2 for k2, v2 in v.items()} for k, v in
                                             data["shortest_paths"].items()}

        # -> Add all other fields from the data that is not graph or pos or env_type
        for key, value in data.items():
            if key not in environment.keys():
                environment[key] = value

        return environment

if __name__ == "__main__":
    # Load graph from JSON
    with open("icare_alloc_config/icare_alloc_config/Environment/environment_graph.json", "r") as f:
        graph_json = f.read()

    env = Environment(name="test", graph=graph_json)
    env_dict = env.asdict()
    print(env_dict)

    env_json = env.to_json()
    print(env_json)

    env_from_json = Environment.from_json(env_json)
    print(env_from_json)

    env_from_dict = Environment.from_dict(env_dict)
    print(env_from_dict)

    # -> Check if the objects are equal
    assert env == env_from_json
    assert env == env_from_dict