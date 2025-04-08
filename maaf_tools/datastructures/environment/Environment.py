
##################################################################################################################

import json
from functools import partial
from pprint import pprint
import warnings

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.tools import loads, dumps

except ImportError:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.tools import loads, dumps, convert_graph_gps_to_ecef

##################################################################################################################


class Environment(MaafItem):
    """
    Environment class to hold environment variables and their values.
    """

    def __init__(self,
                 data: dict = None,
                 name: str = None,
                 graph: str or nx.Graph = None,
                 shortest_paths: dict = None,
                 description: str = None,
                 compute_missing_paths: bool = True,
                 ):
        """
        Initialize the Environment object.

        :param name: Name of the environment.
        :param graph: Graph associated with the environment.
        :param shortest_paths: Shortest paths in the graph.
        :param compute_missing_paths: Whether to compute missing paths if they are not complete.
        :param description: Description of the environment.
        """

        if data is not None:
            name = data.get("name", "Unknown")
            graph = data
            shortest_paths = data.get("shortest_paths", None)
            description = data.get("description", "Unknown")

        self.name = name
        self.description = description
        self.environment_type = "graph"

        # -> Check if the graph is a valid NetworkX graph
        if not isinstance(graph, nx.Graph) and not isinstance(graph, str):
            raise TypeError("Graph must be a NetworkX graph or a json string.")

        # -> If the graph is a string, parse it
        if isinstance(graph, str):
            try:
                environment = self.from_json(json_str=graph)

                self.graph = environment["graph"]
                self.shortest_paths = environment["shortest_paths"]

            except json.decoder.JSONDecodeError:
                raise ValueError("Graph string is not a valid JSON string.")

        else:
            self.graph = graph

        # -> If shortest_paths is not None:
        self.shortest_paths = shortest_paths
        self.check_shortest_paths(compute_missing_paths=compute_missing_paths)

    def __repr__(self):
        """
        String representation of the Environment object.

        :return: String representation of the Environment object.
        """
        return f"Environment(name={self.name}, description={self.description}, graph={self.graph})"

    def __eq__(self, other):
        """
        Check if two Environment objects are equal.

        :param other: The other Environment object to compare with.

        :return: True if the objects are equal, False otherwise.
        """
        return (
            self.name == other.name and
            self.description == other.description and
            self.graph == other.graph and
            self.shortest_paths == other.shortest_paths
        )

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
    def pos3D(self):
        """
        Get the positions of the nodes in the graph.

        :return: The positions of the nodes in the graph.
        """
        if isinstance(self.graph, nx.Graph):
            return nx.get_node_attributes(self.graph, "pos")
        else:
            return None

    @property
    def pos2D(self):
        """
        Get the 2D positions of the nodes in the graph (3d pos minus the last element).

        :return: The 2D positions of the nodes in the graph.
        """
        if isinstance(self.graph, nx.Graph):
            pos3D = nx.get_node_attributes(self.graph, "pos")
            pos2D = {k: v[:-1] for k, v in pos3D.items()}
            return pos2D
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

    def compute_all_shortest_paths(self, recompute_all_shortest_paths) -> dict:
        """
        Compute all shortest paths in the graph.

        :param recompute_all_shortest_paths: Whether to recompute all shortest paths.
        :return: A dictionary containing all shortest paths.
        """
        if recompute_all_shortest_paths or self.shortest_paths is None:
            self.shortest_paths = dict()

        for node in self.graph.nodes:
            # Initialize dictionary for node if not already present
            if node not in self.shortest_paths:
                self.shortest_paths[node] = dict()

            for node2 in self.graph.nodes:
                # Optionally handle self-to-self case (if needed)
                if node == node2:
                    continue

                # If already computed, skip
                if node2 in self.shortest_paths[node]:
                    continue

                # Compute and store the shortest path
                path = self.compute_shortest_paths(source=node, target=node2)
                self.shortest_paths[node][node2] = path

        return self.shortest_paths

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

        # -> Convert the graph to a JSON string
        graph_dict = nx.node_link_data(G=self.graph)

        # -> Add the name and description to the graph JSON
        graph_dict["name"] = self.name
        graph_dict["description"] = self.description

        if self.shortest_paths is not None:
            graph_dict["shortest_paths"] = {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in
                                            self.shortest_paths.items()}

        return graph_dict

    @classmethod
    def from_dict(cls, item_dict: dict, partial: bool = False) -> object:
        """
        Convert a dictionary to a dataclass instance.

        :param item_dict: The dictionary representation of the dataclass
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A task object
        """

        # -> Extract the shortest paths if they exist
        shortest_paths = None

        if "shortest_paths" in item_dict.keys():
            shortest_paths = {eval(k): {eval(k2): v2 for k2, v2 in v.items()} for k, v in
                              item_dict["shortest_paths"].items()}

            # Remove the shortest paths from the graph item_dict
            del item_dict["shortest_paths"]

        # -> Ensure nodes have 3D positions (pos key)
        if any("pos" not in node for node in item_dict["nodes"]):
            item_dict = convert_graph_gps_to_ecef(item_dict)

        # -> Create the graph from node-link item_dict
        # Check if the keys for the edges is links or edges
        if "links" in item_dict.keys():
            graph = nx.node_link_graph(item_dict, edges="links")
        else:
            graph = nx.node_link_graph(item_dict, edges="edges")

        # -> Extract positions dictionary
        positions = {node["id"]: tuple(node["pos"]) for node in item_dict["nodes"]}

        # -> Add positions to the graph
        nx.set_node_attributes(graph, positions, name="pos")

        return cls(
            name=item_dict.get("name", "Unknown"),
            graph=graph,
            shortest_paths=shortest_paths,
            description=item_dict.get("description", "Unknown"),
            compute_missing_paths=True
        )

    def to_json(self, shortest_paths=None, indent: int = 2) -> dict:
        """
        Convert the graph and the positions to a JSON string.

        :param shortest_paths: Whether or not to include shortest paths in the JSON string.
        :param indent: The indentation level for the JSON string.

        :return: The JSON string representation of the graph.
        """

        graph_dict = self.asdict(include_local=False)

        return dumps(graph_dict)

    @classmethod
    def from_json(cls, json_str: str, partial: bool = False, *args, **kwargs) -> (nx.Graph, dict):
        """
        Convert a JSON string to a graph.

        :param json_str: The JSON string representation of the graph.
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A graph object.
        :return: A dictionary containing the positions of the nodes.
        """
        # -> Load the JSON string
        data = loads(json_str)

        return cls.from_dict(item_dict=data, partial=partial)

if __name__ == "__main__":
    # Load graph from JSON
    env = Environment.load_from_file(
        filename="icare_alloc_config/icare_alloc_config/Environment/test_env.json",
        partial=True
    )
    print("Check 0:", env)

    env_dict = env.asdict()
    env_json = env.to_json()

    env.save_to_file("test_env.json")

    env_from_json = Environment.from_json(env_json)
    print("Check 3:", env_from_json)

    env_from_dict = Environment.from_dict(env_dict)
    print("Check 4:", env_from_dict)
