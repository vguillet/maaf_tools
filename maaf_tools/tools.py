
from math import pi
from numpy import arctan2, arcsin
import networkx as nx
from json import loads, dumps
import numpy as np
import pandas as pd

from rclpy.time import Time


def graph_to_json(graph, pos) -> dict:
    """
    Convert the graph and the positions to a JSON string.

    :return: The JSON string representation of the graph.
    """

    graph_json = {
        "env_type": "graph",
        "graph": nx.node_link_data(graph),
        "pos": {str(k): v for k, v in pos.items()}
    }

    return graph_json


def json_to_graph(graph_json: str) -> (nx.Graph, dict):
    """
    Convert a JSON string to a graph.

    :param graph_json: The JSON string representation of the graph.

    :return: A graph object.
    :return: A dictionary containing the positions of the nodes.
    """

    if type(graph_json) == str:
        data = loads(graph_json)
    else:
        data = graph_json

    graph = nx.node_link_graph(data["graph"])
    pos = {eval(k): v for k, v in data["pos"].items()}
    env_type = data["env_type"]

    environment = {
        "env_type": env_type,
        "graph": graph,
        "pos": pos
        }

    return environment


def euler_from_quaternion(quat):
    """
    Convert quaternion (w in last place) to euler roll, pitch, yaw (rad).
    quat = [x, y, z, w]

    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = arctan2(sinr_cosp, cosr_cosp) * 180 / pi

    sinp = 2 * (w * y - z * x)
    pitch = arcsin(sinp) * 180 / pi

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = arctan2(siny_cosp, cosy_cosp) * 180 / pi

    return roll, pitch, yaw


def timestamp_from_ros_time(ros_time: Time) -> float:
    """
    Convert a ROS Time object to a Python datetime timestamp.

    :param ros_time: A ROS Time object.
    :return: A Python datetime timestamp.
    """
    return ros_time.nanosec / 1e9 + ros_time.sec


def timestamp_to_ros_time(timestamp: float) -> Time:
    """
    Convert a Python datetime timestamp to a ROS Time object.

    :param timestamp: A Python datetime timestamp.
    :return: A ROS Time object.
    """
    # Extract seconds and nanoseconds from the timestamp
    seconds = int(timestamp)
    nanoseconds = int((timestamp - seconds) * 1e9)

    # Create a Time object from seconds and nanoseconds
    return Time(seconds=seconds, nanoseconds=nanoseconds)


def consistent_random(string: str, min_value: float, max_value: float) -> float:
    """
    Consistently generate a random number between min and max based on the input string.

    :param string: The input string to generate the random number.
    :param min_value: The minimum value of the random number.
    :param max_value: The maximum value of the random number.
    """
    # -> Use hash() function to hash the input string
    hash_value = hash(string)
    # -> Normalize hash value to be between min and max
    random_value = min_value + (hash_value % 1000000) / 1000000 * (max_value - min_value)

    return random_value


def deep_compare(obj1, obj2):
    """
    Deep comparison function for various types, including NumPy arrays, Pandas DataFrames, and Pandas Series.

    :param obj1: First object to compare.
    :param obj2: Second object to compare.

    :return: True if the objects are deeply equal, False otherwise.
    """
    # If both objects are Pandas DataFrames, Series, or NumPy arrays, use appropriate comparison methods
    if isinstance(obj1, (pd.DataFrame, pd.Series, np.ndarray)) and isinstance(obj2,
                                                                              (pd.DataFrame, pd.Series, np.ndarray)):
        if isinstance(obj1, pd.DataFrame) and isinstance(obj2, pd.DataFrame):
            return obj1.equals(obj2)
        elif isinstance(obj1, pd.Series) and isinstance(obj2, pd.Series):
            return obj1.equals(obj2)
        elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            return np.array_equal(obj1, obj2)

    # If the objects are not of the same type, they are not equal
    if type(obj1) != type(obj2):
        return False

    # If the objects are lists or tuples, recursively compare each element
    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        for item1, item2 in zip(obj1, obj2):
            if not deep_compare(item1, item2):
                return False
        return True

    # If the objects are dictionaries, recursively compare each value
    if isinstance(obj1, dict):
        if len(obj1) != len(obj2):
            return False
        for key in obj1:
            if key not in obj2 or not deep_compare(obj1[key], obj2[key]):
                return False
        return True

    # For other types (e.g., int, float, str), use the equality operator
    return obj1 == obj2


if __name__ == "__main__":
    # > Test deep_compare
    # > Dataframes
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    print(deep_compare(df1, df2))    # True

    dict1 = {"df": df1}
    dict2 = {"df": df2}

    print(deep_compare(dict1, dict2))    # True