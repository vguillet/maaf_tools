
from math import pi
from numpy import arctan2, arcsin
import networkx as nx
import numpy as np
import pandas as pd
import json
import orjson

from pyproj import Transformer
from rclpy.time import Time


def convert_graph_gps_to_ecef(graph_data: dict) -> dict:
    # Create a transformer to convert from WGS84 (EPSG:4326) to ECEF (EPSG:4978)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    # Iterate over each node in the graph.
    for node in graph_data["graph"]["nodes"]:
        metadata = node.get("metadata", {})
        lat = metadata.get("latitude")
        lon = metadata.get("longitude")
        alt = metadata.get("altitude", 0.0)  # Default altitude to 0 if not provided.

        # Only perform conversion if latitude and longitude are provided.
        if lat is not None and lon is not None:
            # Transformer expects (longitude, latitude, altitude)
            x, y, z = transformer.transform(lon, lat, alt)
            node["pos"] = (x, y, z)

    return graph_data


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


def custom_encoder(data):
    if isinstance(data, np.float64):
        return float(data)  # Convert to Python float
    raise TypeError(f"Type {type(data)} not serializable")


def dumps(data):
    return orjson.dumps(data, default=custom_encoder).decode("utf-8")


def loads(data):
    return orjson.loads(data)


if __name__ == "__main__":
    # > Test deep_compare
    # > Dataframes
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    print(deep_compare(df1, df2))    # True

    dict1 = {"df": df1}
    dict2 = {"df": df2}

    print(deep_compare(dict1, dict2))    # True