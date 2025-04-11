
##################################################################################################################

from dataclasses import dataclass, fields, field
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.datastructures.MaafList import MaafList

    # -> Organisation
    from maaf_tools.datastructures.organisation.Organisation import Organisation
    from maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_tools.datastructures.organisation.AllocationSpecification import AllocationSpecification
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.datastructures.organisation.MOISEPlus.StructuralSpecification import StructuralSpecification
    from maaf_tools.datastructures.organisation.MOISEPlus.FunctionalSpecification import FunctionalSpecification
    from maaf_tools.datastructures.organisation.MOISEPlus.DeonticSpecification import DeonticSpecification

    # -> Fleet
    from maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.datastructures.agent.Agent import Agent
    from maaf_tools.datastructures.agent.Fleet import Fleet
    
    # -> Tasklog
    from maaf_tools.datastructures.task.Task import Task
    from maaf_tools.datastructures.task.TaskGraph import TaskGraph
    from maaf_tools.datastructures.task.TaskLog import TaskLog
    
except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.datastructures.MaafList import MaafList

    # -> Organisation
    from maaf_tools.maaf_tools.datastructures.organisation.Organisation import Organisation
    from maaf_tools.maaf_tools.datastructures.organisation.RoleAllocation import RoleAllocation
    from maaf_tools.maaf_tools.datastructures.organisation.AllocationSpecification import AllocationSpecification
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.StructuralSpecification import StructuralSpecification
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.FunctionalSpecification import FunctionalSpecification
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.DeonticSpecification import DeonticSpecification

    # -> Fleet
    from maaf_tools.maaf_tools.datastructures.agent.AgentState import AgentState
    from maaf_tools.maaf_tools.datastructures.agent.Plan import Plan
    from maaf_tools.maaf_tools.datastructures.agent.Agent import Agent
    from maaf_tools.maaf_tools.datastructures.agent.Fleet import Fleet
    
    # -> Tasklog
    from maaf_tools.maaf_tools.datastructures.task.Task import Task
    from maaf_tools.maaf_tools.datastructures.task.TaskGraph import TaskGraph
    from maaf_tools.maaf_tools.datastructures.task.TaskLog import TaskLog

##################################################################################################################

types_str_class_dict = {
    # > Base
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "None": type(None),

    # > Libraries
    "datetime": datetime,

    # Pandas
    "pd.DataFrame": pd.DataFrame,
    "pd.Series": pd.Series,

    # Numpy
    "np.ndarray": np.ndarray,
    "np.array": np.array,
    "np.matrix": np.matrix,
    "np.float64": np.float64,
    "np.int64": np.int64,

    # > Maaf
    # "MaafItem": MaafItem,
    # "MaafList": MaafList,

    # Fleet
    "AgentState": AgentState,
    "Plan": Plan,
    "Agent": Agent,
    "Fleet": Fleet,

    # Tasklog
    "Task": Task,
    "TaskGraph": TaskGraph,
    "TaskLog": TaskLog,

    # Organisation
    "Organisation": Organisation,
    "RoleAllocation": RoleAllocation,
    "MoiseModel": MoiseModel,
    "AllocationSpecification": AllocationSpecification,
    "StructuralSpecification": StructuralSpecification,
    "FunctionalSpecification": FunctionalSpecification,
    "DeonticSpecification": DeonticSpecification
}

# -> Create a dictionary with class types as keys and their string representation as values
types_class_str_dict = {v: k for k, v in types_str_class_dict.items()}


def asdict(item, fields_exclusion_lst: list = []) -> dict:
    """
    Create a dictionary containing the fields of the maaflist data class instance with their current values.
    All hidden fields with __ in the name are excluded from the dictionary.

    :param item: The data class instance to create a dictionary from.
    :param fields_exclusion_lst: A list of fields to exclude from the dictionary. Default is an empty list.

    :return: A dictionary with field names as keys and current values.
    """

    def recursive_get_field_type_and_value(item, field) -> (dict, dict):
        """
        Recursive function to get the field type and value.

        :param item: The item to get the field type and value from.
        :param field: The field to get the type and value from.

        :return: The field type and value dictionaries.
        """

        def get_item_type_and_value(item):
            try:
                item_type = types_class_str_dict[type(item)]
            except KeyError:
                raise KeyError(f"Type {type(item)} not found in types_dict")

            # ----- Dataclasses
            if hasattr(item, "asdict"):
                return item.asdict(), item_type

            # ----- Class specific
            elif isinstance(item, pd.DataFrame):
                return item.to_dict(orient="index"), item_type

            # ----- Base
            else:
                return item, item_type

        # ----- Mutables
        # > If the field is a mutable sequence, apply recursion
        if isinstance(getattr(item, field.name), list):
            fields_dict[field.name] = []
            fields_types[field.name] = []

            for subitem in getattr(item, field.name):
                item_dict, item_type = get_item_type_and_value(subitem)
                fields_dict[field.name].append(item_dict)
                fields_types[field.name].append(item_type)

            return fields_dict, fields_types

        # > If the field is a dictionary, apply recursion
        elif isinstance(getattr(item, field.name), dict):
            fields_dict[field.name] = {}
            fields_types[field.name] = {}

            for key, subitem in getattr(item, field.name).items():
                item_dict, item_type = get_item_type_and_value(subitem)
                fields_dict[field.name][key] = item_dict
                fields_types[field.name][key] = item_type

            return fields_dict, fields_types

        # ----- Base
        # > Else, add the field value to the dictionary
        else:
            fields_dict[field.name], fields_types[field.name] = get_item_type_and_value(getattr(item, field.name))
            return fields_dict, fields_types

    # -> Get the fields of the item class
    item_fields = fields(item)

    # -> Exclude all fields with __ in the name
    item_fields = [f for f in item_fields if "__" not in f.name]

    # -> Exclude all fields in the exclusion list
    item_fields = [f for f in item_fields if f.name not in fields_exclusion_lst]

    # -> Create a dictionary with field names as keys and the value as values
    fields_dict = {}

    # -> Create a dictionary with field names as keys and the types as values
    fields_types = {}

    for field in item_fields:
        fields_dict, fields_types = recursive_get_field_type_and_value(item, field)

    # -> Add field types to fields_dict
    fields_dict["field_types"] = fields_types

    return fields_dict


def from_dict(cls, item_dict: dict, fields_exclusion_lst: list = [], partial: bool = False):
    """
    Convert a dictionary to an item.

    :param item_dict: The dictionary representation of the item.
    :param partial: Whether to allow creation from a dictionary with missing fields.

    :return: An item object
    """

    # def recursive_create_item(field, field_type: dict, partial: bool) -> object:
    #     """
    #     Recursive function to create the item object.
    #
    #     :param field: The dictionary representation of the item.
    #     :param field_type: The types of the fields in the dictionary.
    #
    #     :return: The item object.
    #     """
    #
    #     # pprint(f"--------------------------------------- DOING {field_type}: {field}")
    #
    #     if type(field_type) == str:
    #         field_type_string = field_type
    #     else:
    #         field_type_string = types_class_str_dict[type(field_type)]
    #
    #     field_cls = types_str_class_dict[field_type_string]
    #
    #     if hasattr(field_cls, "from_dict") and field_cls:
    #         try:
    #             return field_cls.from_dict(item_dict=field, partial=partial)
    #         except:
    #             return field_cls.from_dict(field)
    #
    #     # ----- Mutables
    #     elif isinstance(field, list):
    #         return [recursive_create_item(subitem, field_type[i], partial) for i, subitem in enumerate(field)]
    #
    #     elif isinstance(field, dict):
    #         return {key: recursive_create_item(subitem, field_type[key], partial) for key, subitem in field.items()}
    #
    #     # ----- Class specific
    #     elif field_type_string == "pd.DataFrame":
    #         return pd.DataFrame.from_dict(field, orient="index")
    #
    #     elif field_type_string == "None":
    #         return None
    #
    #     # ----- Base
    #     else:
    #         return field_cls(field)

    def recursive_create_item(field, field_type, partial: bool) -> object:
        # Determine the type string first.
        if isinstance(field_type, str):
            field_type_string = field_type
        else:
            field_type_string = types_class_str_dict[type(field_type)]
        field_cls = types_str_class_dict[field_type_string]

        # Specific branch for pandas DataFrame:
        if field_type_string == "pd.DataFrame":
            return pd.DataFrame.from_dict(field, orient="index")

        # Specific branch for a None type:
        elif field_type_string == "None":
            return None

        # Then the generic from_dict dispatch:
        if hasattr(field_cls, "from_dict") and field_cls:
            try:
                return field_cls.from_dict(item_dict=field, partial=partial)
            except Exception:
                # Consider logging the exception details here for debugging.
                return field_cls.from_dict(field)

        # Mutables and built-in types:
        if isinstance(field, list):
            return [recursive_create_item(subitem, field_type[i], partial) for i, subitem in enumerate(field)]
        elif isinstance(field, dict):
            return {key: recursive_create_item(subitem, field_type[key], partial) for key, subitem in field.items()}
        else:
            return field_cls(field)

    # -> Get the fields of the item class
    item_fields = fields(cls)

    # -> Exclude all fields with __ in the name
    item_fields = [f for f in item_fields if "__" not in f.name]

    # -> Exclude all fields in the exclusion list
    item_fields = [f for f in item_fields if f.name not in fields_exclusion_lst]

    # -> Extract field names from the fields
    field_names = {field.name for field in item_fields}

    # -> Extract types from the dictionary
    try:
        fields_types = item_dict.pop("field_types")
    except KeyError:
        raise ValueError("The input dictionary is missing the 'field_types' key.")

    if not partial:
        # -> Check if all required fields are present in the dictionary
        if not field_names.issubset(set(item_dict.keys())):
            raise ValueError(f"!!! {cls.__name__} creation from dictionary failed: {cls.__name__} dictionary is missing required fields: {field_names - set(item_dict.keys())} !!!")

    else:
        # > Remove fields not present in the dictionary
        item_fields = [field for field in item_fields if field.name in item_dict]

    field_values = {}

    for field in item_fields:
        field_values[field.name] = recursive_create_item(item_dict[field.name], fields_types[field.name], partial)

    # -> Create and return an item object
    return cls(**field_values)
